"""查询向量缓存与降级冻结的契约测试（自 evochat test_vector_cache.py 迁移）。

- **Run 内查询向量缓存**：同一 Run 同一查询文本只嵌一次——Run 内重放/跨 worker 恢复
  必命中（ContextVar 仅作进程内一级缓存，主存储是后端）；缓存 key 写入后不可变。
- **降级冻结**：查询嵌入首次失败即写 Run 级降级标记，恢复**不重试**嵌入——降级决定
  随 Run 持久化，重放确定。
- run_id 显式注入（原宿主经 tracer ContextVar 提供，翻译点收在宿主装配层）；
  向量路开关是宿主概念（embedder=None 即 BM25 现状），本包不做 flag 门控。

Redis 交互全走内存替身（实现实际消费的同步客户端面），不依赖真实 Redis。
"""

import json

import pytest

from agent_retrieval.core.ports import MockEmbedder, QueryEmbeddingError
from agent_retrieval.run.cache import (
    clear_run_cache,
    make_backend,
    query_embedder,
    take_degradation_count,
)


class _FakeRedis:
    """同步 redis 客户端的最小替身：只实现查询缓存面实际用到的方法。"""

    def __init__(self):
        self.strings: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    def get(self, key):
        return self.strings.get(key)

    def set(self, key, value, ex=None, nx=False):
        if nx and key in self.strings:
            return False
        self.strings[key] = value
        if ex:
            self.ttls[key] = ex
        return True

    def delete(self, *keys):
        for key in keys:
            self.strings.pop(key, None)
            self.ttls.pop(key, None)
        return len(keys)


RUN_ID = "run-cache-1"


@pytest.fixture()
def backend():
    """替身缓存后端 + 隔离的进程内一级缓存。"""
    fake = _FakeRedis()
    backend = make_backend(client=fake, retention_seconds=7777, key_prefix="test-prefix")
    clear_run_cache()
    yield backend
    clear_run_cache()


class _CountingEmbedder:
    """嵌入调用计数的替身：失败模式由 ``fail_on`` 控制。"""

    def __init__(self, *, fail_on: str | None = None, dimensions: int = 8):
        self.embed_query_calls: list[str] = []
        self.embed_corpus_calls: list[str] = []
        self._fail_on = fail_on
        self._inner = MockEmbedder(dimensions=dimensions)

    def embed_corpus(self, texts):
        self.embed_corpus_calls.extend(texts)
        if self._fail_on:
            raise QueryEmbeddingError(f"corpus 断：{self._fail_on}")
        return [self._inner.embed_query(text) for text in texts]

    def embed_query(self, text):
        self.embed_query_calls.append(text)
        if self._fail_on:
            raise QueryEmbeddingError(f"query 断：{self._fail_on}")
        return self._inner.embed_query(text)


def test_query_embedder_returns_none_without_run_id(backend):
    """无 Run 域不构造向量路：缓存与降级标记都以 Run 为键域，没有 Run 域就没有缓存语义。"""
    assert query_embedder(backend=backend, embedder=_CountingEmbedder(), run_id=None) is None
    assert query_embedder(backend=backend, embedder=_CountingEmbedder(), run_id="") is None


def test_query_embedder_returns_none_without_embedder(backend):
    """嵌入器缺席（宿主 flag off / 未配置）：装配入口返回 None，BM25 现状。"""
    assert query_embedder(backend=backend, embedder=None, run_id=RUN_ID) is None


def test_same_query_within_run_embeds_once(backend):
    """Run 内同 query 二次检索 ``embed_query`` 计数 == 1。"""
    raw = _CountingEmbedder()

    embedder = query_embedder(backend=backend, embedder=raw, run_id=RUN_ID)
    assert embedder is not None
    first = embedder.embed_query("推演剧本编排")
    second = embedder.embed_query("推演剧本编排")

    assert raw.embed_query_calls == ["推演剧本编排"]
    assert first == second


def test_worker_recovery_hits_backend_cache_without_reembedding(backend):
    """模拟换 worker——ContextVar 一级缓存清空 + 后端缓存留存 → 不重嵌。"""
    raw = _CountingEmbedder()

    embedder = query_embedder(backend=backend, embedder=raw, run_id=RUN_ID)
    assert embedder is not None
    embedder.embed_query("推演剧本编排")
    # 换 worker：进程内一级缓存随进程消失；后端里 Run 域的缓存留存。
    clear_run_cache()

    embedder = query_embedder(backend=backend, embedder=raw, run_id=RUN_ID)
    assert embedder is not None
    recovered = embedder.embed_query("推演剧本编排")

    assert raw.embed_query_calls == ["推演剧本编排"]  # 恢复后未重嵌
    query_key = backend.query_key(RUN_ID, "推演剧本编排")
    assert backend.client.get(query_key) is not None
    assert backend.client.ttls.get(query_key) == 7777
    assert isinstance(recovered, tuple)


def test_first_embedding_failure_writes_run_degradation_marker(backend):
    """查询嵌入首次失败 → Run 级降级标记写入 + 降级计数（观测源）。"""
    raw = _CountingEmbedder(fail_on="api 断")

    embedder = query_embedder(backend=backend, embedder=raw, run_id=RUN_ID)
    assert embedder is not None
    with pytest.raises(QueryEmbeddingError):
        embedder.embed_query("任意目标")

    marker_key = backend.degraded_key(RUN_ID)
    assert backend.client.get(marker_key) is not None
    assert backend.client.ttls.get(marker_key) == 7777
    assert take_degradation_count() == 1


def test_degraded_run_never_retries_embedding(backend):
    """降级冻结——标记存在后 ``query_embedder`` 返回 None，恢复不重试嵌入。"""
    raw = _CountingEmbedder(fail_on="api 断")

    embedder = query_embedder(backend=backend, embedder=raw, run_id=RUN_ID)
    assert embedder is not None
    with pytest.raises(QueryEmbeddingError):
        embedder.embed_query("任意目标")

    # 模拟恢复后同 Run 再检索：装配入口看到降级标记直接返回 None（不构造嵌入器）。
    assert query_embedder(backend=backend, embedder=_CountingEmbedder(), run_id=RUN_ID) is None
    # 计数停在首次失败那一次——冻结路径零新增（恢复不重试嵌入，也不再重复记观测）。
    assert take_degradation_count() == 1

    assert raw.embed_query_calls == ["任意目标"]  # 只有首次那一次


def test_corpus_embedding_failure_degrades_the_same_way(backend):
    """embed_corpus 失败同样写标记冻结——两条嵌入入口同一降级决定。"""
    raw = _CountingEmbedder(fail_on="批量断")

    embedder = query_embedder(backend=backend, embedder=raw, run_id=RUN_ID)
    assert embedder is not None
    with pytest.raises(QueryEmbeddingError):
        embedder.embed_corpus(["a", "b"])

    assert backend.client.get(backend.degraded_key(RUN_ID)) is not None
    assert query_embedder(backend=backend, embedder=raw, run_id=RUN_ID) is None


def test_cached_corpus_embedding_reuses_query_cache(backend):
    """语料路与查询路共享同一 Run 缓存：先查过的文本再进语料不再嵌。"""
    raw = _CountingEmbedder()

    embedder = query_embedder(backend=backend, embedder=raw, run_id=RUN_ID)
    assert embedder is not None
    embedder.embed_query("共享文本")
    vectors = embedder.embed_corpus(["共享文本", "新文本"])

    assert raw.embed_query_calls == ["共享文本"]
    assert raw.embed_corpus_calls == ["新文本"]  # 共享文本命中缓存，未进批量
    assert len(vectors) == 2


def test_degradation_marker_shape_is_run_scoped_json(backend):
    """标记与缓存都在 Run 键域下：``{prefix}:run:{run_id}:qdegraded`` / ``:qvec:{hash}``。"""
    raw = _CountingEmbedder(fail_on="断")

    embedder = query_embedder(backend=backend, embedder=raw, run_id=RUN_ID)
    assert embedder is not None
    with pytest.raises(QueryEmbeddingError):
        embedder.embed_query("目标")

    marker = json.loads(backend.client.get(backend.degraded_key(RUN_ID)))
    assert marker["run_id"] == RUN_ID
    assert marker["reason"]

    query_key = backend.query_key(RUN_ID, "别的查询")
    assert query_key.startswith("test-prefix:run:run-cache-1:qvec:")


def test_memory_backend_serves_the_same_contract():
    """内存替身走同一装配入口：无 Redis 的调用方获得同语义的缓存与冻结。"""
    from agent_retrieval.run.cache import MemoryBackend

    clear_run_cache()
    try:
        raw = _CountingEmbedder()
        backend = make_backend(client=MemoryBackend(), key_prefix="mem", retention_seconds=0)

        embedder = query_embedder(backend=backend, embedder=raw, run_id=RUN_ID)
        assert embedder is not None
        embedder.embed_query("同一文本")
        embedder.embed_query("同一文本")
        assert raw.embed_query_calls == ["同一文本"]
    finally:
        clear_run_cache()
