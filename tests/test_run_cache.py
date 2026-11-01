"""查询向量缓存与降级冻结的契约测试（自 evochat test_vector_cache.py 迁移）。

- **Run 内查询向量缓存**：同一 Run 同一查询文本只嵌一次——Run 内重放/跨 worker 恢复
  必命中（ContextVar 仅作进程内一级缓存，主存储是后端）；缓存 key 写入后不可变。
- **入口分键 + 身份隔离（B6）**：查询/语料两入口按 ``entry`` 分键；向量缓存键含
  嵌入器身份摘要段——同 Run 换嵌入器配置互不污染；``embedder_identity=None`` 是
  bypass 模式（零向量缓存读写，但失败冻结语义保留）。
- **降级冻结**：查询嵌入首次失败即写 Run 级降级标记，恢复**不重试**嵌入——降级决定
  随 Run 持久化，重放确定。冻结与身份模式正交（两套键两种生命周期）。
- run_id 显式注入（原宿主经 tracer ContextVar 提供，翻译点收在宿主装配层）；
  向量路开关是宿主概念（embedder=None 即 BM25 现状），本包不做 flag 门控。

Redis 交互全走内存替身（实现实际消费的同步客户端面），不依赖真实 Redis。
"""

import hashlib
import json

import pytest

from agent_retrieval.core.ports import MockEmbedder, QueryEmbeddingError
from agent_retrieval.run.cache import (
    clear_run_cache,
    embedder_identity_digest,
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

#: 缓存断言类用例的装配身份（非默认 flags 之外的常规四元：三元 legacy + 空 flags）。
IDENTITY = ("mock-p", "mock-m", "v1", "")


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
    """Run 内同 query 二次检索 ``embed_query`` 计数 == 1（缓存断言类：显式传身份）。"""
    raw = _CountingEmbedder()

    embedder = query_embedder(backend=backend, embedder=raw, run_id=RUN_ID, embedder_identity=IDENTITY)
    assert embedder is not None
    first = embedder.embed_query("推演剧本编排")
    second = embedder.embed_query("推演剧本编排")

    assert raw.embed_query_calls == ["推演剧本编排"]
    assert first == second


def test_worker_recovery_hits_backend_cache_without_reembedding(backend):
    """模拟换 worker——ContextVar 一级缓存清空 + 后端缓存留存 → 不重嵌。"""
    raw = _CountingEmbedder()

    embedder = query_embedder(backend=backend, embedder=raw, run_id=RUN_ID, embedder_identity=IDENTITY)
    assert embedder is not None
    embedder.embed_query("推演剧本编排")
    # 换 worker：进程内一级缓存随进程消失；后端里 Run 域的缓存留存。
    clear_run_cache()

    embedder = query_embedder(backend=backend, embedder=raw, run_id=RUN_ID, embedder_identity=IDENTITY)
    assert embedder is not None
    recovered = embedder.embed_query("推演剧本编排")

    assert raw.embed_query_calls == ["推演剧本编排"]  # 恢复后未重嵌
    query_key = backend.query_key(RUN_ID, embedder_identity_digest(IDENTITY), "q", "推演剧本编排")
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


def test_query_and_corpus_entries_use_disjoint_cache_keys(backend):
    """q/c 分键（B6 契约反转）：同 run 同文本两入口各自嵌入、互不污染。

    旧契约两入口共享同键（先查过的文本再进语料不再嵌）——与入口分键相反；
    新键域 ``entry ∈ {"q","c"}`` 后查询缓存不再喂给语料路。
    """
    raw = _CountingEmbedder()

    embedder = query_embedder(backend=backend, embedder=raw, run_id=RUN_ID, embedder_identity=IDENTITY)
    assert embedder is not None
    embedder.embed_query("共享文本")
    vectors = embedder.embed_corpus(["共享文本", "新文本"])

    assert raw.embed_query_calls == ["共享文本"]
    assert raw.embed_corpus_calls == ["共享文本", "新文本"]  # q 键对 c 入口不可见：语料路自嵌
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

    # 向量缓存键新形状（逐字节契约）：身份摘要段 + entry 段夹住 sha256(text)。
    digest = embedder_identity_digest(IDENTITY)
    text_digest = hashlib.sha256("别的查询".encode()).hexdigest()
    assert backend.query_key(RUN_ID, digest, "q", "别的查询") == \
        f"test-prefix:run:run-cache-1:{digest}:q:{text_digest}"
    assert backend.query_key(RUN_ID, digest, "c", "别的查询").endswith(f":c:{text_digest}")


def test_memory_backend_serves_the_same_contract():
    """内存替身走同一装配入口：无 Redis 的调用方获得同语义的缓存与冻结。"""
    from agent_retrieval.run.cache import MemoryBackend

    clear_run_cache()
    try:
        raw = _CountingEmbedder()
        backend = make_backend(client=MemoryBackend(), key_prefix="mem", retention_seconds=0)

        embedder = query_embedder(backend=backend, embedder=raw, run_id=RUN_ID, embedder_identity=IDENTITY)
        assert embedder is not None
        embedder.embed_query("同一文本")
        embedder.embed_query("同一文本")
        assert raw.embed_query_calls == ["同一文本"]
    finally:
        clear_run_cache()


def test_same_run_same_text_different_identities_stay_isolated(backend):
    """同 run 同入口同文本、不同嵌入器身份：互不污染（键含身份摘要段）。"""
    raw_small = _CountingEmbedder(dimensions=8)
    raw_large = _CountingEmbedder(dimensions=16)

    small = query_embedder(
        backend=backend, embedder=raw_small, run_id=RUN_ID,
        embedder_identity=("mock-p", "mock-m", "v8", ""),
    )
    large = query_embedder(
        backend=backend, embedder=raw_large, run_id=RUN_ID,
        embedder_identity=("mock-p", "mock-m", "v16", ""),
    )
    assert small is not None and large is not None

    v_small = small.embed_query("同一文本")
    v_large = large.embed_query("同一文本")

    # 两侧都真嵌了（身份 A 的缓存不喂给身份 B），向量各归各的语义。
    assert raw_small.embed_query_calls == ["同一文本"]
    assert raw_large.embed_query_calls == ["同一文本"]
    assert len(v_small) == 8
    assert len(v_large) == 16

    # 再取各自命中：身份维度持续隔离。
    assert small.embed_query("同一文本") == v_small
    assert large.embed_query("同一文本") == v_large
    assert raw_small.embed_query_calls == ["同一文本"]
    assert raw_large.embed_query_calls == ["同一文本"]


def test_backend_hit_writes_back_to_l1_cache(backend):
    """后端命中回写一级缓存：只清一级（换 worker 模拟）后语料路仍命中且不落空。

    旧实现后端命中只跳过重嵌、不回写一级——返回列表按一级缓存取值时落空。
    """
    raw = _CountingEmbedder()

    embedder = query_embedder(backend=backend, embedder=raw, run_id=RUN_ID, embedder_identity=IDENTITY)
    assert embedder is not None
    embedder.embed_corpus(["长语料文本"])
    assert raw.embed_corpus_calls == ["长语料文本"]

    clear_run_cache()  # 只清一级缓存；后端键留存

    recovered = query_embedder(backend=backend, embedder=raw, run_id=RUN_ID, embedder_identity=IDENTITY)
    assert recovered is not None
    vectors = recovered.embed_corpus(["长语料文本"])

    assert raw.embed_corpus_calls == ["长语料文本"]  # 后端命中：未重嵌
    assert len(vectors) == 1 and isinstance(vectors[0], tuple)  # 回写修复：返回不落空


def test_bypass_identity_none_skips_vector_cache_but_keeps_freeze_on_query(backend):
    """bypass（embedder_identity=None）：零向量缓存读写；query 失败冻结逐项完好（R4-P1）。"""
    raw = _CountingEmbedder()
    embedder = query_embedder(backend=backend, embedder=raw, run_id=RUN_ID)
    assert embedder is not None

    embedder.embed_query("任意目标")
    embedder.embed_query("任意目标")
    assert raw.embed_query_calls == ["任意目标", "任意目标"]  # 直通 inner：不查缓存
    assert backend.client.strings == {}  # 零向量缓存读写：后端无任何键

    failing = query_embedder(backend=backend, embedder=_CountingEmbedder(fail_on="api 断"), run_id=RUN_ID)
    assert failing is not None
    with pytest.raises(QueryEmbeddingError):
        failing.embed_query("目标")

    assert backend.client.get(backend.degraded_key(RUN_ID)) is not None  # 冻结标记写入
    assert take_degradation_count() == 1  # 降级计数
    # 同 Run 再装配被阻断（冻结与 bypass 正交：bypass 只免缓存、不免冻结）。
    assert query_embedder(backend=backend, embedder=_CountingEmbedder(), run_id=RUN_ID) is None


def test_bypass_identity_none_keeps_freeze_on_corpus(backend):
    """bypass 模式 embed_corpus 失败同样冻结：标记 + 计数 + 再装配阻断逐项断言。"""
    raw = _CountingEmbedder(fail_on="批量断")
    embedder = query_embedder(backend=backend, embedder=raw, run_id=RUN_ID)
    assert embedder is not None

    with pytest.raises(QueryEmbeddingError):
        embedder.embed_corpus(["a", "b"])

    assert backend.client.get(backend.degraded_key(RUN_ID)) is not None
    assert take_degradation_count() == 1
    assert query_embedder(backend=backend, embedder=_CountingEmbedder(), run_id=RUN_ID) is None
    assert raw.embed_corpus_calls == ["a", "b"]  # 直通 inner：零缓存分拣


def test_pipe_collision_identities_get_distinct_digests_and_stay_isolated(backend):
    """跨字段竖线碰撞：``("a|b","c","v","")`` vs ``("a","b|c","v","")`` 摘要不同且隔离。

    ``"|".join`` 摘要对此碰撞双解（R4-P1）；JSON 规范序列化消除。隔离在清一级缓存
    （后端键留存）前后均成立。
    """
    id_left = ("a|b", "c", "v", "")
    id_right = ("a", "b|c", "v", "")
    assert embedder_identity_digest(id_left) != embedder_identity_digest(id_right)

    raw_left = _CountingEmbedder(dimensions=8)
    raw_right = _CountingEmbedder(dimensions=16)
    left = query_embedder(backend=backend, embedder=raw_left, run_id=RUN_ID, embedder_identity=id_left)
    right = query_embedder(backend=backend, embedder=raw_right, run_id=RUN_ID, embedder_identity=id_right)
    assert left is not None and right is not None

    v_left = left.embed_query("同一文本")
    v_right = right.embed_query("同一文本")
    assert raw_left.embed_query_calls == ["同一文本"]
    assert raw_right.embed_query_calls == ["同一文本"]  # 碰撞对未共享缓存

    clear_run_cache()  # 清一级：后端键留存，隔离必须继续成立

    left_after = query_embedder(backend=backend, embedder=raw_left, run_id=RUN_ID, embedder_identity=id_left)
    right_after = query_embedder(backend=backend, embedder=raw_right, run_id=RUN_ID, embedder_identity=id_right)
    assert left_after is not None and right_after is not None
    assert left_after.embed_query("同一文本") == v_left
    assert right_after.embed_query("同一文本") == v_right
    assert raw_left.embed_query_calls == ["同一文本"]
    assert raw_right.embed_query_calls == ["同一文本"]  # 后端命中按摘要分键：仍零交叉
