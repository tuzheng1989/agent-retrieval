"""查询向量缓存与降级冻结：Run 生命周期的向量路装配（extras: ``redis``）。

查询侧调用点共用的唯一装配入口 :func:`query_embedder`：

- 无 Run 域 / 嵌入器不可用 / **降级已冻结** → ``None``（BM25 现状，缺席降级语义）；
- 否则返回 :class:`RunCachedEmbedder`：``embed_query`` 先进程内（ContextVar）一级
  缓存、再后端二级缓存（键域随 Run），都未命中才真正嵌入；
- **降级冻结**：嵌入首次失败即写 Run 级降级标记（与查询缓存同 TTL），此后该 Run
  恢复**不重试**嵌入——降级决定随 Run 持久化，重放确定。

Run 域（``run_id``）由调用方显式注入：库不读调用方的 ContextVar/tracer——检索调用
发生在很深的调用栈里，但"当前 Run 是什么"是宿主的执行模型概念，翻译点收在宿主的
装配层。
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from contextvars import ContextVar
from typing import Any

from agent_retrieval.core.ports import Embedder, QueryEmbeddingError

log = logging.getLogger("agent_retrieval.run")

#: 查询向量缓存与降级标记 TTL 的包缺省：调用方应按**宿主 Run 的 Redis 保留期**显式
#: 传参（Run 在后端侧的生命周期结束后，缓存也无从被命中——该值跟着宿主走）。
DEFAULT_RETENTION_SECONDS = 7 * 24 * 3600


class QueryCacheBackend:
    """查询缓存与降级标记的键域与读写（同步客户端协议）。

    查询侧调用点通常在同步函数栈里，异步客户端进不来；键域形状是契约：
    ``{prefix}:run:{run_id}:qvec:{sha256(text)}`` 与 ``{prefix}:run:{run_id}:qdegraded``。
    后端任何失败都不阻断检索：读失败按 miss、写失败 best-effort（缓存只是省往返的
    旁路，清空不影响正确性）。
    """

    def __init__(self, *, client: Any, key_prefix: str, retention_seconds: int):
        self.client = client
        self.key_prefix = key_prefix
        self.retention_seconds = retention_seconds

    def query_key(self, run_id: str, text: str) -> str:
        digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
        return f"{self.key_prefix}:run:{run_id}:qvec:{digest}"

    def degraded_key(self, run_id: str) -> str:
        return f"{self.key_prefix}:run:{run_id}:qdegraded"

    def get_vector(self, run_id: str, text: str) -> tuple[float, ...] | None:
        """缓存读：后端异常按 **miss** 处理——不可用时向量路直连嵌入器继续工作。"""
        try:
            raw = self.client.get(self.query_key(run_id, text))
        except Exception as exc:  # noqa: BLE001 — 缓存后端任何失败都不阻断检索
            log.warning("[vector] 查询缓存读取失败，按未命中处理: %s", exc)
            return None
        if raw is None:
            return None
        return tuple(float(item) for item in json.loads(raw))

    def set_vector(self, run_id: str, text: str, vector: tuple[float, ...]) -> None:
        """缓存写：best-effort——写失败只丢一次缓存收益，绝不外抛（嵌入已成功，检索照常）。"""
        if self.retention_seconds <= 0:
            return
        try:
            self.client.set(
                self.query_key(run_id, text),
                json.dumps([float(item) for item in vector]),
                ex=self.retention_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[vector] 查询缓存写入失败（忽略）: %s", exc)

    def mark_degraded(self, run_id: str, reason: str) -> None:
        """冻结标记写：best-effort——标记写不进时下个 Run 会重试嵌入（多一次失败尝试，
        无正确性影响）。"""
        if self.retention_seconds <= 0:
            return
        try:
            self.client.set(
                self.degraded_key(run_id),
                json.dumps({"run_id": run_id, "reason": reason[:200]}),
                ex=self.retention_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[vector] 降级标记写入失败（忽略）: %s", exc)

    def is_degraded(self, run_id: str) -> bool:
        """冻结标记读：后端异常按「未降级」——标记读不到时继续装配向量路，真正的
        嵌入失败会在 embed 链路里被捕获并冻结（fail-open 到重试，不是 fail-closed 到
        恒降级）。"""
        try:
            return self.client.get(self.degraded_key(run_id)) is not None
        except Exception as exc:  # noqa: BLE001
            log.warning("[vector] 降级标记读取失败，按未降级处理: %s", exc)
            return False


class MemoryBackend:
    """:class:`QueryCacheBackend` 的进程内 dict 替身（单测/无 Redis 场景）。

    接口只要求 ``get``/``set``（支持 ``ex=`` 关键字），与 Redis 客户端面同形；
    ``retention_seconds`` 在内存形态下不生效（进程生命周期即保留期）。
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> None:  # noqa: ARG002
        self._store[key] = value


def make_backend(
    *,
    client: Any | None = None,
    redis_url: str | None = None,
    key_prefix: str = "ar",
    retention_seconds: int = DEFAULT_RETENTION_SECONDS,
) -> QueryCacheBackend:
    """生产后端装配；测试注入替身客户端与 TTL（键域形状契约见 :class:`QueryCacheBackend`）。

    ``client`` 优先于 ``redis_url``；两者都缺省时惰性连本机默认 Redis（调用方应显式
    传参，本路径只为兜底存在）。
    """
    if client is None:
        import redis  # 同步客户端：查询侧调用点在同步函数栈内

        client = redis.Redis.from_url(redis_url or "redis://localhost:6379/0", decode_responses=True)
    return QueryCacheBackend(
        client=client,
        key_prefix=key_prefix,
        retention_seconds=retention_seconds,
    )


#: 进程内一级缓存：``{run_id: {query_key: vector}}``。**仅**是省一次后端往返的
#: 一级缓存——主存储是后端（换 worker 恢复必命中），清空它不影响正确性。
_RUN_VECTORS: ContextVar[dict[str, dict[str, tuple[float, ...]]] | None] = ContextVar(
    "agent_retrieval_run_query_vectors", default=None,
)

#: 降级计数（观测源）：调用点收口读走、写进自己的观测体系，或供事件发射判断
#: 「本次降级是否首次」。随 Run 走（ContextVar 作用域），深层调用栈 mutate 同一对象。
_DEGRADATIONS: ContextVar[int | None] = ContextVar("agent_retrieval_vector_degradations", default=None)


def clear_run_cache() -> None:
    """清空进程内一级缓存与降级计数（换 worker 模拟、测试隔离、Run 收尾）。"""
    _RUN_VECTORS.set(None)
    _DEGRADATIONS.set(None)


def record_vector_degradation() -> None:
    """记一次向量路降级（Run 域）。观测链：此处 → 宿主收口消费。"""
    current = _DEGRADATIONS.get() or 0
    _DEGRADATIONS.set(current + 1)


def take_degradation_count() -> int:
    """读走本上下文累积的降级次数（读取后清零，同 Run 下一轮重新计）。"""
    count = _DEGRADATIONS.get() or 0
    _DEGRADATIONS.set(0)
    return count


class RunCachedEmbedder:
    """:class:`Embedder` 的 Run 缓存包装：两级缓存 + 首败冻结 + 降级计数。"""

    def __init__(self, *, inner: Embedder, run_id: str, backend: QueryCacheBackend):
        self._inner = inner
        self._run_id = run_id
        self._backend = backend

    def embed_query(self, text: str) -> tuple[float, ...]:
        cache = _RUN_VECTORS.get()
        if cache is None:
            cache = {}
            _RUN_VECTORS.set(cache)
        vectors = cache.setdefault(self._run_id, {})
        key = self._backend.query_key(self._run_id, text)
        if key in vectors:
            return vectors[key]
        cached = self._backend.get_vector(self._run_id, text)
        if cached is not None:
            vectors[key] = cached
            return cached
        try:
            vector = self._inner.embed_query(text)
        except QueryEmbeddingError as exc:
            self._freeze(exc)
            raise
        vectors[key] = vector
        self._backend.set_vector(self._run_id, text, vector)
        return vector

    def embed_corpus(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        # 语料路与查询路共享同一 Run 缓存：Run 内语料基本不变，重放时命中缓存；
        # 先查过的文本不再进批量。
        cache = _RUN_VECTORS.get()
        if cache is None:
            cache = {}
            _RUN_VECTORS.set(cache)
        vectors = cache.setdefault(self._run_id, {})
        fresh: list[str] = []
        for text in texts:
            key = self._backend.query_key(self._run_id, text)
            if key not in vectors and self._backend.get_vector(self._run_id, text) is None:
                fresh.append(text)
        if fresh:
            try:
                embedded = self._inner.embed_corpus(fresh)
            except QueryEmbeddingError as exc:
                self._freeze(exc)
                raise
            for text, vector in zip(fresh, embedded):
                vectors[self._backend.query_key(self._run_id, text)] = vector
                self._backend.set_vector(self._run_id, text, vector)
        return [
            vectors[self._backend.query_key(self._run_id, text)]
            for text in texts
        ]

    def _freeze(self, exc: QueryEmbeddingError) -> None:
        """首次失败：写 Run 级降级标记（同 TTL）+ 计数。此后该 Run 不再重试嵌入。"""
        self._backend.mark_degraded(self._run_id, str(exc))
        record_vector_degradation()
        log.warning(
            "[vector] Run %s 查询嵌入失败，向量路冻结为 BM25（本 Run 内不重试）: %s",
            self._run_id, exc,
        )


def query_embedder(
    *,
    embedder: Embedder | None,
    run_id: str | None,
    backend: QueryCacheBackend | None = None,
) -> Embedder | None:
    """查询侧向量路的唯一装配入口。

    装配顺序（与降级语义一致）：无 Run 域 → ``None``；**降级已冻结** → ``None``
    （恢复不重试嵌入）；嵌入器缺席 → ``None``；否则返回 Run 缓存包装。

    「向量路总开关」是宿主概念：开关关闭时宿主传 ``embedder=None`` 即得 BM25 现状。
    """
    if not run_id:
        # 无 Run 域就没有缓存与降级标记的键位；缓存语义不可成立即不用向量路。
        return None
    backend = backend or make_backend()
    if backend.is_degraded(run_id):
        return None  # 降级冻结：恢复不重试嵌入
    if embedder is None:
        return None
    return RunCachedEmbedder(inner=embedder, run_id=run_id, backend=backend)
