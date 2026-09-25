"""向量快照的 Redis 存取：write-once 发布原子与查询侧读取面（extras: ``redis``）。

键域（四个键模板是存储契约的一部分，缺省与演进来源项目的生产键域逐字节一致）::

    {prefix}:registry:vector:{snapshot_id}        # hash: "{kind}:{id}" → 向量
    {prefix}:registry:vector-meta:{snapshot_id}   # 元数据字段表
    {prefix}:registry:vector-of:{owner_version}   # owner_version → snapshot 指针
    {prefix}:registry:vector-lock:{snapshot_id}   # single-flight 锁（SET NX）

**write-once**：发布前 ``read_meta`` 预检，同 id 仅续 TTL、零重嵌（发布热路径可能
高频触发，必须只读元数据）；不一致或缺失才在 single-flight 锁内嵌入写入。写入即挂
TTL、republish 顺延；TTL 与宿主的版本保留期同参（由调用方传 ``retention_seconds``）。
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

log = logging.getLogger("agent_retrieval.stores")

#: single-flight 锁的租约上限：拿到锁的进程在这个时间内完成嵌入与写入；进程崩溃时
#: 锁随 PX 过期自动释放，不会永久阻塞后来者。嵌入一批白名单语料（数千条内）远够。
LOCK_LEASE_SECONDS = 300

_VECTOR_KEY = "{prefix}:registry:vector:{snapshot_id}"
_META_KEY = "{prefix}:registry:vector-meta:{snapshot_id}"
_POINTER_KEY = "{prefix}:registry:vector-of:{owner_version}"
_LOCK_KEY = "{prefix}:registry:vector-lock:{snapshot_id}"


def vector_snapshot_id(
    owner_version: str,
    identity: tuple[str, str, str],
    corpus_hash: str,
) -> str:
    """不可变快照身份 = hash(owner_version + 嵌入器身份三元组 + 语料 hash)。

    任一因子变化（候选集、换嵌入模型/版本、语料文本）都产生新 id → 全量重嵌；
    全部一致 → 同 id → write-once 只续 TTL。
    """
    provider, model, model_version = identity
    raw = "|".join((owner_version, provider, model, model_version, corpus_hash))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def corpus_hash(texts: Mapping[str, str]) -> str:
    """白名单语料文本的 hash：键序（``{kind}:{id}``）固定，拼接序因此确定。"""
    joined = "\x00".join(f"{key}\x01{texts[key]}" for key in sorted(texts))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _vector_key(prefix: str, snapshot_id: str) -> str:
    return _VECTOR_KEY.format(prefix=prefix, snapshot_id=snapshot_id)


def _meta_key(prefix: str, snapshot_id: str) -> str:
    return _META_KEY.format(prefix=prefix, snapshot_id=snapshot_id)


def _pointer_key(prefix: str, owner_version: str) -> str:
    return _POINTER_KEY.format(prefix=prefix, owner_version=owner_version)


def _lock_key(prefix: str, snapshot_id: str) -> str:
    return _LOCK_KEY.format(prefix=prefix, snapshot_id=snapshot_id)


class RedisVectorStore:
    """异步 Redis 客户端上的 write-once 向量快照存储。

    ``client`` 优先于 ``redis_url``：传入即复用（连接生命周期归调用方，可用测试替身）；
    只传 ``redis_url`` 时惰性建连（``decode_responses=True``），由 :meth:`aclose` 关闭。
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        redis_url: str | None = None,
        key_prefix: str = "ar",
        lock_lease_seconds: int = LOCK_LEASE_SECONDS,
    ) -> None:
        if client is None and not redis_url:
            raise ValueError("RedisVectorStore 需要 client 或 redis_url 之一")
        self._client = client
        self._redis_url = redis_url
        self.prefix = key_prefix
        self._lock_lease_seconds = lock_lease_seconds

    async def _get_client(self) -> Any:
        if self._client is None:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._client

    async def aclose(self) -> None:
        """关闭惰性建连的客户端；调用方注入的客户端由调用方自行管理。"""
        if self._client is not None and self._redis_url is not None:
            await self._client.aclose()
            self._client = None

    # ---- write-once 预检与锁 -------------------------------------------------

    async def read_meta(self, snapshot_id: str) -> dict[str, str] | None:
        """快照元数据预检（write-once 判据）：缺失返回 ``None``。"""
        client = await self._get_client()
        raw = await client.get(_meta_key(self.prefix, snapshot_id))
        if raw is None:
            return None
        meta = json.loads(raw)
        return meta if isinstance(meta, dict) else None

    async def extend_ttl(
        self, snapshot_id: str, owner_version: str, retention_seconds: int,
    ) -> None:
        """同 id 重复发布的唯一动作：仅续 TTL（向量 + meta + 指针），零重嵌。"""
        if retention_seconds <= 0:
            return
        client = await self._get_client()
        pipeline = client.pipeline(transaction=True)
        for key in (
            _vector_key(self.prefix, snapshot_id),
            _meta_key(self.prefix, snapshot_id),
            _pointer_key(self.prefix, owner_version),
        ):
            pipeline.expire(key, retention_seconds)
        await pipeline.execute()

    async def acquire_lock(self, snapshot_id: str) -> bool:
        """single-flight：Redis SET NX + PX 租约，锁粒度 = vector_snapshot_id。"""
        client = await self._get_client()
        return bool(await client.set(
            _lock_key(self.prefix, snapshot_id),
            "publishing",
            nx=True,
            px=self._lock_lease_seconds * 1000,
        ))

    async def release_lock(self, snapshot_id: str) -> None:
        client = await self._get_client()
        await client.delete(_lock_key(self.prefix, snapshot_id))

    # ---- 发布与读取 ----------------------------------------------------------

    async def publish_vectors(
        self,
        *,
        snapshot_id: str,
        owner_version: str,
        vectors: Mapping[str, Sequence[float]],
        meta: Mapping[str, str],
        retention_seconds: int,
    ) -> None:
        """锁内写入：向量 hash + meta + ``owner_version → id`` 指针，写入即挂 TTL。"""
        client = await self._get_client()
        pipeline = client.pipeline(transaction=True)
        vector_key = _vector_key(self.prefix, snapshot_id)
        pipeline.delete(vector_key)
        if vectors:
            pipeline.hset(vector_key, mapping={
                key: json.dumps([float(item) for item in value])
                for key, value in vectors.items()
            })
        meta_key = _meta_key(self.prefix, snapshot_id)
        pipeline.set(meta_key, json.dumps(dict(meta)))
        pipeline.set(_pointer_key(self.prefix, owner_version), snapshot_id)
        if retention_seconds > 0:
            for key in (vector_key, meta_key, _pointer_key(self.prefix, owner_version)):
                pipeline.expire(key, retention_seconds)
        await pipeline.execute()

    async def load_snapshot_vectors(
        self, owner_version: str, item_keys: Sequence[str],
    ) -> dict[str, tuple[float, ...]]:
        """查询侧读取面：按 ``owner_version`` 找快照，取 ``{kind}:{id}`` 子集的向量。

        快照缺席（发布期嵌入失败 / TTL 过期 / Redis 不可用）返回空 dict——向量路
        缺席，调用方降级 BM25，绝不阻塞检索。
        """
        client = await self._get_client()
        try:
            snapshot_id = await client.get(_pointer_key(self.prefix, owner_version))
            if not snapshot_id:
                return {}
            raw = await client.hmget(_vector_key(self.prefix, snapshot_id), list(item_keys))
            vectors: dict[str, tuple[float, ...]] = {}
            for key, payload in zip(item_keys, raw):
                if payload is None:
                    continue
                vectors[key] = tuple(float(item) for item in json.loads(payload))
            return vectors
        except Exception as exc:  # Redis 语料不可用 → 向量缺席降级
            log.warning("[vector] 读取向量快照失败，向量路按缺席处理: %s", exc)
            return {}
