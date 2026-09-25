"""RedisVectorStore 的契约测试：write-once、single-flight、缺席降级、键域形状。

Redis 交互全走异步替身客户端（实现 pipeline/GET/SET NX/HGETALL 实际消费的异步面），
不依赖真实 Redis。键域模板是存储契约：断言逐字节形状。
"""

import json

import pytest

from agent_retrieval.stores.redis_store import (
    LOCK_LEASE_SECONDS,
    RedisVectorStore,
    corpus_hash,
    vector_snapshot_id,
)


class _FakeAsyncRedis:
    """异步 redis 客户端的最小替身：字符串 + hash + pipeline(transaction=True)。"""

    def __init__(self):
        self.strings: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.ttls: dict[str, int] = {}

    async def get(self, key):
        return self.strings.get(key)

    async def set(self, key, value, nx=False, px=None):
        if nx and key in self.strings:
            return False
        self.strings[key] = value
        if px:
            self.ttls[key] = px // 1000
        return True

    async def delete(self, *keys):
        for key in keys:
            self.strings.pop(key, None)
            self.hashes.pop(key, None)
            self.ttls.pop(key, None)
        return len(keys)

    def pipeline(self, transaction=True):
        return _FakePipeline(self)

    async def hmget(self, key, keys):
        mapping = self.hashes.get(key, {})
        return [mapping.get(k) for k in keys]

    async def aclose(self):
        return None


class _FakePipeline:
    def __init__(self, client: _FakeAsyncRedis):
        self._client = client
        self._ops = []

    def delete(self, key):
        self._ops.append(("delete", key))
        return self

    def hset(self, key, mapping=None):
        self._ops.append(("hset", key, dict(mapping or {})))
        return self

    def set(self, key, value):
        self._ops.append(("set", key, value))
        return self

    def expire(self, key, seconds):
        self._ops.append(("expire", key, seconds))
        return self

    async def execute(self):
        results = []
        for op in self._ops:
            if op[0] == "delete":
                results.append(await self._client.delete(op[1]))
            elif op[0] == "hset":
                self._client.hashes.setdefault(op[1], {}).update(op[2])
                results.append(1)
            elif op[0] == "set":
                results.append(await self._client.set(op[1], op[2]))
            elif op[0] == "expire":
                self._client.ttls[op[1]] = op[2]
                results.append(True)
        self._ops = []
        return results


@pytest.fixture()
def store():
    client = _FakeAsyncRedis()
    return RedisVectorStore(client=client, key_prefix="t"), client


async def test_snapshot_id_changes_with_any_identity_factor():
    base = ("v1", ("p", "m", "mv"), "corpus")
    assert vector_snapshot_id(*base) == vector_snapshot_id(*base)  # 确定性
    assert vector_snapshot_id(*base) != vector_snapshot_id("v2", *base[1:])
    assert vector_snapshot_id(*base) != vector_snapshot_id(base[0], ("p", "m", "mv2"), base[2])
    assert vector_snapshot_id(*base) != vector_snapshot_id(base[0], base[1], "corpus2")


def test_corpus_hash_is_order_independent_and_key_sensitive():
    a = corpus_hash({"tool:x": "alpha", "skill:y": "beta"})
    assert a == corpus_hash({"skill:y": "beta", "tool:x": "alpha"})  # 键序无关
    assert a != corpus_hash({"tool:x": "alpha", "skill:y": "gamma"})


async def test_read_meta_missing_returns_none(store):
    s, _client = store
    assert await s.read_meta("nope") is None


async def test_publish_then_load_roundtrip_with_expected_key_shapes(store):
    """键域逐字节契约：``{prefix}:registry:vector{,meta,}-lock:{snapshot_id}`` 等。"""
    s, client = store
    snapshot_id = vector_snapshot_id("v1", ("p", "m", "mv"), "c1")

    await s.publish_vectors(
        snapshot_id=snapshot_id, owner_version="v1",
        vectors={"tool:x": (0.5, 0.25)},
        meta={"embedding_provider": "p", "dim": "2"},
        retention_seconds=7777,
    )

    vector_key = f"t:registry:vector:{snapshot_id}"
    meta_key = f"t:registry:vector-meta:{snapshot_id}"
    pointer_key = "t:registry:vector-of:v1"
    assert json.loads(client.hashes[vector_key]["tool:x"]) == [0.5, 0.25]
    assert json.loads(client.strings[meta_key])["embedding_provider"] == "p"
    assert client.strings[pointer_key] == snapshot_id
    # 写入即挂 TTL：三个键全部 7777。
    assert all(client.ttls.get(key) == 7777 for key in (vector_key, meta_key, pointer_key))

    vectors = await s.load_snapshot_vectors("v1", ["tool:x", "tool:missing"])
    assert vectors == {"tool:x": (0.5, 0.25)}


async def test_extend_ttl_only_touches_ttl(store):
    s, client = store
    snapshot_id = vector_snapshot_id("v1", ("p", "m", "mv"), "c1")
    await s.publish_vectors(
        snapshot_id=snapshot_id, owner_version="v1",
        vectors={"tool:x": (0.5,)}, meta={}, retention_seconds=10,
    )
    before = dict(client.hashes[f"t:registry:vector:{snapshot_id}"])

    await s.extend_ttl(snapshot_id, "v1", 99)

    assert client.hashes[f"t:registry:vector:{snapshot_id}"] == before  # 零重嵌
    assert client.ttls[f"t:registry:vector-meta:{snapshot_id}"] == 99
    assert client.ttls["t:registry:vector-of:v1"] == 99


async def test_single_flight_lock_is_exclusive_and_released(store):
    s, client = store
    snapshot_id = "snap-lock"

    assert await s.acquire_lock(snapshot_id) is True
    assert await s.acquire_lock(snapshot_id) is False  # SET NX：第二把拿不到
    assert client.ttls[f"t:registry:vector-lock:{snapshot_id}"] == LOCK_LEASE_SECONDS

    await s.release_lock(snapshot_id)
    assert await s.acquire_lock(snapshot_id) is True


async def test_load_snapshot_vectors_missing_owner_version_returns_empty(store):
    """快照缺席（发布期嵌入失败 / TTL 过期）返回空 dict——向量路缺席降级的读取面。"""
    s, _client = store
    assert await s.load_snapshot_vectors("never-published", ["tool:x"]) == {}


async def test_load_snapshot_vectors_redis_outage_degrades_to_empty(caplog):
    """Redis 异常同样返回空 dict：绝不外抛、绝不阻塞检索（降级契约的读取侧）。"""
    class _BrokenClient(_FakeAsyncRedis):
        async def get(self, key):
            raise ConnectionError("redis down")

    s = RedisVectorStore(client=_BrokenClient(), key_prefix="t")

    assert await s.load_snapshot_vectors("v1", ["tool:x"]) == {}
    assert any("缺席" in record.message or "失败" in record.message for record in caplog.records)


async def test_constructor_requires_client_or_url():
    with pytest.raises(ValueError, match="client 或 redis_url"):
        RedisVectorStore()
