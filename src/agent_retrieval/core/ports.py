"""检索内核的存储与嵌入端口：Protocol + 确定性内存/替身实现。

端口只定义检索内核需要的最小面，不 import 平台资源模型——实现方（如 B7 的
Redis 快照与生产嵌入器）在端口外自行装配。
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorStore(Protocol):
    """向量快照的 write-once 存储。

    ``publish`` 同一 ``snapshot_id`` 只允许一次：写入后快照不可变，重复发布抛
    ``ValueError``。读取面返回拷贝，调用方改返回值不影响存储内容。
    """

    def publish(
        self,
        snapshot_id: str,
        vectors: Mapping[str, Sequence[float]],
        meta: Mapping[str, str],
    ) -> None: ...

    def load(self, snapshot_id: str) -> dict[str, tuple[float, ...]] | None: ...

    def read_meta(self, snapshot_id: str) -> dict[str, str] | None: ...


@dataclass
class InMemoryVectorStore:
    """dict 存储的 :class:`VectorStore`，进程内生命周期。

    发布即转存不可变形态（向量 tuple、meta 浅拷贝 dict），此后拒绝对同一
    snapshot_id 的再次发布——write-once 语义是快照确定语义（R3）的存储侧前提。
    """

    _snapshots: dict[str, dict[str, tuple[float, ...]]]
    _meta: dict[str, dict[str, str]]

    def __init__(self) -> None:
        self._snapshots = {}
        self._meta = {}

    def publish(
        self,
        snapshot_id: str,
        vectors: Mapping[str, Sequence[float]],
        meta: Mapping[str, str],
    ) -> None:
        if snapshot_id in self._snapshots:
            raise ValueError(f"vector snapshot already published: {snapshot_id}")
        self._snapshots[snapshot_id] = {
            item_id: tuple(vector) for item_id, vector in vectors.items()
        }
        self._meta[snapshot_id] = dict(meta)

    def load(self, snapshot_id: str) -> dict[str, tuple[float, ...]] | None:
        """返回快照向量的拷贝；快照不存在返回 ``None``（缺失是正常路径，非异常）。"""
        snapshot = self._snapshots.get(snapshot_id)
        return dict(snapshot) if snapshot is not None else None

    def read_meta(self, snapshot_id: str) -> dict[str, str] | None:
        snapshot_meta = self._meta.get(snapshot_id)
        return dict(snapshot_meta) if snapshot_meta is not None else None


class QueryEmbeddingError(RuntimeError):
    """查询/语料嵌入失败——检索面按类型捕获并退回 BM25，不裸穿 requests 异常。"""


@runtime_checkable
class Embedder(Protocol):
    """文本嵌入端口：语料批量与查询单条两个入口。

    实现方在嵌入失败时应抛 :class:`QueryEmbeddingError`（而不是裸穿第三方异常）：
    融合检索面（``core.ranking``）按该类型捕获并自动退化为 BM25 单路。
    """

    def embed_corpus(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...

    def embed_query(self, text: str) -> tuple[float, ...]: ...


class MockEmbedder:
    """文本 hash → 确定性单位向量的真实替身（可运行实现，非打桩）。

    确定性来源是 ``hashlib.shake_256``：同文本任意两次调用逐位一致、跨进程一致，
    评测与测试因此可以离线复现（fusion-ranking §6）。归一化成单位向量是为了与
    真实嵌入器同语义——真实嵌入向量按惯例归一化，cosine 消费方在 Mock 与生产
    嵌入器之间行为一致。嵌入质量为零：只证明管线，不证明检索质量。
    """

    def __init__(self, dimensions: int = 8) -> None:
        self._dimensions = dimensions

    def embed_corpus(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> tuple[float, ...]:
        digest = hashlib.shake_256((text or "").encode("utf-8")).digest(self._dimensions)
        vector = tuple(byte / 255.0 for byte in digest)
        norm = math.sqrt(sum(value * value for value in vector))
        return tuple(value / norm for value in vector)
