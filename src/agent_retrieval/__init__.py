"""agent-retrieval：确定性 BM25 + 向量双路 RRF 融合检索内核与生产参考实现。

分层结构（内层零第三方依赖）：

- :mod:`agent_retrieval.core` — 原子检索内核：BM25 数学内核、RRF 融合、端口
  Protocol、资源索引适配器、候选检索面。仅标准库；
- :mod:`agent_retrieval.embedders` — 嵌入器参考实现（OpenAI 兼容 API / 本地 ONNX）
  与工厂（extras: ``api`` / ``local``）；
- :mod:`agent_retrieval.stores` — 向量快照的 Redis write-once 存储（extras: ``redis``）；
- :mod:`agent_retrieval.run` — Run 生命周期内的查询向量缓存与降级冻结
  （extras: ``redis``）。

库纪律：包不读环境变量、不读配置文件、不读调用方的 ContextVar——一切经构造参数。
"""
from __future__ import annotations

from agent_retrieval.core.bm25 import (
    BM25Index,
    BM25Score,
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    TOKENIZER_VERSION,
    content_length,
    tokenize,
)
from agent_retrieval.core.fusion import BM25Hit, FusedHit, RRF_K, VectorHit, fuse
from agent_retrieval.core.ports import (
    Embedder,
    InMemoryVectorStore,
    MockEmbedder,
    QueryEmbeddingError,
    VectorStore,
)
from agent_retrieval.core.resource_index import (
    ResourceBM25Index,
    ResourceHit,
    clearly_related,
    substantive_term,
)
from agent_retrieval.core.ranking import CandidateHit, rank_candidates

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # 嵌入器参考实现（requests / onnx 依赖，extras: api / local）
    "EmbeddingConfig": ("agent_retrieval.embedders.config", "EmbeddingConfig"),
    "ApiEmbedder": ("agent_retrieval.embedders.api", "ApiEmbedder"),
    "LocalEmbedder": ("agent_retrieval.embedders.local", "LocalEmbedder"),
    "build_embedder": ("agent_retrieval.embedders.factory", "build_embedder"),
    "vector_available": ("agent_retrieval.embedders.factory", "vector_available"),
    # Redis 向量快照存储（redis 依赖，extras: redis）
    "RedisVectorStore": ("agent_retrieval.stores.redis_store", "RedisVectorStore"),
    "corpus_hash": ("agent_retrieval.stores.redis_store", "corpus_hash"),
    "vector_snapshot_id": ("agent_retrieval.stores.redis_store", "vector_snapshot_id"),
    # Run 级查询缓存与降级冻结（redis 依赖，extras: redis）
    "RunCachedEmbedder": ("agent_retrieval.run.cache", "RunCachedEmbedder"),
    "QueryCacheBackend": ("agent_retrieval.run.cache", "QueryCacheBackend"),
    "make_backend": ("agent_retrieval.run.cache", "make_backend"),
    "query_embedder": ("agent_retrieval.run.cache", "query_embedder"),
    "clear_run_cache": ("agent_retrieval.run.cache", "clear_run_cache"),
    "record_vector_degradation": ("agent_retrieval.run.cache", "record_vector_degradation"),
    "take_degradation_count": ("agent_retrieval.run.cache", "take_degradation_count"),
}

__all__ = [
    "BM25Hit",
    "BM25Index",
    "BM25Score",
    "CandidateHit",
    "DEFAULT_BM25_B",
    "DEFAULT_BM25_K1",
    "TOKENIZER_VERSION",
    "Embedder",
    "content_length",
    "EmbeddingConfig",
    "FusedHit",
    "InMemoryVectorStore",
    "LocalEmbedder",
    "ApiEmbedder",
    "MockEmbedder",
    "QueryEmbeddingError",
    "QueryCacheBackend",
    "RRF_K",
    "RedisVectorStore",
    "ResourceBM25Index",
    "ResourceHit",
    "RunCachedEmbedder",
    "VectorHit",
    "VectorStore",
    "build_embedder",
    "clear_run_cache",
    "clearly_related",
    "content_length",
    "corpus_hash",
    "fuse",
    "make_backend",
    "query_embedder",
    "rank_candidates",
    "record_vector_degradation",
    "substantive_term",
    "take_degradation_count",
    "tokenize",
    "vector_available",
    "vector_snapshot_id",
]

__version__ = "0.1.0"


def __getattr__(name: str):  # PEP 562：重依赖模块仅在真正取属性时导入
    entry = _LAZY_EXPORTS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(entry[0]), entry[1])


def __dir__() -> list[str]:
    return sorted(__all__)
