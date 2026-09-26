"""Disk-backed vector cache for benchmark runs (experiments-side host code).

``rank_candidates`` embeds the whole corpus on every call — correct for the
stateless kernel, brutal for a benchmark that re-ranks a 40k-document pool once
per query. This wrapper persists vectors in a local SQLite file keyed by
 ``(identity, sha256(text))`` so the corpus embedding cost becomes a one-time
spend shared across arms and reruns. Stdlib only (sqlite3 + array); vectors are
stored as float64 blobs in insertion order.

Embedding failures propagate as-is: ``rank_candidates`` already catches
``QueryEmbeddingError`` and degrades to BM25-only, and cached vectors survive
the failure for the next run.
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
import time
from array import array
from collections.abc import Sequence
from pathlib import Path

from agent_retrieval import Embedder, QueryEmbeddingError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    identity TEXT NOT NULL,
    key TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,
    PRIMARY KEY (identity, key)
)
"""
#: SQLite bound-parameter ceiling per statement; fetch keys in chunks below it.
_QUERY_CHUNK = 900
#: Persist embedded vectors every N texts. Corpus embeddings for large pools run
#: tens of minutes; committing in chunks keeps the work when a run is interrupted
#: (OOM reap, endpoint failure mid-way) instead of discarding it all.
_COMMIT_CHUNK = 512
#: Pause between chunks (8 batched requests at batch_size=64). Sustained ~460
#: requests/minute tripped the endpoint's rate limit mid-corpus once; this caps
#: us well under common per-minute quotas at near-zero throughput cost.
_CHUNK_PAUSE_SECONDS = 2.0


class CachedEmbedder:
    """:class:`Embedder` protocol implementation with a persistent SQLite cache."""

    def __init__(self, inner: Embedder, *, db_path: Path, identity: str) -> None:
        self._inner = inner
        self._identity = identity
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(db_path)
        self._connection.execute(_SCHEMA)
        self._connection.commit()
        #: Number of texts actually sent to the inner embedder this session.
        self.embedded_texts = 0
        #: Session-level overlay (zero-padded failures and fresh vectors) that
        #: survives across ``embed_corpus`` calls — ``_load_cached`` rebuilds its
        #: dict from disk every call, which would otherwise retry zero-padded
        #: texts on every single query.
        self._overlay: dict[str, tuple[float, ...]] = {}

    def embed_corpus(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed([text])[0]

    def close(self) -> None:
        self._connection.close()

    def _embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        keys = [_key(text) for text in texts]
        cached = self._load_cached(keys)
        for key, vector in self._overlay.items():
            if key not in cached:
                cached[key] = vector
        missing = [index for index, key in enumerate(keys) if key not in cached]
        for start in range(0, len(missing), _COMMIT_CHUNK):
            chunk = missing[start:start + _COMMIT_CHUNK]
            fresh: list[tuple[float, ...]] = []
            try:
                fresh = self._inner.embed_corpus([texts[index] for index in chunk])
                if len(fresh) != len(chunk):
                    raise ValueError(
                        f"inner embedder returned {len(fresh)} vectors for {len(chunk)} texts"
                    )
                self.embedded_texts += len(chunk)
                self._persist(keys, chunk, fresh)
                cached.update(zip((keys[index] for index in chunk), fresh))
                progress = f"{len(chunk)} embedded"
            except QueryEmbeddingError:
                # 批内坏文本（如超端点 token 上限的超长文档）只损失自身向量：
                # 降级逐条，成功者照常入缓存，失败者零向量占位——cosine=0 按
                # 内核契约（cosine ≤ 0 无证据）自动不占向量排名位，BM25 路不受影响。
                fresh = self._embed_one_by_one(texts, keys, chunk, cached)
                progress = f"{len(chunk)} embedded one-by-one (some zero-padded)"
            print(
                f"[vector-cache] {self._identity}: {progress}, {len(cached)} total cached",
                file=sys.stderr,
            )
            time.sleep(_CHUNK_PAUSE_SECONDS)
        return [cached[key] for key in keys]

    def _embed_one_by_one(
        self,
        texts: Sequence[str],
        keys: list[str],
        chunk: list[int],
        cached: dict[str, tuple[float, ...]],
    ) -> list[tuple[float, ...]]:
        fresh: list[tuple[float, ...]] = []
        survivors: list[tuple[int, tuple[float, ...]]] = []
        for index in chunk:
            try:
                vector = self._inner.embed_corpus([texts[index]])[0]
            except QueryEmbeddingError:
                # (0.0,) 与任何查询向量的 zip-点积恒为 0，维度无需已知。
                vector = (0.0,)
            fresh.append(vector)
            # 内存缓存含零占位：返回对齐且本会话不重试；磁盘只写幸存者，
            # 坏文本下次运行仍有机会被重嵌（端点修复后自动补齐）。
            cached[keys[index]] = vector
            if vector == (0.0,):
                self._overlay[keys[index]] = vector
            if vector != (0.0,):  # 正常嵌入是归一化非零向量，不会恰好等于占位值
                survivors.append((index, vector))
        self.embedded_texts += len(survivors)
        if survivors:
            self._persist(keys, [index for index, _ in survivors], [v for _, v in survivors])
        return fresh

    def _persist(self, keys: list[str], chunk: list[int], fresh: list[tuple[float, ...]]) -> None:
        self._connection.executemany(
            "INSERT OR REPLACE INTO vectors (identity, key, dim, vec) VALUES (?, ?, ?, ?)",
            [
                (self._identity, keys[index], len(fresh[position]), _pack(fresh[position]))
                for position, index in enumerate(chunk)
            ],
        )
        # Chunk-level commit: a later endpoint failure keeps everything before it.
        self._connection.commit()

    def _load_cached(self, keys: list[str]) -> dict[str, tuple[float, ...]]:
        cached: dict[str, tuple[float, ...]] = {}
        for start in range(0, len(keys), _QUERY_CHUNK):
            chunk = keys[start:start + _QUERY_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = self._connection.execute(
                f"SELECT key, vec FROM vectors WHERE identity = ? AND key IN ({placeholders})",
                (self._identity, *chunk),
            ).fetchall()
            cached.update({key: _unpack(blob) for key, blob in rows})
        return cached


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pack(vector: tuple[float, ...]) -> bytes:
    return array("d", vector).tobytes()


def _unpack(blob: bytes) -> tuple[float, ...]:
    values = array("d")
    values.frombytes(blob)
    return tuple(values)
