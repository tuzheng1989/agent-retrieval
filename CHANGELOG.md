# Changelog

## 0.1.0 (2026-09-25)

Initial release.

- **Core** (stdlib-only): deterministic BM25 kernel (`BM25Index`, `tokenize`,
  `ideal_score`), dual-path RRF fusion (`fuse`, `RRF_K=60`), ports
  (`Embedder`/`VectorStore` protocols + `InMemoryVectorStore`/`MockEmbedder`),
  generic resource index (`ResourceBM25Index`, `substantive_term`,
  `clearly_related`), candidate ranking face (`rank_candidates`,
  `CandidateHit`).
- **Embedders** (`[api]`/`[local]` extras): OpenAI-compatible `ApiEmbedder`
  (L2-normalized, batched, dimension-checked), local ONNX `LocalEmbedder`
  (mean-pooling), `EmbeddingConfig` value object, `build_embedder` factory and
  `vector_available` tri-state.
- **Stores** (`[redis]` extra): `RedisVectorStore` — write-once vector
  snapshots with content-addressed snapshot ids, single-flight publish lock
  and TTL-pinned reads that degrade to empty on any outage.
- **Run cache** (`[redis]` extra): per-run query embedding cache with
  first-failure degrade freezing (`query_embedder`, `RunCachedEmbedder`,
  `QueryCacheBackend`, `MemoryBackend`, in-process `ContextVar` first-level
  cache, degradation counters).

Provenance: extracted from the shared-retrieval middleware of a production
multi-agent platform (BM25 + GLM-Embedding-3 vector dual-path, gate-verified
by a three-arm offline evaluation with frozen real embeddings), commit
`e0a5845c` of that codebase. Public API was renormalized at extraction
(`rank_admissible_agents` → generic `rank_candidates`; config plumbing made
explicit-constructor-only), so the 0.x series reserves the right to adjust
the API before 1.0.
