# Changelog

## 0.2.0 (2026-11-08)

SDK composition layer: judge ports, declarative corpus projection, docgen
client and adoption-gate statistics — all behind explicit extras, core stays
stdlib-only. Component boundaries frozen per ADR-0001 (a component enters the
SDK only when a scenario-matrix cell needs it; research parts stay in
`experiments/`).

- **Judge ports** (core, no extra): `Judge` protocol, `JudgedCandidate` /
  `Judgment`, `select_with_judge` (abstain semantics, no-default
  `minimum_probability`), `JudgeResponseError`.
- **Jev judge** (`[judge]` extra): `JevJudge` — System One wire protocol over
  bare HTTP (requests; no typesafe-sdk dependency), `judge_state` as the
  single construction point, gateway-retry semantics. Parity against the
  legacy SDK arm frozen in `judge-parity-legacy.json`.
- **Chat judge** (`[judge]` extra): `ChatJudge` — judgment over any chat
  completions endpoint; frozen prompt template (version "1"), temperature
  pinned at 0.0 into `identity()`, markdown-fenced JSON parsing with backoff.
- **Corpus projection**: `CorpusProjection` — declarative field projection
  (include/exclude) for the retrieval face; keeps doc2query expansions out of
  admission evidence (doc2query-- pattern).
- **Docgen client** (`[docgen]` extra): `DocGenClient` / `build_docgen` /
  `docgen_corpus_text` — offline doc2query corpus expansion; thinking-token
  budget pitfalls documented in the retrieval-config guide.
- **Adoption-gate stats** (core, stdlib-only): `agent_retrieval.stats` —
  `paired_bootstrap` (10k resamples, fixed seed), `load_arm_cases`,
  `paired_report` (case-level paired-diff CI95 with improved/tied/worsened).

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
