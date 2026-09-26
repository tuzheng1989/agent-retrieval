# agent-retrieval

Deterministic **BM25 + vector dual-path RRF fusion** retrieval kernel for agent resource discovery, with production reference implementations for embeddings (OpenAI-compatible API / local ONNX) and Redis vector snapshots.

Born inside a production multi-agent platform (BM25 path + GLM-Embedding-3 vector path, gate-verified by a three-arm offline evaluation), extracted as a dependency-light library for any agent that needs natural-language retrieval over a set of candidate resources (tools, skills, flows, agents, documents, …).

## Why

Most retrieval libraries give you a search engine. This one encodes an **opinionated retrieval discipline** for agent systems, where a bad match silently misroutes work:

- **Fusion ordering, not thresholding.** The kernel returns annotated, deterministically ordered candidates; admission/binding decisions stay with the caller ("retrieval advises, the caller decides").
- **Deterministic by construction.** Same input → bit-identical output: fixed accumulation order, lexicographic id tie-breaks, no hidden randomness. Replay-safe.
- **Graceful degradation, never raises.** Embedding failure, snapshot absence, cache outage — all collapse to BM25-only silently. Vector is an *upgrade*, never a dependency.
- **Zero-dependency core.** `agent_retrieval.core` imports nothing beyond the standard library (CI-enforced).

## Behavioral contract

Violating these is how retrieval systems quietly rot. The kernel holds them so you don't have to:

1. **Filter before fusion.** Admission is the caller's job; retrieval can never rank an inadmissible candidate into view.
2. **`k` truncates output, not ranking.** Full recall inside the kernel; truncation happens at the presentation layer.
3. **`cosine ≤ 0` is no evidence.** Non-positive cosine candidates occupy no vector-rank slot.
4. **Exact-match topping is a caller contract.** The kernel sorts `(not exact, -fused, id)` but only sees candidates you put in — feed your exact hits into at least one path.
5. **Absent path = `None`, not `0.0`.** Missing evidence is distinguishable from zero evidence (0.0 is a legal BM25 value for exact hits).
6. **Deterministic tie-breaks.** Fixed-order accumulation + id lexicographic ordering; two calls with the same input are bit-identical.
7. **Degrade, don't throw.** Embedding failure / snapshot absence / cache errors fall back to BM25-only.
8. **Relevance is scale-relative.** Decisions use `score / ideal_score`, not raw BM25 scores (raw scores inflate with corpus size; absolute thresholds only hold at calibration size).

## Installation

```bash
pip install agent-retrieval              # core only (stdlib, zero deps)
pip install "agent-retrieval[api]"       # + OpenAI-compatible embeddings (requests)
pip install "agent-retrieval[redis]"     # + Redis vector snapshot store & run cache
pip install "agent-retrieval[local]"     # + local ONNX embeddings (not available on Windows ARM64)
```

Python ≥ 3.10. Wheels are pure Python — platform-independent.

## Quickstart

### 1. Rank candidates with BM25 (+ optional vector path)

```python
from dataclasses import dataclass
from agent_retrieval import MockEmbedder, rank_candidates

@dataclass(frozen=True)
class Tool:
    id: str
    name: str
    description: str

tools = [
    Tool("geo-query", "Geo Query", "query geospatial data by region and time range"),
    Tool("doc-search", "Doc Search", "full-text search over uploaded documents"),
]

def corpus(t: Tool) -> str:          # what gets indexed: your "declared face"
    return f"{t.id} {t.name} {t.description}"

hits = rank_candidates(
    tools,
    "find population data for California in 2024",
    corpus_text=corpus,
    item_id=lambda t: t.id,
    item_name=lambda t: t.name,
    vector=MockEmbedder(),           # omit for BM25-only; production: build_embedder(...)
)
for hit in hits:
    print(hit.fused_score, hit.item.name, hit.matched_terms)
```

The same `corpus` function feeds both the BM25 index and the vector path — retrieval evidence and admission evidence stay consistent.

### 2. Real embeddings

```python
from agent_retrieval import EmbeddingConfig, build_embedder, vector_available

config = EmbeddingConfig(
    kind="api", provider="zhipu", model_name="embedding-3",
    base_url="https://open.bigmodel.cn/api/paas", api_key="...",
    dimensions=2048,
)
embedder = build_embedder(config)     # None = vector path unavailable, stay BM25
print(vector_available(config))       # "api" | "local" | "none"
```

### 3. Write-once vector snapshots (Redis)

```python
from agent_retrieval import RedisVectorStore, corpus_hash, vector_snapshot_id

store = RedisVectorStore(redis_url="redis://localhost:6379/0", key_prefix="myapp")

snapshot_id = vector_snapshot_id(
    owner_version="resources-v42",           # your candidate-set version
    identity=config.identity(),              # (provider, model, model_version)
    corpus_hash=corpus_hash(corpus_texts),   # {"kind:id": text}
)

if await store.read_meta(snapshot_id) is None and await store.acquire_lock(snapshot_id):
    await store.publish_vectors(             # inside your lock: embed then write
        snapshot_id=snapshot_id,
        owner_version="resources-v42",
        vectors={"tool:geo-query": [0.1, ...]},
        meta={"embedding_provider": config.provider, "dim": "2048"},
        retention_seconds=7 * 24 * 3600,
    )
    await store.release_lock(snapshot_id)
elif ...:
    await store.extend_ttl(snapshot_id, "resources-v42", 7 * 24 * 3600)  # write-once: TTL only, zero re-embedding

vectors = await store.load_snapshot_vectors("resources-v42", ["tool:geo-query"])  # {} → degrade to BM25
```

Snapshot identity is content-addressed: same owner version + same embedder + same corpus → same id → re-publish only extends TTL. Re-embedding happens only when something actually changed.

### 4. Per-run query caching with degrade freezing

```python
from agent_retrieval import MemoryBackend, query_embedder, clear_run_cache, take_degradation_count

wrapped = query_embedder(
    embedder=build_embedder(config),
    run_id="run-123",                        # your execution-scoped id (explicit injection)
    backend=MemoryBackend(),                 # or make_backend(redis_url=..., key_prefix=...)
)
# wrapped embeds each distinct query text once per run; first embedding failure
# freezes the vector path for the rest of the run (no retries, replay-deterministic).
clear_run_cache()                            # at run teardown
```

## Library discipline

The package never reads environment variables, config files, or your ContextVars. Everything arrives via constructor parameters — your wiring layer translates your config system (yaml/env/secrets manager) into `EmbeddingConfig` / store / backend arguments. `run_id` is injected explicitly; the vector on/off flag is a host concept (pass `embedder=None` for BM25-only).

## Known limitations

- `LocalEmbedder` builds an ONNX session per instantiation — construct once and reuse.
- In-process run cache uses `ContextVar`s; call `clear_run_cache()` at run teardown to avoid cross-run leakage within a worker.
- The `local` extra (ONNX) has no Windows ARM64 wheels — use the `api` path there.

## Quality benchmarks

Retrieval quality (kernel + embedder) is evaluated against two public benchmarks —
[ToolRet](https://arxiv.org/abs/2503.01763) (tool retrieval, ~43k tools) and
[SkillRet](https://arxiv.org/abs/2605.05726) (agent-skill retrieval, 6,006-skill
evaluation pool) — with bm25 / fusion arms, recall / completeness / NDCG / MRR
metrics, and a persistent vector cache. See
[experiments/README.md](experiments/README.md#retrieval-benchmarks-toolret-tool-retrieval-and-skillret-skill-retrieval).

## License

MIT
