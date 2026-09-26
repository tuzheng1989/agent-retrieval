# Jev candidate judgment experiment

This evaluates a caller-side Jev judgment after `rank_candidates`. The package's
deterministic retrieval core and dependencies are unchanged.

See [jev-findings.md](jev-findings.md) for the first live run on synthetic cases.

## Run

From the repository root, with the project installed in the active Python environment:

```powershell
python -m experiments.jev_retrieval --output experiments/baseline-result.json
pip install typesafe-sdk
$env:TYPESAFE_API_KEY = "<your key>"
python -m experiments.jev_retrieval --live --output experiments/jev-result.json
```

The baseline command makes no external calls. `--live` sends the query and shortlisted
resource descriptions to TypeSafe. Keep the key in the environment and keep sensitive
resource descriptions out of the sample file. The JSON report includes the query,
shortlist, per-candidate Jev probabilities, and aggregate metrics.

`--dataset path/to/data.json` accepts the schema in `jev_cases.json`. Each case has one
correct resource ID or `null` if no resource should be bound. Use representative real
queries and label them before looking at model outputs. A positive case whose gold ID
is absent from the shortlist cannot be rescued by Jev; inspect `candidate_recall_at_k`
before interpreting reranking results.

`--k` defaults to 5. `--threshold` defaults to an **exploratory** 0.7; choose it using
separate calibration examples before reporting performance on a held-out set. Exact ID
or name matches bypass Jev and stay first. A missing Jev response fails the live run
instead of silently producing a misleading comparison.

This small included dataset is synthetic and only verifies that the experiment runs.
It is not evidence that Jev improves this project's production routing. Compare
positive top-1 accuracy, negative false-bind rate, candidate recall, request latency,
and token usage on a real labeled set. The baseline chooses the first retrieved hit;
it does not model any host-specific `clearly_related` threshold.

On the included 14 cases, the BM25 baseline has 10/11 candidate recall at 5 among
positive cases, 9/11 positive top-1 accuracy, and a 3/3 false-bind count on the
no-match cases. The `unrecalled` case has an empty BM25 shortlist, illustrating
that Jev cannot recover a candidate the retriever never surfaced. These are
synthetic fixture results, not production estimates.

TypeSafe references: [Python SDK](https://docs.typesafe.ai/sdk/python.md),
[Noul](https://docs.typesafe.ai/primitives/noul.md), and
[reranking cookbook](https://docs.typesafe.ai/cookbooks/rerank_typesafe.md).
# Retrieval benchmarks: ToolRet (tool retrieval) and SkillRet (skill retrieval)

Two public benchmarks evaluate retrieval quality of the kernel end to end, one per
resource domain the package targets:

| Benchmark | Domain | Pool | Eval queries | Labels | Official metrics |
|---|---|---|---|---|---|
| [ToolRet](https://arxiv.org/abs/2503.01763) (ACL 2025 Findings) | tool retrieval | ~43k heterogeneous tools | 7.6k tasks (35 sources) | binary | NDCG / Recall @{5,10,20} |
| [SkillRet](https://arxiv.org/abs/2605.05726) | agent-skill retrieval | 6,006 skills (test pool) | 4,392 | binary | NDCG / Recall / Completeness @{5,10,15} |

Both adapters download once into `experiments/benchmarks/data/` (gitignored) and
project each dataset onto the shared `BenchmarkDataset` shape. The runner reports
`recall@k`, `completeness@k`, `ndcg@k`, and `mrr` over the kernel's FULL rankings —
truncation happens in the metrics layer, never inside the kernel. Two arms:

- `bm25` — `rank_candidates` with no vector path. No network, no key.
- `fusion` — BM25 + vector RRF through an OpenAI-compatible embedder.

## Embedder configuration (.env or environment variables)

The package itself never reads the environment (host discipline). This script is the
host; it assembles `EmbeddingConfig` from variables that may live in **`.env` at the
repository root** (gitignored, parsed via optional `python-dotenv`) or be exported in
the shell. Existing environment variables WIN over `.env`, so one-off overrides stay
possible. Copy [.env.example](../.env.example) to `.env` and fill in:

```bash
AGENT_RETRIEVAL_EMBEDDING_API_KEY=<your key>               # required for the fusion arm
# AGENT_RETRIEVAL_EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas   # default
# AGENT_RETRIEVAL_EMBEDDING_MODEL=embedding-3              # default
# AGENT_RETRIEVAL_EMBEDDING_DIMENSIONS=1024                # default
```

Without a key the fusion arm is skipped and the report says so; the BM25 arm always
runs. A configured fusion arm is **preflighted** with one probe embedding — an
unreachable endpoint or rejected key fails the run (exit 2) instead of silently
reporting BM25 numbers under the fusion label. Fusion-arm corpus/query vectors are
cached in `experiments/benchmarks/.cache/vectors.sqlite3` (keyed by embedder identity +
text hash), so the corpus embedding cost is paid once across arms and reruns;
`--no-cache` disables this.

## Run

```powershell
# BM25 arm of SkillRet (downloads ~125 MB on first run; no key needed)
.venv\Scripts\python.exe -m experiments.benchmarks.run_benchmark --bench skillret --limit 300 --output experiments/skillret-benchmark.json

# ToolRet (needs `pip install duckdb` for parquet shards; downloads ~35 MB)
.venv\Scripts\python.exe -m experiments.benchmarks.run_benchmark --bench toolret --limit 200 --output experiments/toolret-benchmark.json

# Both arms with an embedder key set
.venv\Scripts\python.exe -m experiments.benchmarks.run_benchmark --bench skillret --arm both
```

`--limit` samples cases only (prefix of SkillRet's file order, round-robin across
ToolRet's 35 source tasks) and never shrinks the candidate pool — recall against a
shrunken pool would flatter every arm. SkillRet downloads are pinned to dataset
revision `a050ad2` (the Hub head is mutable; see the dataset card). ToolRet corpus
text is the `documentation` field verbatim; SkillRet's declared face is
name + description + taxonomy tags, with the full SKILL.md body deliberately kept
out of the lexical corpus (kernel discipline: the body is a permission decision).

These benchmarks measure the kernel + embedder combination, not the kernel alone;
compare arms under the SAME embedder identity. The package's behavioral contracts
(determinism, degradation, fusion shape) are covered by `tests/`, not here.
