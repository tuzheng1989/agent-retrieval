# Jev on SkillRet and ToolRet recall

Run date: 2026-09-26. This is a paired **pilot**, not a full benchmark run.
The TypeSafe API reported `jev-1.13.0`.

## Method

For every sampled query, `rank_candidates` searched the **full** benchmark resource
pool with the existing GLM-Embedding-3 fusion arm. Jev then judged each of the top
20 candidates with one Noul question in a single request and reordered only those
20 by probability. The remaining ranking was unchanged. The ToolRet query used its
task-aware instruction (`--query-mode instruction`); SkillRet has none. Every case
uses all published binary relevance labels, so multi-resource recall is measured
as a fraction of its gold resources. No threshold or no-match decision is involved.

The oracle column moves all labeled resources that are already in the top 20 to
the front. It is the best possible recall from **this** candidate window. Jev
does not see labels. SkillRet samples 40 cases spaced across its pinned test split;
ToolRet samples one case from each of the first 20 source tasks in its round-robin
order. The sample IDs and case-level scores are in the local JSON reports.

The repository's earlier, larger fusion reports also bound a *top-10-only*
reranker without further calls: SkillRet's 300 cases have Recall@5 `0.6344`
and Recall@10 `0.6983`, leaving at most `0.0639` absolute improvement at 5;
ToolRet's 100 instruction-mode cases have `0.3526` and `0.3980`, leaving at
most `0.0454`. Reordering those same ten candidates cannot change Recall@10.
These reports use different cases from the Jev pilot below.

| Dataset and metric | Fusion | Fusion + Jev | Top-20 oracle |
| --- | ---: | ---: | ---: |
| SkillRet (40 queries, 6,006 skills), Recall@5 | 0.5708 | **0.7583** | 0.7583 |
| SkillRet, Recall@10 | 0.6708 | **0.7583** | 0.7583 |
| SkillRet, Completeness@5 | 0.4250 | **0.6500** | 0.6500 |
| SkillRet, NDCG@5 | 0.5397 | **0.6335** | 0.7848 |
| ToolRet (20 queries, 44,453 tools), Recall@5 | 0.3750 | 0.3750 | 0.4500 |
| ToolRet, Recall@10 | 0.4250 | **0.4500** | 0.4500 |
| ToolRet, Completeness@5 | **0.3500** | 0.3000 | 0.3500 |
| ToolRet, NDCG@5 | 0.3232 | 0.3271 | 0.4726 |

On SkillRet, Recall@5 improved on 9 cases, worsened on 0, and tied on 31.
Recall@10 improved on 4 and worsened on 0. Jev reached the oracle's *recall*
in this sample, but not its NDCG: NDCG@5 worsened on 5 individual cases and MRR
on 4, so its exact ordering is not consistently optimal. Five of 40 cases had
no gold resource in the top 20.

On ToolRet, Recall@5 improved on 2 cases and worsened on 1, canceling in the
mean. Recall@10 improved on 1 and worsened on 0. The single Recall@5 loss also
reduced Completeness@5. **Nine of 20 cases had no gold tool in the top 20**, so
reranking could not help those queries. Across all gold labels, only one third
were in that window (micro coverage); the macro top-20 oracle recall is 0.4500.

A paired, case-bootstrap 95% interval for the observed Jev-minus-fusion delta is
`+0.0750 to +0.3125` for SkillRet Recall@5 and `-0.1250 to +0.1000` for
ToolRet Recall@5. These intervals describe uncertainty in these small sampled
sets; SkillRet's systematic spacing and ToolRet's 20-source subset limit
generalization. The earlier repository reports use different case samples and
must not be subtracted from these numbers.

Jev used 40 requests / 213,551 input tokens on SkillRet and 20 requests /
155,651 input tokens on ToolRet. Mean Jev request latency was 1.24 and 1.28
seconds respectively. These figures exclude the retriever's much larger full-pool
ranking cost and any future concurrency design.

## Interpretation

Jev **can raise Recall@k by moving already retrieved gold resources into the
first k positions**. It cannot increase coverage of its top-20 input window.
The SkillRet pilot is promising for a caller-side optional reranker. ToolRet's
candidate coverage is the larger problem; its Jev Recall@5 showed no net gain
and Completeness@5 declined. Increasing the candidate window or improving the
first-stage retrieval should be evaluated before adding Jev to ToolRet routing.
For both datasets, a larger, held-out paired evaluation and comparison against
a purpose-built reranker are needed before adopting it.

To reproduce the pilot (datasets and embedding vectors are locally cached):

```powershell
python -m experiments.benchmarks.jev_rerank --bench skillret --arm fusion --limit 40 --window 20 --output experiments/jev-skillret-benchmark.json
python -m experiments.benchmarks.jev_rerank --bench toolret --arm fusion --query-mode instruction --limit 20 --window 20 --output experiments/jev-toolret-benchmark.json
```

The JSON reports are gitignored because future benchmark inputs may be private.

## Follow-up full run (2026-09-27): SkillRet, 300 queries, window 20 vs 50

A follow-up confirmed the pilot at scale: 300 cases spread across the pinned test
split, fusion retrieval, Jev-1.13.0, identical paired methodology.

| Metric | Fusion | Jev@20 | Jev@50 | Oracle@20 | Oracle@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MRR | 0.7329 | **0.7830** | 0.7673 | 0.8979 | 0.9237 |
| Recall@5 | 0.6628 | 0.7500 | **0.7589** | 0.7817 | 0.8244 |
| Completeness@5 | 0.5100 | 0.6133 | **0.6233** | 0.6600 | 0.7133 |
| NDCG@5 | 0.6267 | **0.7096** | 0.6982 | 0.8084 | 0.8479 |
| Recall@10 | 0.7267 | 0.7800 | **0.7989** | 0.7817 | 0.8244 |
| gold in window | — | 0.751 | 0.793 | — | — |

Doubling the window to 50 buys ~1pp more Recall@5 / Completeness@5 (more gold
enters the window) but costs 2.3× tokens (3.7M vs 1.6M input) *and* degrades
order-sensitive metrics: MRR −0.016, NDCG@5 −0.011 — judging 50 candidates at
once dilutes per-candidate discrimination. **Window 20 is the better operating
point.** Cost at window 20: ~5.3k input tokens and one ~1 s System One call per
query. Jev captured ~73% of the window-reordable Recall@5 oracle gap.

A separate cross-encoder rerank arm over ToolRet's instruction-mode fusion top-100
(paratera `GLM-Rerank`) was also evaluated and is **reported as invalid**: the
endpoint returned a constant score (1.0) for every document, so "re-ranking"
collapsed to lexicographic order and all metrics dropped below the BM25 baseline.
The runner now detects constant-score responses, keeps the fusion order, and
counts such cases as degraded (`degraded_cases` in the report) instead of
reporting silently shuffled rankings.

Reports: `jev-rerank-skillret-300-w{20,50}.json` and
`toolret-benchmark-rerank.json` (gitignored).
