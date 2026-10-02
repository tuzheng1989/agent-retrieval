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

## Recall-ceiling panorama and two falsified recall fixes (2026-09-28/29)

Recall is the binding constraint of the whole pipeline. Gold-in-window coverage
over the full pools (fusion, ToolRet in instruction mode):

| gold in window | @5 | @10 | @20 | @50 |
| --- | ---: | ---: | ---: | ---: |
| SkillRet (6,006 pool) | 63.4% | 69.8% | 73.5% | 79.9% |
| ToolRet (44,453 pool) | 35.3% | 39.8% | 48.6% | 59.1% |

Two recall improvements were implemented and **falsified** by their own
ablations — reports under `recall-ablation-*.json` / `iterative-*.json`
(gitignored):

1. **Field-level multi-path BM25** (name/description/tags as separate paths,
   RRF-merged) degraded SkillRet across the board: BM25-arm Recall@5 fell
   0.459 → 0.308, MRR 0.563 → 0.395 (300 queries). Unweighted RRF lets a
   narrow field's accidental single hit take that path's top slot (1/61) and
   crowd out candidates accumulated by multi-word description hits. SkillRet's
   paper reports field-separation winning under *learned weighted* fusion —
   that advantage does not transfer to unweighted RRF.
2. **Jev-triggered iterative retrieval** (PRF-expanded second round when the
   window's best Jev probability < τ): trigger rate was **0% across
   τ ∈ {0.3, 0.5, 0.7} on both benchmarks** (SkillRet 30, ToolRet 30). Jev's
   question ("can some resource here perform a step of the task?") is almost
   always answerable by *some* partially-related candidate in a fusion top-20,
   so its confidence never signals the failure mode that matters (gold absent
   from the window). A trigger for this purpose needs a different question.

What the panorama says instead: on ToolRet, 41% of gold is outside even the
top-50 window — the lever is first-stage coverage (a purpose-built or
higher-capacity embedder, doc2query-style corpus expansion), not re-ranking or
iterating over a weak window. On SkillRet the ceiling at top-50 is 79.9% and
Jev already captures most of the re-orderable gap within window 20.

## doc2query corpus expansion works (2026-10-01)

Following the panorama's pointer, `docgen.py` generated 5 realistic user queries
per skill (`glm-5.3-flash` via Zhipu's coding endpoint, 6,006/6,006 resources,
zero failures after switching off thinking tokens and budgeting 2,000
max_tokens — GLM-4.5/5.3 reasoning tokens count against the budget and
silently truncate the answer). Same 300-case prefix, same embedder, k up to 50:

| Arm | Corpus | Recall@5 | Recall@20 | Recall@50 | MRR | NDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | plain | 0.459 | 0.606 | 0.674 | 0.563 | 0.470 |
| **BM25** | **docgen** | **0.695** | **0.803** | **0.851** | **0.793** | **0.695** |
| Fusion | plain | 0.634 | 0.735 | 0.799 | 0.737 | 0.631 |
| Fusion | docgen | 0.691 | 0.801 | **0.879** | 0.775 | 0.681 |

**BM25 + docgen beats plain fusion** on every metric — the vocabulary bridge
in the corpus replaces the vector path's semantic bridging at zero query-time
cost. The recall ceiling itself moved: Recall@50 0.674 → 0.851. Fusion+docgen
still holds the highest Recall@50 (0.879) but its MRR/NDCG trail BM25+docgen
(vector and expansion overlap in benefit).

Caveat: docgen queries and SkillRet's evaluation queries are both LLM-generated
English questions; the same-distribution effect likely inflates the gain versus
messier real users. Directionality (vocabulary bridging lifts recall) is solid;
exact magnitudes are benchmark-specific. Reports:
`recall-ablation-skillret-{single,fields,docgen}.json` (gitignored).
Reproduce: `python -m experiments.benchmarks.docgen --bench skillret --k 5`
then `python -m experiments.benchmarks.run_benchmark --bench skillret --limit
300 --arm both --corpus docgen --k 5 10 20 50`.

## Full-combination matrix (2026-10-01): the optimum is docgen + BM25 + Jev@20

All three docgen × Jev combinations ran on 300 spread cases (comparable to the
full run above; note `run_benchmark`'s 300-case prefix and this spread differ —
compare only within this table):

| Final configuration | Recall@5 | Recall@10 | MRR | NDCG@5 | Extra cost |
| --- | ---: | ---: | ---: | ---: | --- |
| fusion (plain baseline) | 0.663 | 0.727 | 0.733 | 0.627 | — |
| fusion + Jev@20 | 0.750 | 0.780 | 0.783 | 0.710 | ~1.7M tok |
| **docgen + BM25 + Jev@20** | **0.763** | 0.794 | **0.789** | **0.718** | ~1.7M tok |
| docgen + fusion (no Jev) | 0.696 | 0.761 | **0.805** | 0.687 | none |
| docgen + fusion + Jev@50 | **0.769** | **0.834** | 0.774 | 0.711 | ~3.5M tok |

Three findings:

1. **The all-round winner is docgen + BM25 + Jev@20** — best or joint-best on
   all four metrics *and* it needs no embedding infrastructure at all.
2. **After docgen, adding Jev to fusion *lowers* MRR** (0.805 bare → 0.781/0.774
   with Jev@20/@50) even though recall still rises: docgen already fixes the
   vocabulary gap, fusion's top-1 hit rate jumps, and Jev's occasional
   misjudgment now drags down an ordering that was mostly right. Jev pays for
   itself where retrieval is *weak* (the plain-corpus case above), not on top
   of a strong one.
3. **Window 50 becomes worth it under docgen** (gold-in-window 0.793 → 0.868):
   Recall@10 = 0.834 is the best window coverage measured, at 2× token cost and
   slightly softer ordering.

Recommendation by objective: default **docgen + BM25 + Jev@20**; latency-bound
**docgen + BM25** (no Jev: Recall@5 0.724, MRR 0.793, zero per-query calls);
coverage-bound **docgen + fusion + Jev@50** (Recall@10 0.834).

## ToolRet: docgen + Jev validation (2026-10-02)

ToolRet (44,453 pool, 100 instruction-mode round-robin queries) ran the same
ladder with doc2query expansion over the documentation-verbatim corpus
(44,444/44,453 resources expanded; 9 pathological documents skipped):

| Configuration | Recall@5 | Recall@10 | Recall@20 | Recall@50 | MRR | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 plain | 0.317 | 0.362 | 0.435 | 0.486 | 0.336 | 0.303 |
| BM25 docgen | 0.329 | 0.388 | 0.439 | 0.528 | 0.362 | 0.319 |
| Fusion plain | 0.353 | 0.398 | 0.486 | **0.591** | 0.372 | 0.339 |
| Fusion docgen | 0.381 | 0.427 | 0.486 | 0.574 | 0.353 | 0.326 |
| BM25 docgen + Jev@20 | 0.408 | 0.433 | — | — | 0.400 | 0.366 |
| **Fusion docgen + Jev@20** | **0.424** | **0.468** | — | — | **0.414** | **0.388** |

Findings — the mirror image of SkillRet, and a unified rule:

- docgen lifts ToolRet only modestly (BM25 Recall@5 +3.8%, Fusion +8.0%), and
  Fusion Recall@50 slightly *drops* (0.591 → 0.574): expansion noise pushes
  some long-tail gold out of deep positions. ToolRet's documentation JSON is
  already vocabulary-rich, so the colloquial bridge adds less.
- Jev strongly complements the weaker ToolRet retrievals: +24% Recall@5 on
  BM25, +11% on Fusion, +17% MRR on Fusion. **Jev's gain scales with retrieval
  weakness** — docgen made SkillRet retrieval strong enough that Jev's marginal
  turned negative; ToolRet retrieval stays weak, so Jev pays.
- Unlike SkillRet, the vector path stays essential on ToolRet: docgen+Fusion
  beats docgen+BM25 by +16% Recall@5 (parameter semantics in the documentation
  JSON are not covered by colloquial generated queries).

Best ToolRet configuration: **fusion docgen + Jev@20** (Recall@5 0.424,
NDCG@10 0.388) — numerically above the official leaderboard's best published
NDCG@10 (NV-Embed-v1, 0.338; different-embedder caveat applies). Reports:
`jev-toolret-docgen-{bm25,fusion}-w20.json`, `recall-ablation-toolret-{bm25,fusion}.json`,
`recall-ablation-toolret-docgen.json` (gitignored).
