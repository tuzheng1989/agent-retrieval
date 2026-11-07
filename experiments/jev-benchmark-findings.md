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

## Two-tier retrieval (OpenViking scope-before-rank): falsified on SkillRet (2026-10-22)

Borrowed from OpenViking's directory-aware retrieval: groups (18 `sub` categories)
become first-class candidates; the first `rank_candidates` call ranks groups, the
top-G groups' members are pooled, and a second call ranks resources in that pool.
The kernel is untouched — tier two is "pass the subset and call again". Paired
300-case spread sample, BM25 stack, plain corpus; pre-registered M1 (semantic
two-tier vs single tier in the same runner) and M2 (semantic vs seeded random
groups with the same size profile, isolating the pool-shrinkage side effect).
Single-tier rows are bitwise-identical across M1/M2 runs; case sets match.

| Comparison | ΔMRR | Δrecall@10 | Δrecall@20 | reading |
| --- | ---: | ---: | ---: | --- |
| two-tier − single (total) | −0.138 [−0.180, −0.099] | −0.159 [−0.202, −0.118] | −0.197 [−0.241, −0.155] | fails: recall CI upper < 0 |
| two-tier − random two-tier (net) | **+0.119 [+0.072, +0.165]** | +0.152 [+0.100, +0.204] | +0.161 [+0.109, +0.214] | the structural prior is real |

Mechanism: the first tier's entry loss dominates everything. Group hit rate
(`p_any`, ≥1 gold group among top-G=3) is 0.743 — 3× its 0.241 chance baseline,
so the tier-1 signal is genuine — but the 25.7% of cases whose gold group missed
lose their recall entirely, and no second-tier ranking gain can buy that back.
Stratified: single-gold-group cases (n=165) miss their *only* group 36.4% of the
time (name-concatenation surfaces dilute against 6k-char group texts); cross-group
cases (n=135, 45% of the sample) hit any group easily (0.87+) but `p_all` collapses
to 0.29/0.05 — G=3 structurally caps their completeness.

Sensitivity arms (300 spread cases, same single-tier baseline):

| Arm | MRR | R@50 | `p_any` | tier-2 pool |
| --- | ---: | ---: | ---: | ---: |
| single (plain) | 0.554 | 0.685 | — | 6,006 |
| sub G2 names | 0.358 | 0.398 | 0.623 | 1,267 |
| sub G3 names (M1) | 0.416 | 0.478 | 0.743 | 1,811 |
| sub G5 names | 0.464 | 0.571 | 0.843 | 2,692 |
| sub G10 names | 0.528 | 0.653 | 0.953 | 4,363 |
| domain G3 names | 0.391 | 0.508 | 0.773 | 1,697 |
| sub G3 **llm** | 0.208 | 0.237 | 0.417 | 910 |
| single (docgen) | 0.786 | 0.866 | — | 6,006 |
| sub G3 names, docgen tier-2 | 0.592 | 0.563 | 0.743 | 1,811 |

Three sensitivity findings. (1) The G curve is cleanly monotone (G2→G10 climbs
0.358→0.528 MRR as `p_any` climbs 0.623→0.953) and G10 — where 4.7% entry loss
remains — still trails single tier: the mechanism story is internally consistent,
and even a near-perfect first tier cannot beat ranking over the full pool.
(2) The **llm surface collapses** (`p_any` 0.417 vs 0.743): one generated
use-when sentence per group has less inter-group lexical discrimination than
member-name concatenation, and OpenViking's L0-abstract idea presupposes
semantically coherent directories — SkillRet's `sub` labels are post-hoc
clusters of up to 1,416 heterogeneous skills, so no short description represents
them. A generated group description is only as good as the group's cohesion.
(3) The apparent docgen "win" (0.592 > 0.554) is a cross-corpus artifact; paired
against single-tier **docgen** — the correct comparison — two-tier loses
ΔMRR −0.194 [−0.24, −0.15], Δrecall@50 −0.303 [−0.35, −0.26]. docgen strengthens
tier two, which makes the fixed entry loss *relatively* more expensive: two-tier
strictly trails same-corpus single tier under every corpus × surface × G tried.

Verdict: the net structural signal (+0.12 MRR over random) is smaller than the
entry loss (−0.26 vs single). **Scope-before-rank needs either a near-perfect
first tier or gold that lives in one group; SkillRet has neither.** A caller with
hierarchic data should treat the first tier as a *hint source* (e.g. boost chosen
groups' members) rather than a hard filter.
Reports: `two-tier-skillret-{sub,random,domain}-G{2,3,5,10}-*.json` (gitignored).
Reproduce: `python -m experiments.benchmarks.two_tier --groupby sub --g 3
--surface names --corpus plain --limit 300 --output ...` then
`python -m experiments.benchmarks.ablation_stats pair --a <m1> --arm-a two_tier
--b <m1> --arm-b single`.

## Summary-instead-of-clip for the Jev state: premise fails (2026-10-22)

Second OpenViking borrow: replace the 2,000-char description clip in the Jev
state with an offline LLM summary (OpenViking's L1 overview idea), on the theory
that clipping mid-JSON discards exactly the declared-capability evidence Jev
judges. Pre-registered stop gate: measure the truncation rate over the actual
windows first; <20% clipped → the intervention has nothing to bite on.

Measured over the 100-case fusion docgen w20 windows (1,682 unique resources):
**2.6% clipped (44 resources)** — and the window's median description is 542
chars, *shorter than the 800–1,500-char summary target*. The clip almost never
bites on the current best stack, and a summary would lengthen 97% of inputs.
All 44 clips do land mid-JSON (splitting the documentation), so on a domain with
longer documents the motivation would stand — but here the premise gate stops the
experiment before ~1,600 generation calls and a 3-arm Jev comparison. Runner
support (`--state-desc {truncate,summary,truncate-matched}` with a
length-matched control arm) is in place and tested for the day a longer-document
domain shows up. Report: `summarize --measure-only` (see the tool's output).

## SiliconFlow bge-reranker-v2-m3 finally measured: valid endpoint, harmful direction — rejected (2026-10-31)

The default rerank endpoint had never been exercised (the only tested one, paratera
`GLM-Rerank`, was invalid). Pre-registered gates in issue #3: per-tier paired
bootstrap CI, NDCG@10 primary, seven-metric zero-regression, constant-score
precheck, stop gates. SiliconFlow `BAAI/bge-reranker-v2-m3` over fusion-docgen
top-100 windows, same case sampler as the Jev reports (cross-report pairing by
case id works — the SkillRet baseline metrics reproduce the Jev report's bit for
bit).

The endpoint is *valid* this time: 420 calls, zero constant-score responses, zero
failures (`degraded_cases` = 0 on all three tiers; the probe discriminates 0.087
vs 0.0). But the direction is *harmful everywhere*, significantly:

| Tier | cases | NDCG@10 Δ (rerank−fusion) | CI95 | verdict |
|---|---|---|---|---|
| ToolRet w/ instruction | 20 (S2 stop gate) | −0.078 | [−0.146, −0.017] | harmful (upper < 0) |
| ToolRet w/o instruction | 100 | −0.053 | [−0.107, −0.000] | harmful (upper < 0) |
| SkillRet | 300 | −0.128 | [−0.162, −0.093] | harmful (all-negative CI) |

All seven metrics ≤ 0 on every tier (adopt gate needed CI lower bound > 0 on at
least one tier — zero tiers came close). Against Jev on the shared cases the
cross-encoder also loses everywhere: ToolRet instruction −0.116 NDCG@10;
SkillRet −0.142 (vs w20) / −0.152 (vs w50), all-negative CIs.

Interpretation (extends Law 1): a stage-2 ranker pays only when it judges a
dimension stage 1 cannot see. bge's training objective is query–document
*semantic relatedness* — largely the dimension BM25+fusion already rank by —
while both benchmarks' gold is *task executability*, exactly the dimension Jev's
capability framing adds. On top of that the sigmoid scores cluster (0.087 vs 0.0
on the probe), ties reorder by noise, and the 100-wide window churns a head the
first stage already ordered well.

Cost (measured): ~1.9 s per call, consistent across tiers; the endpoint reports
no token usage (free tier). Char volume ≈ 79–110k chars per case (~5× the Jev
state); at the Opper reference $0.11/M input ≈ 26k tokens/case ≈ $2.9 per 1k
queries — cost was never the constraint, direction was.

paratera `GLM-Rerank` re-tested during setup: 4/20 constant-score degraded cases
and every metric below baseline — the earlier invalid verdict stands.

**Verdict: the API reranker lever is not adopted on either benchmark's evidence
domain; Jev@20 remains the only validated stage-2 component.** Runner:
`experiments.benchmarks.rerank_paired` (per-case paired rows, constant-score and
abort semantics inherited from `run_rerank_arm`; `rerank_detailed` also passes
through endpoint usage). Reports: `rerank-pilot-sf-toolret-inst-20.json`,
`rerank-sf-toolret-query-100.json`, `rerank-sf-skillret-300.json` (regenerable,
gitignored). Reproduce: `python -m experiments.benchmarks.rerank_paired --bench
toolret --query-mode instruction --limit 20 --output ...` etc.

## Domain-tuned embedding arm: Edge adopted on the SkillRet face, ToolRet face is hardware-blocked (2026-11-01)

Ticket #8's paired-embedding study, run under the pre-registered gates (issue #9
comment: per-arm calibration stop gate, paired bootstrap 10k / seed 20261022,
NDCG@10 primary + six-metric zero-regression). Outcome: **one arm survives and
is adopted; everything else is falsified by measurement, not by intuition.**

**S1 calibration stop gate killed three candidates before any ranking run:**

| Candidate | Contract | Measured ms/doc | 44k (or 6k) extrapolation | Verdict |
|---|---|---|---|---|
| SKILLRET-Embedding-0.6B | Qwen3 decoder, bf16, 8k ctx | 79,702 | ≈ 40 days | dead |
| harrier-oss-v1-0.6b | same decoder family | not measured | — | dead by family inference |
| ToolRet-trained-bge-large (fp32 forced) | BERT 335M, CLS, 512 | 7,608 | ≈ 94 h | dead (gate: 6 h) |

Decoder-family embedding models are categorically corpus-infeasible on pure
CPU; even the sanctioned encoder replacement is 15x over the feasibility line
(a solo-machine re-test is estimated at 15–30 h, still far over). **ToolRet's
embedding-face upgrade is hardware-blocked, not evidence-blocked** — that is a
roadmap input (API-hosted domain-tuned embedders or GPU), recorded as such.

**Gate-implementation correction, disclosed before any new embedding run** (issue
#9 comment): the pre-registered gate text is "44k extrapolated >6 h"; the code
had hard-coded 500 ms/doc, which agrees for 44k faces but wrongly kills
face-restricted models. Amended to the extrapolation semantics (single-face
≤360 min). Under it, the replacement arm passed: SKILLRET-Edge-109M (82 MB,
CLS/256/bare-query, teacher-parity on the public split) calibrated at
837 ms/doc → 6,006-doc face ≈ 84 min, embedded, done.

**SkillRet face result (300 paired cases, docgen corpus, fusion arm with the
embedder as the only changed factor vs GLM-Embedding-3):**

| Metric | Δ (Edge − GLM) | CI95 | improved/tied/worsened |
|---|---|---|---|
| MRR | +0.0406 | [+0.0156, +0.0654] | 56/222/22 |
| recall@5 | +0.0967 | [+0.0683, +0.1261] | 60/233/7 |
| completeness@5 | +0.1300 | [+0.0900, +0.1733] | 43/253/4 |
| ndcg@5 | +0.0777 | [+0.0541, +0.1016] | 87/186/27 |
| recall@10 | +0.0822 | [+0.0533, +0.1122] | 53/240/7 |
| completeness@10 | +0.1067 | [+0.0667, +0.1500] | 37/258/5 |
| **ndcg@10 (primary)** | **+0.0730** | **[+0.0523, +0.0940]** | 104/168/28 |

All seven paired-difference CIs clear zero — the adoption gate (primary CI
lower bound > 0 + every secondary mean ≥ 0) passes with margin. Absolute
levels: Edge fusion NDCG@10 0.7833 / recall@5 0.7928 vs GLM 0.7104 / 0.6961.
The first-stage coverage bottleneck moves too: **gold-in-window@20 = 0.9667,
@50 = 0.99** (GLM's window-20 coverage was 0.793) — the 82 MB local model
nearly saturates the Jev window at a fraction of the API model's size.

This is the repo's first measured instance of **domain fine-tuning beating a
larger general API embedder inside the RRF fusion architecture** ("fine-tuning
beats scale" reproduces here with compounding, not cancellation).

Run architecture note: embedding (model-resident, memory-heavy) runs detached
in user PowerShell or background shells; retrieval consumes only the sqlite
vector cache via a miss-fails-fast lookup embedder, so ranking sessions never
load a model. fp32 loading is forced for encoder checkpoints — CPU has no
native fp16/bf16 and the emulated path was measured drastically slower.
Cache identity is `{model repo}:{dim}`; dtype is not in the identity — when a
dtype changes, the old rows are purged instead (edge's vectors are bf16-loaded,
documented; internally consistent per arm).

Cost (measured): Edge face embedding 84 min CPU once, then free forever;
GLM baseline = per-query API calls; bge = 94 h CPU — not a cost but a
feasibility wall. All numbers reproducible via
`experiments.benchmarks.embedding_paired` (`--embed-only` then default mode).

## Jev question type: score rubric vs noul probability — rejected; P(3)-fold follow-up candidate (2026-11-07)

Pre-registered in issue #13 (gates S0–S3 published before any run). Question: does replacing the
binary Noul question with a 4-level ordered Score rubric (0 Unrelated / 1 Topical only /
2 Partial coverage / 3 Direct coverage) improve reranking? Noul's answer is already a continuous
probability and `select_with_jev` argmaxes it, so the real comparison is **rubric degree-expression
vs binary-probability expression** under identical state/model/batching. Rubric level 2 shares the
noul criteria.true phrase verbatim ("perform a specific operation needed for the task"), making
P(≥2) the noul-aligned fold. Primary fold (pre-registered): expectation/3 normalized; sensitivity
folds recomputed offline from the stored distributions, not in the gate.

**Main surface (SkillRet docgen + fusion w20, 600 paired cases — one pre-registered expansion
after pilot power said n = 399):**

| Metric | Δ (score − noul) | CI95 | improved/tied/worsened |
|---|---|---|---|
| MRR | +0.0122 | [−0.0082, +0.0324] | 83/442/75 |
| recall@5 | −0.0017 | [−0.0158, +0.0125] | 14/567/19 |
| completeness@5 | +0.0017 | [−0.0150, +0.0183] | 13/575/12 |
| ndcg@5 | +0.0146 | [+0.0002, +0.0291] | 97/414/89 |
| recall@10 | −0.0044 | [−0.0114, +0.0028] | 2/588/10 |
| completeness@10 | −0.0083 | [−0.0183, +0.0017] | 2/591/7 |
| **ndcg@10 (primary)** | **+0.0130** | **[+0.0004, +0.0255]** | 102/388/110 |

Primary CI clears zero, but three secondary means are negative (recall@5, recall@10,
completeness@10) → the zero-regression requirement fails → adoption gate not met.

**Secondary surface (ToolRet docgen + fusion w20 instruction mode, 100 cases):** no signal —
NDCG@10 −0.0069 [−0.0354, +0.0214]; all seven CIs cross zero. gold-in-window@20 = 0.396: the
recall-bottleneck surface leaves the judge little to reorder (consistent with the earlier
"Jev gains grow as stage-1 recall weakens" law only up to the point where gold is absent).

**Well-formedness (S0/S1, all passed):** 0 malformed score distributions across 752 judged cases;
legend echoed verbatim in 100% of answers; per-answer fold vs endpoint's own expected score max
disagreement 0.01 (tolerance 0.02). Fold self-jitter (3 cases × 2 passes, 60 candidate scores):
max 0.0367 / mean 0.0083 — noul's own re-run jitter was max 0.05 / mean 0.0127. Non-gate
reproduction check: this run's jev_noul vs the published `jev-skillret-docgen-w20.json` jev arm,
300 shared cases — mean |Δ| ≤ 0.0051, all CIs straddle zero (frozen arm did not drift).

**Sensitivity folds (offline refold, NOT in the gate):**

| Fold | NDCG@10 Δ vs noul | CI95 | recall@10 Δ | CI95 |
|---|---|---|---|---|
| expected (primary) | +0.0130 | [+0.0004, +0.0255] | −0.0044 | [−0.0114, +0.0028] |
| P(≥2) (noul-aligned) | −0.0548 | [−0.0677, −0.0421] | −0.0325 | [−0.0456, −0.0208] |
| P(3) (top level only) | +0.0279 | [+0.0132, +0.0430] | +0.0067 | [−0.0008, +0.0150] |
| endpoint score | +0.0142 | [+0.0014, +0.0270] | −0.0044 | [−0.0114, +0.0028] |

**Abstention smoke (descriptive, no gate)** — 6 no-match queries against the SkillRet full-pool
fusion w20 window, window-max readings vs the 0.7 line: noul crossed 3/6, P(3) 1/6, P(≥2) 4/6.
On `chart_ambiguous` and `purple elephant` P(3) collapsed (0.36 / 0.39) where noul bound
(0.82 / 0.71). But on the bare greeting ("Hello, good morning") both bind — the score endpoint
asserts P(3) = 0.96 on no-task text, so score is not intrinsically safer against no-match
queries; its conservatism is specific to the direct-vs-partial discrimination, not to nonsense
detection.

Mechanism: the rubric's ranking value concentrates in the 2-vs-3 (partial vs direct coverage)
distinction. The expectation fold dilutes it with partial-coverage mass; the noul-aligned P(≥2)
fold is actively harmful (it re-creates noul's "at least one step" anchor inside a rubric that
also scores topical candidates at level 1... and partials at 2, drowning the direct signal);
the P(3) fold is stronger than the primary fold on every graded metric and nearly non-negative
on recall@10.

**Verdict: rejected** — score does not replace noul as the default question type (primary CI
passes, zero-regression fails; secondary surface shows no generalization). Follow-up candidate
(requires its own pre-registration): P(3)-fold score rubric on the main surface — the
pre-registered sensitivity data already show NDCG@10 CI > 0 with recall@10 CI barely crossing
zero at n = 600. The expected and P(≥2) folds should not be revisited.

Cost (measured): ≈ 8.3M input / ≈ 0.5M output tokens total (smoke ×2 + pilot 40 + main 600 +
ToolRet 100 + abstain). score/noul input ratio 1.38, output ratio 0.83, latency comparable —
below the 3× disclosure threshold.

Reports: `jev-qt-smoke.json`, `jev-qt-smoke2.json`, `jev-qt-skillret-pilot-40.json`,
`jev-qt-skillret-600.json`, `jev-qt-toolret-100.json`, `jev-qt-abstain.json`,
`jev-qt-skillret-600-refold-{at_least_2,top,endpoint}.json`.

Reproduce:

```powershell
python -m experiments.benchmarks.jev_question_paired --bench skillret --arm fusion --corpus docgen --limit 600 --window 20 --output experiments/jev-qt-skillret-600.json
python -m experiments.benchmarks.ablation_stats pair --a experiments/jev-qt-skillret-600.json --arm-a jev_score --b experiments/jev-qt-skillret-600.json --arm-b jev_noul --metrics mrr ndcg@5 ndcg@10 recall@5 recall@10 completeness@5 completeness@10
python -m experiments.benchmarks.jev_question_paired --bench toolret --arm fusion --corpus docgen --query-mode instruction --limit 100 --window 20 --output experiments/jev-qt-toolret-100.json
python -m experiments.benchmarks.jev_question_abstain --arm fusion --output experiments/jev-qt-abstain.json
```
