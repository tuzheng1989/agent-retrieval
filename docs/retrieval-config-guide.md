# Retrieval Configuration Guide

How to configure the retrieval stack for *your* domain. Every recommendation
here is backed by a paired ablation on public benchmarks (SkillRet, 6,006-skill
pool; ToolRet, 44,453-tool pool; 300/100 query samples) — see
[experiments/jev-benchmark-findings.md](../experiments/jev-benchmark-findings.md)
for the full numbers. **There is no universally best configuration**: the two
benchmarks picked *different* winners, and the rule that explains both is the
real deliverable.

## The configuration surface

| # | Knob | Where it lives | Cost profile |
|---|---|---|---|
| 1 | Corpus projection (`corpus_text`) | caller function | free |
| 2 | doc2query corpus expansion (`docgen.py`, offline) | index build time | one-time LLM batch |
| 3 | Vector path (`vector=` on/off, embedder choice) | `rank_candidates` | one-time corpus embedding + per-query compute |
| 4 | Jev re-rank (System One, top-N window) | caller-side, post-retrieval | per-query LLM call (~5k tok, ~1 s) |
| 5 | Window size N for stage-2 | caller parameter | token-linear |
| 6 | BM25 parameters (k1/b, term filters) | `BM25Index` constructor | free |

The kernel stays stateless and dependency-free; every knob above is a caller
decision. That is deliberate — the ablations below are exactly why these
cannot be hard-coded.

## Three laws (learned by falsification, not intuition)

**Law 1 — Jev's gain scales with retrieval weakness.**
SkillRet (weak→strong after docgen): +13% recall@5 before docgen, *negative*
MRR marginal after. ToolRet (still weak): +24% recall@5 on BM25, +17% MRR on
fusion. A stage-2 judge pays for itself exactly where stage-1 is weakest.

**Law 2 — docgen's gain scales with vocabulary poverty.**
SkillRet (one-line descriptions): BM25 Recall@5 **+51%**, and it *overtook*
plain fusion. ToolRet (vocabulary-rich documentation JSON): +3.8% only, and
Fusion Recall@50 slightly dropped (expansion noise pushes long-tail gold out
of deep positions).

**Law 3 — the vector path survives or dies by domain ablation.**
After docgen, SkillRet's vector path was fully redundant (BM25 alone beat
plain fusion). ToolRet still needs it (+16% Recall@5 over docgen+BM25) —
parameter semantics inside documentation JSON are not covered by colloquial
generated queries. Never assume; run the ablation.

Falsified on both benchmarks (do not reach for these): field-level multi-path
RRF (unweighted merge lets narrow-field noise win), Jev-triggered iterative
retrieval (Jev confidence does not signal "gold absent from window" — 0%
trigger rate across τ ∈ {0.3, 0.5, 0.7} on both benchmarks).

## Decision guide

Start from the row that matches your constraints, then verify with the
validation workflow at the end:

| Your situation | Recommended stack | Expected (SkillRet / ToolRet R@5) |
|---|---|---|
| Default, balanced | docgen + BM25 + Jev@20 | 0.763 / 0.424 |
| Latency-bound (no per-query LLM) | docgen + BM25 (no Jev) | 0.724 / 0.329 |
| Coverage-bound (answer must be present) | docgen + fusion + Jev@50 | 0.769* / 0.424†, R@10 0.834† |
| No LLM budget at all | docgen + BM25 | 0.724 / 0.329 |
| Baseline (do not ship this if you can do better) | plain fusion | 0.634 / 0.353 |

\* SkillRet measured; † ToolRet measured (the two benchmarks are the range
boundaries — your domain lands between them).

## Component reference

### doc2query expansion — use when descriptions are short or jargon-heavy
- Generate 3–5 colloquial questions per resource offline; append to the
  **retrieval face only** (doc2query-- pattern). Keep generated text out of
  admission/judgment evidence — the kernel's `declared=` projection is the
  hook for that separation in production.
- Turn OFF / skip when: documentation is already vocabulary-rich (ToolRet-like)
  and you only care about deep recall (Fusion R@50 can drop ~1.7pp from
  expansion noise).
- Model note: reasoning-style models (GLM-4.5/5.3) spend their max_tokens
  budget on hidden thinking — disable thinking tokens and budget ≥2,000, or
  content comes back silently empty (finish_reason=length).
- Results persist with the resource; re-embedding on corpus change is handled
  by content-addressed snapshots.

### Vector path — keep where the domain is heterogeneous or jargon-dense
- OFF is a legitimate production choice on docgen-expanded, short-declaration
  domains (SkillRet: BM25 alone beat plain fusion on every metric).
- ON where documentation embeds structured semantics (parameters, schemas):
  ToolRet docgen+Fusion beats docgen+BM25 by +16% Recall@5.
- Embedder choice: swap via `EmbeddingConfig` identity — content-addressed
  snapshots make old/new coexist and rollback trivial.

### Jev re-rank — use where retrieval is weak; skip where it is strong
- Judge "can this resource perform a concrete step of the task", NOT keyword
  relevance — the capability framing is what adds a dimension lexical and
  vector scores do not have.
- Window 20 is the operating point (window 50 bought +1pp recall at 2× tokens
  and −0.02 MRR from diluted discrimination).
- Know the failure signal problem: Jev's confidence does NOT mean "gold is in
  the window" — it is almost always ≥0.5 when *any* partially-related candidate
  exists. Do not use it as a recall-failure trigger.
- Cap per-candidate descriptions (~2,000 chars) — long-tail documents blow the
  System One input budget (HTTP 400).

### Reranker API (cross-encoder) — validated path, unvalidated model
- The client and two-stage arm are production-shaped (sparse-result alignment,
  retry with backoff, constant-score detection); the only endpoint tested
  (paratera GLM-Rerank) returned constant 1.0 scores and is reported invalid.
  Before trusting any reranker, check its scores are not constant — the runner
  now does this for you (`degraded_cases`).

## Validation workflow (run this for your own domain)

1. Establish the two anchors with `run_benchmark`: `--arm both --retriever
   single --k 5 10 20 50` on plain, then on docgen-expanded corpus. Read
   `recall@20/50` — that is your recall ceiling per configuration.
2. Apply **Law 1**: if first-stage Recall@5 < ~0.5, add the Jev window arm
   (`jev_rerank`); if > ~0.7, expect negative MRR marginal and skip it.
3. Apply **Law 2**: short/jargon descriptions → docgen first, it is the
   cheapest big win; rich documentation → expect single-digit gains, decide by
   whether R@50 moves.
4. Apply **Law 3**: with the winning corpus, compare `--retriever single` vs
   `fields` is NOT the vector test — compare BM25 vs fusion arms directly; if
   fusion ≤ BM25, drop the vector path and its embedding budget.
5. Never compare across sampling schemes or embedder identities; the kernel's
   determinism means every number above is exactly reproducible.

## Pitfalls worth their weight

- Reasoning models + small max_tokens = silently empty outputs
  (finish_reason=length). Disable thinking or raise the budget.
- Gateways report rate limits as generic 400s (not 429) — retry with backoff
  on any 4xx except auth, and pace under ~30 RPM for Flash-tier models.
- Unweighted RRF across many paths lets narrow-field noise win. Field-level
  fusion needs learned weights to beat a single well-projected corpus.
- Same-distribution caveat: LLM-generated evaluation queries flatter
  LLM-generated expansion (docgen). Direction holds; magnitudes are
  benchmark-specific.
