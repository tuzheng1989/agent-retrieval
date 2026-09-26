# Jev candidate judgment: synthetic smoke test

Run date: 2026-09-25. Model reported by the API: `jev-1.13.0`.

## Setup

- Eight synthetic resources and fourteen labeled queries in `jev_cases.json`.
- Existing `rank_candidates` in BM25-only mode supplies up to five candidates.
- One batched TypeSafe request per nonempty, non-exact shortlist; one Noul per candidate.
- A candidate is selected when its probability is at least the exploratory threshold
  of 0.7. Exact ID/name hits bypass Jev. No model call is made for an empty shortlist.
- The baseline selects the first retrieved candidate, or none if retrieval is empty.
  It does **not** include any host-specific `clearly_related` gate.

| Measure | BM25 baseline | BM25 + Jev |
| --- | ---: | ---: |
| Correct final selection, all 14 cases | 9/14 | 13/14 |
| Correct final selection, 11 positive cases | 9/11 | 10/11 |
| Incorrect binding, 3 no-match cases | 3/3 | 0/3 |
| Gold resource in top-five shortlist | 10/11 | 10/11 |

Jev corrected the `paraphrase` case, where BM25 ranked `doc-search` over
`geo-query`. It declined all three no-match cases. It could not answer the
`unrecalled` case, where BM25 returned an empty shortlist.

The run made 12 TypeSafe requests. API-reported usage was 10,674 input tokens and
864 output tokens. Total measured request time was 16.11 seconds, averaging
1.34 seconds per request; this is the sum of sequential request durations, not a
concurrent end-to-end latency measurement. No monetary cost is reported because
pricing was not measured in this experiment.

The stored probabilities yield the same 13/14 result with thresholds 0.3, 0.5,
and 0.7. A threshold of 0.9 drops to 12/14 because one correct positive was 0.86.
This is a sensitivity check on the synthetic cases, **not** threshold calibration.

## What this establishes

The TypeSafe API, SDK request format, candidate batching, result parsing, and
fallback boundaries work in this repository's experiment runner. Jev judged the
synthetic candidate descriptions as intended on this one run.

These results cannot establish production improvement: the examples are small,
synthetic, and were written for this experiment. The baseline also omits each
host's existing admission policy. Next, supply a separately labeled, representative
query set and the host's actual eligible resources. Calibrate the decision threshold
on a development split, then report recall, positive top-one accuracy, no-match
false-binding rate, latency, and token usage on a held-out split. If many positive
cases are absent from BM25 shortlists, improve recall before tuning Jev.

The detailed local JSON report is `jev-result.json` (gitignored because real
datasets may contain private queries). Reproduce with:

```powershell
python -m experiments.jev_retrieval --live --output experiments/jev-result.json
```
