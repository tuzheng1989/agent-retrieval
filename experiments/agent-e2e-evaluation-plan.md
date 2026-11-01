# Agent task evaluation using agent-retrieval

Detailed DeepAgents implementation design: [agent-e2e-evaluation-design.md](agent-e2e-evaluation-design.md).

## Scope and decision rule

ToolRet and SkillRet measure candidate rankings. The product decision is whether
retrieval helps a **host Agent complete a user's task**. Run this evaluation in a
host that can execute the Agent and observe its tool calls and resulting state;
this repository alone has no Agent loop or tool runtime.

Use **safe task success rate** as the primary measure:

```
successful episodes with all required end-state checks true,
the expected terminal behavior, no forbidden action, and completion
within the agreed budget
----------------------------------------------------------------------
all evaluated episodes
```

For an unsupported request, the required result can be a clear refusal or
clarifying question **without** an irrelevant tool call. A correct final sentence
does not make an incorrect side effect successful. Define budgets per host before
the comparison; there is no universal time or token cutoff.

## Evaluation unit and labels

Each case is one replayable Agent episode, not just a query string. Store:

```json
{
  "id": "schedule-001",
  "slice": "multi_step",
  "user_request": "Find a free hour with Mei next Tuesday and schedule a meeting",
  "clock": "2026-09-27T09:00:00+08:00",
  "context_fixture": "calendar-fixture-v3",
  "catalog_snapshot": "tools-v42",
  "permissions": ["calendar.read", "calendar.write"],
  "required_state": [
    {"kind": "calendar_event_exists", "attendees": ["Mei"], "day": "next Tuesday"}
  ],
  "forbidden_effects": ["email_sent", "event_on_wrong_day"],
  "acceptable_behavior": "complete"
}
```

`required_state` and `forbidden_effects` are evaluated against the tool sandbox
or replayed external state. For a no-capability case, use an empty required state
and `acceptable_behavior: "decline_or_clarify"`. For ambiguous requests, specify
which missing fact requires clarification. Multiple valid tool sequences are
allowed; do not use one exact sequence as the gold answer.

Build a first set of roughly 50–100 labeled episodes from de-identified real
requests. Include simple requests, multi-step tasks, exact resource mentions,
paraphrases, no matching resource, ambiguous requests, permission-denied cases,
tool failures, and multilingual wording if present in traffic. Oversample rare
failures for diagnosis, but report production-weighted totals separately. Keep
calibration and held-out cases separate; do not tune Jev questions or thresholds
on the held-out set.

## Paired experiment

For each episode, run at least these variants against the **same** Agent model,
prompt, catalog, permissions, tool fixtures, execution budget, and embedding
snapshot:

1. Reference fusion retrieval with fixed eligibility filtering and no extra semantic admission gate; the repository does not contain a production caller policy.
2. Same fusion candidates plus Jev reranking only; evaluate Jev admission in a separate arm.
3. If relevant, a dedicated reranker at the same candidate window and budget.

Freeze the retrieval window (for example, top 20) across reranking variants.
Run each variant on the same episodes; repeat stochastic Agent runs (for example,
three times) and report both mean success and the share of cases that succeed on
**every** repeat. Use paired per-episode differences with confidence intervals.
State clearly which runs share cached embeddings and which incur cold-start cost.

## Reported measures

| Layer | Measure | Why it matters |
| --- | --- | --- |
| Outcome (primary) | Safe task success rate | Did the user goal reach the right state without a forbidden effect? |
| Outcome | Correct abstain/clarify rate | Unsupported or underspecified requests must not trigger unrelated tools. |
| Retrieval diagnostic | Gold or sufficient-resource coverage@window | Separates missing candidates from Agent decision failures; use only where a resource label is meaningful. |
| Agent decision | Invalid/unauthorized tool-call rate; wrong-argument rate | Identifies failures after a correct candidate was available. |
| Efficiency | End-to-end p50/p95 latency, tokens, cost, tool calls | Measures the actual tradeoff of Jev or a reranker. |
| Reliability | Per-run success and all-repeats success | Exposes stochastic regressions hidden by a single successful run. |

For open-ended answers, add an evidence-grounding rubric only when mechanical
state checks cannot decide success. Calibrate any model judge against human labels
and blind it to variant names. Retain negative-case and permission-case rates as
separate guardrails even if the overall success rate improves.

## Required trace fields

Record `case_id`, variant and version IDs, query/context snapshot, eligible
resource IDs, full retrieval order and scores, reranker/Jev judgments, chosen
resource and arguments, tool observations, final state/output, elapsed time,
token/cost totals, and errors. Keep raw traces for case-by-case attribution:

```
candidate absent → candidate present but not selected → wrong arguments/tool
→ tool failure → final answer/state failure
```

The retrieval package should continue to own ranking evidence. The host owns
eligibility, binding, execution, and this end-to-end outcome evaluation.
