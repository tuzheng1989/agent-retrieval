"""Compare the existing retrieval order with optional Jev candidate judgments.

This is a caller-side experiment. It deliberately leaves agent_retrieval.core untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from agent_retrieval import CandidateHit, rank_candidates


@dataclass(frozen=True)
class Resource:
    id: str
    name: str
    description: str
    use_when: str = ""
    avoid_when: str = ""


@dataclass(frozen=True)
class Case:
    id: str
    query: str
    gold: str | None


@dataclass(frozen=True)
class Judgment:
    probabilities: dict[str, float]
    model: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class Judge(Protocol):
    def judge(self, query: str, candidates: list[Resource]) -> Judgment: ...


def load_dataset(path: Path) -> tuple[list[Resource], list[Case]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    resources = [Resource(**row) for row in data["resources"]]
    cases = [Case(**row) for row in data["cases"]]
    resource_ids = [resource.id for resource in resources]
    case_ids = [case.id for case in cases]
    if not resources or not cases or len(resource_ids) != len(set(resource_ids)):
        raise ValueError("Dataset needs resources and cases with unique resource IDs")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Case IDs must be unique")
    if any(not resource.id or not resource.name or not resource.description for resource in resources):
        raise ValueError("Each resource needs an ID, name, and description")
    if any(not case.query or (case.gold is not None and case.gold not in resource_ids) for case in cases):
        raise ValueError("Each case needs a query and a known gold resource or null")
    return resources, cases


def retrieval_text(resource: Resource) -> str:
    # Exclusions inform Jev but must not promote a resource in lexical retrieval.
    return " ".join((resource.id, resource.name, resource.description, resource.use_when))


def shortlist(resources: list[Resource], query: str, k: int) -> list[CandidateHit[Resource]]:
    return rank_candidates(
        resources,
        query,
        corpus_text=retrieval_text,
        item_id=lambda resource: resource.id,
        item_name=lambda resource: resource.name,
    )[:k]


class JevJudge:
    def __init__(self) -> None:
        if not os.environ.get("TYPESAFE_API_KEY"):
            raise RuntimeError("Set TYPESAFE_API_KEY to run live Jev judgments")
        try:
            from typesafe_sdk import Noul, TypeSafeClient
        except ImportError as exc:
            raise RuntimeError("Install the optional SDK: pip install typesafe-sdk") from exc
        self._noul = Noul
        self._client_type = TypeSafeClient

    def judge(self, query: str, candidates: list[Resource]) -> Judgment:
        state = {
            "user_goal": query,
            "candidates": {
                f"c{index}": {
                    "name": resource.name,
                    "description": resource.description,
                    "use_when": resource.use_when,
                    "avoid_when": resource.avoid_when,
                }
                for index, resource in enumerate(candidates)
            },
        }
        questions = {
            f"c{index}": self._noul(
                instructions=(
                    f"Given `user_goal` and `candidates.c{index}`, can this resource "
                    "directly perform the user's requested task within its declared capabilities?"
                ),
                criteria={
                    "true": "The stated capability and intended use directly cover the task.",
                    "false": (
                        "Only similar wording or a related topic, an excluded use, or insufficient "
                        "evidence of the required capability."
                    ),
                },
            )
            for index in range(len(candidates))
        }
        started = perf_counter()
        with self._client_type() as client:
            response = client.system_one(state=state, questions=questions)
        elapsed = perf_counter() - started
        probabilities = {
            resource.id: float(response.answers[f"c{index}"].noul)
            for index, resource in enumerate(candidates)
        }
        usage = response.usage
        return Judgment(
            probabilities=probabilities,
            model=response.model,
            latency_seconds=elapsed,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )


def select_with_jev(
    hits: list[CandidateHit[Resource]], judgment: Judgment, threshold: float
) -> str | None:
    if hits and hits[0].exact:
        return hits[0].item.id
    if not hits:
        return None
    if set(judgment.probabilities) != {hit.item.id for hit in hits}:
        raise ValueError("Jev must return one probability per shortlisted candidate")
    if any(not 0 <= probability <= 1 for probability in judgment.probabilities.values()):
        raise ValueError("Jev probabilities must be within [0, 1]")
    # Python max preserves the retrieval order when probabilities tie.
    best = max(hits, key=lambda hit: judgment.probabilities[hit.item.id])
    return best.item.id if judgment.probabilities[best.item.id] >= threshold else None


def _metrics(rows: list[dict[str, object]], field: str) -> dict[str, float | int | None]:
    positives = [row for row in rows if row["gold"] is not None]
    negatives = [row for row in rows if row["gold"] is None]
    return {
        "top1_accuracy": sum(row[field] == row["gold"] for row in rows) / len(rows),
        "positive_top1_accuracy": (
            sum(row[field] == row["gold"] for row in positives) / len(positives)
            if positives else None
        ),
        "negative_false_bind_rate": (
            sum(row[field] is not None for row in negatives) / len(negatives)
            if negatives else None
        ),
    }


def run_experiment(
    resources: list[Resource],
    cases: list[Case],
    *,
    k: int,
    threshold: float,
    judge: Judge | None = None,
) -> dict[str, object]:
    if k < 1 or not 0 <= threshold <= 1:
        raise ValueError("k must be positive and threshold must be within [0, 1]")
    rows: list[dict[str, object]] = []
    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    for case in cases:
        hits = shortlist(resources, case.query, k)
        ids = [hit.item.id for hit in hits]
        row: dict[str, object] = {
            "id": case.id,
            "query": case.query,
            "gold": case.gold,
            "shortlist": ids,
            "gold_in_shortlist": case.gold in ids if case.gold is not None else None,
            "baseline": ids[0] if ids else None,
        }
        if judge is not None:
            if hits and not hits[0].exact:
                judgment = judge.judge(case.query, [hit.item for hit in hits])
                row["jev"] = select_with_jev(hits, judgment, threshold)
                row["probabilities"] = judgment.probabilities
                row["model"] = judgment.model
                row["latency_seconds"] = judgment.latency_seconds
                total_latency += judgment.latency_seconds
                total_input_tokens += judgment.input_tokens or 0
                total_output_tokens += judgment.output_tokens or 0
            else:
                row["jev"] = row["baseline"]
                row["jev_skipped"] = "exact_match" if hits else "empty_shortlist"
        rows.append(row)

    positives = [row for row in rows if row["gold"] is not None]
    result: dict[str, object] = {
        "dataset": {"resources": len(resources), "cases": len(cases), "positives": len(positives)},
        "configuration": {"k": k, "jev_threshold": threshold, "retriever": "BM25"},
        "candidate_recall_at_k": (
            sum(row["gold_in_shortlist"] is True for row in positives) / len(positives)
            if positives else None
        ),
        "baseline": _metrics(rows, "baseline"),
        "cases": rows,
    }
    if judge is not None:
        result["jev"] = _metrics(rows, "jev")
        result["jev_usage"] = {
            "requests": sum("probabilities" in row for row in rows),
            "latency_seconds": total_latency,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("jev_cases.json"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Exploratory Jev decision threshold; calibrate on labeled data")
    parser.add_argument("--live", action="store_true", help="Call Jev (requires typesafe-sdk and API key)")
    parser.add_argument("--output", type=Path, help="Write the full JSON report to this file")
    args = parser.parse_args()
    try:
        resources, cases = load_dataset(args.dataset)
        judge = JevJudge() if args.live else None
        result = run_experiment(resources, cases, k=args.k, threshold=args.threshold, judge=judge)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Experiment error: {exc}", file=sys.stderr)
        return 2
    report = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
