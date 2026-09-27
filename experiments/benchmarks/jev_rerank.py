"""Measure Jev reranking on ToolRet or SkillRet without changing the retrieval pool.

The same full ranking supplies three paired arms: baseline, Jev-reordered top N,
and an oracle that moves all labeled items in that window first. The oracle is a
recall ceiling, not an implementable system. No relevance threshold is applied:
binary benchmark recall measures ranking, not admission decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from agent_retrieval import QueryEmbeddingError, build_embedder, rank_candidates, vector_available

from experiments.benchmarks.dataset import Case, Resource
from experiments.benchmarks.embedder_cache import CachedEmbedder
from experiments.benchmarks.metrics import aggregate, score_case
from experiments.benchmarks.run_benchmark import (
    _DEFAULT_CACHE_PATH,
    _DEFAULT_DATA_DIR,
    case_query_text,
    embedder_from_env,
    load_benchmark,
    load_env_file,
)


def rerank_window(ranked: list[str], scores: dict[str, float], window: int) -> list[str]:
    head = ranked[:window]
    if set(head) != set(scores):
        raise ValueError("Jev must score every item in the reranking window")
    if any(not 0 <= score <= 1 for score in scores.values()):
        raise ValueError("Jev probabilities must be within [0, 1]")
    return sorted(head, key=lambda item_id: -scores[item_id]) + ranked[window:]


def oracle_window(ranked: list[str], golds: frozenset[str], window: int) -> list[str]:
    return sorted(ranked[:window], key=lambda item_id: item_id not in golds) + ranked[window:]


def jev_scores(case: Case, resources: list[Resource], *, bench: str) -> tuple[dict[str, float], dict]:
    from typesafe_sdk import Noul, TypeSafeClient

    state = {
        "task": case.query,
        "retrieval_instruction": case.instruction if bench == "toolret" else "",
        "candidates": {
            f"c{index}": {
                "name": resource.name,
                "description": resource.description,
                "tags": resource.tags,
            }
            for index, resource in enumerate(resources)
        },
    }
    questions = {
        f"c{index}": Noul(
            instructions=(
                f"Considering `task`, `retrieval_instruction`, and `candidates.c{index}`, "
                "is this resource suitable for at least one concrete step needed to "
                "fulfill the task? Judge its declared capability, not shared keywords."
            ),
            criteria={
                "true": "The resource can perform a specific operation needed for the task.",
                "false": "Only a related topic, a wrong operation, or insufficient stated capability.",
            },
        )
        for index in range(len(resources))
    }
    started = perf_counter()
    with TypeSafeClient() as client:
        response = client.system_one(state=state, questions=questions)
    elapsed = perf_counter() - started
    scores = {
        resource.id: float(response.answers[f"c{index}"].noul)
        for index, resource in enumerate(resources)
    }
    usage = {
        "model": response.model,
        "latency_seconds": elapsed,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return scores, usage


def _sample(dataset, limit: int):
    if limit < 1:
        raise ValueError("--limit must be positive")
    if limit >= len(dataset.cases):
        return dataset.cases
    if dataset.name == "toolret":
        # The ToolRet loader has already interleaved its 35 source tasks.
        return dataset.cases[:limit]
    # Spread SkillRet cases through its pinned test split rather than taking a
    # potentially category-concentrated prefix.
    return [dataset.cases[(index * len(dataset.cases)) // limit] for index in range(limit)]


def load_sampled_benchmark(name: str, data_dir: Path, limit: int):
    # SkillRet needs the full split in memory before sampling across it.
    # ToolRet limits during loading to retain its round-robin task sampling.
    dataset = load_benchmark(name, data_dir, None if name == "skillret" else limit)
    return dataset, _sample(dataset, limit)


def run(args: argparse.Namespace) -> dict:
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise RuntimeError("TYPESAFE_API_KEY is required")
    if args.window < max(args.k) or any(k < 1 for k in args.k):
        raise ValueError("--window must be at least the largest positive --k")
    if args.query_mode == "instruction" and args.bench != "toolret":
        raise ValueError("instruction query mode is ToolRet-only")

    dataset, cases = load_sampled_benchmark(args.bench, args.data_dir, args.limit)
    embedder = None
    cache = None
    embedder_identity = None
    if args.arm == "fusion":
        config = embedder_from_env()
        if vector_available(config) != "api":
            raise RuntimeError("AGENT_RETRIEVAL_EMBEDDING_API_KEY is required for fusion")
        live_embedder = build_embedder(config)
        assert live_embedder is not None
        try:
            probe = live_embedder.embed_query("benchmark preflight")
        except QueryEmbeddingError as exc:
            raise RuntimeError(f"Embedding preflight failed: {exc}") from exc
        config = replace(config, dimensions=len(probe))
        cache = CachedEmbedder(
            live_embedder,
            db_path=args.cache_path,
            identity=f"{config.provider}:{config.model_name}:{config.dimensions}",
        )
        embedder = cache
        embedder_identity = config.identity()

    k_values = tuple(sorted(set(args.k)))
    rows: list[dict] = []
    by_id = {resource.id: resource for resource in dataset.resources}
    started = perf_counter()
    try:
        for index, case in enumerate(cases, start=1):
            query = case_query_text(case, args.query_mode)
            hits = rank_candidates(
                dataset.resources,
                query,
                corpus_text=dataset.corpus_text,
                item_id=lambda resource: resource.id,
                item_name=lambda resource: resource.name,
                vector=embedder,
            )
            baseline = [hit.item.id for hit in hits]
            head = baseline[:args.window]
            if head:
                scores, usage = jev_scores(case, [by_id[item_id] for item_id in head], bench=args.bench)
                reranked = rerank_window(baseline, scores, args.window)
            else:
                scores, usage, reranked = {}, None, baseline
            oracle = oracle_window(baseline, case.golds, args.window)
            rows.append({
                "case_id": case.id,
                "gold_count": len(case.golds),
                "gold_in_window": len(case.golds.intersection(head)),
                "baseline": score_case(baseline, case.golds, k_values),
                "jev": score_case(reranked, case.golds, k_values),
                "oracle": score_case(oracle, case.golds, k_values),
                "baseline_top": baseline[:args.window],
                "jev_top": reranked[:args.window],
                "jev_scores": scores,
                "usage": usage,
            })
            print(f"[{args.bench}/{args.arm}] {index}/{len(cases)} cases", file=sys.stderr)
    finally:
        if cache is not None:
            cache.close()

    usage_rows = [row["usage"] for row in rows if row["usage"] is not None]
    return {
        "dataset": {
            "name": dataset.name,
            "revision": dataset.revision,
            "resources": len(dataset.resources),
            "cases": len(cases),
            "case_ids": [case.id for case in cases],
        },
        "configuration": {
            "arm": args.arm,
            "query_mode": args.query_mode,
            "window": args.window,
            "k_values": list(k_values),
            "sampling": (
                "spread across pinned test split" if args.bench == "skillret"
                else "round-robin source-task prefix"
            ),
            "embedder": embedder_identity,
            "model": sorted({row["model"] for row in usage_rows}),
        },
        "metrics": {
            name: aggregate([row[name] for row in rows], k_values)
            for name in ("baseline", "jev", "oracle")
        },
        "gold_in_window_fraction": (
            sum(row["gold_in_window"] for row in rows)
            / sum(row["gold_count"] for row in rows)
        ),
        "usage": {
            "requests": len(usage_rows),
            "latency_seconds": sum(row["latency_seconds"] for row in usage_rows),
            "input_tokens": sum(row["input_tokens"] or 0 for row in usage_rows),
            "output_tokens": sum(row["output_tokens"] or 0 for row in usage_rows),
            "total_wall_seconds": perf_counter() - started,
        },
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", choices=("toolret", "skillret"), required=True)
    parser.add_argument("--arm", choices=("bm25", "fusion"), default="fusion")
    parser.add_argument("--query-mode", choices=("query", "instruction"), default="query")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--k", nargs="+", type=int, default=(5, 10))
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--cache-path", type=Path, default=_DEFAULT_CACHE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_env_file()
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Jev benchmark error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("dataset", "configuration", "metrics", "usage")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
