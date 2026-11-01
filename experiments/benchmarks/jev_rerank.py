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
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from agent_retrieval import QueryEmbeddingError, build_embedder, rank_candidates, vector_available

from experiments.benchmarks.dataset import Case, Resource
from experiments.benchmarks.docgen import docgen_corpus_text, load_docgen
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


#: Per-candidate description cap in the Jev state. ToolRet's ``description`` is
#: the raw documentation JSON (long tail to ~14k chars); a 20-candidate window
#: of those blows System One's input budget (HTTP 400 ``max_tokens_exceeded``).
#: 2,000 chars per candidate keeps a 20-window near ~10k tokens — under the
#: ~12k the SkillRet w50 run already proved the endpoint accepts. SkillRet
#: descriptions are 10–2.3k chars, so only a tiny tail gets clipped there.
JEV_DESCRIPTION_MAX_CHARS = 2000


def jev_scores(
    case: Case,
    resources: list[Resource],
    *,
    bench: str,
    desc_provider: Callable[[Resource], str] | None = None,
) -> tuple[dict[str, float], dict]:
    from typesafe_sdk import Noul, TypeSafeClient

    resolve_desc: Callable[[Resource], str] = desc_provider or (
        lambda resource: resource.description[:JEV_DESCRIPTION_MAX_CHARS]
    )

    state = {
        "task": case.query,
        "retrieval_instruction": case.instruction if bench == "toolret" else "",
        "candidates": {
            f"c{index}": {
                "name": resource.name,
                "description": resolve_desc(resource),
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


def load_sampled_benchmark(name: str, data_dir: Path, limit: int,
                           corpus: str = "plain"):
    # SkillRet needs the full split in memory before sampling across it.
    # ToolRet limits during loading to retain its round-robin task sampling.
    dataset = load_benchmark(name, data_dir, None if name == "skillret" else limit)
    if corpus == "docgen":
        expansions = load_docgen(data_dir, name)
        if not expansions:
            raise ValueError(
                f"no docgen expansions found for {name} — run the docgen CLI first",
            )
        dataset = replace(
            dataset,
            revision=f"{dataset.revision}+docgen",
            corpus_text=docgen_corpus_text(dataset.corpus_text, expansions),
        )
    return dataset, _sample(dataset, limit)


def state_desc_provider(
    mode: str,
    summary_file: Path | None,
    matched_chars: int | None,
) -> tuple[Callable[[Resource], str] | None, Callable[[list[Resource]], int] | None]:
    """Build the state-description provider for ``--state-desc``.

    Returns ``(provider, fallback_counter)``; ``(None, None)`` is the default
    truncate-at-2,000 behavior. Summary mode falls back to the clip for
    resources without a generated summary and *counts* the misses — a >5%
    fallback rate would quietly turn the summary arm into the truncate arm.
    """
    if mode == "truncate":
        return None, None
    if mode == "truncate-matched":
        if not matched_chars or matched_chars < 1:
            raise ValueError("--state-desc truncate-matched needs a positive --matched-chars")
        return (lambda resource: resource.description[:matched_chars]), None
    if mode == "summary":
        if summary_file is None:
            raise ValueError("--state-desc summary needs --summary-file")
        from experiments.benchmarks.summarize import load_summaries

        summaries = load_summaries(summary_file)
        if not summaries:
            raise ValueError(f"no summaries loaded from {summary_file} — run summarize first")

        def provider(resource: Resource) -> str:
            return summaries.get(resource.id, resource.description[:JEV_DESCRIPTION_MAX_CHARS])

        def fallback_counter(resources: list[Resource]) -> int:
            return sum(1 for resource in resources if resource.id not in summaries)

        return provider, fallback_counter
    raise ValueError(f"unknown state-desc mode: {mode}")


def run(args: argparse.Namespace) -> dict:
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise RuntimeError("TYPESAFE_API_KEY is required")
    if args.window < max(args.k) or any(k < 1 for k in args.k):
        raise ValueError("--window must be at least the largest positive --k")
    if args.query_mode == "instruction" and args.bench != "toolret":
        raise ValueError("instruction query mode is ToolRet-only")
    desc_provider, summary_fallback_counter = state_desc_provider(
        args.state_desc, args.summary_file, args.matched_chars,
    )

    dataset, cases = load_sampled_benchmark(args.bench, args.data_dir, args.limit,
                                            corpus=args.corpus)
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
            identity="|".join(config.identity()),  # R5-P1：身份串含 pooling/前缀 flags
        )
        embedder = cache
        embedder_identity = config.identity()

    k_values = tuple(sorted(set(args.k)))
    rows: list[dict] = []
    by_id = {resource.id: resource for resource in dataset.resources}
    missing_summary_fallbacks = 0
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
                head_resources = [by_id[item_id] for item_id in head]
                scores, usage = jev_scores(
                    case, head_resources, bench=args.bench, desc_provider=desc_provider,
                )
                if summary_fallback_counter is not None:
                    missing_summary_fallbacks += summary_fallback_counter(head_resources)
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
    if args.state_desc == "summary" and missing_summary_fallbacks:
        window_slots = sum(len(row["baseline_top"]) for row in rows)
        fallback_rate = missing_summary_fallbacks / window_slots if window_slots else 0.0
        if fallback_rate > 0.05:
            print(f"[jev] WARNING: {fallback_rate:.1%} of window candidates had no "
                  "summary and fell back to the truncate clip", file=sys.stderr)

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
            "corpus": args.corpus,
            "window": args.window,
            "k_values": list(k_values),
            "state_desc": args.state_desc,
            "matched_chars": args.matched_chars if args.state_desc == "truncate-matched" else None,
            "summary_file": str(args.summary_file) if args.summary_file else None,
            "missing_summary_fallbacks": missing_summary_fallbacks,
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
    parser.add_argument("--corpus", choices=("plain", "docgen"), default="plain",
                        help="docgen mode appends doc2query generated queries to each "
                             "resource's retrieval corpus (docgen CLI output required)")
    parser.add_argument("--query-mode", choices=("query", "instruction"), default="query")
    parser.add_argument("--state-desc", choices=("truncate", "summary", "truncate-matched"),
                        default="truncate",
                        help="what each Jev-state candidate's description is: the 2,000-char "
                             "clip (default), a generated summary from --summary-file, or "
                             "the clip re-run at --matched-chars (length-matched control)")
    parser.add_argument("--summary-file", type=Path, default=None,
                        help="summarize CLI output jsonl (required with --state-desc summary)")
    parser.add_argument("--matched-chars", type=int, default=None,
                        help="clip length for --state-desc truncate-matched "
                             "(use the summary arm's p50 length)")
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
