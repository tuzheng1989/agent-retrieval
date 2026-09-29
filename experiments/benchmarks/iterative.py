"""Jev-triggered iterative retrieval with PRF query expansion (experiments side).

Second retrieval improvement arm (plan: adaptive-hugging-candle). The first
round retrieves with the benchmark retriever; Jev then judges the top window.
When the window's best probability is below an explicit confidence threshold —
Jev's way of saying "nothing here can do the task" — the query is expanded with
pseudo-relevance feedback (Rocchio-style: frequent non-query terms of the first
round's top documents) and retrieval runs a second round. Both windows are
merged, first round first.

PRF on purpose: it is deterministic, free, and isolates the *mechanism* (does a
second, better-informed round add recall?) from the cost of an LLM rewrite. A
low confidence never silently passes: the trigger rate and every degraded case
are counted in the report. ``--tau`` has no default — a threshold that can be
forgotten is a threshold that gets copied between call sites uncalibrated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

from agent_retrieval import Embedder, build_embedder, rank_candidates, vector_available

# ``_sample`` is jev_rerank's spread/prefix sampler — reused so both Jev
# evaluators measure the same 300-case spread for the same ``--limit``.
from experiments.benchmarks.embedder_cache import CachedEmbedder
from experiments.benchmarks.jev_rerank import _sample, jev_scores
from experiments.benchmarks.multipath import FieldRetriever
from experiments.benchmarks.run_benchmark import (
    case_query_text,
    embedder_from_env,
    field_paths,
    load_benchmark,
    load_env_file,
)
from experiments.benchmarks.metrics import aggregate, score_case

_DATA_DIR = Path(__file__).with_name("data")
_CACHE_PATH = Path(__file__).with_name(".cache").joinpath("vectors.sqlite3")


class QueryContext(NamedTuple):
    """What ``jev_scores`` needs to know about a query."""

    query: str
    instruction: str = ""


def prf_expand(
    query: str, documents: list[str], *, top_docs: int, max_terms: int,
) -> tuple[str, tuple[str, ...]]:
    """Append the most frequent non-query substantive terms of the top documents.

    Returns the expanded query and the terms added (in frequency order).
    """
    from agent_retrieval.core.bm25 import tokenize
    from agent_retrieval.core.resource_index import substantive_term

    query_terms = {term for term in tokenize(query) if substantive_term(term)}
    counts: Counter[str] = Counter()
    for text in documents[:top_docs]:
        for term in tokenize(text):
            if substantive_term(term) and term not in query_terms:
                counts[term] += 1
    terms = tuple(term for term, _ in counts.most_common(max_terms))
    return " ".join((query, *terms)).strip(), terms


def run_case(
    *,
    case,
    query: str,
    retrieve: Callable[[str], list[str]],
    judge: Callable[[str, list[str]], tuple[dict[str, float], dict]],
    window: int,
    tau: float,
    top_docs: int,
    max_terms: int,
    corpus_by_id: dict[str, str],
    k_values: tuple[int, ...],
) -> tuple[dict, dict | None]:
    """Baseline vs iterative vs oracle for one case, with trigger accounting."""
    first = retrieve(query)
    head = first[:window]
    usage: dict | None = None
    triggered = False
    terms: tuple[str, ...] = ()
    expanded = query
    if head:
        scores, usage = judge(query, head)
        best = max(scores.values())
        if best < tau:
            triggered = True
            expanded, terms = prf_expand(
                query,
                [corpus_by_id[item_id] for item_id in head[:top_docs]],
                top_docs=top_docs,
                max_terms=max_terms,
            )
    second = retrieve(expanded) if triggered else first
    merged: list[str] = []
    seen: set[str] = set()
    for item_id in [*first, *second]:
        if item_id not in seen:
            seen.add(item_id)
            merged.append(item_id)
    # ``first`` 就是完整的第一轮全池排序（确定性检索，无需重算）：
    # baseline 指标与 oracle 窗口都从它读取。
    oracle = sorted(first[:window], key=lambda item_id: item_id not in case.golds) + first[window:]
    row = {
        "case_id": case.id,
        "triggered": triggered,
        "expansion_terms": list(terms),
        "baseline": score_case(first, case.golds, k_values),
        "iterative": score_case(merged, case.golds, k_values),
        "oracle": score_case(oracle, case.golds, k_values),
    }
    return row, usage


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bench", choices=("toolret", "skillret"), required=True)
    parser.add_argument("--arm", choices=("bm25", "fusion"), default="fusion")
    parser.add_argument("--retriever", choices=("single", "fields"), default="single")
    parser.add_argument("--query-mode", choices=("query", "instruction"), default="query")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--tau", type=float, required=True,
                        help="Jev confidence gate: windows whose best probability is below "
                             "tau trigger a PRF-expanded second round (no default — "
                             "calibrate per query shape, then pin the value)")
    parser.add_argument("--top-docs", type=int, default=3)
    parser.add_argument("--max-terms", type=int, default=10)
    parser.add_argument("--k", nargs="+", type=int, default=(5, 10, 20, 50))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_env_file()
    k_values = tuple(sorted(set(args.k)))
    try:
        if not 0 <= args.tau <= 1:
            raise ValueError("--tau must be within [0, 1]")
        if args.window < max(k_values):
            raise ValueError("--window must be at least the largest --k")
        if not os.environ.get("TYPESAFE_API_KEY"):
            raise RuntimeError("TYPESAFE_API_KEY is required for the Jev confidence gate")
        dataset = load_benchmark(
            args.bench, _DATA_DIR, None if args.bench == "skillret" else args.limit,
        )
        cases = _sample(dataset, args.limit)
    except (OSError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"Iterative benchmark error: {exc}", file=sys.stderr)
        return 2

    field_retriever = (
        FieldRetriever(dataset.resources, field_paths(dataset))
        if args.retriever == "fields" else None
    )
    cache: CachedEmbedder | None = None
    embedder: Embedder | None = None
    if args.arm == "fusion":
        config = embedder_from_env()
        if vector_available(config) != "api":
            print("AGENT_RETRIEVAL_EMBEDDING_API_KEY is required for --arm fusion", file=sys.stderr)
            return 2
        live = build_embedder(config)
        assert live is not None
        try:
            probe = live.embed_query("preflight")
        except Exception as exc:
            print(f"Embedding preflight failed: {exc}", file=sys.stderr)
            return 2
        config = replace(config, dimensions=len(probe))
        cache = CachedEmbedder(
            live, db_path=_CACHE_PATH,
            identity=f"{config.provider}:{config.model_name}:{config.dimensions}",
        )
        embedder = cache

    def retrieve(query: str) -> list[str]:
        if args.retriever == "fields":
            assert field_retriever is not None
            return field_retriever.rank(query, embedder=embedder,
                                        corpus_text=dataset.corpus_text)
        hits = rank_candidates(
            dataset.resources, query, corpus_text=dataset.corpus_text,
            item_id=lambda resource: resource.id, item_name=lambda resource: resource.name,
            vector=embedder,
        )
        return [hit.item.id for hit in hits]

    def judge(query: str, item_ids: list[str]) -> tuple[dict[str, float], dict]:
        by_id = {resource.id: resource for resource in dataset.resources}
        return jev_scores(QueryContext(query), [by_id[i] for i in item_ids], bench=args.bench)

    corpus_by_id = {resource.id: dataset.corpus_text(resource) for resource in dataset.resources}
    rows: list[dict] = []
    started = perf_counter()
    triggered_count = 0
    try:
        for index, case in enumerate(cases, start=1):
            query = case_query_text(case, args.query_mode)
            row, usage = run_case(
                case=case, query=query, retrieve=retrieve, judge=judge,
                window=args.window, tau=args.tau, top_docs=args.top_docs,
                max_terms=args.max_terms, corpus_by_id=corpus_by_id, k_values=k_values,
            )
            triggered_count += row["triggered"]
            row["usage"] = usage
            rows.append(row)
            print(f"[{args.bench}/{args.arm}] {index}/{len(cases)} cases "
                  f"(triggered={row['triggered']})", file=sys.stderr)
    finally:
        if cache is not None:
            cache.close()

    report = {
        "dataset": {
            "name": dataset.name,
            "revision": dataset.revision,
            "resources": len(dataset.resources),
            "cases": len(cases),
            "case_ids": [case.id for case in cases],
        },
        "configuration": {
            "arm": args.arm,
            "retriever": args.retriever,
            "query_mode": args.query_mode,
            "window": args.window,
            "tau": args.tau,
            "top_docs": args.top_docs,
            "max_terms": args.max_terms,
            "k_values": list(k_values),
        },
        "metrics": {
            name: aggregate([row[name] for row in rows], k_values)
            for name in ("baseline", "iterative", "oracle")
        },
        "triggered_cases": triggered_count,
        "trigger_rate": triggered_count / len(cases) if cases else 0.0,
        "usage": {
            "requests": sum(1 for row in rows if row["usage"] is not None),
            "input_tokens": sum(row["usage"]["input_tokens"] or 0 for row in rows if row["usage"]),
            "output_tokens": sum(row["usage"]["output_tokens"] or 0 for row in rows if row["usage"]),
            "total_wall_seconds": round(perf_counter() - started, 1),
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("configuration", "metrics", "triggered_cases", "usage")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
