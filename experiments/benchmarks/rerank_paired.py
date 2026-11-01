"""Paired cross-encoder rerank experiment on the production-shaped rerank arm.

一次运行同时产出逐 case 的 fusion 基线与 rerank 序指标（运行内配对）；采样与既有
Jev@20 报告完全同源（``jev_rerank.load_sampled_benchmark``），因此 ``ablation_stats
pair`` 既能配 ``rerank`` vs ``baseline``（同报告），也能按 case_id 跨报告配
``rerank`` vs ``jev``（Jev@20 已发布数字）。常数分与连续失败的处理沿用
``run_benchmark.run_rerank_arm`` 的语义：保留 fusion 原序、计入 ``degraded``、
连续 5 败中止。成本数据走 :meth:`ApiReranker.rerank_detailed` 的实测 usage。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from agent_retrieval import (
    QueryEmbeddingError,
    build_embedder,
    rank_candidates,
    vector_available,
)

from experiments.benchmarks.dataset import BenchmarkDataset, Case
from experiments.benchmarks.embedder_cache import CachedEmbedder
from experiments.benchmarks.jev_rerank import load_sampled_benchmark
from experiments.benchmarks.metrics import aggregate, score_case
from experiments.benchmarks.reranker import (
    RERANK_DOC_FLOOR_CHARS,
    RERANK_PAIR_BUDGET_CHARS,
    ApiReranker,
    RerankError,
)
from experiments.benchmarks.run_benchmark import (
    _DEFAULT_CACHE_PATH,
    _DEFAULT_DATA_DIR,
    case_query_text,
    embedder_from_env,
    load_env_file,
    reranker_from_env,
)


def constant_scores(scores: list[float]) -> bool:
    """无区分度判定：有效分数不足两个，或极差小于阈值（实测 paratera GLM-Rerank
    恒返 1.0）。未被打分的 ``-inf`` 不参与极差——端点漏打分不是常数分。"""
    finite = [score for score in scores if score > float("-inf")]
    return len(finite) < 2 or max(finite) - min(finite) < 1e-9


def build_fusion_embedder() -> tuple[CachedEmbedder, str]:
    config = embedder_from_env()
    if vector_available(config) != "api":
        raise RuntimeError("AGENT_RETRIEVAL_EMBEDDING_API_KEY is required (fusion first stage)")
    live = build_embedder(config)
    assert live is not None  # vector_available said "api"
    try:
        probe = live.embed_query("benchmark preflight")
    except QueryEmbeddingError as exc:
        raise RuntimeError(f"Embedding preflight failed: {exc}") from exc
    config = replace(config, dimensions=len(probe))
    cache = CachedEmbedder(
        live,
        db_path=_DEFAULT_CACHE_PATH,
        identity=f"{config.provider}:{config.model_name}:{config.dimensions}",
    )
    return cache, config.identity()


def run_paired(
    dataset: BenchmarkDataset,
    cases: list[Case],
    *,
    embedder: CachedEmbedder,
    reranker: ApiReranker,
    rerank_candidates: int,
    k_values: tuple[int, ...],
    label: str,
    query_mode: str,
) -> dict:
    rows: list[dict] = []
    degraded = consecutive_failures = 0
    usage_totals = {"requests": 0, "prompt_tokens": 0, "total_tokens": 0,
                    "rerank_latency_seconds": 0.0, "pair_chars": 0}
    started = perf_counter()
    for index, case in enumerate(cases, start=1):
        query = case_query_text(case, query_mode)
        hits = rank_candidates(
            dataset.resources,
            query,
            corpus_text=dataset.corpus_text,
            item_id=lambda resource: resource.id,
            item_name=lambda resource: resource.name,
            vector=embedder,
        )
        baseline = [hit.item.id for hit in hits]
        window = hits[:rerank_candidates]
        # 精排输入按 query+document 总长预算截断（网关 code 1214 拒收超长对）。
        doc_cap = max(RERANK_DOC_FLOOR_CHARS, RERANK_PAIR_BUDGET_CHARS - len(query))
        case_degraded = False
        scores: list[float] | None = None
        usage = {"prompt_tokens": 0, "total_tokens": 0, "latency_seconds": 0.0}
        if window:
            docs = [dataset.corpus_text(hit.item)[:doc_cap] for hit in window]
            usage_totals["pair_chars"] += len(query) + sum(len(doc) for doc in docs)
            call_started = perf_counter()
            try:
                scores, usage_raw = reranker.rerank_detailed(query, docs)
            except RerankError as exc:
                consecutive_failures += 1
                case_degraded = True
                degraded += 1
                print(f"[{label}] case {case.id} rerank failed, keeping fusion order: {exc}",
                      file=sys.stderr)
                if consecutive_failures >= 5:
                    print(f"[{label}] 5 consecutive failures — rerank endpoint is down, aborting",
                          file=sys.stderr)
                    raise
                # 连续失败大概率仍在限流窗口内：冷却后再进下一个 case。
                time.sleep(min(10.0 * consecutive_failures, 30.0))
            else:
                consecutive_failures = 0
                usage = {
                    "prompt_tokens": usage_raw["prompt_tokens"],
                    "total_tokens": usage_raw["total_tokens"],
                    "latency_seconds": round(perf_counter() - call_started, 4),
                }
                usage_totals["requests"] += 1
                usage_totals["prompt_tokens"] += usage["prompt_tokens"]
                usage_totals["total_tokens"] += usage["total_tokens"]
                usage_totals["rerank_latency_seconds"] += usage["latency_seconds"]
                if constant_scores(scores):
                    case_degraded = True
                    degraded += 1
                    print(f"[{label}] case {case.id}: rerank scores are constant — "
                          "no discrimination, keeping fusion order", file=sys.stderr)
                    scores = None
        if scores is None:
            reranked = baseline
        else:
            head = sorted(
                zip(window, scores), key=lambda pair: (-pair[1], pair[0].item.id),
            )
            reranked = [hit.item.id for hit, _ in head]
            reranked += [hit.item.id for hit in hits[rerank_candidates:]]
        rows.append({
            "case_id": case.id,
            "gold_count": len(case.golds),
            "baseline": score_case(baseline, case.golds, k_values),
            "rerank": score_case(reranked, case.golds, k_values),
            "degraded": case_degraded,
            "rerank_usage": usage if window else None,
        })
        if index % 25 == 0:
            # 全池无状态检索每次重建索引，分代 GC 回收不及时——压住内存峰值。
            gc.collect()
        if index % 10 == 0:
            print(f"[{label}] {index}/{len(cases)} cases", file=sys.stderr)
    return {
        "metrics": {
            name: aggregate([row[name] for row in rows], k_values)
            for name in ("baseline", "rerank")
        },
        "degraded_cases": degraded,
        "usage": {
            **usage_totals,
            "rerank_latency_seconds": round(usage_totals["rerank_latency_seconds"], 1),
            "total_wall_seconds": round(perf_counter() - started, 1),
        },
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", choices=("toolret", "skillret"), required=True)
    parser.add_argument("--query-mode", choices=("query", "instruction"), default="query")
    parser.add_argument("--corpus", choices=("plain", "docgen"), default="docgen",
                        help="与 Jev@20 报告同语料（docgen），保证跨报告可比")
    parser.add_argument("--limit", type=int, default=20,
                        help="case 数：ToolRet 全量 100、SkillRet 全量 300（与 Jev 报告一致）")
    parser.add_argument("--rerank-candidates", type=int, default=100,
                        help="rerank 窗口（fusion top-N），默认 100 = ToolRet 官方二阶段口径")
    parser.add_argument("--k", nargs="+", type=int, default=(5, 10))
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.query_mode == "instruction" and args.bench != "toolret":
        print("Benchmark error: --query-mode instruction is ToolRet-only", file=sys.stderr)
        return 2
    k_values = tuple(sorted({k for k in args.k if k > 0})) or (5, 10)
    load_env_file()

    try:
        reranker = reranker_from_env()
        if not reranker._api_key:  # noqa: SLF001 — host 侧装配检查，与 run_benchmark 同判
            raise RuntimeError("AGENT_RETRIEVAL_RERANK_API_KEY not set")
        embedder, embedder_identity = build_fusion_embedder()
        dataset, cases = load_sampled_benchmark(
            args.bench, args.data_dir, args.limit, corpus=args.corpus,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Rerank paired error: {exc}", file=sys.stderr)
        return 2

    # 常数分预检：探针查询 + 三份梯度相关文档，端点无区分度则直接判无效（票面第一道闸）。
    try:
        probe_scores, _ = reranker.rerank_detailed(
            "benchmark preflight", ["alpha document", "beta document", "unrelated cooking recipe"],
        )
    except RerankError as exc:
        print(f"[rerank] preflight failed — fix the rerank config: {exc}", file=sys.stderr)
        return 2
    if constant_scores(probe_scores):
        print("[rerank] preflight scores are constant — endpoint gives no discrimination, "
              "aborting before spending the budget", file=sys.stderr)
        return 2

    try:
        report = run_paired(
            dataset, cases,
            embedder=embedder, reranker=reranker,
            rerank_candidates=args.rerank_candidates, k_values=k_values,
            label=f"{args.bench}/{args.query_mode}", query_mode=args.query_mode,
        )
    except RerankError as exc:
        print(f"Rerank paired error: endpoint aborted — {exc}", file=sys.stderr)
        return 2
    finally:
        embedder.close()

    output = {
        "dataset": {
            "name": dataset.name,
            "revision": dataset.revision,
            "resources": len(dataset.resources),
            "cases": len(cases),
            "case_ids": [case.id for case in cases],
        },
        "configuration": {
            "query_mode": args.query_mode,
            "corpus": args.corpus,
            "rerank_candidates": args.rerank_candidates,
            "k_values": list(k_values),
            "embedder": embedder_identity,
            "reranker": os.environ.get("AGENT_RETRIEVAL_RERANK_MODEL",
                                       "BAAI/bge-reranker-v2-m3"),
        },
        **report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(
        {"dataset": output["dataset"], "configuration": output["configuration"],
         "metrics": report["metrics"], "degraded_cases": report["degraded_cases"],
         "usage": report["usage"]},
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
