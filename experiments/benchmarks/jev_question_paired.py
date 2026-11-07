"""noul vs score 题型配对实验 runner（运行内配对，同 state 仅 questions 不同）。

一次运行对同一 case 的同一 w20 窗口各发一次 noul 与 score 的 System One 请求
（两次独立调用、state 严格同一份），产出四臂逐 case 指标：baseline / jev_noul
（冻结模板，采纳对照）/ jev_score（4 档量规，主折叠 expected 归一）/ oracle。
统计出口直接吃本报告：``ablation_stats pair --a <本报告> --arm-a jev_score
--b <本报告> --arm-b jev_noul``（``agent_retrieval.stats.paired_report`` 按
case_id 配对）。

良构闸（预注册 S0/S1 同一条规则）：score 分布键集/和值出界即为 malformed，
malformed 占比 > 5% 退出码 2（端点批量形态不适用——不否决假设本身）。
judge 传输异常沿用 rerank_paired 语义：该 case 两判定臂退回 fusion 原序、冷却
重试下一个 case、连续 5 败中止。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

from agent_retrieval import QueryEmbeddingError, build_embedder, rank_candidates, vector_available

from experiments.benchmarks.embedder_cache import CachedEmbedder
from experiments.benchmarks.jev_questions import (
    SCORE_LEVELS,
    SCORE_RUBRIC,
    build_judge_state,
    build_noul_questions,
    build_score_questions,
    distribution_problems,
    fold_at_least,
    fold_expected,
    fold_top,
)
from experiments.benchmarks.jev_rerank import (
    JEV_DESCRIPTION_MAX_CHARS,
    load_sampled_benchmark,
    oracle_window,
    rerank_window,
)
from experiments.benchmarks.metrics import aggregate, score_case
from experiments.benchmarks.run_benchmark import (
    _DEFAULT_CACHE_PATH,
    _DEFAULT_DATA_DIR,
    case_query_text,
    embedder_from_env,
    load_env_file,
)

#: judge 传输异常的连续失败中止阈值（与 rerank_paired 同判）。
_MAX_CONSECUTIVE_FAILURES = 5
#: malformed score 分布占比闸（预注册 S0/S1：>5% → 端点批量形态不适用）。
_MALFORMED_GATE = 0.05
#: 主折叠与端点自算期望分的一致容差（量化噪声界，judge_parity 同源）。
_ENDPOINT_DISAGREEMENT_TOLERANCE = 0.02


@dataclass(frozen=True)
class PairJudgment:
    """一次双题型判定的结果（同一 state 的两次独立请求）。"""

    noul_scores: dict[str, float]
    #: 每候选的完整档位分布 {resource_id: {level: probability}}（折叠与敏感性分析的原料）。
    score_distributions: dict[str, dict[int, float]]
    #: 端点自算期望分（ScoreAnswer.score，∈ [0, 3] 未归一）。
    endpoint_scores: dict[str, float]
    #: legend 回显与量规逐字一致的候选占比（S0 读数）。
    legend_echo_fraction: float
    model: str
    noul_usage: dict
    score_usage: dict


def judge_pair(state: dict, count: int) -> PairJudgment:
    """测试缝：同 state 两次独立 system_one 调用（noul 与 score 各一请求）。

    两请求不混型（questions 理论上可同请求混型，但 noul 请求保持与冻结仪器
    逐位同形、usage 按题型分账、避免混型响应解析的未知风险）。
    """
    from typesafe_sdk import TypeSafeClient

    noul_questions = build_noul_questions(count)
    score_questions = build_score_questions(count)
    with TypeSafeClient() as client:
        started = perf_counter()
        noul_response = client.system_one(state=state, questions=noul_questions)
        noul_latency = perf_counter() - started
        started = perf_counter()
        score_response = client.system_one(state=state, questions=score_questions)
        score_latency = perf_counter() - started
    models = {noul_response.model, score_response.model}
    if len(models) != 1:
        raise RuntimeError(f"noul/score served by different models: {sorted(models)}")
    distributions: dict[str, dict[int, float]] = {}
    endpoint_scores: dict[str, float] = {}
    legend_hits = 0
    for index in range(count):
        answer = score_response.answers[f"c{index}"]
        distributions[f"c{index}"] = {int(level): float(p) for level, p in answer.probabilities.items()}
        endpoint_scores[f"c{index}"] = float(answer.score)
        legend = {int(level): str(text) for level, text in answer.legend.items()}
        if legend == dict(enumerate(SCORE_RUBRIC)):
            legend_hits += 1
    noul_scores = {
        f"c{index}": float(noul_response.answers[f"c{index}"].noul)
        for index in range(count)
    }

    def _usage(response, latency: float) -> dict:
        return {
            "model": response.model,
            "latency_seconds": latency,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    return PairJudgment(
        noul_scores=noul_scores,
        score_distributions=distributions,
        endpoint_scores=endpoint_scores,
        legend_echo_fraction=legend_hits / count,
        model=noul_response.model,
        noul_usage=_usage(noul_response, noul_latency),
        score_usage=_usage(score_response, score_latency),
    )


#: 主折叠选择（#14 起 P(3)=top 为确认臂；expected 是 #13 的原主折叠，保留向后兼容）。
_FOLD_FUNCTIONS = {"expected": fold_expected, "top": fold_top}


def fresh_sampled_benchmark(
    bench: str, data_dir: Path, limit: int, *, corpus: str, exclude_report: Path,
):
    """fresh 切片采样：pinned split 滤除 ``exclude_report`` 已用 case 后再 spread。

    P(3) 确认实验（#14）的确定性采样规则：fold 选择发生在 #13 的 600 case 上，
    确认样本必须与选样零重叠。滤除后列表仍是全 split 覆盖的分层 spread 样本。
    skillret-only：toolret 的 loader 在加载期就截断，滤除后无法保证剩余量。
    """
    if bench != "skillret":
        raise ValueError("--exclude-report sampling is SkillRet-only")
    dataset, _sampled = load_sampled_benchmark(bench, data_dir, limit, corpus=corpus)
    excluded = set(json.loads(exclude_report.read_text(encoding="utf-8"))["dataset"]["case_ids"])
    remaining = [case for case in dataset.cases if case.id not in excluded]
    if len(remaining) < limit:
        raise ValueError(
            f"only {len(remaining)} cases left after excluding {len(excluded)} from "
            f"{exclude_report.name} — need {limit}"
        )
    sampled = [remaining[(index * len(remaining)) // limit] for index in range(limit)]
    return dataset, sampled


def run(args: argparse.Namespace) -> dict:
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise RuntimeError("TYPESAFE_API_KEY is required")
    if args.window < max(args.k) or any(k < 1 for k in args.k):
        raise ValueError("--window must be at least the largest positive --k")
    if args.query_mode == "instruction" and args.bench != "toolret":
        raise ValueError("instruction query mode is ToolRet-only")

    fold_function = _FOLD_FUNCTIONS[args.fold]
    if args.exclude_report is not None:
        dataset, cases = fresh_sampled_benchmark(
            args.bench, args.data_dir, args.limit,
            corpus=args.corpus, exclude_report=args.exclude_report,
        )
    else:
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
            identity="|".join(config.identity()),
        )
        embedder = cache
        embedder_identity = config.identity()

    k_values = tuple(sorted(set(args.k)))
    by_id = {resource.id: resource for resource in dataset.resources}
    rows: list[dict] = []
    usage_totals = {
        "noul": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "latency_seconds": 0.0},
        "score": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "latency_seconds": 0.0},
    }
    malformed_score_cases = 0
    judge_error_cases = 0
    legend_echo_min = 1.0
    endpoint_disagreements = 0
    max_endpoint_disagreement = 0.0
    consecutive_failures = 0
    models: set[str] = set()
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
            noul_reranked = score_reranked = baseline
            noul_scores: dict[str, float] = {}
            score_distributions: dict[str, dict[int, float]] = {}
            endpoint_scores: dict[str, float] = {}
            disagreements: dict[str, float] = {}
            case_usage = None
            try:
                if head:
                    head_resources = [by_id[item_id] for item_id in head]
                    state = build_judge_state(
                        case.query,
                        case.instruction if args.bench == "toolret" else "",
                        head_resources,
                    )
                    judgment = judge_pair(state, len(head))
                    models.add(judgment.model)
                    legend_echo_min = min(legend_echo_min, judgment.legend_echo_fraction)
                    case_usage = {"noul": judgment.noul_usage, "score": judgment.score_usage}
                    for arm_name, arm_usage in case_usage.items():
                        usage_totals[arm_name]["requests"] += 1
                        usage_totals[arm_name]["input_tokens"] += arm_usage["input_tokens"] or 0
                        usage_totals[arm_name]["output_tokens"] += arm_usage["output_tokens"] or 0
                        usage_totals[arm_name]["latency_seconds"] += arm_usage["latency_seconds"]
                    # c{index} -> resource id 映射（窗口顺序即 state 候选顺序）。
                    noul_scores = {
                        head[int(key[1:])]: probability
                        for key, probability in judgment.noul_scores.items()
                    }
                    score_distributions = {
                        head[int(key[1:])]: distribution
                        for key, distribution in judgment.score_distributions.items()
                    }
                    endpoint_scores = {
                        head[int(key[1:])]: raw
                        for key, raw in judgment.endpoint_scores.items()
                    }
                    noul_reranked = rerank_window(baseline, noul_scores, args.window)
                    problems = [
                        resource_id
                        for resource_id, distribution in score_distributions.items()
                        if distribution_problems(distribution)
                    ]
                    if problems:
                        malformed_score_cases += 1
                        print(f"[{args.bench}] case {case.id}: {len(problems)} malformed score "
                              "distributions — score arm keeps fusion order", file=sys.stderr)
                    else:
                        fold_scores = {
                            resource_id: fold_function(distribution)
                            for resource_id, distribution in score_distributions.items()
                        }
                        score_reranked = rerank_window(baseline, fold_scores, args.window)
                        # 端点期望分对照只对 expected 折叠有意义（同定义）。
                        if args.fold == "expected":
                            disagreements = {
                                resource_id: abs(
                                    fold_expected(distribution)
                                    - endpoint_scores[resource_id] / (SCORE_LEVELS - 1)
                                )
                                for resource_id, distribution in score_distributions.items()
                            }
                            endpoint_disagreements += sum(
                                1 for gap in disagreements.values()
                                if gap > _ENDPOINT_DISAGREEMENT_TOLERANCE
                            )
                            max_endpoint_disagreement = max(
                                max_endpoint_disagreement, *disagreements.values(),
                            )
                        else:
                            disagreements = {}
                    consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001 — 传输层失败统一走降级路径
                consecutive_failures += 1
                judge_error_cases += 1
                print(f"[{args.bench}] case {case.id} judge failed, keeping fusion order: {exc}",
                      file=sys.stderr)
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    print(f"[{args.bench}] {consecutive_failures} consecutive failures — "
                          "endpoint is down, aborting", file=sys.stderr)
                    raise
                time.sleep(min(10.0 * consecutive_failures, 30.0))
            tail_mrr = _tail_mrr(baseline, case.golds, args.window)
            rows.append({
                "case_id": case.id,
                "gold_count": len(case.golds),
                "gold_in_window": len(case.golds.intersection(head)),
                "golds_in_window": sorted(case.golds.intersection(head)),
                "tail_mrr": tail_mrr,
                "baseline": score_case(baseline, case.golds, k_values),
                "jev_noul": score_case(noul_reranked, case.golds, k_values),
                "jev_score": score_case(score_reranked, case.golds, k_values),
                "oracle": score_case(oracle_window(baseline, case.golds, args.window),
                                     case.golds, k_values),
                "baseline_top": head,
                "noul_top": noul_reranked[:args.window],
                "score_top": score_reranked[:args.window],
                "noul_scores": noul_scores,
                "score_distributions": score_distributions,
                "endpoint_score": endpoint_scores,
                "fold_endpoint_max_disagreement": (
                    max(disagreements.values(), default=0.0)
                ),
                "usage": case_usage,
            })
            if index % 25 == 0:
                gc.collect()
            if index % 10 == 0:
                print(f"[{args.bench}/{args.arm}] {index}/{len(cases)} cases", file=sys.stderr)
    finally:
        if cache is not None:
            cache.close()

    judged_cases = sum(1 for row in rows if row["usage"] is not None)
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
            "state_desc": "truncate",
            "state_desc_max_chars": JEV_DESCRIPTION_MAX_CHARS,
            "question_types": ["noul", "score"],
            "score_rubric_levels": SCORE_LEVELS,
            "primary_fold": args.fold,
            "exclude_report": str(args.exclude_report) if args.exclude_report else None,
            "sampling": (
                ("fresh spread on pinned split minus excluded case ids"
                 if args.exclude_report else "spread across pinned test split")
                if args.bench == "skillret"
                else "round-robin source-task prefix"
            ),
            "embedder": embedder_identity,
            "model": sorted(models),
        },
        "metrics": {
            name: aggregate([row[name] for row in rows], k_values)
            for name in ("baseline", "jev_noul", "jev_score", "oracle")
        },
        "gold_in_window_fraction": (
            sum(row["gold_in_window"] for row in rows)
            / sum(row["gold_count"] for row in rows)
        ),
        "degradation": {
            "malformed_score_cases": malformed_score_cases,
            "judge_error_cases": judge_error_cases,
            "judged_cases": judged_cases,
            "malformed_fraction": (
                malformed_score_cases / judged_cases if judged_cases else 0.0
            ),
            "legend_echo_min": legend_echo_min,
            "endpoint_disagreements": endpoint_disagreements,
            "max_endpoint_disagreement": round(max_endpoint_disagreement, 4),
        },
        "usage": {
            **usage_totals,
            "total_wall_seconds": perf_counter() - started,
        },
        "cases": rows,
    }


def _tail_mrr(baseline: list[str], golds: frozenset[str], window: int) -> float:
    """尾部（窗口外）首个 gold 的完整序倒数排名——refold 重构 mrr 的尾部补给。"""
    for position, item_id in enumerate(baseline[window:], start=window + 1):
        if item_id in golds:
            return 1.0 / position
    return 0.0


def malformed_gate(report: dict) -> bool:
    """预注册 S0/S1 良构闸：malformed 占比 > 5% → True（端点批量形态不适用）。"""
    degradation = report["degradation"]
    return degradation["malformed_fraction"] > _MALFORMED_GATE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", choices=("toolret", "skillret"), required=True)
    parser.add_argument("--arm", choices=("bm25", "fusion"), default="fusion")
    parser.add_argument("--corpus", choices=("plain", "docgen"), default="docgen")
    parser.add_argument("--query-mode", choices=("query", "instruction"), default="query")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--fold", choices=("expected", "top"), default="expected",
                        help="score 臂的主折叠：expected = 期望分÷3（#13 原主折叠），"
                             "top = P(3)（#14 确认臂）")
    parser.add_argument("--exclude-report", type=Path, default=None,
                        help="fresh 采样：pinned split 滤除该报告已用 case 后再 spread"
                             "（skillret-only，fold 选择样本与确认样本零重叠）")
    parser.add_argument("--k", nargs="+", type=int, default=(5, 10))
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--cache-path", type=Path, default=_DEFAULT_CACHE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_env_file()
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Jev question-type benchmark error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if malformed_gate(report):
        degradation = report["degradation"]
        print(f"malformed gate tripped: {degradation['malformed_score_cases']}/"
              f"{degradation['judged_cases']} cases had malformed score distributions "
              f"({degradation['malformed_fraction']:.1%}) — score endpoint not usable "
              "in batch form (hypothesis NOT refuted)", file=sys.stderr)
        return 2
    print(json.dumps({key: report[key] for key in
                      ("dataset", "configuration", "metrics", "degradation", "usage")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
