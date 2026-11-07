"""noul vs score 题型对比实验的构造与折叠纯函数（零 IO、零环境读取）。

noul 侧逐字复刻 ``jev_rerank.jev_scores``（L83-96）的冻结模板——与
``judge_parity.legacy_judge_payload`` 的存档逐字节可对照（机器证明见单测）。
score 侧是 4 档有序量规：第 2 档逐字吸收 noul 的 TRUE 判据，使 ``P(>=2)`` 与
noul 概率语义对齐（敏感性折叠）；第 3 档锚定"直接完成整任务"（量规新增的
排序信息）；第 0+1 档分解 noul FALSE 判据的两个分支。

折叠规则：score 响应含全档分布，排序标量 = 折叠(分布)。主折叠
``expected``（期望分归一）预注册定死；其余折叠事后经 :func:`refold_report`
离线重算，零 API 重跑。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

from experiments.benchmarks.jev_rerank import JEV_DESCRIPTION_MAX_CHARS

#: noul 冻结模板（与 jev_rerank L83-96 逐字一致；措辞冻结，不得改动）。
NOUL_INSTRUCTIONS_TEMPLATE = (
    "Considering `task`, `retrieval_instruction`, and `candidates.c{index}`, "
    "is this resource suitable for at least one concrete step needed to "
    "fulfill the task? Judge its declared capability, not shared keywords."
)
NOUL_CRITERIA = {
    "true": "The resource can perform a specific operation needed for the task.",
    "false": "Only a related topic, a wrong operation, or insufficient stated capability.",
}

#: 与 noul 共享的前导段与收尾 guard（两臂差异集中在问句本身，措辞框架不变）。
_INSTRUCTIONS_PREFIX = (
    "Considering `task`, `retrieval_instruction`, and `candidates.c{index}`, "
)
_INSTRUCTIONS_GUARD = "Judge its declared capability, not shared keywords."
SCORE_INSTRUCTIONS_TEMPLATE = (
    _INSTRUCTIONS_PREFIX
    + "how far do this resource's declared capabilities go toward fulfilling the task? "
    + _INSTRUCTIONS_GUARD
)

#: 4 档量规（得分从 0 起）。第 2 档前半句逐字 = noul criteria.true；第 1 档与
#: 第 0 档合力逐分支分解 noul criteria.false；第 3 档 = 直接完成整任务。
SCORE_RUBRIC: tuple[str, str, str, str] = (
    (
        "Unrelated: the resource shares no meaningful connection with the task, "
        "or performs a wrong operation, or its stated capabilities exclude what "
        "the task needs."
    ),
    (
        "Topical only: the resource belongs to the same domain or shares salient "
        "keywords with the task, but states no capability needed for any concrete "
        "step of the task."
    ),
    (
        "Partial coverage: the resource can perform a specific operation needed "
        "for the task, but cannot directly complete the user's requested task by "
        "itself."
    ),
    (
        "Direct coverage: the resource itself directly performs the user's "
        "requested task within its declared capabilities."
    ),
)
SCORE_LEVELS = len(SCORE_RUBRIC)
#: 折叠归一分母（期望分 ∈ [0, 1] 直接满足 rerank_window 的值域不变式）。
_SCORE_SCALE = SCORE_LEVELS - 1

#: 良构容差：端点概率按 0.01 量化（judge_parity 实测），4 档求和的量化漂移界。
PROBABILITY_SUM_TOLERANCE = 0.03


def build_noul_questions(count: int) -> dict[str, object]:
    """noul 冻结模板逐字构造（count 个候选问题，键 c{index}）。"""
    from typesafe_sdk import Noul

    if count < 1:
        raise ValueError("question count must be positive")
    return {
        f"c{index}": Noul(
            instructions=NOUL_INSTRUCTIONS_TEMPLATE.format(index=index),
            criteria=dict(NOUL_CRITERIA),
        )
        for index in range(count)
    }


def build_score_questions(count: int) -> dict[str, object]:
    """score 量规构造（count 个候选问题，键与 noul 同型）。"""
    from typesafe_sdk import Score

    if count < 1:
        raise ValueError("question count must be positive")
    return {
        f"c{index}": Score(
            instructions=SCORE_INSTRUCTIONS_TEMPLATE.format(index=index),
            criteria=SCORE_RUBRIC,
        )
        for index in range(count)
    }


def build_judge_state(
    task: str,
    retrieval_instruction: str,
    resources,
    *,
    desc_provider: Callable | None = None,
) -> dict:
    """判定 state 构造（与 jev_rerank L71-82 逐字段一致：2000 字符截断 + tags）。"""
    resolve_desc = desc_provider or (lambda resource: resource.description[:JEV_DESCRIPTION_MAX_CHARS])
    return {
        "task": task,
        "retrieval_instruction": retrieval_instruction,
        "candidates": {
            f"c{index}": {
                "name": resource.name,
                "description": resolve_desc(resource),
                "tags": resource.tags,
            }
            for index, resource in enumerate(resources)
        },
    }


def distribution_problems(distribution: dict[int, float]) -> list[str]:
    """score 档位分布的良构检查：返回问题描述清单（空清单 = 良构）。"""
    problems: list[str] = []
    keys = set(distribution)
    if not keys:
        return ["empty distribution"]
    if not keys <= set(range(SCORE_LEVELS)):
        problems.append(f"keys outside [0, {SCORE_LEVELS - 1}]: {sorted(keys - set(range(SCORE_LEVELS)))}")
    if any(p < 0 for p in distribution.values()):
        problems.append("negative probability")
    total = sum(distribution.values())
    if not abs(total - 1.0) <= PROBABILITY_SUM_TOLERANCE:
        problems.append(f"probabilities sum to {total:.4f}")
    return problems


def fold_expected(distribution: dict[int, float]) -> float:
    """主折叠：期望分归一 ``(0*p0 + 1*p1 + 2*p2 + 3*p3) / 3`` ∈ [0, 1]。"""
    _check(distribution)
    return sum(level * p for level, p in distribution.items()) / _SCORE_SCALE


def fold_at_least(distribution: dict[int, float], level: int) -> float:
    """P(>= level)：缺失档按 0 计（分布允许键为子集）。"""
    _check(distribution)
    if not 0 <= level < SCORE_LEVELS:
        raise ValueError(f"fold level must be within [0, {SCORE_LEVELS - 1}]")
    return sum(p for key, p in distribution.items() if key >= level)


def fold_top(distribution: dict[int, float]) -> float:
    """P(顶档) = P(3)。"""
    return fold_at_least(distribution, SCORE_LEVELS - 1)


def _check(distribution: dict[int, float]) -> None:
    problems = distribution_problems(distribution)
    if problems:
        raise ValueError(f"malformed score distribution: {'; '.join(problems)}")


#: 敏感性折叠名 -> 每候选的折叠取值来源。``endpoint`` 用端点自算期望分
#: （ScoreAnswer.score ∈ [0,3]，同除以 3 归一），其余从落盘分布重算。
FOLD_NAMES = ("expected", "at_least_2", "top", "at_least_1", "one_minus_unrelated", "endpoint")


def refold_scores(
    fold: str,
    distributions: dict[str, dict[int, float]],
    endpoint_scores: dict[str, float],
) -> dict[str, float]:
    """按折叠名把每候选的判定读数折成 [0, 1] 排序标量（离线，不触 API）。"""
    if fold not in FOLD_NAMES:
        raise ValueError(f"unknown fold: {fold} (choose from {FOLD_NAMES})")
    scores: dict[str, float] = {}
    for resource_id in distributions:
        if fold == "endpoint":
            raw = endpoint_scores[resource_id]
            scores[resource_id] = raw / _SCORE_SCALE
        elif fold == "expected":
            scores[resource_id] = fold_expected(distributions[resource_id])
        elif fold == "at_least_2":
            scores[resource_id] = fold_at_least(distributions[resource_id], 2)
        elif fold == "top":
            scores[resource_id] = fold_top(distributions[resource_id])
        elif fold == "at_least_1":
            scores[resource_id] = fold_at_least(distributions[resource_id], 1)
        else:  # one_minus_unrelated
            scores[resource_id] = 1.0 - distributions[resource_id].get(0, 0.0)
    return scores


def _head_metrics(
    head: list[str],
    golds_in_window: list[str],
    gold_count: int,
    tail_mrr: float,
    k_values: tuple[int, ...],
) -> dict[str, float]:
    """窗口重排序上的七指标（完整序的精确重构，不需要尾部清单）。

    k <= window 时各指标只依赖窗口内位置与 gold 总数：recall 分母用 gold 总数，
    ndcg 的 idcg 用 min(gold_count, k)——故不能把 golds_in_window 直接喂
    score_case（它两个分母都取传入集合的大小）。mrr 的首个 gold 落在尾部时取
    ``tail_mrr``（尾部与各臂相同，位置不变）。
    """
    if not gold_count:
        metrics = {"mrr": 0.0}
        for k in k_values:
            metrics[f"recall@{k}"] = 0.0
            metrics[f"completeness@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
        return metrics
    golds = set(golds_in_window)
    mrr = next((1.0 / position for position, item in enumerate(head, start=1)
                if item in golds), 0.0)
    metrics = {"mrr": mrr or tail_mrr}
    for k in k_values:
        metrics[f"recall@{k}"] = len(golds.intersection(head[:k])) / gold_count
        # gold 有窗口外部分时 completeness 结构性不可达（窗口外 gold 不在任何
        # k <= window 前缀里）。
        metrics[f"completeness@{k}"] = (
            1.0 if gold_count == len(golds) and golds.issubset(head[:k]) else 0.0
        )
        dcg = sum(1.0 / math.log2(position + 2)
                  for position, item in enumerate(head[:k]) if item in golds)
        idcg = sum(1.0 / math.log2(position + 2) for position in range(min(gold_count, k)))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return metrics


def refold_report(
    report_path: Path,
    fold: str,
    *,
    output_path: Path | None = None,
) -> dict:
    """从已落盘报告离线重折叠出敏感性臂（零 API）。

    返回（并可写 ``output_path``）``{"cases": [{"case_id", "jev_score_<fold>":
    {七指标}}]}`` 形状的报告，臂名供 ``ablation_stats pair --arm-a`` 直接消费。
    重算折叠与报告记录的主折叠（``configuration.primary_fold``）同名时，逐 case
    断言重算值与主折叠臂存档一致——重构数学（窗口子集指标 + tail_mrr）的自校验。
    """
    report = json.loads(report_path.read_text(encoding="utf-8"))
    primary_fold = report.get("configuration", {}).get("primary_fold", "expected")
    k_values = tuple(report["configuration"]["k_values"])
    window = report["configuration"]["window"]
    rows: list[dict] = []
    for row in report["cases"]:
        distributions = {
            resource_id: {int(level): p for level, p in dist.items()}
            for resource_id, dist in row.get("score_distributions", {}).items()
        }
        endpoint = row.get("endpoint_score", {})
        head = []
        if distributions and not any(distribution_problems(dist) for dist in distributions.values()):
            # 含 malformed 分布的 case 与 runner 主折叠同语义：整 case 保 fusion 序。
            scores = refold_scores(fold, distributions, endpoint)
            head = sorted(row["baseline_top"], key=lambda resource_id: -scores[resource_id])
        else:
            head = list(row["baseline_top"])
        metrics = _head_metrics(
            head, row.get("golds_in_window", []), row["gold_count"],
            row.get("tail_mrr", 0.0), k_values,
        )
        if fold == primary_fold and distributions:
            stored = row["jev_score"]
            for name, value in metrics.items():
                if abs(value - stored[name]) > 1e-9:
                    raise ValueError(
                        f"refold reconstruction mismatch on {row['case_id']}/{name}: "
                        f"{value} != {stored[name]}"
                    )
        arm = f"jev_score_{fold}"
        rows.append({"case_id": row["case_id"], arm: metrics, "window": window})
    refolded: dict = {
        "configuration": {
            "refolded_from": report_path.name,
            "fold": fold,
            "arm": f"jev_score_{fold}",
        },
        "cases": rows,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(refolded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return refolded
