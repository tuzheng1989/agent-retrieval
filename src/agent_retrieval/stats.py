"""采纳门槛的配对统计工具。

配对 bootstrap 是仓采纳门槛（GLOSSARY「Adoption Gate」：配对样本 + bootstrap 95%
置信区间 + 预注册停止门槛 + 逐指标零退化）的执行工具。本模块自
``experiments/benchmarks/ablation_stats.py`` 逐字收编而来（B5 批次，行为逐位
平移、不改任何数值逻辑）：

* :func:`paired_bootstrap` — case 级配对差的 percentile bootstrap（种子固定）；
* :func:`load_arm_cases` — 从 runner 报告 JSON 读单臂的逐 case 分数；
* :func:`paired_report` — 两臂共享 case 配对、逐指标 bootstrap CI。

two_tier 实验专用的分析函数（``chance_baseline`` / ``hit_rate_report`` /
``power_report``）与 CLI 仍留在 experiments 侧，不属库面。

库纪律：纯标准库，不读环境变量、不读配置文件——一切经调用方传参。
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def paired_bootstrap(
    diffs: list[float],
    *,
    iters: int = 10_000,
    seed: int = 20261022,
) -> tuple[float, float, float]:
    """(mean, lo95, hi95) of case-level paired differences via percentile bootstrap.

    Resampling cases (not scores) preserves the pairing; the fixed seed keeps
    every invocation on this data reproducible bit-for-bit.
    """
    if not diffs:
        raise ValueError("no paired differences to bootstrap")
    rng = random.Random(seed)
    n = len(diffs)
    means: list[float] = []
    for _ in range(iters):
        sample = (diffs[rng.randrange(n)] for _ in range(n))
        means.append(sum(sample) / n)
    means.sort()
    mean = sum(diffs) / n
    return (
        mean,
        means[int(0.025 * (iters - 1))],
        means[int(0.975 * (iters - 1))],
    )


def load_arm_cases(report_path: Path, arm: str) -> dict[str, dict[str, float]]:
    """``{case_id: per-case score dict}`` for one arm of a runner report."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = report["cases"]
    return {row["case_id"]: row[arm] for row in cases}


def paired_report(
    path_a: Path,
    arm_a: str,
    path_b: Path,
    arm_b: str,
    metrics: list[str],
) -> dict:
    """Bootstrap CI for arm_a - arm_b on shared cases, per metric."""
    cases_a = load_arm_cases(path_a, arm_a)
    cases_b = load_arm_cases(path_b, arm_b)
    shared = sorted(set(cases_a).intersection(cases_b))
    if not shared:
        raise ValueError("the two reports share no case ids — check samplers")
    missing_a = len(cases_a) - len(shared)
    missing_b = len(cases_b) - len(shared)
    if missing_a or missing_b:
        print(
            f"[stats] warning: {missing_a} case ids only in A, {missing_b} only in B "
            f"— pairing on {len(shared)} shared cases",
        )
    result: dict[str, object] = {
        "pairing": {"a": str(path_a.name), "arm_a": arm_a, "b": str(path_b.name),
                    "arm_b": arm_b, "shared_cases": len(shared)},
        "metrics": {},
    }
    for metric in metrics:
        diffs = [cases_a[case][metric] - cases_b[case][metric] for case in shared]
        mean, lo, hi = paired_bootstrap(diffs)
        improved = sum(1 for diff in diffs if diff > 0)
        worsened = sum(1 for diff in diffs if diff < 0)
        result["metrics"][metric] = {
            "mean": round(mean, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "improved": improved,
            "tied": len(shared) - improved - worsened,
            "worsened": worsened,
        }
    return result
