"""Paired-difference statistics for the ablation runners.

Everything here consumes the per-case score dicts the runners already emit, so
the same code answers three questions:

* is arm A better than arm B, and is the paired difference's bootstrap CI away
  from zero? (``--a/--b`` with arm names)
* how does the per-case group hit rate compare to its chance baseline, case by
  case and stratified by gold-group span? (``--hit-rate``)
* is the sampled case count large enough to detect a minimum credible
  difference? (``--power``, feeding the "extend to 600" decision)

Percentile bootstrap with a fixed seed; the repository's earlier findings used
the same estimator (case-level paired deltas, 95% interval).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# B5 收编：配对统计三函数下沉为库面 ``agent_retrieval.stats``，此处仅转发名。
# 模块内引用（power_report 的 load_arm_cases、main() 的 paired_report）经转发名
# 走包实现，行为不变。
from agent_retrieval.stats import (  # noqa: F401 —— 转发面：paired_bootstrap 模块内已无直调方，但既有导入路径必须保持可用
    load_arm_cases,
    paired_bootstrap,
    paired_report,
)


def chance_baseline(n_groups: int, g: int, gold_group_span: int) -> float:
    """P(>=1 gold group among G uniformly random groups) for one case.

    Hypergeometric tail: 1 - C(n - span, G) / C(n, G). Callers aggregate this
    per case (span varies) instead of using a single pool-wide number — the
    47% of SkillRet cases whose gold spans 2-3 groups carry a different
    baseline than single-group cases.
    """
    if gold_group_span < 0 or gold_group_span > n_groups:
        raise ValueError("gold_group_span outside the group count")
    if gold_group_span == 0:
        return 0.0
    if g > n_groups - gold_group_span:
        return 1.0
    miss = math.comb(n_groups - gold_group_span, g)
    total = math.comb(n_groups, g)
    return 1.0 - miss / total


def hit_rate_report(report_path: Path, arm: str = "two_tier") -> dict:
    """Group-hit rates vs their chance baselines, overall and stratified by span.

    Expects the two-tier runner's per-case rows: ``group_hit_any`` /
    ``group_hit_all`` / ``group_chance_any`` / ``gold_group_span``.
    """
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["cases"]
    spans: dict[int, dict[str, list[float]]] = {}
    counts: dict[int, int] = {}
    for row in rows:
        span = int(row["gold_group_span"])
        bucket = spans.setdefault(span, {"any": [], "all": [], "chance": []})
        bucket["any"].append(float(row["group_hit_any"]))
        bucket["all"].append(float(row["group_hit_all"]))
        bucket["chance"].append(float(row["group_chance_any"]))
        counts[span] = counts.get(span, 0) + 1

    def summarize(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    strata = {
        str(span): {
            "cases": counts[span],
            "hit_any": summarize(bucket["any"]),
            "hit_all": summarize(bucket["all"]),
            "chance_any": summarize(bucket["chance"]),
        }
        for span, bucket in sorted(spans.items())
    }
    every = [row for row in rows if int(row["gold_group_span"]) == 1]
    crossed = [row for row in rows if int(row["gold_group_span"]) > 1]
    return {
        "arm": arm,
        "cases": len(rows),
        "overall": {
            "hit_any": summarize([float(r["group_hit_any"]) for r in rows]),
            "hit_all": summarize([float(r["group_hit_all"]) for r in rows]),
            "chance_any": summarize([float(r["group_chance_any"]) for r in rows]),
        },
        "single_group_cases": len(every),
        "cross_group_cases": len(crossed),
        "by_span": strata,
    }


def power_report(
    path_a: Path,
    arm_a: str,
    path_b: Path,
    arm_b: str,
    metric: str,
    mcid: float,
) -> dict:
    """From the realized paired-difference SD, the n needed for 80% power at MCID.

    Normal approximation n ≈ ((1.96 + 0.84)² · SD²) / MCID² — deliberately the
    same rule of thumb the findings document's CI widths imply, so the
    "extend to 600?" decision is reproducible without hand-fitting.
    """
    cases_a = load_arm_cases(path_a, arm_a)
    cases_b = load_arm_cases(path_b, arm_b)
    shared = sorted(set(cases_a).intersection(cases_b))
    if len(shared) < 2:
        raise ValueError("need at least two shared cases to estimate an SD")
    diffs = [cases_a[case][metric] - cases_b[case][metric] for case in shared]
    mean = sum(diffs) / len(diffs)
    variance = sum((diff - mean) ** 2 for diff in diffs) / (len(diffs) - 1)
    sd = math.sqrt(variance)
    required = ((1.96 + 0.84) ** 2) * sd * sd / (mcid * mcid) if mcid > 0 else 0.0
    # 阈值吞掉配对差浮点残渣（SD ~1e-17 时 ceil 会虚报 1 个样本）。
    needed = math.ceil(required) if required > 1e-12 else 0
    return {
        "metric": metric,
        "mcid": mcid,
        "observed_cases": len(shared),
        "observed_mean_delta": round(mean, 4),
        "paired_sd": round(sd, 4),
        "n_for_80pct_power": needed,
        "sufficient": len(shared) >= needed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    pair = sub.add_parser("pair", help="paired bootstrap CI between two report arms")
    pair.add_argument("--a", type=Path, required=True)
    pair.add_argument("--arm-a", default="two_tier")
    pair.add_argument("--b", type=Path, required=True)
    pair.add_argument("--arm-b", default="single")
    pair.add_argument("--metrics", nargs="+",
                      default=("mrr", "ndcg@10", "recall@20"))

    hits = sub.add_parser("hit-rate", help="group hit rates vs chance baselines")
    hits.add_argument("--report", type=Path, required=True)
    hits.add_argument("--arm", default="two_tier")

    power = sub.add_parser("power", help="is the sample size enough for an MCID?")
    power.add_argument("--a", type=Path, required=True)
    power.add_argument("--arm-a", default="two_tier")
    power.add_argument("--b", type=Path, required=True)
    power.add_argument("--arm-b", default="single")
    power.add_argument("--metric", default="mrr")
    power.add_argument("--mcid", type=float, default=0.03)

    args = parser.parse_args()
    try:
        if args.mode == "pair":
            report = paired_report(args.a, args.arm_a, args.b, args.arm_b, args.metrics)
        elif args.mode == "hit-rate":
            report = hit_rate_report(args.report, args.arm)
        else:
            report = power_report(
                args.a, args.arm_a, args.b, args.arm_b, args.metric, args.mcid,
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[stats] error: {exc}")
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
