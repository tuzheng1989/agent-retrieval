"""e2e 汇报（E3）：:class:`EpisodeResult` 聚合与 JSON 落盘（方案 E3 节）。"""

import json
from pathlib import Path

from experiments.agent_e2e.loading import Case
from experiments.agent_e2e.scorer import EpisodeResult


def summarize(results: list[EpisodeResult], cases: list[Case] | None = None) -> dict:
    """聚合评分结果：总数/成功数/失败数/失败分项（按 acceptable_behavior）/归因分布/逐例表。

    ``cases`` 与 ``results`` 按位置一一对应（冒烟串行执行口径，长度不一致显式报错）；
    缺省时失败分项为 ``{}``、逐例表 ``case_id`` 以 ``#序号`` 占位。分项/归因 dict 键按
    排序输出，逐例表 ``attributions`` 排序——报告 JSON 逐字节可复现（可重放证据面）。
    """
    if cases is not None and len(cases) != len(results):
        raise ValueError(f"cases 与 results 长度不一致: {len(cases)} != {len(results)}")

    failure_by_behavior: dict[str, int] = {}
    attribution_counts: dict[str, int] = {}
    rows: list[dict] = []
    for index, result in enumerate(results):
        if not result.success and cases is not None:
            behavior = cases[index].acceptable_behavior
            failure_by_behavior[behavior] = failure_by_behavior.get(behavior, 0) + 1
        for attribution in result.attributions:
            attribution_counts[attribution] = attribution_counts.get(attribution, 0) + 1
        rows.append(
            {
                "case_id": cases[index].id if cases is not None else f"#{index}",
                "success": result.success,
                "failure_reasons": list(result.failure_reasons),
                "attributions": sorted(result.attributions),
            }
        )

    return {
        "total": len(results),
        "success": sum(1 for result in results if result.success),
        "failure": sum(1 for result in results if not result.success),
        "failure_by_behavior": dict(sorted(failure_by_behavior.items())),
        "attribution_counts": dict(sorted(attribution_counts.items())),
        "cases": rows,
    }


def write_report(report: dict, path: Path) -> None:
    """JSON 落盘（父目录不存在则创建；UTF-8、ensure_ascii=False、缩进 2）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
