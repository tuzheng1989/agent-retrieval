"""弃权冒烟：负例/对抗查询上两题型的窗口最大判定读数（描述性，无闸、非统计）。

复现 iterative 触发率 0% 的机制场景：无 gold 的查询在 SkillRet 全池 fusion
w20 检索下，窗口内全是主题相关候选——noul 窗口最大概率历史行为是始终 >= 0.7
（无法弃权）；score 的机制预言是 P(3) / P(>=2) 在这类窗口上塌向低档，使
0.7 线可区分"确定无关"与"边界模糊"。

读数：每条查询 x {noul p_max, P(>=2)_max, P(3)_max, expected_max} 及其相对
0.7 线的位置。写进 findings 作机制证据；统计验证另立预注册 issue，不在本
实验内。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

from agent_retrieval import build_embedder, rank_candidates, vector_available

from experiments.benchmarks.jev_question_paired import judge_pair
from experiments.benchmarks.jev_questions import (
    build_judge_state,
    fold_at_least,
    fold_expected,
    fold_top,
)
from experiments.benchmarks.jev_rerank import load_sampled_benchmark
from experiments.benchmarks.run_benchmark import (
    _DEFAULT_DATA_DIR,
    embedder_from_env,
    load_env_file,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_JEV_CASES = _REPO_ROOT / "experiments" / "jev_cases.json"

#: 弃权判读线（与既有 noul 探索阈值同源，jev-findings 的 0.7）。
ABSTAIN_LINE = 0.7
_WINDOW = 20


def _negative_queries() -> list[dict]:
    """jev_cases 的 no-match 负例 + golden fixtures 的对抗查询。"""
    from experiments.golden.build_fixtures import ADVERSARIAL_QUERIES

    cases = json.loads(_JEV_CASES.read_text(encoding="utf-8"))["cases"]
    entries = [
        {"source": f"jev_cases/{case['id']}", "query": case["query"]}
        for case in cases if case["gold"] is None
    ]
    entries.extend(
        {"source": "golden/adversarial", "query": query} for query in ADVERSARIAL_QUERIES
    )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("bm25", "fusion"), default="fusion")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_env_file()
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is required", file=sys.stderr)
        return 2

    embedder = None
    if args.arm == "fusion":
        config = embedder_from_env()
        if vector_available(config) != "api":
            print("AGENT_RETRIEVAL_EMBEDDING_API_KEY is required for fusion", file=sys.stderr)
            return 2
        embedder = build_embedder(config)

    # 只取全池资源与 docgen 语料投影（采样出的 case 不用——查询来自负例集合）。
    dataset, _cases = load_sampled_benchmark("skillret", _DEFAULT_DATA_DIR, 1, corpus="docgen")
    by_id = {resource.id: resource for resource in dataset.resources}
    rows = []
    started = perf_counter()
    for entry in _negative_queries():
        hits = rank_candidates(
            dataset.resources,
            entry["query"],
            corpus_text=dataset.corpus_text,
            item_id=lambda resource: resource.id,
            item_name=lambda resource: resource.name,
            vector=embedder,
        )
        head = [hit.item.id for hit in hits[:_WINDOW]]
        if not head:
            rows.append({**entry, "window": 0, "readings": None})
            continue
        state = build_judge_state(entry["query"], "", [by_id[item_id] for item_id in head])
        judgment = judge_pair(state, len(head))
        distributions = {
            head[int(key[1:])]: dist for key, dist in judgment.score_distributions.items()
        }
        readings = {
            "noul_max": max(judgment.noul_scores.values()),
            "p_at_least_2_max": max(fold_at_least(dist, 2) for dist in distributions.values()),
            "p_top_max": max(fold_top(dist) for dist in distributions.values()),
            "expected_max": max(fold_expected(dist) for dist in distributions.values()),
            "legend_echo_fraction": judgment.legend_echo_fraction,
        }
        readings["crosses_line"] = {
            key: value >= ABSTAIN_LINE
            for key, value in readings.items()
            if key != "legend_echo_fraction" and isinstance(value, float)
        }
        rows.append({**entry, "window": len(head), "readings": readings})
        print(f"[{entry['source']}] {json.dumps(readings, ensure_ascii=False)}", file=sys.stderr)
    report = {
        "configuration": {
            "arm": args.arm,
            "bench": "skillret",
            "corpus": "docgen",
            "window": _WINDOW,
            "abstain_line": ABSTAIN_LINE,
            "wall_seconds": perf_counter() - started,
        },
        "queries": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
