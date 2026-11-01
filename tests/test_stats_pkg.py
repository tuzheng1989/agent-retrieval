"""包侧 ``agent_retrieval.stats`` 的收编回归测试。

种子回归的期望值是**平移前旧实现**（``experiments/benchmarks/ablation_stats.py``）
在本次收编前实跑记录的输出，作为独立真值写死在字面量里——本批红线是
「平移不改任何数值逻辑」，因此用精确相等（逐位一致）而非近似断言。
"""

from __future__ import annotations

import json

import pytest

from agent_retrieval.stats import (
    load_arm_cases,
    paired_bootstrap,
    paired_report,
)


class TestPairedBootstrapSeedRegression:
    def test_seed_regression_bitwise(self):
        # 期望值 = 平移前旧实现实跑输出（seed 20261022、1 万次、percentile 分位取法不变）。
        result = paired_bootstrap([0.1, -0.1, 0.3])
        assert result == (0.09999999999999999, -0.10000000000000002, 0.3)

    def test_default_seed_reproducible(self):
        diffs = [0.1, -0.1, 0.3]
        assert paired_bootstrap(diffs) == paired_bootstrap(diffs)

    def test_empty_diffs_rejected(self):
        with pytest.raises(ValueError):
            paired_bootstrap([])


class TestPairedReportPkgSide:
    """包侧端到端：两份最小报告 JSON 的共享 case 配对。"""

    @pytest.fixture
    def report_paths(self, tmp_path):
        cases_a = [
            {"case_id": "c1", "x": {"mrr": 0.8}},
            {"case_id": "c2", "x": {"mrr": 0.5}},
            {"case_id": "c3", "x": {"mrr": 0.2}},
        ]
        cases_b = [
            {"case_id": "c2", "y": {"mrr": 0.4}},
            {"case_id": "c3", "y": {"mrr": 0.3}},
            {"case_id": "c4", "y": {"mrr": 0.9}},
        ]
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        path_a.write_text(json.dumps({"cases": cases_a}), encoding="utf-8")
        path_b.write_text(json.dumps({"cases": cases_b}), encoding="utf-8")
        return path_a, path_b

    def test_missing_cases_warn_and_pair_on_shared(self, report_paths, capsys):
        # c1 只在 A、c4 只在 B → 警告打到 stdout，配对只落在共享集 {c2, c3}。
        path_a, path_b = report_paths
        report = paired_report(path_a, "x", path_b, "y", ["mrr"])
        captured = capsys.readouterr()
        assert "[stats] warning: 1 case ids only in A, 1 only in B" in captured.out
        assert report["pairing"] == {
            "a": "a.json",
            "arm_a": "x",
            "b": "b.json",
            "arm_b": "y",
            "shared_cases": 2,
        }
        # 共享序 c2 < c3，配对差 [0.5-0.4, 0.2-0.3] = [+0.1, -0.1]。
        metric = report["metrics"]["mrr"]
        assert metric["mean"] == 0.0
        assert metric["improved"] == 1
        assert metric["tied"] == 0
        assert metric["worsened"] == 1

    def test_full_overlap_pairs_all_cases(self, tmp_path, capsys):
        # 两份报告 case 完全一致 → 无警告，共享集 = 全部 case。
        cases = [
            {"case_id": "c1", "x": {"mrr": 0.6}, "y": {"mrr": 0.4}},
            {"case_id": "c2", "x": {"mrr": 0.3}, "y": {"mrr": 0.3}},
        ]
        path_a = tmp_path / "same_a.json"
        path_b = tmp_path / "same_b.json"
        path_a.write_text(json.dumps({"cases": cases}), encoding="utf-8")
        path_b.write_text(json.dumps({"cases": cases}), encoding="utf-8")
        report = paired_report(path_a, "x", path_b, "y", ["mrr"])
        captured = capsys.readouterr()
        assert captured.out == ""
        assert report["pairing"]["shared_cases"] == 2
        metric = report["metrics"]["mrr"]
        assert metric["mean"] == 0.1
        assert metric["improved"] == 1
        assert metric["tied"] == 1
        assert metric["worsened"] == 0

    def test_disjoint_case_sets_raise(self, tmp_path):
        # 共享 case 为空 → 沿用原实现的 ValueError。
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        path_a.write_text(
            json.dumps({"cases": [{"case_id": "a1", "x": {"mrr": 0.1}}]}), encoding="utf-8",
        )
        path_b.write_text(
            json.dumps({"cases": [{"case_id": "b1", "y": {"mrr": 0.1}}]}), encoding="utf-8",
        )
        with pytest.raises(ValueError, match="share no case ids"):
            paired_report(path_a, "x", path_b, "y", ["mrr"])

    def test_load_arm_cases_reads_caller_path(self, tmp_path):
        # load_arm_cases 读调用方传入路径的 JSON，返回 {case_id: 该臂分数字典}。
        path = tmp_path / "r.json"
        path.write_text(
            json.dumps({"cases": [{"case_id": "c1", "x": {"mrr": 0.7}}]}), encoding="utf-8",
        )
        assert load_arm_cases(path, "x") == {"c1": {"mrr": 0.7}}
