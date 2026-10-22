"""Unit tests for the shared ablation statistics."""

from __future__ import annotations

from experiments.benchmarks.ablation_stats import (
    chance_baseline,
    paired_bootstrap,
    power_report,
)

import json

import pytest


class TestChanceBaseline:
    def test_single_gold_one_of_eighteen_g_three(self):
        # 1 - C(17,3)/C(18,3) = 3/18 = 1/6 — the hand-checkable case.
        assert chance_baseline(18, 3, 1) == pytest.approx(1 / 6)

    def test_span_zero_means_zero(self):
        assert chance_baseline(18, 3, 0) == 0.0

    def test_span_ge_slack_means_certain(self):
        # A gold set so wide that G draws cannot avoid it.
        assert chance_baseline(18, 3, 16) == 1.0

    def test_monotone_in_span_and_g(self):
        wider = chance_baseline(18, 3, 2)
        assert wider > chance_baseline(18, 3, 1)
        assert chance_baseline(18, 5, 1) > chance_baseline(18, 3, 1)

    def test_rejects_out_of_range_span(self):
        with pytest.raises(ValueError):
            chance_baseline(18, 3, 19)


class TestPairedBootstrap:
    def test_constant_positive_diff_gives_positive_ci(self):
        mean, lo, hi = paired_bootstrap([3.0] * 40)
        assert mean == pytest.approx(3.0)
        assert lo > 0 and hi == pytest.approx(3.0)

    def test_symmetric_diff_straddles_zero(self):
        mean, lo, hi = paired_bootstrap([1.0, -1.0] * 50, iters=2000)
        assert mean == pytest.approx(0.0, abs=1e-9)
        assert lo < 0 < hi

    def test_deterministic_under_fixed_seed(self):
        diffs = [((index % 7) - 3) / 10 for index in range(60)]
        assert paired_bootstrap(diffs) == paired_bootstrap(diffs)

    def test_empty_diffs_rejected(self):
        with pytest.raises(ValueError):
            paired_bootstrap([])


class TestPowerReport:
    @pytest.fixture
    def report_pair(self, tmp_path):
        rows_a, rows_b = [], []
        for index in range(50):
            score = 0.5 + ((index % 5) - 2) * 0.1
            rows_a.append({"case_id": f"q{index}", "arm": {"mrr": score + 0.03}})
            rows_b.append({"case_id": f"q{index}", "arm": {"mrr": score}})
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        path_a.write_text(json.dumps({"cases": rows_a}), encoding="utf-8")
        path_b.write_text(json.dumps({"cases": rows_b}), encoding="utf-8")
        return path_a, path_b

    def test_recovers_known_delta_and_flags_sufficiency(self, report_pair):
        path_a, path_b = report_pair
        report = power_report(path_a, "arm", path_b, "arm", "mrr", mcid=0.03)
        # Every paired diff is exactly +0.03 → SD 0, trivially sufficient.
        assert report["observed_mean_delta"] == pytest.approx(0.03)
        assert report["paired_sd"] == 0.0
        assert report["n_for_80pct_power"] == 0
        assert report["sufficient"] is True

    def test_zero_sd_small_mcid_needs_zero_extra(self, report_pair):
        path_a, path_b = report_pair
        report = power_report(path_a, "arm", path_b, "arm", "mrr", mcid=0.0)
        assert report["n_for_80pct_power"] == 0
