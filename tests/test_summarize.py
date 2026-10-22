"""Unit tests for the summarize tool's measurement and summary loading."""

from __future__ import annotations

import json

import pytest

from experiments.benchmarks.dataset import Resource
from experiments.benchmarks.jev_rerank import JEV_DESCRIPTION_MAX_CHARS
from experiments.benchmarks.summarize import (
    load_summaries,
    measure_truncation,
    window_unique_ids,
)


def _fake_dataset():
    """A minimal toolret-shaped dataset object for direct injection."""
    resources = [
        Resource(id="short", name="s", description="tiny doc", tags=""),
        Resource(id="long", name="l", description='{"a": "' + "x" * 3000 + '"}', tags=""),
    ]
    return type("Dataset", (), {"resources": resources})()


def _window_report(tmp_path, windows):
    path = tmp_path / "window.json"
    rows = [
        {
            "case_id": f"q{index}",
            "baseline_top": ids,
        }
        for index, ids in enumerate(windows)
    ]
    path.write_text(json.dumps({"cases": rows}), encoding="utf-8")
    return path


class TestMeasureTruncation:
    def test_clipped_fraction_and_json_split(self, tmp_path):
        report = _window_report(tmp_path, [["short", "long"], ["long"]])
        stats = measure_truncation(report, _fake_dataset())
        assert stats["window_resources"] == 2
        # Only `long` (3,008 chars) exceeds the clip, and the clip lands mid-JSON.
        assert stats["clipped_count"] == 1
        assert stats["clipped_fraction"] == 0.5
        assert stats["clip_splits_mid_json"] == 1
        assert stats["chars_max"] == len('{"a": "' + "x" * 3000 + '"}')

    def test_under_clip_nothing_splits(self, tmp_path):
        report = _window_report(tmp_path, [["short"]])
        stats = measure_truncation(report, _fake_dataset())
        assert stats["clipped_count"] == 0
        assert stats["clip_splits_mid_json"] == 0


class TestWindowUniqueIds:
    def test_union_across_cases_is_sorted_and_unique(self, tmp_path):
        report = _window_report(tmp_path, [["b", "a"], ["a", "c"]])
        assert window_unique_ids(report) == ["a", "b", "c"]


class TestLoadSummaries:
    def test_absent_file_is_empty(self, tmp_path):
        assert load_summaries(tmp_path / "nope.jsonl") == {}

    def test_round_trip(self, tmp_path):
        path = tmp_path / "summaries.jsonl"
        path.write_text(
            json.dumps({"resource_id": "a", "summary": "s", "model": "m", "source_chars": 10})
            + "\n",
            encoding="utf-8",
        )
        assert load_summaries(path) == {"a": "s"}


class TestClipConstant:
    def test_clip_constant_unchanged(self):
        # The truncate arm IS the incumbent behavior — a silent change here
        # would invalidate every cross-arm comparison.
        assert JEV_DESCRIPTION_MAX_CHARS == 2000
