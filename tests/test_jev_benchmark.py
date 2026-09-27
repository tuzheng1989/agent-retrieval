"""Validate benchmark reranking and its recall ceiling without API calls."""

import pytest

import experiments.benchmarks.jev_rerank as jev_benchmark
from experiments.benchmarks.dataset import BenchmarkDataset, Case, Resource
from experiments.benchmarks.jev_rerank import oracle_window, rerank_window
from experiments.benchmarks.metrics import recall_at_k


def test_jev_can_improve_recall_inside_window_without_changing_coverage():
    baseline = ["noise-a", "noise-b", "noise-c", "gold-a", "gold-b", "tail"]
    scores = {"noise-a": 0.1, "noise-b": 0.2, "noise-c": 0.3,
              "gold-a": 0.9, "gold-b": 0.8}
    reranked = rerank_window(baseline, scores, window=5)

    assert reranked == ["gold-a", "gold-b", "noise-c", "noise-b", "noise-a", "tail"]
    assert recall_at_k(baseline, {"gold-a", "gold-b"}, 2) == 0.0
    assert recall_at_k(reranked, {"gold-a", "gold-b"}, 2) == 1.0
    assert recall_at_k(reranked, {"gold-a", "gold-b"}, 5) == 1.0


def test_oracle_cannot_recover_a_gold_outside_window():
    ranked = ["noise-a", "noise-b", "gold-a", "noise-c", "gold-b"]
    oracle = oracle_window(ranked, frozenset({"gold-a", "gold-b"}), window=3)

    assert oracle == ["gold-a", "noise-a", "noise-b", "noise-c", "gold-b"]
    assert recall_at_k(oracle, {"gold-a", "gold-b"}, 3) == 0.5


@pytest.mark.parametrize("scores", [{"a": 0.2}, {"a": 1.1, "b": 0.2},
                                     {"a": float("nan"), "b": 0.2}])
def test_incomplete_or_invalid_scores_fail(scores):
    with pytest.raises(ValueError):
        rerank_window(["a", "b"], scores, window=2)


def test_skillret_sampling_loads_full_split_before_spacing_cases(monkeypatch):
    cases = [Case(str(index), f"query {index}", frozenset({"gold"})) for index in range(100)]
    dataset = BenchmarkDataset("skillret", "test", lambda resource: resource.name,
                               [Resource("gold", "Gold", "description")], cases)
    calls = []

    def fake_load(name, data_dir, limit):
        calls.append((name, limit))
        return dataset

    monkeypatch.setattr(jev_benchmark, "load_benchmark", fake_load)
    _, sampled = jev_benchmark.load_sampled_benchmark("skillret", None, 5)

    assert calls == [("skillret", None)]
    assert [case.id for case in sampled] == ["0", "20", "40", "60", "80"]
