"""Unit tests for field-level multi-path retrieval (experiments side)."""

from __future__ import annotations

from agent_retrieval import MockEmbedder

from experiments.benchmarks.dataset import Resource
from experiments.benchmarks.multipath import FieldRetriever, rrf_merge, vector_ranking
from experiments.benchmarks.run_benchmark import field_paths

_RESOURCES = [
    Resource("geo", "Geo Query", "query population data by region and year", "geo:demographics"),
    Resource("doc", "Doc Search", "full-text search over uploaded documents", "doc:search"),
    Resource("weather", "Weather", "current weather forecasts and alerts", "weather:alerts"),
]
_PATHS = [
    ("name", lambda r: r.name),
    ("description", lambda r: r.description),
    ("tags", lambda r: r.tags),
]


def test_rrf_merge_accumulates_across_paths_and_breaks_ties_by_id():
    merged = rrf_merge([["b", "a"], ["a", "c"]])
    # a 双路命中（1/62 + 1/61 ≈ 0.0325）压过单路头名 b（1/61 ≈ 0.0164）——
    # RRF 的业务语义：两路共识优先于单路头部；c 仅单路末位排最后。
    assert merged == ["a", "b", "c"]
    # 完全并列的路径由字典序决胜，保证确定性。
    assert rrf_merge([["x", "y"]], k=1) == ["x", "y"]
    assert rrf_merge([]) == []


def test_field_retriever_merges_cross_field_hits_and_filters_zero_scores():
    retriever = FieldRetriever(_RESOURCES, _PATHS)
    # "population" 只在 geo 的 description 里 → doc/weather 无候选，被 score>0 过滤。
    ranked = retriever.bm25_ranking("population data by region")
    assert ranked[0] == "geo"
    assert "doc" not in ranked and "weather" not in ranked
    # 完全无关查询：无任何候选。
    assert retriever.bm25_ranking("quantum entanglement machinery") == []


def test_field_retriever_name_path_survives_when_description_is_noisy():
    # 查询词命中 weather 的 name 与 tags，但 geo 的 description 更长更吵：
    # 分路后 name 命中不被 description 词海稀释。
    retriever = FieldRetriever(_RESOURCES, _PATHS)
    assert retriever.bm25_ranking("weather alerts")[0] == "weather"


def test_field_retriever_rank_merges_vector_path():
    retriever = FieldRetriever(_RESOURCES, _PATHS)
    # MockEmbedder 无语义：只验证向量路参与合并且不崩溃、结果确定性。
    first = retriever.rank("population data", embedder=MockEmbedder(),
                           corpus_text=lambda r: f"{r.name} {r.description}")
    second = retriever.rank("population data", embedder=MockEmbedder(),
                            corpus_text=lambda r: f"{r.name} {r.description}")
    assert first == second and set(first) == {r.id for r in _RESOURCES}


def test_vector_ranking_excludes_non_positive_cosine():
    embedder = MockEmbedder()
    ranked = vector_ranking(_RESOURCES, lambda r: r.description, "population data", embedder)
    assert set(ranked) <= {r.id for r in _RESOURCES}
    # 重复调用确定性。
    assert ranked == vector_ranking(_RESOURCES, lambda r: r.description, "population data", embedder)


def test_field_paths_covers_both_benchmarks():
    from experiments.benchmarks.dataset import BenchmarkDataset

    skill = BenchmarkDataset("skillret", "t", lambda r: "", list(_RESOURCES), [])
    tool = BenchmarkDataset("toolret", "t", lambda r: "", list(_RESOURCES), [])
    assert [label for label, _ in field_paths(skill)] == ["name", "description", "tags"]
    assert [label for label, _ in field_paths(tool)] == ["name", "documentation"]
