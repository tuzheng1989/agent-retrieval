"""Unit tests for the ToolRet / SkillRet benchmark adapters.

No network access: downloaders are untested here; adapters are exercised on
synthetic rows shaped exactly like the published datasets.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from agent_retrieval import MockEmbedder

import experiments.benchmarks.run_benchmark as benchmark_module
from experiments.benchmarks.dataset import BenchmarkDataset, Case, Resource
from experiments.benchmarks.embedder_cache import CachedEmbedder
from experiments.benchmarks.metrics import (
    aggregate,
    completeness_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
    score_case,
)
from experiments.benchmarks.reranker import RerankError, parse_rerank_response
from experiments.benchmarks.run_benchmark import case_query_text, load_env_file, run_arm, run_rerank_arm
from experiments.benchmarks.skillret import skill_case, skill_corpus_text, skill_resource
from experiments.benchmarks.toolret import round_robin, tool_case, tool_corpus_text, tool_resource


# ---- metrics -----------------------------------------------------------------


def test_recall_at_k_counts_fraction_of_golds_found():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"a", "c"}, 2) == 0.5
    assert recall_at_k(ranked, {"a", "c"}, 4) == 1.0
    assert recall_at_k(ranked, {"z"}, 4) == 0.0


def test_completeness_requires_every_gold_in_top_k():
    ranked = ["a", "b", "c"]
    assert completeness_at_k(ranked, {"a", "b"}, 2) == 1.0
    assert completeness_at_k(ranked, {"a", "c"}, 2) == 0.0


def test_mrr_uses_first_gold_position():
    assert mrr(["x", "gold", "y"], {"gold"}) == 0.5
    assert mrr(["x", "y", "gold"], {"gold"}) == 1 / 3
    assert mrr(["x", "y"], {"gold"}) == 0.0


def test_ndcg_binary_gain_matches_hand_computed_values():
    # 单 gold 排第一：DCG = 1，IDCG = 1。
    assert ndcg_at_k(["g", "x", "y"], {"g"}, 3) == 1.0
    # 单 gold 排第二：DCG = 1/log2(3)，IDCG = 1。
    assert ndcg_at_k(["x", "g", "y"], {"g"}, 3) == 1 / __import__("math").log2(3)
    # 双 gold 排 1、3 位：IDCG = 1 + 1/log2(3)。
    golds = {"g1", "g2"}
    dcg = 1.0 + 1 / __import__("math").log2(4)
    idcg = 1.0 + 1 / __import__("math").log2(3)
    assert abs(ndcg_at_k(["g1", "x", "g2"], golds, 3) - dcg / idcg) < 1e-12


def test_score_case_and_aggregate_produce_all_metric_names():
    scores = score_case(["g"], {"g"}, k_values=(5, 10))
    assert set(scores) == {"mrr", "recall@5", "completeness@5", "ndcg@5",
                           "recall@10", "completeness@10", "ndcg@10"}
    means = aggregate([scores, scores], k_values=(5, 10))
    assert means == scores  # 同值平均即原值，round(4) 不失真


# ---- SkillRet adapter --------------------------------------------------------


def test_skill_resource_maps_fields_and_joins_tags():
    row = {
        "id": "s1", "name": "Coder", "description": "reviews code",
        "namespace": "acme/tools", "major": "Dev", "sub": "Review",
        "primary_action": "review", "primary_object": "code", "domain": "software",
    }
    resource = skill_resource(row)
    assert (resource.id, resource.name, resource.description) == ("s1", "Coder", "reviews code")
    assert "acme/tools" in resource.tags and "software" in resource.tags


def test_skill_case_keeps_only_queries_with_pool_golds():
    golds_by_query = {"q1": {"s1"}}
    case = skill_case({"id": "q1", "query": "review my code"}, golds_by_query)
    assert case == Case(id="q1", query="review my code", golds=frozenset({"s1"}))
    assert skill_case({"id": "q2", "query": "anything"}, golds_by_query) is None


def test_skill_corpus_text_keeps_markdown_body_out():
    resource = Resource(id="s1", name="N", description="D", tags="T1 T2")
    assert skill_corpus_text(resource) == "N D T1 T2"


# ---- ToolRet adapter ---------------------------------------------------------


def test_tool_resource_extracts_name_from_documentation_json():
    row = {"id": "t1", "documentation": json.dumps({"name": "bacterial_growth", "x": 1})}
    resource = tool_resource(row)
    assert resource.name == "bacterial_growth"
    assert tool_corpus_text(resource) == row["documentation"]  # 官方协议：原文进语料


def test_tool_resource_survives_unparseable_documentation():
    assert tool_resource({"id": "t2", "documentation": "{oops"}).name == ""


def test_tool_case_parses_labels_json_and_applies_pool_and_relevance():
    row = {
        "id": "q1",
        "query": "calculate bacterial growth",
        "instruction": "Given a `growth` task, retrieve tools that calculate population.",
        "labels": json.dumps([
            {"id": "t1", "relevance": 1},
            {"id": "t9", "relevance": 1},   # 池外 → 丢弃
            {"id": "t3", "relevance": 0},   # 非相关 → 丢弃
        ]),
    }
    case = tool_case(row, {"t1", "t3"})
    assert case is not None and case.golds == frozenset({"t1"})
    assert case.instruction.startswith("Given a `growth` task")
    assert tool_case({"id": "q2", "query": "x", "labels": "[]"}, {"t1"}) is None


def test_case_query_text_by_mode():
    plain = Case("q1", "find anime", frozenset({"t1"}))
    assert case_query_text(plain, "query") == "find anime"
    instructed = Case("q2", "find anime", frozenset({"t1"}),
                      instruction="Given a `media` task, retrieve search tools.")
    assert case_query_text(instructed, "instruction") == (
        "Instruct: Given a `media` task, retrieve search tools.\nQuery: find anime"
    )
    try:
        case_query_text(plain, "instruction")
        raised = False
    except ValueError:
        raised = True
    assert raised  # 无 instruction 的 case 在 instruction 模式必须显式失败


def test_round_robin_balances_across_tasks_and_respects_limit():
    rows_by_task = {"a": [{"i": 1}, {"i": 2}], "b": [{"i": 3}, {"i": 4}]}
    assert [row["i"] for row in round_robin(rows_by_task, 2)] == [1, 3]
    assert [row["i"] for row in round_robin(rows_by_task, 3)] == [1, 3, 2]
    assert len(round_robin(rows_by_task, 99)) == 4  # 超额即全量
    assert len(round_robin(rows_by_task, None)) == 4
    assert [row["i"] for row in round_robin({"a": [], "b": [{"i": 9}]}, 5)] == [9]


# ---- vector cache ------------------------------------------------------------


class CountingEmbedder:
    """Deterministic fake inner embedder counting every text it sees."""

    def __init__(self) -> None:
        self.texts_seen: list[str] = []

    def embed_corpus(self, texts):
        self.texts_seen.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        self.texts_seen.append(text)
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> tuple[float, ...]:
        return (float(len(text) % 5) + 0.25, 0.5)


def test_cached_embedder_persists_vectors_across_instances(tmp_path: Path):
    db = tmp_path / "vectors.sqlite3"
    first_inner, second_inner = CountingEmbedder(), CountingEmbedder()
    first = CachedEmbedder(first_inner, db_path=db, identity="m:1")
    vectors = first.embed_corpus(["alpha", "beta"])
    first.close()

    second = CachedEmbedder(second_inner, db_path=db, identity="m:1")
    assert second.embed_corpus(["alpha", "beta"]) == vectors
    assert second.embed_query("alpha") == vectors[0]
    assert second_inner.texts_seen == []  # 全部命中缓存，inner 零调用


def test_cached_embedder_isolates_identities(tmp_path: Path):
    db = tmp_path / "vectors.sqlite3"
    inner = CountingEmbedder()
    CachedEmbedder(inner, db_path=db, identity="m:1").embed_query("same text")
    CachedEmbedder(inner, db_path=db, identity="m:2").embed_query("same text")
    assert inner.texts_seen == ["same text", "same text"]  # 换身份必重嵌


def test_cached_embedder_keeps_committed_chunks_after_failure(tmp_path: Path):
    """Inner failing mid-corpus must not discard already-committed chunks:
    chunk 1 persists normally, chunk 2 degrades to zero-padding, and reruns
    reuse chunk 1 from disk without re-embedding it."""
    from agent_retrieval import QueryEmbeddingError

    class FailingAfter(CountingEmbedder):
        def __init__(self, limit: int) -> None:
            super().__init__()
            self.limit = limit

        def embed_corpus(self, texts):
            if len(self.texts_seen) + len(texts) > self.limit:
                raise QueryEmbeddingError("boom")
            return super().embed_corpus(texts)

    db = tmp_path / "vectors.sqlite3"
    texts = [f"text-{index}" for index in range(600)]  # 两个 chunk（512 + 88）
    inner = FailingAfter(limit=550)  # 第二个 chunk（512+88 > 550）触发失败
    first = CachedEmbedder(inner, db_path=db, identity="m:1")
    vectors = first.embed_corpus(texts)  # 不抛：失败 chunk 逐条降级，能救一条是一条
    assert len(vectors) == 600
    assert all(v != (0.0,) for v in vectors[:512])       # 第一个 chunk 完整
    assert any(v == (0.0,) for v in vectors[512:])       # 失败部分零占位
    assert any(v != (0.0,) for v in vectors[512:])       # 逐条降级救回了前几条

    # 重跑：磁盘已有的命中缓存（inner 零调用），零占位的重试这次全部成功。
    second_inner = CountingEmbedder()
    second = CachedEmbedder(second_inner, db_path=db, identity="m:1")
    vectors = second.embed_corpus(texts)
    assert all(v != (0.0,) for v in vectors)
    assert len(second_inner.texts_seen) < 100            # 只补嵌了零占位的那部分


def test_cached_embedder_zero_pads_unembeddable_texts(tmp_path: Path):
    """A per-text endpoint rejection must not poison its chunk: survivors are
    cached, the bad text is zero-padded (cosine 0 → no vector rank)."""
    from agent_retrieval import QueryEmbeddingError

    class RejectingOne(CountingEmbedder):
        def embed_corpus(self, texts):
            if "bad" in texts:
                raise QueryEmbeddingError("400 Bad Request")
            return super().embed_corpus(texts)

    db = tmp_path / "vectors.sqlite3"
    inner = RejectingOne()
    embedder = CachedEmbedder(inner, db_path=db, identity="m:1")
    texts = ["good-1", "bad", "good-2"]  # batch_size=3 的整批都会因 "bad" 失败
    vectors = embedder.embed_corpus(texts)
    assert len(vectors) == 3
    assert vectors[1] == (0.0,)                      # 坏文本零占位
    assert vectors[0] == embedder.embed_query("good-1")  # 好文本已入缓存且一致
    assert vectors[0] != (0.0,)
    # 零占位在会话内跨调用存续：下一次语料嵌入不再重试坏文本。
    inner.texts_seen.clear()
    vectors = embedder.embed_corpus(texts + ["good-3"])
    assert inner.texts_seen == ["good-3"]            # 只嵌新文本，"bad" 不重试
    assert vectors[1] == (0.0,)


# ---- rerank arm ----------------------------------------------------------------


def test_parse_rerank_response_aligns_sparse_results_by_index():
    payload = {"results": [{"index": 2, "relevance_score": 0.9},
                           {"index": 0, "relevance_score": 0.1}]}
    assert parse_rerank_response(payload, 3) == [0.1, float("-inf"), 0.9]
    try:
        parse_rerank_response({"results": [{"index": 5, "relevance_score": 1.0}]}, 3)
        raised = False
    except RerankError:
        raised = True
    assert raised  # 越界 index 显式失败


def test_run_rerank_arm_reorders_window_and_keeps_tail_order():
    class PreferDocsReranker:
        """把包含 'alpha' 的文档排第一的确定性替身。"""

        def rerank(self, query, documents):
            order = sorted(range(len(documents)),
                           key=lambda i: (0 if "alpha" in documents[i] else 1, i))
            scores = [0.0] * len(documents)
            for rank, index in enumerate(order):
                scores[index] = 1.0 / (rank + 1)
            return scores

    resources = [
        Resource("doc-a", "Alpha Doc", "alpha full-text search document"),
        Resource("doc-b", "Beta Doc", "beta unrelated content"),
        Resource("doc-c", "Gamma Doc", "gamma alpha also mentioned here"),
        Resource("doc-d", "Delta Doc", "delta filler body text"),
    ]
    cases = [Case("c1", "alpha document", frozenset({"doc-c"}))]
    dataset = BenchmarkDataset("synthetic", "test", skill_corpus_text, resources, cases)
    # MockEmbedder 的向量路是零语义 hash：fusion 给出某个初始序，reranker 强制把
    # 含 "alpha" 的文档（doc-a、doc-c）提到窗口最前——gold doc-c 因此必然改善。
    result = run_rerank_arm(dataset, (5, 10), embedder=MockEmbedder(),
                            reranker=PreferDocsReranker(), candidates=3,
                            label="test")
    assert result.metrics["recall@5"] == 1.0
    assert result.metrics["mrr"] == 0.5  # doc-c 被排到第 2（doc-a 的 "alpha" 在语料更前）


def test_run_rerank_arm_keeps_fusion_order_on_constant_scores():
    """常量分 = reranker 无区分度：必须保留 fusion 序而非塌缩成 id 字典序。"""

    class ConstantReranker:
        def rerank(self, query, documents):
            return [1.0] * len(documents)

    resources = [
        Resource("z-doc", "Zeta", "relevant zeta document about calendars"),
        Resource("a-doc", "Alpha", "unrelated alpha filler text"),
        Resource("m-doc", "Mid", "another unrelated filler body"),
    ]
    # BM25 会把 z-doc（唯一含查询词）排第一；若 rerank 塌缩成字典序，a-doc 反超。
    cases = [Case("c1", "calendars", frozenset({"z-doc"}))]
    dataset = BenchmarkDataset("synthetic", "test", skill_corpus_text, resources, cases)
    result = run_rerank_arm(dataset, (5, 10), embedder=None,
                            reranker=ConstantReranker(), candidates=3, label="test")
    assert result.metrics["mrr"] == 1.0       # z-doc 仍在第一（fusion 序保留）
    assert result.degraded_cases == 1          # 且如实计入退化


# ---- .env loading -------------------------------------------------------------


def test_load_env_file_injects_missing_variables_and_keeps_existing_ones(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        'AGENT_RETRIEVAL_TEST_FROM_ENV="shell value"\n'
        "AGENT_RETRIEVAL_TEST_FROM_FILE=file value\n"
        "# comment line\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_RETRIEVAL_TEST_FROM_ENV", "shell value")
    monkeypatch.delenv("AGENT_RETRIEVAL_TEST_FROM_FILE", raising=False)
    monkeypatch.setattr(benchmark_module, "_REPO_ROOT", tmp_path)
    load_env_file()
    try:
        # 文件值注入缺失变量；已有环境变量不被覆盖（shell 临时覆盖优先）。
        assert os.environ["AGENT_RETRIEVAL_TEST_FROM_FILE"] == "file value"
        assert os.environ["AGENT_RETRIEVAL_TEST_FROM_ENV"] == "shell value"
    finally:
        os.environ.pop("AGENT_RETRIEVAL_TEST_FROM_FILE", None)


def test_load_env_file_is_silent_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_module, "_REPO_ROOT", tmp_path)  # 无 .env 的目录
    load_env_file()  # 不抛异常即可


# ---- end-to-end arm ----------------------------------------------------------


def _dataset() -> BenchmarkDataset:
    resources = [
        Resource("geo", "Geo Query", "query population data by region and year"),
        Resource("weather", "Weather", "current weather forecasts"),
        Resource("docs", "Doc Search", "full-text search over documents"),
    ]
    cases = [
        Case("c1", "population data for California", frozenset({"geo"})),
        Case("c2", "full text document search", frozenset({"docs"})),
    ]
    return BenchmarkDataset("synthetic", "test", skill_corpus_text, resources, cases)


def test_run_arm_bm25_recovers_all_golds_and_is_deterministic():
    dataset = _dataset()
    first = run_arm(dataset, (5, 10), embedder=None, label="test")
    second = run_arm(dataset, (5, 10), embedder=None, label="test")
    assert first.metrics == second.metrics
    assert first.metrics["recall@5"] == 1.0
    assert first.metrics["completeness@5"] == 1.0
    assert first.metrics["mrr"] == 1.0
    assert first.cases == 2


def test_run_arm_with_mock_vector_path_stays_in_full_recall():
    result = run_arm(_dataset(), (5, 10), embedder=MockEmbedder(), label="test")
    assert result.metrics["recall@10"] == 1.0
    assert result.metrics["mrr"] > 0.0
