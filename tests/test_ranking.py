"""候选检索面 rank_candidates 的契约测试。

自 evochat test_agent_search_face.py 纯函数部分迁移：宿主资源模型（AgentResource）
换成本文件的最小 dataclass——投影函数（corpus_text/item_id/item_name）本来就是
检索面消费的全部接口，这正是泛型化的意义。接线用例（准入池、planner 清单渲染、
flag 工厂）留在宿主，不属于本包。
"""

from dataclasses import dataclass

from agent_retrieval.core.ports import MockEmbedder
from agent_retrieval.core.ranking import CandidateHit, rank_candidates


@dataclass(frozen=True)
class Candidate:
    id: str
    name: str
    description: str
    body: str = ""  # 永不进语料的"正文"——验证语料投影只认声明面


def _candidate(candidate_id: str, desc: str | None = None) -> Candidate:
    return Candidate(
        id=candidate_id,
        name=candidate_id,
        description=desc or f"{candidate_id} 的领域说明",
        body="# 正文\n该正文绝不能进检索语料。",
    )


def _declared(candidate: Candidate) -> str:
    """宿主语料投影惯例：声明面字段拼接（id/name/description）。"""
    return f"{candidate.id} {candidate.name} {candidate.description}"


def test_rank_orders_by_match_strength_and_annotates_corpus_terms():
    """强含查询词的候选先于弱含者；零命中语料不入排名；命中词条必须来自投影语料。

    「正文绝不能进」若能命中 body，词条断言就会翻车——语料只由 corpus_text 决定。
    """
    strong = _candidate("scenario-agent", desc="负责推演剧本编排与裁评的执行体")
    weak = _candidate("doc-agent", desc="负责编排文档结构")

    hits = rank_candidates(
        [strong, weak],
        "推演剧本编排",
        corpus_text=_declared,
        item_id=lambda c: c.id,
        item_name=lambda c: c.name,
    )

    assert [hit.item.id for hit in hits] == ["scenario-agent", "doc-agent"]
    by_id = {hit.item.id: hit for hit in hits}
    assert by_id["scenario-agent"].fused_score > by_id["doc-agent"].fused_score > 0
    assert by_id["scenario-agent"].matched_terms  # ≥1 命中语料词条
    assert all("绝不能进" not in term for hit in hits for term in hit.matched_terms)
    assert all(isinstance(hit, CandidateHit) for hit in hits)


def test_rank_is_deterministic_and_full_recall_with_vector_path():
    """向量路注入：同输入两次调用逐位一致；全量召回不截断。"""
    candidates = [
        _candidate(f"agent-{i:02d}", desc=f"执行体{i}，负责推演剧本编排第{i}环节")
        for i in range(15)
    ]

    first = rank_candidates(
        candidates, "推演剧本编排",
        corpus_text=_declared, item_id=lambda c: c.id, item_name=lambda c: c.name,
        vector=MockEmbedder(),
    )
    second = rank_candidates(
        candidates, "推演剧本编排",
        corpus_text=_declared, item_id=lambda c: c.id, item_name=lambda c: c.name,
        vector=MockEmbedder(),
    )

    assert len(first) == 15  # 全量召回，k 截断是查询面的事
    assert [(h.item.id, h.fused_score, h.matched_terms) for h in first] == [
        (h.item.id, h.fused_score, h.matched_terms) for h in second
    ]
    assert all(h.fused_score > 0 for h in first)
    scores = [h.fused_score for h in first]
    assert scores == sorted(scores, reverse=True)


def test_rank_empty_pool_returns_empty():
    assert rank_candidates([], "推演剧本编排", corpus_text=_declared, item_id=lambda c: c.id) == []


def test_embedding_failure_degrades_to_bm25_only_without_raising():
    """嵌入失败按 QueryEmbeddingError 捕获：退化 BM25 单路，不外抛（缺席降级契约）。"""
    from agent_retrieval.core.ports import QueryEmbeddingError

    class _BrokenEmbedder:
        def embed_corpus(self, texts):
            raise QueryEmbeddingError("端点不可达")

        def embed_query(self, text):
            raise QueryEmbeddingError("端点不可达")

    candidates = [_candidate("alpha", desc="负责推演剧本编排"), _candidate("beta", desc="邮件收发")]

    hits = rank_candidates(
        candidates, "推演剧本编排",
        corpus_text=_declared, item_id=lambda c: c.id,
        vector=_BrokenEmbedder(),  # type: ignore[arg-type]
    )

    assert [hit.item.id for hit in hits] == ["alpha"]
    assert hits[0].cosine is None  # 向量路缺席 = None（区别于 0.0）


def test_exact_identity_tops_independent_of_scores():
    """exact 全等命中置顶是调用方契约：查询串与 id 归一化全等 → 排序键第一层生效。"""
    target = _candidate("chart-tool", desc="与查询完全无关的说明")

    hits = rank_candidates(
        [target, _candidate("other", desc="图表统计报表柱状图绘图")],
        "chart-tool",
        corpus_text=_declared, item_id=lambda c: c.id, item_name=lambda c: c.name,
    )

    assert hits[0].item.id == "chart-tool"
    assert hits[0].exact is True
