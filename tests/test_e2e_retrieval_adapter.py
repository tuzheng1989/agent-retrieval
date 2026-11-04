"""E2 检索适配器测试：资格过滤 → rank_candidates 全量排序 → 前 k 展示窗口（E-D3）。

用真实 catalog-v1 数据驱动；evidence 是宿主留存的全量排序证据（未截断），
shown 只是 evidence 排序的前 k 前缀——两条断言面分开钉住。
"""

from agent_retrieval import rank_candidates

from experiments.agent_e2e.loading import Catalog, default_paths, load_catalog
from experiments.agent_e2e.retrieval_adapter import RetrievalAdapter


def _adapter() -> tuple[RetrievalAdapter, Catalog]:
    catalog = load_catalog(default_paths()["catalogs"] / "catalog-v1.json")
    return RetrievalAdapter(catalog), catalog


def _ranking_ids(result) -> list[str]:
    return [entry["id"] for entry in result.evidence["ranking"]]


# ---------------------------------------------------------------------------
# 资格过滤（E2 验收 1：allowed_ids 排除者全量 evidence 也不含）
# ---------------------------------------------------------------------------


def test_excluded_id_absent_from_shown_and_full_evidence():
    adapter, catalog = _adapter()
    allowed = frozenset(catalog.by_id) - {"geo-query"}

    result = adapter.search(
        "population and demographic data by geographic region", allowed
    )

    assert "geo-query" not in _ranking_ids(result)
    assert "geo-query" not in [r.id for r in result.shown]


def test_included_id_ranks_when_query_matches():
    """对照锚：同一查询在未排除时 geo-query 可检索到——排除断言因此有意义。"""
    adapter, _ = _adapter()
    result = adapter.search(
        "population and demographic data by geographic region",
        frozenset(adapter.catalog.by_id),
    )
    assert "geo-query" in _ranking_ids(result)


# ---------------------------------------------------------------------------
# kind 过滤
# ---------------------------------------------------------------------------


def test_kind_filter_only_skill_resources_enter_ranking():
    adapter, _ = _adapter()
    skill_ids = {"meeting-notes-skill", "data-summary-skill"}

    result = adapter.search("会议安排 数据摘要 报告", frozenset(adapter.catalog.by_id), kind="skill")

    assert set(_ranking_ids(result)) <= skill_ids
    assert all(r.kind == "skill" for r in result.shown)


def test_kind_filter_tool_excludes_skills():
    adapter, _ = _adapter()
    result = adapter.search(
        "population weather email calendar", frozenset(adapter.catalog.by_id), kind="tool"
    )
    assert "meeting-notes-skill" not in _ranking_ids(result)
    assert "data-summary-skill" not in _ranking_ids(result)


# ---------------------------------------------------------------------------
# k 截断与 evidence 全量
# ---------------------------------------------------------------------------


def test_k_truncates_shown_but_evidence_keeps_full_ranking():
    adapter, _ = _adapter()
    result = adapter.search(
        "population demographic weather forecast email calendar query data",
        frozenset(adapter.catalog.by_id),
        k=3,
    )

    ranking = result.evidence["ranking"]
    assert len(result.shown) == 3
    assert len(ranking) > 3  # 全量排序证据未被截断
    assert _ranking_ids(result)[:3] == [r.id for r in result.shown]  # shown 是前缀


def test_default_k_is_five():
    adapter, _ = _adapter()
    result = adapter.search(
        "population demographic weather forecast email calendar query data",
        frozenset(adapter.catalog.by_id),
    )
    assert len(result.shown) == 5
    assert result.evidence["ranking"] and len(result.evidence["ranking"]) >= len(result.shown)


def test_exact_id_query_tops_ranking_and_flags_exact():
    adapter, _ = _adapter()
    result = adapter.search("email-send", frozenset(adapter.catalog.by_id))

    top = result.evidence["ranking"][0]
    assert top["id"] == "email-send"
    assert top["exact"] is True
    assert result.shown[0].id == "email-send"


def test_evidence_entries_carry_score_matched_terms_exact():
    adapter, _ = _adapter()
    result = adapter.search(
        "population and demographic data", frozenset(adapter.catalog.by_id)
    )
    for entry in result.evidence["ranking"]:
        assert set(entry) == {"id", "fused_score", "matched_terms", "exact"}
        assert isinstance(entry["fused_score"], float)
        assert isinstance(entry["matched_terms"], list)
        assert isinstance(entry["exact"], bool)


# ---------------------------------------------------------------------------
# 投影对照（E2 验收 5：adapter 投影与手写 getattr 拼接 hits 逐位一致）
# ---------------------------------------------------------------------------


def test_projection_parity_with_manual_getattr_join():
    adapter, catalog = _adapter()
    eligible = list(catalog.resources)
    query = "population demographic weather forecast send email calendar meeting"

    manual_hits = rank_candidates(
        eligible,
        query,
        corpus_text=lambda r: " ".join(
            part for part in (getattr(r, f) for f in ("id", "name", "description")) if part
        ),
        item_id=lambda r: r.id,
        item_name=lambda r: r.name,
    )
    adapted = adapter.search(query, frozenset(catalog.by_id))

    manual = [
        (h.item.id, h.fused_score, h.matched_terms, h.exact) for h in manual_hits
    ]
    evidence = [
        (e["id"], e["fused_score"], tuple(e["matched_terms"]), e["exact"])
        for e in adapted.evidence["ranking"]
    ]
    assert manual == evidence


# ---------------------------------------------------------------------------
# 空查询 / 无匹配
# ---------------------------------------------------------------------------


def test_empty_query_returns_empty():
    adapter, catalog = _adapter()
    result = adapter.search("", frozenset(catalog.by_id))
    assert result.shown == []
    assert result.evidence["ranking"] == []


def test_no_match_query_returns_empty():
    adapter, catalog = _adapter()
    result = adapter.search("zzqqxxww 完全不存在的词", frozenset(catalog.by_id))
    assert result.shown == []
    assert result.evidence["ranking"] == []


def test_search_result_is_frozen_dataclass():
    adapter, catalog = _adapter()
    result = adapter.search("email", frozenset(catalog.by_id))
    try:
        result.shown = []  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("SearchResult 应为 frozen dataclass")
