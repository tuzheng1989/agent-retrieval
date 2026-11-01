"""Judge 端口与判停辅助的契约测试（SDK P0 / B1）。

测试面只有四个公共接缝：``JudgedCandidate`` / ``Judgment`` / ``Judge`` /
``select_with_judge``，全部经 ``agent_retrieval`` 顶层导入——顺带验证顶层导出契约。
L2 编排用例照 ``experiments/jev_retrieval.py`` 的宿主编排模式复刻：exact 短路发生在
调 judge 之前，判停归属编排层。
"""

from __future__ import annotations

import pytest

from agent_retrieval import (
    CandidateHit,
    Judge,
    JudgedCandidate,
    Judgment,
    rank_candidates,
    select_with_judge,
)

# 非对象宿主数据的最小形态：(id, 语料文本) 元组——证明提取层不依赖属性访问。
HostResource = tuple[str, str]


class StubJudge:
    """带调用计数与入参记录的 Judge 替身：返回构造时给定的固定概率。"""

    def __init__(self, probabilities: dict[str, float]) -> None:
        self._probabilities = probabilities
        self.calls: list[dict[str, object]] = []

    def judge(
        self, query: str, candidates: list[JudgedCandidate], *, instruction: str = ""
    ) -> Judgment:
        self.calls.append(
            {
                "query": query,
                "instruction": instruction,
                "candidate_ids": [candidate.id for candidate in candidates],
            }
        )
        return Judgment(
            probabilities=dict(self._probabilities), model="stub", latency_seconds=0.0
        )


def _hit(item: str, *, exact: bool = False) -> CandidateHit[str]:
    return CandidateHit(item=item, fused_score=1.0, matched_terms=(), exact=exact)


def _judgment(probabilities: dict[str, float]) -> Judgment:
    return Judgment(probabilities=probabilities, model="stub", latency_seconds=0.0)


def _identity(item: str) -> str:
    return item


def _select(
    hits: list[CandidateHit[str]], judgment: Judgment, *, minimum_probability: float
) -> str | None:
    """字符串 item 的便捷调用：item_id 取恒等映射。"""
    return select_with_judge(
        hits, judgment, minimum_probability=minimum_probability, item_id=_identity
    )


def _host_id(item: HostResource) -> str:
    return item[0]


def _host_shortlist(items: list[HostResource], query: str) -> list[CandidateHit[HostResource]]:
    return rank_candidates(items, query, corpus_text=lambda item: item[1], item_id=_host_id)


def _to_candidates(items: list[HostResource]) -> list[JudgedCandidate]:
    return [JudgedCandidate(id=item[0], name=item[0], description=item[1]) for item in items]


def test_stub_judge_satisfies_judge_protocol():
    """@runtime_checkable：实现方只需提供 judge 方法即满足 Judge。"""
    assert isinstance(StubJudge({"a": 0.9}), Judge)


def test_instruction_is_passed_through_to_judge():
    """instruction 是逐次调用参数：透传对替身可见（与 query 并列的接缝）。"""
    judge = StubJudge({"a": 0.9})
    candidates = [JudgedCandidate(id="a", name="工具甲", description="能直接完成任务的工具")]

    judge.judge("部署服务", candidates, instruction="仅当候选能直接完成任务时判 true")

    assert judge.calls == [
        {
            "query": "部署服务",
            "instruction": "仅当候选能直接完成任务时判 true",
            "candidate_ids": ["a"],
        }
    ]


def test_exact_hit_stops_before_judge_is_called():
    """L2 编排判停：exact 命中时短路发生在调 judge 之前——StubJudge 调用次数为 0。

    编排模式照 experiments/jev_retrieval.py 复刻：先查 ``hit.exact``，命中则不构造
    Judgment、不调判定；只有非 exact 路径才进 judge + select_with_judge。
    """
    judge = StubJudge({"chart-tool": 0.99})
    items = [("chart-tool", "图表绘制与统计报表"), ("doc-agent", "文档结构编排")]
    hits = _host_shortlist(items, "chart-tool")
    assert hits[0].exact  # 前置：确为 exact 全等命中

    selected: str | None
    if hits and not hits[0].exact:
        judgment = judge.judge("chart-tool", _to_candidates([hit.item for hit in hits]))
        selected = select_with_judge(hits, judgment, minimum_probability=0.7, item_id=_host_id)
    else:
        selected = _host_id(hits[0].item) if hits else None

    assert judge.calls == []  # 判停：exact 命中路径零判定调用
    assert selected == "chart-tool"


def test_tie_preserves_retrieval_order():
    """概率并列时 max 保留 hits 检索序：返回检索序在前者，不因 dict 遍历序漂移。"""
    hits = [_hit("a"), _hit("b")]
    assert _select(hits, _judgment({"a": 0.9, "b": 0.9}), minimum_probability=0.7) == "a"


def test_below_minimum_probability_abstains():
    """最佳候选概率低于门槛 → 返回 None（abstain 是正常返回值，不是异常）。"""
    hits = [_hit("a"), _hit("b")]
    assert (
        _select(hits, _judgment({"a": 0.6, "b": 0.5}), minimum_probability=0.7) is None
    )


def test_probability_equal_to_threshold_returns_hit():
    """概率恰等于门槛 → 返回该候选（与原实现 `>= threshold` 对齐，R4 审计边界）。"""
    hits = [_hit("a"), _hit("b")]
    assert _select(hits, _judgment({"a": 0.7, "b": 0.1}), minimum_probability=0.7) == "a"


def test_missing_candidate_key_raises():
    """probabilities 缺候选 id → ValueError（键集必须与候选一一对应）。"""
    hits = [_hit("a"), _hit("b")]
    with pytest.raises(ValueError):
        _select(hits, _judgment({"a": 0.9}), minimum_probability=0.7)


def test_extra_probability_key_raises():
    """probabilities 含候选外 id（多键方向）→ ValueError（与缺键同一不等式的另一侧）。"""
    hits = [_hit("a"), _hit("b")]
    with pytest.raises(ValueError):
        _select(hits, _judgment({"a": 0.9, "b": 0.9, "ghost": 0.5}), minimum_probability=0.7)


def test_out_of_range_probability_raises():
    """概率越界（1.2）→ ValueError。"""
    hits = [_hit("a")]
    with pytest.raises(ValueError):
        _select(hits, _judgment({"a": 1.2}), minimum_probability=0.7)


def test_nan_probability_is_rejected_as_out_of_range():
    """NaN 概率按越界拦截：``not 0 <= p <= 1`` 对 NaN 恒真（NaN 与任何数比较为 False）。"""
    assert not 0 <= float("nan") <= 1  # 钉住拦截机制本身
    hits = [_hit("a")]
    with pytest.raises(ValueError):
        _select(hits, _judgment({"a": float("nan")}), minimum_probability=0.7)


def test_duplicate_candidate_ids_raise():
    """候选 [a, a] → ValueError：重复检测在键集比较之前——纯集合比较会漏此形态。"""
    hits = [_hit("a"), _hit("a")]
    with pytest.raises(ValueError):
        _select(hits, _judgment({"a": 0.9}), minimum_probability=0.7)


def test_empty_shortlist_returns_none():
    """空 shortlist → None，不抛（空是正常路径）。"""
    assert _select([], _judgment({}), minimum_probability=0.7) is None


def test_minimum_probability_has_no_default():
    """门槛必填（照 clearly_related 必填关键字风格）：漏传直接 TypeError。"""
    with pytest.raises(TypeError):
        select_with_judge([_hit("a")], _judgment({"a": 0.9}), item_id=_identity)


def test_rank_candidates_hits_flow_through_selector_with_tuple_host_data():
    """组合用例：真实 rank_candidates 产出 hits（宿主数据为元组）经 item_id 提取层正确选取。

    ``hit.item`` 持有的是宿主元组而非带 id 属性的对象——ID 只能经回调提取，
    这正是 item_id 回调接收宿主数据 T 而非 hit 的接缝证明。
    """
    items = [("jev-flow", "Jev 判定候选精排流程"), ("bm25-flow", "BM25 判定评分的检索内核")]
    hits = _host_shortlist(items, "Jev 判定精排")
    assert len(hits) == 2  # 前置：全量召回，两候选都在 shortlist
    assert not hits[0].exact  # 前置：非 exact，编排层才会调 judge

    judge = StubJudge({"jev-flow": 0.8, "bm25-flow": 0.4})
    judgment = judge.judge("Jev 判定精排", _to_candidates([hit.item for hit in hits]))
    selected = select_with_judge(hits, judgment, minimum_probability=0.7, item_id=_host_id)

    assert selected == "jev-flow"
