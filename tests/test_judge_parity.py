"""B3 对拍脚本判定一的离线契约测试（SDK P0，构造层语义保留）。

判定二（传输层等价）是 live 对拍——网络调用属 ``judge_parity`` 脚本实跑，不进
pytest（CI 不跑 live）。此处只测离线可复跑的构造层对照：legacy 内联构造
（``jev_rerank.jev_scores`` :71-95 的忠实复刻）与 ``judge_state`` 的字段级 diff
必须恰为已登记项（候选投影 ``tags`` 键删除一项）；超长 description 两侧截断逐位
一致；instruction 通道两侧贯通；响应键→id 映射随候选顺序一致；diff 机制本身对
未登记差异有检出力（负面用例）。
"""

from __future__ import annotations

import pytest

from experiments.benchmarks.judge_parity import (
    JEV_DESCRIPTION_MAX_CHARS,
    build_window,
    diff_payloads,
    legacy_judge_payload,
    long_description,
    new_judge_payload,
    registered_tag_removals,
    response_id_mapping,
    unregistered_diffs,
)

_TASK = "Find population data for California in 2024"
_INSTRUCTION = "示例检索指令"

#: 合成窗口池：短候选（use_when/avoid_when 非空——R3-P1 样本要求）+ 长描述候选。
_POOL_ENTRIES = [
    {"id": "geo-query", "name": "Geo Query",
     "description": "Query population and demographic data by region.",
     "tags": "geo;census",
     "use_when": "A user needs population statistics for a place and time.",
     "avoid_when": "Do not use for weather forecasts or map images."},
    {"id": "email-send", "name": "Email Sender",
     "description": "Send an email message to named recipients.",
     "tags": "email",
     "use_when": "A user asks to email someone a message.",
     "avoid_when": "Do not use to search an inbox."},
    {"id": "weather-forecast", "name": "Weather Forecast",
     "description": "Report current weather and short-term forecasts.",
     "tags": "weather",
     "use_when": "A user asks about upcoming weather.",
     "avoid_when": "Do not use for historical climate data."},
]
_LONG_PROBE = {"id": "long-doc", "name": "Long Doc", "description": long_description(),
               "tags": "docs"}


def _window():
    """合成窗口（含长描述探针）的两型适配：resources 与 candidates 同源分叉。"""
    return build_window([*_POOL_ENTRIES, _LONG_PROBE])


def test_diff_is_exactly_registered_tags_removal():
    """判定一核心断言：diff 恰为每候选一条 tags 键删除，无任何未登记差异。"""
    resources, candidates = _window()
    legacy = legacy_judge_payload(_TASK, "", resources)
    new = new_judge_payload(_TASK, "", candidates)
    diffs = diff_payloads(legacy, new)
    assert unregistered_diffs(diffs) == []
    assert len(registered_tag_removals(diffs)) == len(resources)


def test_use_when_and_avoid_when_absent_from_both_sides():
    """非空 use_when/avoid_when 两侧都不进 state（R3-P1 样本要求证实）。"""
    resources, candidates = _window()
    legacy = legacy_judge_payload(_TASK, "", resources)
    new = new_judge_payload(_TASK, "", candidates)
    for payload in (legacy, new):
        for projected in payload["state"]["candidates"].values():
            assert "use_when" not in projected
            assert "avoid_when" not in projected


def test_long_description_truncation_bitwise_identical():
    """>2000 字符 description：同一文本两型适配，两侧截断后逐位一致。"""
    text = long_description()
    assert len(text) > JEV_DESCRIPTION_MAX_CHARS
    resources, candidates = _window()
    legacy = legacy_judge_payload(_TASK, "", resources)
    new = new_judge_payload(_TASK, "", candidates)
    legacy_desc = legacy["state"]["candidates"]["c3"]["description"]
    new_desc = new["state"]["candidates"]["c3"]["description"]
    assert new_desc == text[:JEV_DESCRIPTION_MAX_CHARS]
    assert legacy_desc == new_desc


def test_instruction_channel_flows_to_both_sides():
    """非空 instruction 验证通道：两侧 retrieval_instruction 逐位一致。"""
    resources, candidates = _window()
    legacy = legacy_judge_payload(_TASK, _INSTRUCTION, resources)
    new = new_judge_payload(_TASK, _INSTRUCTION, candidates)
    assert legacy["state"]["retrieval_instruction"] == _INSTRUCTION
    assert new["state"]["retrieval_instruction"] == _INSTRUCTION
    assert unregistered_diffs(diff_payloads(legacy, new)) == []


def test_response_id_mapping_follows_candidate_order():
    """响应键→id 映射分开记录：c{index} → id 两侧随候选顺序一致。"""
    resources, candidates = _window()
    legacy_mapping = response_id_mapping([resource.id for resource in resources])
    new_mapping = response_id_mapping([candidate.id for candidate in candidates])
    assert legacy_mapping == new_mapping
    assert list(legacy_mapping.values()) == [entry["id"] for entry in _POOL_ENTRIES] + ["long-doc"]


def test_question_shape_and_candidate_order_match():
    """每问题 instructions/criteria 逐字段一致；候选顺序由 c{index} 键序核验。"""
    resources, candidates = _window()
    legacy = legacy_judge_payload(_TASK, "", resources)
    new = new_judge_payload(_TASK, "", candidates)
    assert list(legacy["state"]["candidates"]) == list(new["state"]["candidates"])
    assert list(legacy["questions"]) == list(new["questions"])
    assert legacy["questions"]["c0"] == new["questions"]["c0"]
    assert "candidates.c0" in new["questions"]["c0"]["instructions"]


def test_unregistered_diff_is_detected():
    """负面用例：篡改 new 侧 name → diff 机制必须报出未登记差异。"""
    resources, candidates = _window()
    legacy = legacy_judge_payload(_TASK, "", resources)
    new = new_judge_payload(_TASK, "", candidates)
    new["state"]["candidates"]["c0"]["name"] = "Tampered Name"
    tampered = diff_payloads(legacy, new)
    paths = [diff["path"] for diff in unregistered_diffs(tampered)]
    assert "state.candidates.c0.name" in paths


def test_extra_new_side_key_is_unregistered():
    """反向缺口（new 独有的多余键）不属已登记形态——必须报未登记差异。"""
    resources, candidates = _window()
    legacy = legacy_judge_payload(_TASK, "", resources)
    new = new_judge_payload(_TASK, "", candidates)
    new["state"]["candidates"]["c1"]["extra"] = "unexpected"
    assert unregistered_diffs(diff_payloads(legacy, new)) != []


@pytest.mark.parametrize("missing", ["name", "description"])
def test_new_side_key_gap_is_unregistered(missing: str):
    """登记项只在 legacy→new 方向的 tags 上：new 侧缺任一保留键都是未登记差异。"""
    resources, candidates = _window()
    legacy = legacy_judge_payload(_TASK, "", resources)
    new = new_judge_payload(_TASK, "", candidates)
    del new["state"]["candidates"]["c2"][missing]
    paths = [diff["path"] for diff in unregistered_diffs(diff_payloads(legacy, new))]
    assert f"state.candidates.c2.{missing}" in paths
