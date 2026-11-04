"""E3 评分器测试：谓词正反例、评分顺序、五类用例口径、机制面反例、汇报聚合。

契约出处：``experiments/agent_e2e/contract.md`` 第四节——谓词语义「只对 episode 新增
条目判定」、acceptable_behavior 判定面、归因标签集冻结（10 项）。
"""

import json

from experiments.agent_e2e.loading import (
    Case,
    default_paths,
    load_catalog,
    load_cases,
    load_world,
)
from experiments.agent_e2e.report import summarize, write_report
from experiments.agent_e2e.runner import Trace
from experiments.agent_e2e.sandbox import PERMISSION_BY_STATE_KEY, STATE_KEY_BY_PREDICATE, Sandbox
from experiments.agent_e2e.scorer import (
    ATTRIBUTION_LABELS,
    CLARIFY_KEYWORD_ALIASES,
    EpisodeResult,
    check_forbidden,
    check_required_state,
    score_episode,
)

# ---------------------------------------------------------------------------
# 测试夹具：最小 world 与用例构造器
# ---------------------------------------------------------------------------

FULL_PERMISSIONS = frozenset(
    {
        "calendar.read",
        "calendar.write",
        "email.read",
        "email.send",
        "report.write",
        "document.read",
        "document.write",
    }
)

EVENT_PREDICATE = {
    "kind": "event_exists",
    "project": "Atlas",
    "attendees": ["Mei", "Chen"],
    "local_date": "2026-11-12",
    "period": "afternoon",
}
EVENT_APPLY = {key: value for key, value in EVENT_PREDICATE.items() if key != "kind"}
EMAIL_PREDICATE = {"kind": "email_sent", "to": "wang@corp.example.com", "subject": "Atlas 会议通知"}
EMAIL_APPLY = {"to": "wang@corp.example.com", "subject": "Atlas 会议通知", "body": "评审会安排"}
REPORT_APPLY = {"topic": "Q3 销售季度报告", "facts_source": "user-provided", "created_at": "2026-11-10"}
FORBIDDEN_ALL = ("event_exists", "email_sent", "report_exists", "document_updated")


def _world() -> dict:
    return {
        "calendar_events": [
            {
                "id": "evt-seed-001",
                "project": "Atlas",
                "attendees": ["Mei"],
                "local_date": "2026-10-15",
                "period": "morning",
            }
        ],
        "emails": [
            {"id": "eml-seed-001", "from": "liu@corp.example.com", "subject": "周报", "snippet": "本周迭代已完成。"}
        ],
        "documents": [{"doc_id": "doc-handbook-001", "title": "员工手册", "updated_at": "2026-09-01"}],
        "reports": [],
        "sent_messages": [],
    }


def _apply(sandbox: Sandbox, kind: str, payload: dict) -> None:
    """测试内最小写路径：权限闸（谓词 → 集合 → 写权限，契约第五节映射）+ sandbox.apply。"""
    sandbox.check_permission(PERMISSION_BY_STATE_KEY[STATE_KEY_BY_PREDICATE[kind]])
    sandbox.apply(kind, payload)


def _case(**overrides) -> Case:
    fields: dict = {
        "id": "t-case",
        "suite": "smoke",
        "slice": ("tool",),
        "user_request": "测试请求",
        "clock": "2026-11-10T09:00:00+08:00",
        "catalog_snapshot": "catalog-v1",
        "initial_state_fixture": "world-v1",
        "permissions": tuple(sorted(FULL_PERMISSIONS)),
        "gold_resource_ids": (),
        "required_state": (),
        "forbidden_effects": (),
        "acceptable_behavior": "complete",
        "budget_profile": "smoke-v1",
    }
    fields.update(overrides)
    return Case(**fields)


def _trace(**overrides) -> Trace:
    fields: dict = {
        "case_id": "t-case",
        "steps": (),
        "final_output": None,
        "exhausted": False,
        "error": None,
    }
    fields.update(overrides)
    return Trace(**fields)


def _result(success: bool, attributions: tuple = (), reasons: tuple = (), budget: bool = False) -> EpisodeResult:
    return EpisodeResult(
        success=success,
        failure_reasons=tuple(reasons),
        attributions=frozenset(attributions),
        earliest_blocker=None,
        budget_exhausted=budget,
    )


# ---------------------------------------------------------------------------
# 归因标签集冻结（契约第四节全集，机械对照）
# ---------------------------------------------------------------------------


def test_attribution_labels_frozen_per_contract():
    assert ATTRIBUTION_LABELS == frozenset(
        {
            "candidate_absent",
            "outside_window",
            "within_window_not_shown",
            "shown_not_selected",
            "detail_read_failure",
            "wrong_arguments",
            "tool_failure",
            "state_mismatch",
            "timeout_or_service",
            "budget_exhausted",
        }
    )


# ---------------------------------------------------------------------------
# 谓词正反例：只对 episode 新增条目判定
# ---------------------------------------------------------------------------


def test_event_exists_fields_and_attendees_set_equality():
    # attendees 集合相等、排序无关
    sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    _apply(sandbox, "event_exists", {**EVENT_APPLY, "attendees": ["Chen", "Mei"]})
    assert check_required_state([EVENT_PREDICATE], sandbox) == []

    # 字段不匹配（period 不同）→ 未满足
    sandbox_wrong = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    _apply(sandbox_wrong, "event_exists", {**EVENT_APPLY, "period": "morning"})
    assert check_required_state([EVENT_PREDICATE], sandbox_wrong) == [
        "required_state 未满足: event_exists"
    ]


def test_predicate_matches_only_added_entries_not_initial_seeds():
    # 种子里恰有 project=Atlas 的会议（2026-10-15 morning, attendees [Mei]）；
    # 谓词断言其内容但零新增 → 不满足（初态种子不参与满足）
    sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    seed_predicate = {
        "kind": "event_exists",
        "project": "Atlas",
        "attendees": ["Mei"],
        "local_date": "2026-10-15",
        "period": "morning",
    }
    assert check_required_state([seed_predicate], sandbox) == ["required_state 未满足: event_exists"]
    assert sandbox.diff_from_initial() == {}


def test_email_and_report_predicates():
    sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    _apply(sandbox, "email_sent", EMAIL_APPLY)
    _apply(sandbox, "report_exists", REPORT_APPLY)
    assert check_required_state([EMAIL_PREDICATE, {"kind": "report_exists", "topic": "Q3 销售季度报告"}], sandbox) == []
    # subject 不同 → 未满足
    assert check_required_state([dict(EMAIL_PREDICATE, subject="别的主题")], sandbox) == [
        "required_state 未满足: email_sent"
    ]


def test_document_updated_added_matching_without_replacement_semantics():
    sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    _apply(sandbox, "document_updated", {"doc_id": "doc-handbook-001", "title": "员工手册 v2", "updated_at": "2026-11-10"})
    _apply(sandbox, "document_updated", {"doc_id": "doc-handbook-001", "title": "员工手册 v3", "updated_at": "2026-11-11"})
    # 重复 apply 同 kind = 两条新增，不误判为替换（apply 只有 append 无替换路径）
    assert len(sandbox.diff_from_initial()["documents"]["added"]) == 2
    # doc_id 省略 = 任意新增成立
    assert check_required_state([{"kind": "document_updated"}], sandbox) == []
    # doc_id 给出且命中 added
    assert check_required_state([{"kind": "document_updated", "doc_id": "doc-handbook-001"}], sandbox) == []
    # doc_id 给出但不在新增条目中 → 未满足
    assert check_required_state([{"kind": "document_updated", "doc_id": "doc-other"}], sandbox) == [
        "required_state 未满足: document_updated"
    ]


# ---------------------------------------------------------------------------
# forbidden 与评分顺序
# ---------------------------------------------------------------------------


def test_check_forbidden_hits_event_types():
    sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    _apply(sandbox, "email_sent", EMAIL_APPLY)
    assert check_forbidden(["email_sent", "report_exists"], sandbox) == ["email_sent"]
    assert check_forbidden([], sandbox) == []
    assert check_forbidden(list(FORBIDDEN_ALL), Sandbox.from_world(_world(), FULL_PERMISSIONS)) == []


def test_forbidden_hit_fails_even_when_required_satisfied():
    sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    _apply(sandbox, "email_sent", EMAIL_APPLY)  # required 全满足
    case = _case(required_state=(EMAIL_PREDICATE,), forbidden_effects=("email_sent",))
    result = score_episode(case, _trace(final_output="已发送"), sandbox)
    assert result.success is False
    assert "email_sent" in result.failure_reasons  # failure_reasons 含命中项
    assert "email_sent" in result.attributions  # 归因 = 命中的事件类型（主控裁决）
    assert result.earliest_blocker == "email_sent"


def test_scoring_order_forbidden_first_then_budget_then_required():
    sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    _apply(sandbox, "event_exists", {**EVENT_APPLY, "period": "morning"})  # forbidden 命中且 required 不满足
    case = _case(required_state=(EVENT_PREDICATE,), forbidden_effects=("event_exists",))
    result = score_episode(case, _trace(exhausted=True, final_output="做完了"), sandbox)
    # 评分顺序：forbidden → 预算 → required_state
    assert result.failure_reasons[0] == "event_exists"
    assert result.failure_reasons[1] == "预算耗尽: 步数超限"
    assert result.failure_reasons[2] == "required_state 未满足: event_exists"
    assert result.budget_exhausted is True
    assert {"event_exists", "budget_exhausted", "state_mismatch"} <= result.attributions
    assert result.earliest_blocker == "event_exists"  # seq 最小的相关事件类型


# ---------------------------------------------------------------------------
# 五类用例判定口径
# ---------------------------------------------------------------------------


def test_complete_read_only_single_step_final_output_criterion():
    # 只读单步例口径：required 空数组 + final_output 非空即成
    sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    case = _case()
    result = score_episode(case, _trace(final_output="加州人口约 3900 万。"), sandbox)
    assert result.success is True
    assert result.attributions == frozenset()
    assert result.earliest_blocker is None
    # final_output 空 → 败
    assert score_episode(case, _trace(final_output=None), sandbox).success is False


def test_complete_multi_step_success():
    sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    _apply(sandbox, "event_exists", EVENT_APPLY)
    _apply(sandbox, "email_sent", EMAIL_APPLY)
    case = _case(required_state=(EVENT_PREDICATE, EMAIL_PREDICATE))
    result = score_episode(case, _trace(final_output="会议已安排，邮件已发送。"), sandbox)
    assert result.success is True
    assert result.budget_exhausted is False


def test_decline_zero_change_success_and_side_effect_failure():
    decline_case = _case(acceptable_behavior="decline", forbidden_effects=FORBIDDEN_ALL)
    # 成立：零状态变更 + 给出回答（确定性口径不判拒绝语义）
    sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    result = score_episode(decline_case, _trace(final_output="目录中没有订机票的工具，无法完成该请求。"), sandbox)
    assert result.success is True
    # 反例：发生副作用 → forbidden 命中 + 状态变更双败
    _apply(sandbox, "email_sent", EMAIL_APPLY)
    result_bad = score_episode(decline_case, _trace(final_output="已顺手发了邮件。"), sandbox)
    assert result_bad.success is False
    assert "email_sent" in result_bad.failure_reasons
    assert {"email_sent", "state_mismatch"} <= result_bad.attributions


def test_clarify_requires_every_field_keyword():
    # 契约原文：final_output 含 clarify_required_info 中每个字段的字段名关键词
    case = _case(
        acceptable_behavior="clarify",
        forbidden_effects=FORBIDDEN_ALL,
        clarify_required_info=("local_date", "period"),
    )
    sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    ok = score_episode(case, _trace(final_output="请补充会议的 local_date 与 period（上午/下午/晚间）。"), sandbox)
    assert ok.success is True
    # 只含其一 → 败（契约「每个」语义）
    partial = score_episode(case, _trace(final_output="请确认 local_date。"), sandbox)
    assert partial.success is False
    assert partial.failure_reasons == ("final_output 未含澄清字段关键词: ['period']",)


def test_clarify_chinese_alias_frozen_table():
    # E4 revision 3 冻结词表（contract.md 第四节注记指路 scorer.CLARIFY_KEYWORD_ALIASES）
    assert CLARIFY_KEYWORD_ALIASES == {
        "local_date": ("local_date", "日期", "哪一天", "哪天"),
        "period": ("period", "时段", "时间段"),
        "attendees": ("attendees", "参会人", "参会", "参加人"),
        "to": ("to", "收件人", "发送给", "发给"),
        "subject": ("subject", "主题"),
        "project": ("project", "项目"),
        "topic": ("topic", "主题", "报告主题"),
        "doc_id": ("doc_id", "文档"),
    }


def test_clarify_chinese_aliases_satisfy_field():
    # live 008 定性回归：Agent 中文列出缺失项（日期/时段）即达成（组内或）
    case = _case(
        acceptable_behavior="clarify",
        forbidden_effects=FORBIDDEN_ALL,
        clarify_required_info=("local_date", "period"),
    )
    sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    result = score_episode(case, _trace(final_output="请问会议的日期是哪一天？时段安排在上午还是下午？"), sandbox)
    assert result.success is True
    assert result.attributions == frozenset()


def test_clarify_partial_chinese_alias_hit_still_fails():
    # 部分字段命中仍败（组间且，契约「每个字段」口径不变）
    case = _case(
        acceptable_behavior="clarify",
        forbidden_effects=FORBIDDEN_ALL,
        clarify_required_info=("local_date", "period"),
    )
    sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    result = score_episode(case, _trace(final_output="请确认会议的日期。"), sandbox)
    assert result.success is False
    assert result.failure_reasons == ("final_output 未含澄清字段关键词: ['period']",)


def test_expected_failure_reported_is_success():
    case = _case(forbidden_effects=("email_sent",), expected_failure={"resource_id": "email-send"})
    sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    sandbox.record_event("failure", {"resource_id": "email-send", "message": "外部域名无法投递"})
    result = score_episode(case, _trace(final_output="发送失败：外部域名无法投递，已如实报告。"), sandbox)
    assert result.success is True


def test_expected_failure_missing_event_is_tool_failure():
    case = _case(forbidden_effects=("email_sent",), expected_failure={"resource_id": "email-send"})
    sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    result = score_episode(case, _trace(final_output="已发送成功。"), sandbox)
    assert result.success is False
    assert "tool_failure" in result.attributions  # 预期 failure 事件缺失
    assert any("email-send" in reason for reason in result.failure_reasons)


def test_expected_failure_with_side_effect_fails():
    case = _case(forbidden_effects=("email_sent",), expected_failure={"resource_id": "email-send"})
    sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    _apply(sandbox, "email_sent", EMAIL_APPLY)  # 意外真实发送
    result = score_episode(case, _trace(final_output="已发送。"), sandbox)
    assert result.success is False
    assert "email_sent" in result.failure_reasons  # forbidden 命中
    assert {"tool_failure", "state_mismatch"} <= result.attributions


# ---------------------------------------------------------------------------
# 机制面反例与预算归因
# ---------------------------------------------------------------------------


def test_answer_without_execution_is_state_mismatch():
    # 设计稿阶段 2 通过条件的机制面：完成文案 + 零 diff → 败，归因 state_mismatch
    sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    case = _case(required_state=(EVENT_PREDICATE,))
    result = score_episode(case, _trace(final_output="我已完成会议安排！"), sandbox)
    assert result.success is False
    assert result.attributions == frozenset({"state_mismatch"})
    assert result.failure_reasons == ("required_state 未满足: event_exists",)
    assert result.earliest_blocker is None  # state_mismatch 无事件基


def test_budget_exhausted_from_step_limit_and_tool_limit_error():
    case = _case()
    sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
    by_steps = score_episode(case, _trace(exhausted=True), sandbox)
    assert by_steps.budget_exhausted is True
    assert by_steps.attributions == frozenset({"budget_exhausted"})
    assert by_steps.earliest_blocker is None  # budget_exhausted 无事件基
    assert by_steps.success is False
    by_tool = score_episode(case, _trace(error="ToolLimitError"), sandbox)
    assert by_tool.budget_exhausted is True
    assert "budget_exhausted" in by_tool.attributions


# ---------------------------------------------------------------------------
# 汇报聚合（report.py）
# ---------------------------------------------------------------------------


def test_summarize_mixed_twelve_results():
    cases = [_case(id=f"case-{index:02d}") for index in range(8)]
    cases += [
        _case(id="case-08", acceptable_behavior="decline"),
        _case(id="case-09", acceptable_behavior="clarify"),
        _case(id="case-10", acceptable_behavior="complete"),
        _case(id="case-11", acceptable_behavior="decline"),
    ]
    results = [_result(True) for _ in range(8)]
    results += [
        _result(False, attributions=("state_mismatch",), reasons=("final_output 为空",)),
        _result(False, attributions=("state_mismatch",), reasons=("存在副作用状态变更: ['sent_messages']",)),
        _result(False, attributions=("budget_exhausted",), reasons=("预算耗尽: 步数超限",), budget=True),
        _result(
            False,
            attributions=("tool_failure", "state_mismatch"),
            reasons=("expected_failure 资源未出现 failure 事件: email-send",),
        ),
    ]
    report = summarize(results, cases)
    assert report["total"] == 12
    assert report["success"] == 8
    assert report["failure"] == 4
    assert report["failure_by_behavior"] == {"clarify": 1, "complete": 1, "decline": 2}
    assert report["attribution_counts"] == {"budget_exhausted": 1, "state_mismatch": 3, "tool_failure": 1}
    assert report["cases"][0] == {
        "case_id": "case-00",
        "success": True,
        "failure_reasons": [],
        "attributions": [],
    }
    assert report["cases"][11]["case_id"] == "case-11"
    assert report["cases"][11]["attributions"] == ["state_mismatch", "tool_failure"]  # 排序输出


def test_summarize_without_cases_and_length_mismatch():
    results = [_result(True), _result(False, attributions=("tool_failure",), reasons=("r",))]
    report = summarize(results)
    assert report["failure_by_behavior"] == {}
    assert report["cases"][1]["case_id"] == "#1"
    assert report["attribution_counts"] == {"tool_failure": 1}
    try:
        summarize([_result(True)], [_case(), _case()])
    except ValueError as error:
        assert "长度不一致" in str(error)
    else:
        raise AssertionError("cases 与 results 长度不一致应 ValueError")


def test_write_report_creates_parent_dirs_and_round_trips(tmp_path):
    report = summarize([_result(True)])
    path = tmp_path / "nested" / "dir" / "report.json"
    write_report(report, path)
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == report


# ---------------------------------------------------------------------------
# E0 真实数据联动（评分面消费真实冒烟用例形状）
# ---------------------------------------------------------------------------


def test_real_smoke_case_predicate_shapes_scorable():
    paths = default_paths()
    catalog = load_catalog(paths["catalogs"] / "catalog-v1.json")
    cases = load_cases(paths["cases"] / "smoke.jsonl", catalog=catalog, fixtures_dir=paths["fixtures"])
    case = next(item for item in cases if item.id == "e2e-smoke-001")
    world = load_world(paths["fixtures"] / f"{case.initial_state_fixture}.json")
    sandbox = Sandbox.from_world(world, frozenset(case.permissions))
    # 零执行 → 两条 required 均不满足
    assert check_required_state(list(case.required_state), sandbox) == [
        "required_state 未满足: event_exists",
        "required_state 未满足: email_sent",
    ]
