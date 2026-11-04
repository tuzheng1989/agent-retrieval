"""E2 受控四工具测试：授权矩阵、invoke 全链路、generated 确定性、搜索限额。

单元面用合成 catalog（failure_modes 携带机器可读 match 字段，契约 §5 变更后形状）；
另设一条真实 catalog + real world 的装配编排（「搜索 → 详情 → 调用」L2 面）。
"""

import json

import pytest

from experiments.agent_e2e.loading import (
    Catalog,
    ResourceSpec,
    default_paths,
    load_catalog,
    load_world,
)
from experiments.agent_e2e.retrieval_adapter import RetrievalAdapter
from experiments.agent_e2e.sandbox import KNOWN_PERMISSIONS, Sandbox, SandboxPermissionError
from experiments.agent_e2e.tools import ToolLimitError, build_tools, match_failure_mode

CLOCK = "2026-11-10T09:00:00+08:00"
FULL = frozenset(KNOWN_PERMISSIONS)


# ---------------------------------------------------------------------------
# 合成目录（match 字段齐备）与 world 替身
# ---------------------------------------------------------------------------


def _catalog() -> Catalog:
    email = ResourceSpec(
        id="email-send",
        name="Email Sender",
        description="Send an email message to named recipients.",
        use_when="A user asks to email someone a message.",
        avoid_when="Do not use to search an inbox.",
        kind="tool",
        params_schema={
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        invoke_effect=(
            {
                "kind": "email_sent",
                "state_key": "sent_messages",
                "fields": {
                    "to": "<args.to>",
                    "subject": "<args.subject>",
                    "body": "<args.body>",
                    "sent_at": "generated:clock",
                },
            },
        ),
        failure_modes=(
            {
                "condition": "收件人域名属于未知投递域（以 .invalid 结尾）",
                "match": {"arg": "to", "op": "endswith", "value": "example.invalid"},
                "message": "发送失败：收件人域名无法投递，消息未发出。",
            },
        ),
    )
    calendar = ResourceSpec(
        id="calendar-create",
        name="Calendar Event Creator",
        description="Create a meeting event with a date, time, and attendees.",
        use_when="A user wants to schedule a meeting.",
        avoid_when="Do not use to send an email.",
        kind="tool",
        params_schema={
            "project": {"type": "string"},
            "attendees": {"type": "array"},
            "local_date": {"type": "string"},
            "period": {"type": "string"},
        },
        invoke_effect=(
            {
                "kind": "event_exists",
                "state_key": "calendar_events",
                "fields": {
                    "id": "generated:event-id",
                    "project": "<args.project>",
                    "attendees": "<args.attendees>",
                    "local_date": "<args.local_date>",
                    "period": "<args.period>",
                },
            },
        ),
        failure_modes=(),
    )
    geo = ResourceSpec(
        id="geo-query",
        name="Geo Query",
        description="Query population data by geographic region and year.",
        use_when="A user needs population statistics.",
        avoid_when="Do not use for weather forecasts.",
        kind="tool",
        params_schema={"region": {"type": "string"}, "year": {"type": "integer"}},
        invoke_effect=(),
        failure_modes=(),
    )
    skill = ResourceSpec(
        id="notes-skill",
        name="Meeting Notes Skill",
        description="会议安排方法。",
        use_when="用户要求安排会议时。",
        avoid_when="不要用于数据报告。",
        kind="skill",
        body="会议安排方法\n\n步骤一：确认四要素。步骤二：calendar-create。",
    )
    resources = (email, calendar, geo, skill)
    return Catalog(resources=resources, by_id={r.id: r for r in resources})


def _world() -> dict:
    return {
        "calendar_events": [
            {
                "id": "evt-1",
                "project": "Atlas",
                "attendees": ["Mei"],
                "local_date": "2026-10-15",
                "period": "morning",
            }
        ],
        "emails": [],
        "documents": [],
        "reports": [],
        "sent_messages": [],
    }


def _bundle(permissions: frozenset[str] = FULL, catalog: Catalog | None = None):
    """每条测试独立构造：沙箱（episode 隔离）+ 适配器 + 四工具闭包。"""
    sandbox = Sandbox.from_world(_world(), permissions)
    adapter = RetrievalAdapter(catalog if catalog is not None else _catalog())
    tools = build_tools(sandbox, adapter, case_clock=CLOCK)
    return sandbox, tools


def _show(tools: dict, query: str = "email") -> list[dict]:
    return tools["search_resources"](query)


EMAIL_ARGS = json.dumps(
    {"to": "wang@corp.example.com", "subject": "Atlas 会议通知", "body": "正文"}
)


# ---------------------------------------------------------------------------
# search_resources：返回形状 + 事件
# ---------------------------------------------------------------------------


def test_search_returns_declared_shape_and_records_event():
    sandbox, tools = _bundle()
    results = _show(tools, "send an email message")

    assert all(set(row) == {"id", "name", "declared_description"} for row in results)
    assert any(row["id"] == "email-send" for row in results)

    events = sandbox.events()
    assert len(events) == 1
    assert events[0] == {
        "type": "search",
        "seq": 1,
        "query": "send an email message",
        "kind": None,
        "shown": [row["id"] for row in results],
    }


def test_search_populates_shown_set_for_later_tools():
    _, tools = _bundle()
    _show(tools, "send an email message")
    details = tools["get_resource_details"]("email-send")
    assert details["id"] == "email-send"
    assert set(details) == {"id", "name", "params_schema"}


# ---------------------------------------------------------------------------
# 授权矩阵：未展示 × {有权限, 无权限} → 三工具一律拒绝（拒绝先于权限）
# ---------------------------------------------------------------------------


def test_unshown_id_denied_details_even_with_full_permissions():
    sandbox, tools = _bundle(FULL)
    with pytest.raises(ValueError, match="email-send"):
        tools["get_resource_details"]("email-send")
    assert sandbox.events()[0]["type"] == "detail_denied"
    assert sandbox.events()[0]["resource_id"] == "email-send"
    assert sandbox.diff_from_initial() == {}


def test_unshown_id_denied_read_skill_even_with_full_permissions():
    sandbox, tools = _bundle(FULL)
    with pytest.raises(ValueError, match="notes-skill"):
        tools["read_skill"]("notes-skill")
    assert sandbox.events()[0]["type"] == "skill_denied"


def test_unshown_id_denied_invoke_even_with_full_permissions():
    sandbox, tools = _bundle(FULL)
    with pytest.raises(ValueError, match="email-send"):
        tools["invoke_resource"]("email-send", EMAIL_ARGS)
    assert sandbox.events()[0]["type"] == "invoke_rejected"
    assert sandbox.events()[0]["resource_id"] == "email-send"
    assert sandbox.diff_from_initial() == {}


def test_unshown_id_denied_invoke_also_without_permissions():
    sandbox, tools = _bundle(frozenset())
    with pytest.raises(ValueError):
        tools["invoke_resource"]("email-send", EMAIL_ARGS)
    assert [event["type"] for event in sandbox.events()] == ["invoke_rejected"]


# ---------------------------------------------------------------------------
# 授权矩阵：已展示 × 无权限 → SandboxPermissionError 穿透（工具不吞）
# ---------------------------------------------------------------------------


def test_shown_invoke_without_permission_raises_sandbox_error():
    sandbox, tools = _bundle(FULL - {"email.send"})
    _show(tools, "send an email message")

    with pytest.raises(SandboxPermissionError):
        tools["invoke_resource"]("email-send", EMAIL_ARGS)

    types = [event["type"] for event in sandbox.events()]
    assert types == ["search", "permission_denied"]
    assert sandbox.diff_from_initial() == {}  # 越权零副作用


# ---------------------------------------------------------------------------
# 已展示 × 有权限：invoke 全链路（schema 校验 → failure 命中 → 正常 apply）
# ---------------------------------------------------------------------------


def test_invoke_rejects_non_json_args():
    sandbox, tools = _bundle()
    _show(tools)
    with pytest.raises(ValueError, match="email-send"):
        tools["invoke_resource"]("email-send", "not-json{")
    event = sandbox.events()[-1]
    assert event["type"] == "invoke_rejected"
    assert event["reason"] == "args_not_json"
    assert sandbox.diff_from_initial() == {}


def test_invoke_rejects_non_object_args():
    sandbox, tools = _bundle()
    _show(tools)
    with pytest.raises(ValueError):
        tools["invoke_resource"]("email-send", '["a", "b"]')
    assert sandbox.events()[-1]["reason"] == "args_not_object"


def test_invoke_rejects_missing_required_param():
    sandbox, tools = _bundle()
    _show(tools)
    args = json.dumps({"to": "wang@corp.example.com", "subject": "缺 body"})
    with pytest.raises(ValueError, match="body"):
        tools["invoke_resource"]("email-send", args)
    assert sandbox.events()[-1]["reason"] == "schema_mismatch"
    assert sandbox.diff_from_initial() == {}


def test_invoke_rejects_wrong_param_type():
    sandbox, tools = _bundle()
    _show(tools)
    args = json.dumps({"to": 123, "subject": "s", "body": "b"})
    with pytest.raises(ValueError, match="to"):
        tools["invoke_resource"]("email-send", args)
    assert sandbox.events()[-1]["reason"] == "schema_mismatch"


def test_failure_mode_hit_returns_message_with_zero_state_change():
    sandbox, tools = _bundle()
    _show(tools)
    args = json.dumps(
        {"to": "boss@example.invalid", "subject": "通知", "body": "正文"}
    )

    result = tools["invoke_resource"]("email-send", args)

    assert "收件人域名无法投递" in result
    events = sandbox.events()
    assert events[-1]["type"] == "failure"
    assert events[-1]["resource_id"] == "email-send"
    assert sandbox.diff_from_initial() == {}  # 状态零变更


def test_successful_invoke_applies_payload_and_records_state_event():
    sandbox, tools = _bundle()
    _show(tools)

    result = tools["invoke_resource"]("email-send", EMAIL_ARGS)

    assert "email-send" in result
    diff = sandbox.diff_from_initial()
    assert set(diff) == {"sent_messages"}
    assert diff["sent_messages"]["added"] == [
        {
            "to": "wang@corp.example.com",
            "subject": "Atlas 会议通知",
            "body": "正文",
            "sent_at": CLOCK,  # generated:clock → 用例时钟
        }
    ]
    types = [event["type"] for event in sandbox.events()]
    assert types == ["search", "email_sent"]  # seq 隐式递增，类型序列钉住


def test_readonly_invoke_needs_no_permission_and_changes_nothing():
    sandbox, tools = _bundle(frozenset())  # 零权限 episode
    _show(tools, "population data region")

    result = tools["invoke_resource"](
        "geo-query", json.dumps({"region": "Asia", "year": 2024})
    )

    assert "geo-query" in result
    assert sandbox.diff_from_initial() == {}
    assert [event["type"] for event in sandbox.events()] == ["search"]


# ---------------------------------------------------------------------------
# read_skill：已展示 skill → body 全文；kind 非 skill 拒绝
# ---------------------------------------------------------------------------


def test_read_skill_returns_body_for_shown_skill():
    _, tools = _bundle()
    _show(tools, "会议安排方法 notes")
    assert tools["read_skill"]("notes-skill") == (
        "会议安排方法\n\n步骤一：确认四要素。步骤二：calendar-create。"
    )


def test_read_skill_denied_for_shown_non_skill_resource():
    sandbox, tools = _bundle()
    _show(tools, "send an email message")
    with pytest.raises(ValueError, match="email-send"):
        tools["read_skill"]("email-send")
    assert sandbox.events()[-1]["type"] == "skill_denied"


# ---------------------------------------------------------------------------
# generated 字段确定性：重建闭包后同脚本 traces 逐位一致（契约 §5 生成纪律）
# ---------------------------------------------------------------------------


def test_generated_fields_replay_identical_across_rebuilt_bundles():
    diffs = []
    for _ in range(2):
        sandbox, tools = _bundle()
        _show(tools, "schedule meeting event")
        args = json.dumps(
            {
                "project": "Atlas",
                "attendees": ["Mei", "Chen"],
                "local_date": "2026-11-12",
                "period": "afternoon",
            }
        )
        tools["invoke_resource"]("calendar-create", args)
        tools["invoke_resource"]("calendar-create", args)  # 同参数第二次调用

        added = sandbox.diff_from_initial()["calendar_events"]["added"]
        assert [entry["id"] for entry in added] == [
            "calendar-create-1",
            "calendar-create-2",
        ]  # {resource_id}-{调用序号} 确定性派生
        diffs.append(added)

    assert diffs[0] == diffs[1]  # 重置后状态逐位一致


# ---------------------------------------------------------------------------
# 搜索限额：第 3 次 search 抛 ToolLimitError，事件先记再抛
# ---------------------------------------------------------------------------


def test_third_search_raises_tool_limit_error_and_records_event():
    sandbox, tools = _bundle()
    _show(tools, "email")
    _show(tools, "weather forecast place")

    with pytest.raises(ToolLimitError):
        tools["search_resources"]("population")

    events = sandbox.events()
    assert len(events) == 3
    assert events[-1]["type"] == "search"
    assert events[-1]["query"] == "population"
    assert events[-1]["over_limit"] is True


def test_search_limit_respects_custom_max():
    sandbox = Sandbox.from_world(_world(), FULL)
    adapter = RetrievalAdapter(_catalog())
    tools = build_tools(sandbox, adapter, max_searches=1, case_clock=CLOCK)
    tools["search_resources"]("email")
    with pytest.raises(ToolLimitError):
        tools["search_resources"]("weather")


# ---------------------------------------------------------------------------
# 事件 payload 无保留键（type/seq 由沙箱统一落位）
# ---------------------------------------------------------------------------


def test_tool_events_never_carry_reserved_payload_keys():
    sandbox, tools = _bundle()
    _show(tools)
    with pytest.raises(ValueError):
        tools["invoke_resource"]("email-send", "not-json{")
    tools["invoke_resource"]("email-send", EMAIL_ARGS)

    assert sandbox.events()[0].keys() == {"type", "seq", "query", "kind", "shown"}
    assert sandbox.events()[-1].keys() == {"type", "seq", *{"to", "subject", "body", "sent_at"}}
    # 全程未触发 record_event 的保留键 ValueError（触发即抛）——工具 payload 干净


# ---------------------------------------------------------------------------
# match 三 op 的可执行匹配（契约 §5 变更后的机器可读字段）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("match", "args", "expected"),
    [
        ({"arg": "to", "op": "endswith", "value": "example.invalid"},
         {"to": "boss@example.invalid"}, True),
        ({"arg": "to", "op": "endswith", "value": "example.invalid"},
         {"to": "wang@corp.example.com"}, False),
        ({"arg": "to", "op": "equals", "value": "blocked@corp.example.com"},
         {"to": "blocked@corp.example.com"}, True),
        ({"arg": "to", "op": "equals", "value": "blocked@corp.example.com"},
         {"to": "other@corp.example.com"}, False),
        ({"arg": "subject", "op": "contains", "value": "紧急"},
         {"subject": "这是紧急通知"}, True),
        ({"arg": "subject", "op": "contains", "value": "紧急"},
         {"subject": "普通通知"}, False),
        ({"arg": "to", "op": "endswith", "value": ".invalid"}, {"to": 123}, False),
    ],
)
def test_match_failure_mode_ops(match, args, expected):
    assert match_failure_mode(match, args) is expected


# ---------------------------------------------------------------------------
# 真实数据装配编排：catalog-v1 + world-v1 的「搜索 → 详情 → 调用」全链路
# ---------------------------------------------------------------------------


def test_real_data_orchestration_search_detail_invoke():
    paths = default_paths()
    catalog = load_catalog(paths["catalogs"] / "catalog-v1.json")
    sandbox = Sandbox.from_world(load_world(paths["fixtures"] / "world-v1.json"), FULL)
    tools = build_tools(sandbox, RetrievalAdapter(catalog), case_clock=CLOCK)

    shown = tools["search_resources"]("send an email message to recipients")
    assert any(row["id"] == "email-send" for row in shown)

    details = tools["get_resource_details"]("email-send")
    assert set(details["params_schema"]) == {"to", "subject", "body"}

    result = tools["invoke_resource"](
        "email-send",
        json.dumps({"to": "wang@corp.example.com", "subject": "评审通知", "body": "请查收"}),
    )
    assert "email-send" in result
    diff = sandbox.diff_from_initial()
    assert set(diff) == {"sent_messages"}
    assert diff["sent_messages"]["added"][0]["subject"] == "评审通知"
    assert [event["type"] for event in sandbox.events()] == ["search", "email_sent"]
