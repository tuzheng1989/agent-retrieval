"""E3 运行器测试：脚本化步进器回放、步数耗尽、ToolLimitError 终止、工具异常续跑 + L2 离线编排。

L2（方案 E3 节验证面）：E0 真实用例（loading 加载 e2e-smoke-001）+ 测试内替身工具
（直接调 ``sandbox.apply``，不经 E2 tools.py——E3 声明面与验证面均不依赖 E2，W3 并行
成立）+ 脚本化步进器 → ``run_episode`` → ``score_episode``。
"""

from experiments.agent_e2e.loading import (
    Case,
    default_paths,
    load_catalog,
    load_cases,
    load_world,
)
from experiments.agent_e2e.runner import Trace, run_episode
from experiments.agent_e2e.sandbox import PERMISSION_BY_STATE_KEY, STATE_KEY_BY_PREDICATE, Sandbox
from experiments.agent_e2e.scorer import score_episode


class ToolLimitError(Exception):
    """镜像 E2 ``tools.ToolLimitError`` 的类名：runner 按异常类名识别（不导入 E2 模块）。"""


# ---------------------------------------------------------------------------
# 测试夹具：最小 world、替身工具、脚本化步进器
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


def _tools(sandbox: Sandbox) -> dict:
    """通用替身工具：search（第 3 次抛 ToolLimitError）+ apply_named（直写沙箱）+ boom。"""
    searches = {"n": 0}

    def search_resources(query, kind=None):
        searches["n"] += 1
        if searches["n"] >= 3:
            raise ToolLimitError("搜索超过每 episode 2 次上限（smoke-v1）")
        sandbox.record_event("search", {"query": query})
        return [{"id": "geo-query", "name": "geo-query", "declared_description": "查询地理数据"}]

    def apply_named(kind, payload):
        _apply(sandbox, kind, payload)
        return f"已执行 {kind}"

    def boom(kind, payload):
        raise ValueError("boom")

    return {"search_resources": search_resources, "apply_named": apply_named, "boom": boom}


def _stepper(decisions: list[dict]):
    """脚本化步进器：按序放行 decisions（越界复用末项）。返回 (agent_step, calls, histories)。"""
    calls = {"n": 0}
    histories: list[list[dict]] = []
    state = {"i": 0}

    def agent_step(tools, history):
        calls["n"] += 1
        histories.append([dict(entry) for entry in history])
        decision = decisions[state["i"]] if state["i"] < len(decisions) else decisions[-1]
        state["i"] += 1
        return decision

    return agent_step, calls, histories


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


# ---------------------------------------------------------------------------
# 回放一致性 / 预算终止 / 异常续跑
# ---------------------------------------------------------------------------


def _run_scripted_success(case: Case) -> tuple[Trace, object, Sandbox]:
    """同 builder 语义跑一遍成功脚本：每 episode 新建沙箱 + 工具闭包。"""
    sandboxes: list[Sandbox] = []

    def tools_builder() -> dict:
        sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
        sandboxes.append(sandbox)
        return _tools(sandbox)

    decisions = [
        {"tool": "search_resources", "args": {"query": "会议 邮件"}},
        {"tool": "apply_named", "args": {"kind": "event_exists", "payload": EVENT_APPLY}},
        {"tool": "apply_named", "args": {"kind": "email_sent", "payload": EMAIL_APPLY}},
        {"tool": "done", "output": "会议已安排，邮件已发送。", "final": True},
    ]
    agent_step, _, _ = _stepper(decisions)
    trace = run_episode(case, tools_builder, agent_step)
    return trace, score_episode(case, trace, sandboxes[0]), sandboxes[0]


def test_scripted_success_replay_is_bit_identical():
    # 同一 case + 同一脚本步进器连跑两次 → Trace 与 EpisodeResult 逐位一致（可重放证据）
    case = _case(required_state=(EVENT_PREDICATE, EMAIL_PREDICATE))
    trace1, result1, _ = _run_scripted_success(case)
    trace2, result2, _ = _run_scripted_success(case)
    assert trace1 == trace2
    assert result1 == result2
    assert result1.success is True
    assert result1.budget_exhausted is False
    assert trace1.final_output == "会议已安排，邮件已发送。"
    assert trace1.exhausted is False
    assert trace1.error is None
    assert len(trace1.steps) == 3


def test_step_exhaustion_maps_budget_exhausted():
    case = _case()
    sandboxes: list[Sandbox] = []

    def tools_builder() -> dict:
        sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
        sandboxes.append(sandbox)
        return _tools(sandbox)

    agent_step, calls, _ = _stepper(
        [{"tool": "apply_named", "args": {"kind": "event_exists", "payload": EVENT_APPLY}}]
    )
    trace = run_episode(case, tools_builder, agent_step, step_limit=3)
    assert trace.exhausted is True
    assert trace.final_output is None
    assert calls["n"] == 3  # 步数耗尽后不再调用步进器
    assert len(trace.steps) == 3
    result = score_episode(case, trace, sandboxes[0])
    assert result.budget_exhausted is True
    assert result.attributions == frozenset({"budget_exhausted"})
    assert result.earliest_blocker is None
    assert result.success is False


def test_tool_limit_error_terminates_episode():
    case = _case()
    sandboxes: list[Sandbox] = []

    def tools_builder() -> dict:
        sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
        sandboxes.append(sandbox)
        return _tools(sandbox)

    decisions = [
        {"tool": "search_resources", "args": {"query": "a"}},
        {"tool": "search_resources", "args": {"query": "b"}},
        {"tool": "search_resources", "args": {"query": "c"}},  # 第 3 次调 search 触发
        {"tool": "done", "output": "不该到达", "final": True},
    ]
    agent_step, calls, _ = _stepper(decisions)
    trace = run_episode(case, tools_builder, agent_step)
    assert trace.error == "ToolLimitError"
    assert trace.exhausted is False
    assert calls["n"] == 3  # 终止后不再调用步进器
    assert trace.steps[-1]["error"] == "ToolLimitError"
    result = score_episode(case, trace, sandboxes[0])
    assert result.budget_exhausted is True
    assert result.success is False


def test_final_step_terminates_early_without_tool_execution():
    case = _case()
    sandboxes: list[Sandbox] = []

    def tools_builder() -> dict:
        sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
        sandboxes.append(sandbox)
        return _tools(sandbox)

    decisions = [
        {"tool": "search_resources", "args": {"query": "人口"}},
        {"tool": "done", "output": "加州人口约 3900 万。", "final": True},
    ]
    agent_step, calls, _ = _stepper(decisions)
    trace = run_episode(case, tools_builder, agent_step)
    assert calls["n"] == 2
    assert len(trace.steps) == 1  # final 步不执行工具、不进 steps
    assert trace.final_output == "加州人口约 3900 万。"
    assert trace.exhausted is False
    assert trace.error is None
    # 只读单步口径：required 空 + final_output 非空即成
    assert score_episode(case, trace, sandboxes[0]).success is True


def test_tool_exception_is_recorded_and_episode_continues():
    case = _case()
    sandboxes: list[Sandbox] = []

    def tools_builder() -> dict:
        sandbox = Sandbox.from_world(_world(), FULL_PERMISSIONS)
        sandboxes.append(sandbox)
        return _tools(sandbox)

    decisions = [
        {"tool": "boom", "args": {"kind": "x", "payload": {}}},
        {"tool": "apply_named", "args": {"kind": "email_sent", "payload": EMAIL_APPLY}},
        {"tool": "done", "output": "重试后完成。", "final": True},
    ]
    agent_step, _, histories = _stepper(decisions)
    trace = run_episode(case, tools_builder, agent_step)
    assert trace.error is None  # 普通工具异常不终止 episode
    assert trace.steps[0]["error"].startswith("ValueError")
    assert trace.steps[1]["output"] == "已执行 email_sent"
    assert trace.final_output == "重试后完成。"
    assert histories[1][0]["error"].startswith("ValueError")  # 异常步进入 history 供下一步决策
    assert score_episode(case, trace, sandboxes[0]).success is True


# ---------------------------------------------------------------------------
# L2 离线编排：E0 真实用例 + 替身工具（直接 sandbox.apply，不经 E2）+ 脚本步进器
# ---------------------------------------------------------------------------


def _smoke001_tools(sandbox: Sandbox) -> dict:
    """e2e-smoke-001 专用替身工具：search + invoke_resource（直写沙箱状态）。"""

    def search_resources(query, kind=None):
        sandbox.record_event("search", {"query": query})
        return [
            {"id": "calendar-create", "name": "日程创建", "declared_description": "创建会议日程"},
            {"id": "email-send", "name": "邮件发送", "declared_description": "发送邮件"},
        ]

    def invoke_resource(id, **fields):
        if id == "calendar-create":
            sandbox.check_permission("calendar.write")
            sandbox.apply(
                "event_exists",
                {
                    "project": fields["project"],
                    "attendees": list(fields["attendees"]),
                    "local_date": fields["local_date"],
                    "period": fields["period"],
                },
            )
            return "日程已创建"
        if id == "email-send":
            sandbox.check_permission("email.send")
            sandbox.apply(
                "email_sent",
                {"to": fields["to"], "subject": fields["subject"], "body": fields.get("body", "")},
            )
            return "邮件已发送"
        raise ValueError(f"未知资源: {id}")

    return {"search_resources": search_resources, "invoke_resource": invoke_resource}


def _load_smoke_case(case_id: str) -> tuple[Case, dict]:
    paths = default_paths()
    catalog = load_catalog(paths["catalogs"] / "catalog-v1.json")
    cases = load_cases(paths["cases"] / "smoke.jsonl", catalog=catalog, fixtures_dir=paths["fixtures"])
    case = next(item for item in cases if item.id == case_id)
    world = load_world(paths["fixtures"] / f"{case.initial_state_fixture}.json")
    return case, world


def _run_smoke001(decisions: list[dict]) -> tuple[object, Trace, Sandbox]:
    case, world = _load_smoke_case("e2e-smoke-001")
    sandboxes: list[Sandbox] = []

    def tools_builder() -> dict:
        sandbox = Sandbox.from_world(world, frozenset(case.permissions))
        sandboxes.append(sandbox)
        return _smoke001_tools(sandbox)

    agent_step, _, _ = _stepper(decisions)
    trace = run_episode(case, tools_builder, agent_step)
    return score_episode(case, trace, sandboxes[0]), trace, sandboxes[0]


def test_l2_smoke001_success_path():
    result, trace, sandbox = _run_smoke001(
        [
            {"tool": "search_resources", "args": {"query": "Atlas 评审会"}},
            {
                "tool": "invoke_resource",
                "args": {
                    "id": "calendar-create",
                    "project": "Atlas",
                    "attendees": ["Mei", "Chen"],
                    "local_date": "2026-11-12",
                    "period": "afternoon",
                },
            },
            {
                "tool": "invoke_resource",
                "args": {
                    "id": "email-send",
                    "to": "wang@corp.example.com",
                    "subject": "Atlas 会议通知",
                    "body": "评审会安排已确认",
                },
            },
            {"tool": "done", "output": "已安排 Atlas 评审会并邮件通知 wang@corp.example.com。", "final": True},
        ]
    )
    assert result.success is True
    assert result.attributions == frozenset()
    assert result.earliest_blocker is None
    assert result.budget_exhausted is False
    # 副作用面：恰两条新增（日程 + 邮件），事件日志可观测
    diff = sandbox.diff_from_initial()
    assert len(diff["calendar_events"]["added"]) == 1
    assert len(diff["sent_messages"]["added"]) == 1
    assert trace.case_id == "e2e-smoke-001"
    assert len(trace.steps) == 3


def test_l2_smoke001_talk_only_negative():
    # 反例脚本：只说完成不执行（final_output 完成文案 + 零 diff）→ 败，归因 state_mismatch
    result, _, sandbox = _run_smoke001(
        [
            {
                "tool": "done",
                "output": "好的，会议已安排，邮件已发送，任务完成！",
                "final": True,
            }
        ]
    )
    assert result.success is False
    assert result.attributions == frozenset({"state_mismatch"})
    assert result.failure_reasons == (
        "required_state 未满足: event_exists",
        "required_state 未满足: email_sent",
    )
    assert result.earliest_blocker is None
    assert sandbox.diff_from_initial() == {}
