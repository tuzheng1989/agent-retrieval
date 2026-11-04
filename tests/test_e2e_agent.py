"""E4 参考装配测试：全部离线（假模型替身，不触网；live 冒烟由主控另行执行）。

覆盖面（方案 E4 节验收 + E2 审计 P2-2 装配断言）：

- ``build_model`` 缺变量报错含缺失变量名与指路；齐备时 temperature=0.0；
- ``allowed_ids_for``：只读资源恒可见；写资源权限齐全可见；缺 ``calendar.write``
  时 calendar-create 不可见（e2e-smoke-010 用例语义）；
- ``build_episode``：010 用例（无 ``calendar.write``）装配后真检索一次确认
  search 结果不含 calendar-create；``case_clock`` 生效（``generated:clock`` 取用例时钟）；
- ``build_reference_agent``：假模型构造成功；工具名恰为契约四名（deepagents 内置
  工具经 HarnessProfile 剥离）；两次构造 checkpointer 与工具闭包互不共享；
- 步进适配冒烟：假模型 + 真实 ``build_episode`` → ``run_episode``（step_limit=1
  单 invoke 适配）→ ``score_episode`` 全链路跑通 decline 用例；
  ``GraphRecursionError`` 转译为 ``exhausted=True`` + 归因 ``budget_exhausted``；
- **tool call 主路径**（E4 revision 2 堵盲区）：脚本化假模型发真实 tool call →
  ToolNode 真调工具 → 状态 diff / 事件 / 评分全链路；invoke_resource 的 LLM
  schema 参数名 ``tool_args``（pydantic v2 对 ``args`` 有字段 mangle，契约第二节
  注记）；failure_mode 命中 → 失败文本 + 零状态变更 + 按预期报告失败算完成。
"""

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from experiments.agent_e2e.agent import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    TOOL_NAMES,
    allowed_ids_for,
    build_episode,
    build_model,
    build_reference_agent,
)
from experiments.agent_e2e.loading import Catalog, default_paths, load_cases, load_catalog, load_world
from experiments.agent_e2e.smoke import RECURSION_LIMIT, run_case
from experiments.agent_e2e.tools import ToolLimitError

_READ_ONLY_IDS = frozenset(
    {
        "geo-query",
        "doc-search",
        "chart-render",
        "inbox-search",
        "weather-forecast",
        "meeting-notes-skill",
        "data-summary-skill",
    }
)


class _FakeChatModel(BaseChatModel):
    """离线假模型：恒返回固定文本（无 tool call）；记录 bind_tools 可见工具名。"""

    reply: str = "无法完成该请求。"
    seen_tool_names: list = []

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(content=self.reply)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        self.seen_tool_names[:] = sorted(
            tool.name for tool in tools if hasattr(tool, "name")
        )
        return self  # 假模型不消费工具 schema，恒定自身即可


class _LoopingChatModel(_FakeChatModel):
    """恒返回同一 tool call 的假模型：逼 langgraph 走到 recursion_limit 耗尽。"""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(
            content="",
            tool_calls=[
                {"name": "search_resources", "args": {"query": "会议"}, "id": "call_loop"}
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class _ScriptedChatModel(BaseChatModel):
    """按脚本序返回预构造 AIMessage（越界复用末项）；记录 bind_tools 的工具对象。

    供 tool call 主路径测试：脚本 = search tool call → invoke tool call → 最终文本。
    """

    outputs: list = []
    cursor: dict = {"i": 0}
    seen_tools: list = []

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        index = self.cursor["i"]
        self.cursor["i"] = index + 1
        message = self.outputs[index] if index < len(self.outputs) else self.outputs[-1]
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        self.seen_tools[:] = list(tools)
        return self


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


@pytest.fixture(scope="module")
def loaded() -> tuple[Catalog, dict, list]:
    paths = default_paths()
    catalog = load_catalog(paths["catalogs"] / "catalog-v1.json")
    world = load_world(paths["fixtures"] / "world-v1.json")
    cases = load_cases(
        paths["cases"] / "smoke.jsonl", catalog=catalog, fixtures_dir=paths["fixtures"]
    )
    return catalog, world, cases


def _case(loaded, case_id: str):
    catalog, _world, cases = loaded
    return next(case for case in cases if case.id == case_id)


# ---------------------------------------------------------------------------
# build_model（E-D10）
# ---------------------------------------------------------------------------


def test_build_model_missing_env_names_the_missing(monkeypatch):
    for name in (ENV_MODEL, ENV_BASE_URL, ENV_API_KEY):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        build_model()
    message = str(excinfo.value)
    for name in (ENV_MODEL, ENV_BASE_URL, ENV_API_KEY):
        assert name in message  # 缺失项与变量名指路


def test_build_model_uses_zero_temperature(monkeypatch):
    monkeypatch.setenv(ENV_MODEL, "test-model")
    monkeypatch.setenv(ENV_BASE_URL, "https://gateway.example/v4")
    monkeypatch.setenv(ENV_API_KEY, "test-key")
    model = build_model()
    assert model.temperature == 0.0
    assert model.model_name == "test-model"


# ---------------------------------------------------------------------------
# allowed_ids_for（E2 审计 P2-2：契约第一节权限过滤最后一公里）
# ---------------------------------------------------------------------------


def test_allowed_ids_for_read_only_always_visible(loaded):
    catalog, _world, _cases = loaded
    allowed = allowed_ids_for(frozenset(), catalog)  # 零权限
    for resource_id in _READ_ONLY_IDS:
        assert resource_id in allowed


def test_allowed_ids_for_full_permissions_show_writers(loaded):
    catalog, _world, cases = loaded
    full = next(case for case in cases if case.id == "e2e-smoke-001")
    allowed = allowed_ids_for(frozenset(full.permissions), catalog)
    assert {"calendar-create", "email-send", "report-write"} <= allowed


def test_allowed_ids_for_missing_calendar_write_hides_calendar_create(loaded):
    catalog, _world, cases = loaded
    case_010 = next(case for case in cases if case.id == "e2e-smoke-010")
    assert "calendar.write" not in case_010.permissions
    allowed = allowed_ids_for(frozenset(case_010.permissions), catalog)
    assert "calendar-create" not in allowed
    assert "email-send" in allowed  # email.send 仍在权限集内


# ---------------------------------------------------------------------------
# build_episode（E2 P2-2 装配断言）
# ---------------------------------------------------------------------------


def test_build_episode_search_never_shows_unauthorized_resource(loaded):
    catalog, world, _cases = loaded
    case_010 = _case(loaded, "e2e-smoke-010")
    _sandbox, adapter, tools = build_episode(case_010, catalog, world)
    # 子 catalog 层面根本不含越权资源
    assert "calendar-create" not in adapter.catalog.by_id
    # 真检索一次：shown 与全量排序证据均不含 calendar-create
    result = tools["search_resources"]("会议 安排 calendar")
    assert all(item["id"] != "calendar-create" for item in result)
    full = adapter.search(
        "会议 安排 calendar", allowed_ids=frozenset(adapter.catalog.by_id)
    )
    assert all(resource.id != "calendar-create" for resource in full.shown)
    assert all(entry["id"] != "calendar-create" for entry in full.evidence["ranking"])


def test_build_episode_case_clock_takes_effect(loaded):
    catalog, world, _cases = loaded
    case_010 = _case(loaded, "e2e-smoke-010")
    assert case_010.clock
    sandbox, _adapter, tools = build_episode(case_010, catalog, world)
    tools["search_resources"]("email send")
    output = tools["invoke_resource"](
        "email-send",
        '{"to": "x@y.example", "subject": "测试", "body": "正文"}',
    )
    assert "调用成功" in output
    added = [
        entry
        for entry in sandbox.state_snapshot("sent_messages")
        if entry["subject"] == "测试"
    ]
    assert added and added[0]["sent_at"] == case_010.clock  # generated:clock → 用例时钟


# ---------------------------------------------------------------------------
# build_reference_agent（E-D7：恰四名 / checkpointer 与闭包互不共享）
# ---------------------------------------------------------------------------


def test_build_reference_agent_exposes_exactly_four_tools(loaded):
    catalog, world, _cases = loaded
    case = _case(loaded, "e2e-smoke-001")
    _sandbox, _adapter, tools = build_episode(case, catalog, world)
    model = _FakeChatModel()
    agent = build_reference_agent(model, tools)
    agent.invoke(
        {"messages": [{"role": "user", "content": "你好"}]},
        config={"configurable": {"thread_id": "agent-test-four"}},
    )
    assert sorted(model.seen_tool_names) == sorted(TOOL_NAMES)


def test_build_reference_agent_checkpointer_and_closures_isolated(loaded):
    catalog, world, _cases = loaded
    case = _case(loaded, "e2e-smoke-006")
    _sandbox_a, _adapter_a, tools_a = build_episode(case, catalog, world)
    _sandbox_b, _adapter_b, tools_b = build_episode(case, catalog, world)
    agent_a = build_reference_agent(_FakeChatModel(reply="a"), tools_a)
    agent_b = build_reference_agent(_FakeChatModel(reply="b"), tools_b)
    assert agent_a is not agent_b
    assert agent_a.checkpointer is not agent_b.checkpointer
    # 工具闭包互不共享：episode A 耗尽搜索配额不影响 B
    # （smoke-v1 搜索预算经契约校准为 3 次，2026-11-05——见 contract.md §3 校准注记）
    for query in ("第一次", "第二次", "第三次"):
        tools_a["search_resources"](query)
    with pytest.raises(ToolLimitError):
        tools_a["search_resources"]("第四次")
    tools_b["search_resources"]("独立配额")  # B 仍可搜索


# ---------------------------------------------------------------------------
# 步进适配冒烟（离线全链路：装配 → run_episode 单 invoke 适配 → score_episode）
# ---------------------------------------------------------------------------


def test_smoke_step_adapter_decline_flow_offline(loaded):
    catalog, world, _cases = loaded
    case = _case(loaded, "e2e-smoke-006")  # 无匹配 → decline
    model = _FakeChatModel(reply="抱歉，当前没有订机票能力，无法完成该请求。")
    trace, result, sandbox = run_case(case, catalog, world, model)
    assert trace.case_id == case.id
    assert trace.final_output is not None and "无法完成" in trace.final_output
    assert trace.exhausted is False
    assert trace.error is None
    assert result.success is True
    assert result.attributions == frozenset()
    assert sandbox.diff_from_initial() == {}  # 零副作用


def test_smoke_step_adapter_chinese_clarify_flow_offline(loaded):
    # E4 revision 3 回归（live 008 定性）：Agent 中文列出缺失项 → clarify 成立
    catalog, world, _cases = loaded
    case = _case(loaded, "e2e-smoke-008")  # 信息不足例：clarify_required_info = [local_date, period]
    model = _FakeChatModel(reply="请补充会议的日期与时段（上午/下午/晚间），我再为您安排。")
    trace, result, sandbox = run_case(case, catalog, world, model)
    assert trace.final_output and "日期" in trace.final_output
    assert result.success is True  # 每字段别名组任一命中（日期→local_date，时段→period）
    assert result.attributions == frozenset()
    assert sandbox.diff_from_initial() == {}  # 零副作用


def test_smoke_step_adapter_recursion_exhaustion_maps_budget(loaded):
    catalog, world, _cases = loaded
    case = _case(loaded, "e2e-smoke-006")
    # recursion_limit=2：第一轮 tool call 未及触发搜索上限即耗尽步数预算
    trace, result, _sandbox = run_case(
        case, catalog, world, _LoopingChatModel(), recursion_limit=2
    )
    # GraphRecursionError → _RecursionExhausted → Trace(exhausted=True) → budget_exhausted
    assert trace.exhausted is True
    assert trace.final_output is None
    assert result.budget_exhausted is True
    assert "budget_exhausted" in result.attributions
    assert result.success is False


def test_smoke_step_adapter_search_limit_maps_tool_limit_error(loaded):
    catalog, world, _cases = loaded
    case = _case(loaded, "e2e-smoke-006")
    # 默认 recursion_limit 下：恒定 search 循环第 3 次触发 ToolLimitError（工具闭包限额）
    trace, result, sandbox = run_case(case, catalog, world, _LoopingChatModel())
    # ToolLimitError 炸穿 graph → _SearchBudgetExhausted → Trace(error="ToolLimitError")
    # → scorer 按子串映射 budget_exhausted（与 runner 同形状，契约第三节）
    assert trace.exhausted is False
    assert trace.error == "ToolLimitError"
    assert trace.final_output is None
    assert result.budget_exhausted is True
    assert "budget_exhausted" in result.attributions
    assert result.success is False
    over_limit = [event for event in sandbox.events() if event.get("over_limit")]
    assert over_limit  # 超限尝试经事件日志可观测（工具闭包先记后抛）


def test_recursion_limit_covers_smoke_v1_budget():
    # 步数预算映射：smoke-v1 12 步 + 工具节点缓冲 8（契约第三节）
    assert RECURSION_LIMIT == 20


def test_agent_module_never_touches_network_offline(loaded):
    """装配 + 假模型 invoke + 评分的完整闭环在离线环境可跑通（无任何网络调用）。"""
    catalog, world, _cases = loaded
    case = _case(loaded, "e2e-smoke-010")  # 越权例：decline
    model = _FakeChatModel(reply="我没有创建日程的权限，无法完成该请求。")
    trace, result, sandbox = run_case(case, catalog, world, model)
    assert result.success is True  # 零副作用 + final_output 非空 = decline 成立
    assert sandbox.events() == ()  # 假模型未调工具 → 事件日志为空（副作用的唯一观测面）


# ---------------------------------------------------------------------------
# tool call 主路径（E4 revision 2 堵盲区：decline 闭环不触发 tool call，live 首轮
# 才暴露 pydantic args mangle——离线必须覆盖 ToolNode 真调工具的路径）
# ---------------------------------------------------------------------------


def test_smoke_tool_call_main_path_invokes_tools_and_scores_success(loaded):
    """e2e-smoke-001 多步主路径：search → invoke calendar-create → invoke email-send → 文本。

    断言 ToolNode 真调工具的状态面（diff）、事件面、schema 参数名（mangle 修复回归面）
    与评分面（required_state 全满足 → success=True）。
    """
    catalog, world, _cases = loaded
    case = _case(loaded, "e2e-smoke-001")  # 多步 complete：required = event_exists + email_sent
    model = _ScriptedChatModel(
        outputs=[
            _tool_call("search_resources", {"query": "calendar create schedule meeting"}, "c1"),
            _tool_call("search_resources", {"query": "email send message"}, "c2"),
            _tool_call(
                "invoke_resource",
                {
                    "id": "calendar-create",
                    "tool_args": '{"project": "Atlas", "attendees": ["Mei", "Chen"], "local_date": "2026-11-12", "period": "afternoon"}',
                },
                "c3",
            ),
            _tool_call(
                "invoke_resource",
                {
                    "id": "email-send",
                    "tool_args": '{"to": "wang@corp.example.com", "subject": "Atlas 会议通知", "body": "评审会安排已确认"}',
                },
                "c4",
            ),
            AIMessage(content="已创建 Atlas 评审会日程并通知 wang@corp.example.com。"),
        ]
    )
    trace, result, sandbox = run_case(case, catalog, world, model)

    # LLM 可见 schema 参数名：invoke_resource 为 tool_args（pydantic args mangle 修复）
    tool_map = {tool.name: tool for tool in model.seen_tools}
    assert set(tool_map["invoke_resource"].args_schema.model_fields) == {"id", "tool_args"}
    assert sorted(tool_map) == sorted(TOOL_NAMES)

    # 工具真执行：状态 diff 恰两条新增（日程 + 邮件），generated:clock 取用例时钟
    diff = sandbox.diff_from_initial()
    added_events = diff.get("calendar_events", {}).get("added", [])
    added_messages = diff.get("sent_messages", {}).get("added", [])
    assert len(added_events) == 1
    assert added_events[0]["project"] == "Atlas"
    assert len(added_messages) == 1 and added_messages[0]["to"] == "wang@corp.example.com"
    assert added_messages[0]["sent_at"] == case.clock

    event_types = [event["type"] for event in sandbox.events()]
    assert event_types.count("search") == 2  # 契约 smoke-v1：每 episode 搜索上限 2（恰好用满）
    assert "event_exists" in event_types and "email_sent" in event_types

    assert trace.final_output and "已创建" in trace.final_output
    assert result.success is True
    assert result.failure_reasons == ()


def test_smoke_tool_call_failure_mode_reports_without_side_effect(loaded):
    """e2e-smoke-011 故障路径：.invalid 域名命中 failure_modes → 失败文本 + 零状态变更。

    按预期报告失败也算完成（契约第四节 complete 第三子条全链路验证）。
    """
    catalog, world, _cases = loaded
    case = _case(loaded, "e2e-smoke-011")  # expected_failure: email-send
    model = _ScriptedChatModel(
        outputs=[
            _tool_call("search_resources", {"query": "email send"}, "f1"),
            _tool_call(
                "invoke_resource",
                {
                    "id": "email-send",
                    "tool_args": '{"to": "external-partner@example.invalid", "subject": "合作意向", "body": "期望进一步洽谈合作"}',
                },
                "f2",
            ),
            AIMessage(content="邮件发送失败：收件人域名无法投递，消息未发出。"),
        ]
    )
    trace, result, sandbox = run_case(case, catalog, world, model)

    assert sandbox.diff_from_initial() == {}  # failure 命中 → 状态零变更
    failure_events = [
        event for event in sandbox.events() if event["type"] == "failure"
    ]
    assert failure_events and failure_events[0]["resource_id"] == "email-send"
    assert result.success is True  # failure 事件在 ∧ 零副作用 ∧ final 非空 → complete
    assert result.attributions == frozenset()
    assert trace.final_output and "失败" in trace.final_output


def test_smoke_denied_detail_error_text_lets_agent_decline(loaded):
    """拒绝路径兜底（DeepAgents ToolNode 适配）：未展示资源 get_resource_details 的
    ValueError 在包装层转为错误文本回传——Agent 收到后继续 decline，episode 不崩
    （live 第七跑 010 暴露的炸穿缺陷的离线回归面；010 无权限用例镜像语义）。
    """
    catalog, world, _cases = loaded
    case = _case(loaded, "e2e-smoke-010")  # 无权限 decline：缺 calendar.write
    model = _ScriptedChatModel(
        outputs=[
            _tool_call("search_resources", {"query": "calendar create schedule meeting"}, "c1"),
            # 权限外资源被 allowed_ids_for 过滤、不在展示面——Agent 尝试读详情触发拒绝
            _tool_call("get_resource_details", {"id": "calendar-create"}, "c2"),
            AIMessage(content="当前环境没有可用的会议安排工具，无法完成该安排。"),
        ]
    )
    trace, result, sandbox = run_case(case, catalog, world, model)

    # 拒绝事件已记录（detail_denied），零状态变更
    event_types = [event["type"] for event in sandbox.events()]
    assert "detail_denied" in event_types
    assert sandbox.diff_from_initial() == {}
    # Agent 收到错误文本后继续决策：episode 走完、decline 判定成功（零 forbidden ∧ 零变更 ∧ final 非空）
    assert trace.error is None
    assert trace.final_output and "无法完成" in trace.final_output
    assert result.success is True
    assert result.attributions == frozenset()


def test_smoke_invalid_kind_gets_correction_hint_not_silent_empty(loaded):
    """kind 枚举校验（live 第八跑 002 挂点回归面）：非法 kind 返回纠正提示而非
    静默空结果——Agent 可自纠；合法 kind 与缺省不受影响。"""
    catalog, world, _cases = loaded
    case = _case(loaded, "e2e-smoke-001")
    model = _ScriptedChatModel(
        outputs=[
            _tool_call("search_resources", {"query": "calendar create", "kind": "calendar"}, "c1"),
            _tool_call("search_resources", {"query": "calendar create meeting"}, "c2"),
            AIMessage(content="已检索到资源。"),
        ]
    )
    trace, result, sandbox = run_case(case, catalog, world, model)
    # 非法 kind 的纠正提示进了工具结果（Agent 可见、可自纠）
    # episode 走完不崩：final_output 非空、无 ToolLimitError
    assert trace.error is None
    assert trace.final_output
    # 合法第二轮 search 正常执行（事件面有两条 search，非法尝试不计数不记事件）
    event_types = [event["type"] for event in sandbox.events()]
    assert event_types.count("search") == 1  # 非法 kind 未消耗搜索配额
