"""e2e 参考装配（E4）：DeepAgents 参考实现 + 单 episode 装配（方案 E4 节，E-D7/E-D10）。

- :func:`build_model` 读 ``AGENT_RETRIEVAL_E2E_{MODEL,BASE_URL,API_KEY}``（缺任一
  ``RuntimeError`` 列出缺失项与变量名指路——E-D10 无回落链）；``temperature=0.0``；
- :data:`AGENT_SYSTEM_PROMPT` 冻结冒烟系统提示词（职责：可用工具说明、先搜索后调用、
  无匹配明确拒绝、信息不足澄清、不虚构执行结果；不写任务专属内容）；
- :func:`allowed_ids_for` 把用例权限翻译成 search 可见资源集（契约第一节「search 先按
  用例权限过滤」的装配层落地，E2 审计 P2-2）；
- :func:`build_episode` 单 episode 装配：权限子 catalog → 沙箱 → 检索适配器 → 四工具
  闭包（``case_clock`` 必传——``generated:clock`` 确定性取用例时钟，契约第五节）；
- :func:`build_reference_agent` DeepAgents 参考装配：四工具经
  ``langchain_core.tools.tool`` 包装（名字/docstring/参数 schema 继承闭包——工具描述
  冻结面），checkpointer 与工具闭包每 episode 新建（E-D7）。

**deepagents==0.7.32 实源核对（2026-11-05）**：``tools=`` 参数是叠加而非替换——默认
注入 ``ls`` / ``read_file`` / ``write_file`` / ``edit_file`` / ``glob`` / ``grep`` /
``task`` / ``delete`` 八个内置工具（``FilesystemMiddleware`` + 默认 general-purpose
subagent），``middleware=()`` 不改变这一点。故 :func:`build_reference_agent` 内部经
``register_harness_profile``（deepagents beta API）注册排除 profile：内置工具全部从
模型可见集剥离、general-purpose subagent 停用——模型可见工具恰为契约四名（E-D7
「未用于真实任务的能力不开放」）。注册键 = 模型 provider（``ChatOpenAI`` 实例解析为
``"openai"``；测试假模型解析为类名小写），每次构造幂等覆盖。
"""

import os
from collections.abc import Callable
from typing import Any

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents._models import get_model_provider
from deepagents.backends import StateBackend
from langchain_core.tools import tool as langchain_tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from experiments.agent_e2e.loading import Catalog, Case
from experiments.agent_e2e.retrieval_adapter import RetrievalAdapter
from experiments.agent_e2e.sandbox import PERMISSION_BY_STATE_KEY, Sandbox
from experiments.agent_e2e.tools import build_tools

# E-D10：e2e 运行时模型配置独立命名，无回落链
ENV_MODEL = "AGENT_RETRIEVAL_E2E_MODEL"
ENV_BASE_URL = "AGENT_RETRIEVAL_E2E_BASE_URL"
ENV_API_KEY = "AGENT_RETRIEVAL_E2E_API_KEY"
_REQUIRED_ENV_VARS = (ENV_MODEL, ENV_BASE_URL, ENV_API_KEY)

# 契约四工具名（tools.build_tools 产出的闭包键）
TOOL_NAMES: tuple[str, ...] = (
    "search_resources",
    "get_resource_details",
    "read_skill",
    "invoke_resource",
)

# deepagents==0.7.32 默认注入的内置工具全集（实源核对 2026-11-05：StateBackend 非
# sandbox 后端故 execute 不注入，仍预防性列入——excluded_tools 按名过滤，多列无害）
_DEEPAGENTS_BUILTIN_TOOLS: frozenset[str] = frozenset(
    {"ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute", "task", "delete"}
)

AGENT_SYSTEM_PROMPT = (
    "你是一个通过工具完成用户请求的助手。\n"
    "\n"
    "可用工具恰四个：\n"
    "- search_resources：按关键词检索可用资源（工具与技能），返回前 5 个候选；"
    "参数 kind 只能填 \"tool\"、\"skill\" 或不传——填其他值会被拒绝；\n"
    "- get_resource_details：查看已展示资源的参数规格（params_schema）；\n"
    "- read_skill：读取已展示技能资源的方法正文；\n"
    "- invoke_resource：执行已展示的资源动作（args 为 JSON 字符串，按目标资源的"
    "参数规格填写）。\n"
    "\n"
    "行为准则：\n"
    "1. 先搜索后调用：只能使用 search_resources 返回过的资源 id；调用前先用 "
    "get_resource_details 确认参数规格；技能资源先经 read_skill 读取方法再操作。\n"
    "2. 检索无匹配时，明确告知用户无法完成该请求，不得借助其他途径拼凑结果。\n"
    "3. 请求信息不足时，逐项列出缺失的信息向用户澄清，不要猜测默认值。\n"
    "4. 如实汇报执行结果：工具返回失败就报告失败；未实际执行的动作不得声称已完成。"
)


def build_model() -> ChatOpenAI:
    """从 ``AGENT_RETRIEVAL_E2E_{MODEL,BASE_URL,API_KEY}`` 构造模型（E-D10 无回落链）。

    任一变量缺失即 ``RuntimeError``，消息列出缺失项与全部变量名指路。
    """
    missing = [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"缺少 e2e 运行时配置环境变量: {missing}；"
            f"请设置 {', '.join(_REQUIRED_ENV_VARS)} 后重试"
            f"（smoke-v1 运行时配置独立命名，无回落链）"
        )
    return ChatOpenAI(
        model=os.environ[ENV_MODEL],
        base_url=os.environ[ENV_BASE_URL],
        api_key=os.environ[ENV_API_KEY],
        temperature=0.0,
    )


def allowed_ids_for(case_permissions: frozenset[str], catalog: Catalog) -> frozenset[str]:
    """把用例权限翻译成 search 可见资源 id 集（契约第一节权限过滤的装配层落地）。

    资源 ``invoke_effect`` 为空（只读）→ 恒可见；非空 → 其涉及的每个 ``state_key``
    经 ``PERMISSION_BY_STATE_KEY`` 反查的写权限全部在 ``case_permissions`` 中才可见。
    """
    allowed: set[str] = set()
    for resource in catalog.resources:
        state_keys = {effect["state_key"] for effect in resource.invoke_effect}
        if not state_keys:
            allowed.add(resource.id)
            continue
        if all(
            PERMISSION_BY_STATE_KEY[state_key] in case_permissions
            for state_key in state_keys
        ):
            allowed.add(resource.id)
    return frozenset(allowed)


def build_episode(
    case: Case, catalog: Catalog, world: dict
) -> tuple[Sandbox, RetrievalAdapter, dict[str, Callable]]:
    """单 episode 装配：子 catalog（权限过滤）→ 沙箱 → 检索适配器 → 四工具闭包。

    ``build_tools`` 显式传 ``case_clock=case.clock``（``generated:clock`` 字段取值，
    禁默认空串静默）与 ``max_searches=2``（契约第三节 smoke-v1 搜索上限）。
    """
    allowed_ids = allowed_ids_for(frozenset(case.permissions), catalog)
    sub_resources = [resource for resource in catalog.resources if resource.id in allowed_ids]
    sub_catalog = Catalog(
        resources=tuple(sub_resources),
        by_id={resource.id: resource for resource in sub_resources},
    )
    sandbox = Sandbox.from_world(world, frozenset(case.permissions))
    adapter = RetrievalAdapter(sub_catalog)
    tools = build_tools(sandbox, adapter, max_searches=3, case_clock=case.clock)
    return sandbox, adapter, tools


def _restrict_builtin_tools(model: Any) -> None:
    """注册 HarnessProfile：剥离 deepagents 内置工具、停用默认 subagent（E-D7）。

    deepagents==0.7.32 的 ``tools=`` 是叠加语义，内置工具的收敛唯一入口是
    ``excluded_tools``（``_ToolExclusionMiddleware`` 按名过滤全部工具，含 middleware
    注入项）。注册键 = 模型 provider，幂等覆盖；仅影响本进程内该 provider 的装配。
    """
    profile = HarnessProfile(
        excluded_tools=_DEEPAGENTS_BUILTIN_TOOLS,
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    )
    register_harness_profile(get_model_provider(model), profile)


def _wrap_tools(closures: dict[str, Callable]) -> list[Any]:
    """四工具闭包 → langchain tool 列表（名字/docstring/参数 schema 继承闭包）。

    ``invoke_resource`` 例外：pydantic v2 对参数名 ``args`` 有字段 mangle（实测
    ``@tool`` 装饰后 schema 字段变 ``v__args``，LLM 按 schema 发参 ``v__args`` →
    闭包 ``args`` 不识别 → TypeError）。暴露给 LLM 的 schema 参数名改 ``tool_args``，
    内部转发原闭包（闭包签名不动——tools.py 是 E2 收口文件零改动；契约第二节
    tool_args 注记见 contract.md）。

    拒绝兜底（DeepAgents ToolNode 适配）：langgraph ``ToolNode`` 默认只把
    ``ToolInvocationError`` 转错误消息、其余异常 re-raise（实源
    ``_default_handle_tool_errors``）——闭包的拒绝 ``ValueError``（未展示/
    schema 不符，事件已在闭包内先记）会炸穿 graph 使 episode 崩溃。包装层就地
    兜住 ``ValueError`` 返回错误文本，Agent 收到后继续决策（无权限用例的
    decline 路径可达）；``ToolLimitError``/``SandboxPermissionError`` 是
    RuntimeError 子类不受影响，保持抛出（终止 episode 的转译通道契约）。
    """
    def _search_resources(query: str, kind: str | None = None) -> str:
        # kind 枚举前置校验（纠正性交互：不计数、不记事件、返回提示让 Agent 自纠）——
        # 非法 kind 在闭包内会静默过滤成空结果（live 第八跑 002 的挂点），Agent 无从自纠。
        if kind is not None and kind not in ("tool", "skill"):
            return (
                f"调用被拒绝：kind={kind!r} 非法——kind 只能是 \"tool\"、\"skill\" 或不传"
                "（不传时在全部资源中检索）。请修正后重试。"
            )
        try:
            return closures["search_resources"](query, kind)
        except ValueError as exc:
            return f"调用被拒绝：{exc}"

    def _get_resource_details(id: str) -> str:
        try:
            return closures["get_resource_details"](id)
        except ValueError as exc:
            return f"调用被拒绝：{exc}"

    def _read_skill(id: str) -> str:
        try:
            return closures["read_skill"](id)
        except ValueError as exc:
            return f"调用被拒绝：{exc}"

    def _invoke_resource(id: str, tool_args: str) -> str:
        try:
            return closures["invoke_resource"](id, tool_args)
        except ValueError as exc:
            return f"调用被拒绝：{exc}"

    # 具名签名是 langchain schema 推导的来源（泛型 *args/**kwargs 会让 schema 退化
    # 成 args/kwargs——签名照契约第二节）；工具描述冻结面与 E2 闭包 docstring 同源。
    specs = {
        "search_resources": _search_resources,
        "get_resource_details": _get_resource_details,
        "read_skill": _read_skill,
        "invoke_resource": _invoke_resource,
    }
    wrapped = []
    for name, fn in specs.items():
        fn.__name__ = name
        fn.__doc__ = closures[name].__doc__
        wrapped.append(langchain_tool(fn))
    return wrapped


def build_reference_agent(
    model: Any, tools: dict[str, Callable]
) -> CompiledStateGraph:
    """DeepAgents 参考装配（E-D7）：四工具 + 冻结系统提示词 + 每 episode 新建 checkpointer。

    ``tools`` 为 :func:`build_tools` 产出的闭包字典（每 episode 新建——搜索计数、
    已展示集合、generated 序号都在闭包内）；``checkpointer`` 在本函数内新建
    ``InMemorySaver``，两次构造互不共享。
    """
    _restrict_builtin_tools(model)
    return create_deep_agent(
        model=model,
        tools=_wrap_tools(tools),
        system_prompt=AGENT_SYSTEM_PROMPT,
        backend=StateBackend(),
        checkpointer=InMemorySaver(),
    )
