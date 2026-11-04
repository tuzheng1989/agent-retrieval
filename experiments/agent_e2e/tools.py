"""e2e 受控四工具（E2）：契约第二节冻结签名，工具闭包持有 episode 内状态。

- ``build_tools`` 每 episode 调一次，产出四工具闭包：已展示集合、搜索计数、generated
  调用序号都由闭包持有（E-D5/D9）；权限单一持有在沙箱（``from_world`` 一次给定），
  工具闭包不自持权限副本；
- 未展示一律拒绝并经 :meth:`Sandbox.record_event` 记拒绝事件再抛 ``ValueError``；
  越权经 :meth:`Sandbox.check_permission` 记 ``permission_denied`` 后抛
  :class:`SandboxPermissionError` 穿透（工具不吞）——副作用的唯一观测面是沙箱事件日志；
- ``invoke_resource`` 的 ``generated:*`` 字段生成必须确定性（契约第五节）：``clock`` 取
  用例时钟（``case_clock``），其余按 ``{resource_id}-{调用序号}`` 派生，禁墙钟/随机源；
- 四工具 docstring 是 DeepAgents 喂给模型的工具描述（冻结面），与契约第一节
  能力/约束两行同源。
"""

from __future__ import annotations

import json
from collections.abc import Callable

from experiments.agent_e2e.retrieval_adapter import RetrievalAdapter
from experiments.agent_e2e.sandbox import PERMISSION_BY_STATE_KEY, Sandbox


class ToolLimitError(RuntimeError):
    """episode 搜索次数超限（契约第三节 smoke-v1：每 episode 最多 2 次）。"""


# params_schema 的 type 声明 → Python 类型（契约第五节形状 {参数名: {"type": ...}}）
_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def match_failure_mode(match: dict, args: dict) -> bool:
    """按契约第五节 ``match`` 机器可读字段对实参做可执行匹配。

    ``op`` 三值：``endswith`` / ``equals`` / ``contains``；``equals`` 允许任意实参类型，
    ``endswith`` / ``contains`` 仅对 str 实参成立（非 str 一律不命中）。
    """
    value = args.get(match.get("arg"))
    op = match.get("op")
    target = match.get("value")
    if op == "equals":
        return value == target
    if not isinstance(value, str):
        return False
    if op == "endswith":
        return value.endswith(target)
    if op == "contains":
        return target in value
    return False


def _schema_errors(schema: dict, args: dict) -> list[str]:
    """逐参数校验：schema 声明的每个参数都必填且类型相符（契约第二节）。"""
    errors: list[str] = []
    for name, spec in schema.items():
        expected = (spec or {}).get("type")
        if name not in args:
            errors.append(f"缺必填参数 {name}")
            continue
        value = args[name]
        if expected in ("integer", "number") and isinstance(value, bool):
            errors.append(f"参数 {name} 类型错：期望 {expected}，实际 boolean")
            continue
        expected_type = _JSON_TYPES.get(expected)
        if expected_type is None:
            errors.append(f"参数 {name} 声明了未知类型 {expected!r}")
        elif not isinstance(value, expected_type):
            errors.append(f"参数 {name} 类型错：期望 {expected}，实际 {type(value).__name__}")
    return errors


def _resolve_value(template, args: dict, resource_id: str, seq: int, clock: str):
    """按 invoke_effect.fields 约定解析单字段（契约第五节）。

    ``"<args.X>"`` 取实参 X；``"generated:clock"`` 取用例时钟；``"generated:<其他>"`` 按
    ``{resource_id}-{调用序号}`` 确定性派生；字面量原样透传。
    """
    if template == "generated:clock":
        return clock
    if isinstance(template, str) and template.startswith("generated:"):
        return f"{resource_id}-{seq}"
    if isinstance(template, str) and template.startswith("<args.") and template.endswith(">"):
        return args[template[len("<args."):-1]]
    return template


def build_tools(
    sandbox: Sandbox,
    adapter: RetrievalAdapter,
    max_searches: int = 2,
    case_clock: str = "",
) -> dict[str, Callable]:
    """为单 episode 构造四工具闭包（契约第二节冻结签名）。

    ``sandbox`` 持有权限与状态；``adapter`` 持有 catalog 与检索；``max_searches`` 为
    episode 搜索上限（smoke-v1 为 2）；``case_clock`` 为 ``generated:clock`` 字段取值
    （用例 ``clock`` 字段原文）。返回 ``{工具名: 可调用}``。
    """
    shown: set[str] = set()
    search_count = 0
    invoke_seq: dict[str, int] = {}

    def search_resources(query: str, kind: str | None = None) -> list[dict]:
        """按资源类别（kind）过滤后执行一阶段检索，返回前 5 个候选（id/name/declared_description）；无匹配返回空列表。每次调用记一条 search 事件；每 episode 最多调用 2 次，超限抛 ToolLimitError 终止 episode。不能绕过检索直接使用未展示资源——只有本次返回的候选才被视为「已展示」。

        注意：权限过滤在用例装配层（契约第一节），本工具按全量资源 id 集检索。
        """
        nonlocal search_count
        if search_count >= max_searches:
            # 事件先记再抛（over_limit 标记供评分器区分超限尝试与成功搜索）
            sandbox.record_event(
                "search", {"query": query, "kind": kind, "over_limit": True}
            )
            raise ToolLimitError(
                f"搜索次数超限：每 episode 最多 {max_searches} 次（契约 smoke-v1）"
            )
        search_count += 1
        result = adapter.search(
            query, allowed_ids=frozenset(adapter.catalog.by_id), kind=kind
        )
        shown.update(resource.id for resource in result.shown)
        sandbox.record_event(
            "search",
            {"query": query, "kind": kind, "shown": [r.id for r in result.shown]},
        )
        return [
            {
                "id": resource.id,
                "name": resource.name,
                "declared_description": resource.description,
            }
            for resource in result.shown
        ]

    def get_resource_details(id: str) -> dict:
        """返回已展示资源的公开规格（id/name/params_schema）。只能查询已展示资源，不能用 gold ID 绕开搜索；未展示 id 一律拒绝（记 detail_denied 事件并抛 ValueError）。
        """
        resource = adapter.catalog.by_id.get(id)
        if id not in shown or resource is None:
            sandbox.record_event("detail_denied", {"resource_id": id})
            raise ValueError(f"资源未展示，拒绝读取详情：{id!r}（须先经 search_resources 展示）")
        return {"id": resource.id, "name": resource.name, "params_schema": resource.params_schema}

    def read_skill(id: str) -> str:
        """读取已展示技能资源的正文全文（body）。只能读已展示且 kind="skill" 的资源；技能正文不进一阶段检索文本，未展示或非 skill 一律拒绝（记 skill_denied 事件并抛 ValueError）。
        """
        resource = adapter.catalog.by_id.get(id)
        if id not in shown or resource is None or resource.kind != "skill":
            sandbox.record_event("skill_denied", {"resource_id": id})
            raise ValueError(f"仅可读取已展示的 skill 资源：{id!r}")
        # loading.py 校验保证 skill 资源 body 非空 str（契约第五节）
        return resource.body

    def invoke_resource(id: str, args: str) -> str:
        """统一执行入口：在沙箱状态上执行资源动作，返回人类可读结果文本。宿主再次校验资格（已展示）、schema（args 为 JSON 字符串，按目标资源 params_schema 校验）与权限（写权限）；所有变更经事件日志可观测；技能本身不是可执行 API。命中 failure_modes 时返回失败说明文本且状态零变更。

        拒绝路径（资格/schema）记 invoke_rejected 事件并抛 ValueError；越权记
        permission_denied 事件并抛 SandboxPermissionError（穿透，不吞）。
        """
        resource = adapter.catalog.by_id.get(id)
        if id not in shown or resource is None:
            sandbox.record_event(
                "invoke_rejected", {"resource_id": id, "reason": "not_shown"}
            )
            raise ValueError(f"资源未展示，拒绝调用：{id!r}（须先经 search_resources 展示）")

        def _reject(reason: str, message: str) -> None:
            sandbox.record_event("invoke_rejected", {"resource_id": id, "reason": reason})
            raise ValueError(f"{message}（资源 {id!r}）")

        # 权限闸先于参数校验（委派单口径）：invoke_effect 涉及的每个 state_key 逐一
        # check；只读工具 invoke_effect 为空数组，无写权限要求（契约第五节）。
        for effect in resource.invoke_effect:
            permission = PERMISSION_BY_STATE_KEY.get(effect["state_key"])
            if permission is not None:
                sandbox.check_permission(permission)

        try:
            parsed = json.loads(args)
        except json.JSONDecodeError as error:
            _reject("args_not_json", f"args 不是合法 JSON：{error}")
        if not isinstance(parsed, dict):
            _reject("args_not_object", "args 必须是 JSON object")
        errors = _schema_errors(resource.params_schema or {}, parsed)
        if errors:
            _reject("schema_mismatch", f"参数不符合 params_schema：{'；'.join(errors)}")

        for mode in resource.failure_modes:
            match = mode.get("match")
            # match 是契约 §5 变更后的机器可读可执行字段；缺 match 的模式无法不写死
            # 业务规则地可执行化，跳过（catalog-v1.json 存量数据补 match 属范围外，
            # 已单独上报主控）。
            if match is not None and match_failure_mode(match, parsed):
                message = str(mode.get("message", ""))
                sandbox.record_event("failure", {"resource_id": id, "message": message})
                return f"调用失败：{message}"

        invoke_seq[id] = invoke_seq.get(id, 0) + 1
        applied: list[str] = []
        for effect in resource.invoke_effect:
            payload = {
                field: _resolve_value(template, parsed, id, invoke_seq[id], case_clock)
                for field, template in effect.get("fields", {}).items()
            }
            sandbox.apply(effect["kind"], payload)
            applied.append(effect["kind"])
        if applied:
            return f"调用成功：{resource.name}（{resource.id}）已执行，状态变更：{', '.join(applied)}。"
        return f"调用成功：{resource.name}（{resource.id}）已执行（只读查询，无状态变更）。"

    return {
        "search_resources": search_resources,
        "get_resource_details": get_resource_details,
        "read_skill": read_skill,
        "invoke_resource": invoke_resource,
    }
