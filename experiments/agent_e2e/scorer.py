"""e2e 评分器（E3）：契约 ``contract.md`` 第四节成功规则的机械落地。

评分顺序（契约第四节，逐条执行、失败原因按序累计、不短路，``attributions`` 可重叠）：

1. **forbidden 检查**——:func:`check_forbidden` 事件类型命中即败。归因按主控裁决
   直接记命中的事件类型字符串（如 ``event_exists``），语义「发生了被禁止的副作用」
   （``state_mismatch`` 不适用——forbidden 是行为面不是状态面）；
2. **预算检查**——``trace.exhausted``（runner 步数截断）或 ``trace.error`` 含
   ``ToolLimitError``（工具闭包搜索超限，经字符串穿越，E3 不导入 E2 模块）→
   ``budget_exhausted=True`` + 归因 ``budget_exhausted``；
3. **required_state 检查**——:func:`check_required_state` 未满足 → 败 + 归因
   ``state_mismatch``；
4. **终止行为检查**——``acceptable_behavior`` ∈ complete/decline/clarify 判定面
   （clarify 要求 ``final_output`` 对 ``clarify_required_info`` 中**每个**字段经
   :data:`CLARIFY_KEYWORD_ALIASES` 冻结别名组任一命中——以落盘契约原文为唯一
   权威，E4 revision 3 起含中文等价词）；
5. 全部通过 → ``success=True``。

谓词只对 episode **新增条目**判定（经 ``sandbox.diff_from_initial()`` 的 ``added``；
``document_updated`` 按「added 中存在匹配条目」实现，不假设替换语义——apply 只有
append 无替换路径）。

``earliest_blocker`` = attributions 相关事件中 seq 最小者的类型。归因 → 事件类型映射：
与事件类型同名直接匹配；``tool_failure`` → ``failure`` 事件；``state_mismatch`` →
四状态谓词事件；其余标签无事件基（对 ``earliest_blocker`` 无贡献，全无贡献 → ``None``）。

因此 ``attributions`` 值域 = :data:`ATTRIBUTION_LABELS`（10 项冻结全集）∪ forbidden
命中的事件类型字符串（主控裁决专用通道）。
"""

from dataclasses import dataclass

from experiments.agent_e2e.loading import Case
from experiments.agent_e2e.runner import Trace
from experiments.agent_e2e.sandbox import EVENT_TYPES, STATE_KEY_BY_PREDICATE, Sandbox

# 契约第四节失败归因标签集（跨阶段冻结全集，10 项）；forbidden 命中归因经事件类型
# 字符串通道进入 attributions（见模块 docstring），不在本集内。
ATTRIBUTION_LABELS: frozenset[str] = frozenset(
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


@dataclass(frozen=True)
class EpisodeResult:
    """单 episode 评分结果：二值成功 + 失败原因 + 可重叠归因 + 互斥最早阻断者。"""

    success: bool
    failure_reasons: tuple[str, ...]
    attributions: frozenset[str]
    earliest_blocker: str | None
    budget_exhausted: bool


# 契约第四节 clarify 判定的字段别名组（E4 revision 3：live e2e-smoke-008 定性——
# Agent 用中文正确列出缺失项，评分器只找英文字段名判败）。组内或、组间且——维持
# 契约「每个字段」口径；词表外字段名回落为仅字段名原文（子串匹配口径不变）。
CLARIFY_KEYWORD_ALIASES: dict[str, tuple[str, ...]] = {
    "local_date": ("local_date", "日期", "哪一天", "哪天"),
    "period": ("period", "时段", "时间段"),
    "attendees": ("attendees", "参会人", "参会", "参加人"),
    "to": ("to", "收件人", "发送给", "发给"),
    "subject": ("subject", "主题"),
    "project": ("project", "项目"),
    "topic": ("topic", "主题", "报告主题"),
    "doc_id": ("doc_id", "文档"),
}


def _match_event_exists(added: list[dict], predicate: dict) -> bool:
    want_attendees = set(predicate["attendees"])
    return any(
        entry.get("project") == predicate["project"]
        and set(entry.get("attendees") or ()) == want_attendees
        and entry.get("local_date") == predicate["local_date"]
        and entry.get("period") == predicate["period"]
        for entry in added
    )


def _match_email_sent(added: list[dict], predicate: dict) -> bool:
    return any(
        entry.get("to") == predicate["to"] and entry.get("subject") == predicate["subject"]
        for entry in added
    )


def _match_report_exists(added: list[dict], predicate: dict) -> bool:
    return any(entry.get("topic") == predicate["topic"] for entry in added)


def _match_document_updated(added: list[dict], predicate: dict) -> bool:
    doc_id = predicate.get("doc_id")
    if doc_id is None:
        return bool(added)  # doc_id 省略 = 任意新增条目成立
    return any(entry.get("doc_id") == doc_id for entry in added)


_PREDICATE_MATCHERS = {
    "event_exists": _match_event_exists,
    "email_sent": _match_email_sent,
    "report_exists": _match_report_exists,
    "document_updated": _match_document_updated,
}


def check_required_state(required: list[dict], sandbox: Sandbox) -> list[str]:
    """返回未满足谓词的描述列表（空列表 = 全部满足）。

    谓词语义照契约第四节：只对 episode 新增条目判定——经 ``sandbox.diff_from_initial()``
    取各集合 ``added`` 匹配，初态种子不参与满足；``document_updated`` 按「added 中存在
    匹配条目」实现（append-only，无替换路径）。
    """
    diff = sandbox.diff_from_initial()
    unsatisfied: list[str] = []
    for predicate in required:
        kind = predicate["kind"]
        added = diff.get(STATE_KEY_BY_PREDICATE[kind], {}).get("added", [])
        if not _PREDICATE_MATCHERS[kind](added, predicate):
            unsatisfied.append(f"required_state 未满足: {kind}")
    return unsatisfied


def check_forbidden(forbidden: list[str], sandbox: Sandbox) -> list[str]:
    """返回 forbidden 模式中被事件日志命中的项（保持 ``forbidden`` 实参顺序）。"""
    events = sandbox.events()
    return [pattern for pattern in forbidden if any(event.get("type") == pattern for event in events)]


def _final_text(trace: Trace) -> str | None:
    """规范化最终回答：``None`` 或仅空白视同空（complete/decline/clarify 均要求非空）。"""
    text = trace.final_output
    if text is None or not text.strip():
        return None
    return text


def _check_behavior(
    case: Case,
    trace: Trace,
    sandbox: Sandbox,
    reasons: list[str],
    attributions: set[str],
) -> None:
    """终止行为判定面（契约第四节 acceptable_behavior 三值枚举），就地追加原因与归因。"""
    behavior = case.acceptable_behavior
    changed_keys = sorted(sandbox.diff_from_initial())
    final_text = _final_text(trace)
    events = sandbox.events()

    if behavior == "complete":
        if case.expected_failure is not None:
            # 工具故障例：按预期报告失败也算完成（契约第四节 complete 第三子条）
            resource_id = case.expected_failure["resource_id"]
            has_failure_event = any(
                event.get("type") == "failure" and event.get("resource_id") == resource_id
                for event in events
            )
            if not has_failure_event:
                reasons.append(f"expected_failure 资源未出现 failure 事件: {resource_id}")
                attributions.add("tool_failure")
            if changed_keys:
                reasons.append(f"存在副作用状态变更: {changed_keys}")
                attributions.add("state_mismatch")
            if final_text is None:
                reasons.append("final_output 为空")
            return
        if final_text is None:
            reasons.append("final_output 为空")
        return

    # decline / clarify 共用「正确不行动」判定面：零副作用变更 + 给出回答
    if changed_keys:
        reasons.append(f"存在副作用状态变更: {changed_keys}")
        attributions.add("state_mismatch")
    if final_text is None:
        reasons.append("final_output 为空")
        return
    if behavior == "clarify":
        keywords = case.clarify_required_info or ()
        if not keywords:
            reasons.append("clarify 缺少 clarify_required_info 字段")
            return
        missing = [
            keyword
            for keyword in keywords
            if not any(
                alias in final_text
                for alias in CLARIFY_KEYWORD_ALIASES.get(keyword, (keyword,))
            )
        ]
        if missing:
            reasons.append(f"final_output 未含澄清字段关键词: {missing}")


def _related_event_types(attribution: str) -> frozenset[str] | None:
    """归因标签 → 相关事件类型集；无事件基的标签返回 ``None``。"""
    if attribution in EVENT_TYPES:
        return frozenset({attribution})
    if attribution == "tool_failure":
        return frozenset({"failure"})
    if attribution == "state_mismatch":
        return frozenset(STATE_KEY_BY_PREDICATE)
    return None


def _earliest_blocker(attributions: frozenset[str], events: tuple[dict, ...]) -> str | None:
    """attributions 相关事件中 seq 最小者的类型；全无事件基 → ``None``。"""
    candidates: list[tuple[int, str]] = []
    for attribution in attributions:
        related = _related_event_types(attribution)
        if related is None:
            continue
        for event in events:
            if event.get("type") in related:
                candidates.append((event["seq"], event["type"]))
    if not candidates:
        return None
    return min(candidates)[1]


def score_episode(case: Case, trace: Trace, sandbox: Sandbox) -> EpisodeResult:
    """按契约第四节顺序评分：forbidden → 预算 → required_state → 终止行为 → 二值成功。

    评分器只读 trace 与沙箱终态（``diff_from_initial``/``events`` 均为只读拷贝视图），
    不改状态；失败原因按评分顺序累计，归因可重叠。
    """
    reasons: list[str] = []
    attributions: set[str] = set()

    # ① forbidden：failure_reasons 优先记录命中项（验收口径：failure_reasons 含命中项），
    #    归因 = 命中的事件类型字符串（主控裁决）
    for hit in check_forbidden(list(case.forbidden_effects), sandbox):
        reasons.append(hit)
        attributions.add(hit)

    # ② 预算：步数耗尽或 ToolLimitError（契约第三节两类限额，执行主体不同）
    budget_reasons: list[str] = []
    if trace.exhausted:
        budget_reasons.append("预算耗尽: 步数超限")
    if trace.error and "ToolLimitError" in trace.error:
        budget_reasons.append("预算耗尽: ToolLimitError")
    budget_exhausted = bool(budget_reasons)
    if budget_exhausted:
        reasons.extend(budget_reasons)
        attributions.add("budget_exhausted")

    # ③ required_state：未满足 → 败 + 归因 state_mismatch
    for reason in check_required_state(list(case.required_state), sandbox):
        reasons.append(reason)
        attributions.add("state_mismatch")

    # ④ 终止行为判定面
    _check_behavior(case, trace, sandbox, reasons, attributions)

    return EpisodeResult(
        success=not reasons,
        failure_reasons=tuple(reasons),
        attributions=frozenset(attributions),
        earliest_blocker=_earliest_blocker(frozenset(attributions), sandbox.events()),
        budget_exhausted=budget_exhausted,
    )
