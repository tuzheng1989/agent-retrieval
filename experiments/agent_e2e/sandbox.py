"""e2e 状态沙箱（E1）：进程内虚拟状态机，episode 间零共享（契约第四节/第五节落地）。

- 状态变更**唯一入口**是 :meth:`Sandbox.apply`（四状态谓词 → world 集合映射写入）；
- 非状态事件（search / 拒绝 / failure 等）统一经 :meth:`Sandbox.record_event` 记录；
- 副作用的唯一观测面是 append-only 事件日志（:meth:`Sandbox.events`，不给删除口）；
- 初态与当前态各持独立深拷贝，:meth:`Sandbox.diff_from_initial` 产出终态检查器的输入；
- 越权观测：:meth:`Sandbox.check_permission` 缺权限时先记 ``permission_denied``
  事件再抛 :class:`SandboxPermissionError`。

谓词/权限/集合映射常量为本仓库单一出处（契约第五节），loading.py 复用本模块常量。
"""

import copy


class SandboxPermissionError(RuntimeError):
    """越权调用专用异常：要求的写权限不在 episode 授权集内。"""


# 契约第五节权限词表（冒烟全量权限集，七项）
KNOWN_PERMISSIONS: frozenset[str] = frozenset(
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

# 契约第四节 state_key 映射：状态谓词 → world 写集合
STATE_KEY_BY_PREDICATE: dict[str, str] = {
    "event_exists": "calendar_events",
    "email_sent": "sent_messages",
    "report_exists": "reports",
    "document_updated": "documents",
}

# 契约第五节写权限映射：world 写集合 → 写权限（emails 只读无写路径，不在映射内）
PERMISSION_BY_STATE_KEY: dict[str, str] = {
    "calendar_events": "calendar.write",
    "sent_messages": "email.send",
    "reports": "report.write",
    "documents": "document.write",
}

# 事件类型词表（主控 revision 2 冻结）：前六为非状态事件，后四与 STATE_KEY_BY_PREDICATE
# 同源（状态谓词事件）；评分器 earliest_blocker 依赖全部事件共用同一 seq 计数器。
EVENT_TYPES: frozenset[str] = frozenset(
    {
        "search",
        "permission_denied",
        "detail_denied",
        "skill_denied",
        "invoke_rejected",
        "failure",
        *STATE_KEY_BY_PREDICATE,
    }
)


class Sandbox:
    """单 episode 沙箱：初态快照 + 当前态 + append-only 事件日志。

    通过 :meth:`from_world` 构造；world 原件与沙箱内部零共享（两份独立深拷贝）。
    """

    def __init__(
        self,
        initial: dict[str, list[dict]],
        state: dict[str, list[dict]],
        permissions: frozenset[str],
    ) -> None:
        self._initial = initial
        self._state = state
        self._permissions = permissions
        self._events: list[dict] = []
        self._seq: int = 0

    @classmethod
    def from_world(cls, world: dict, permissions: frozenset[str]) -> "Sandbox":
        """从 world-v1 集合构造沙箱：深拷贝两份（初态 + 当前态），episode 间零共享。

        重置语义 = 对同一 world 再调一次 ``from_world``。
        """
        state = copy.deepcopy(world)
        initial = copy.deepcopy(state)
        return cls(initial=initial, state=state, permissions=frozenset(permissions))

    def record_event(self, event_type: str, payload: dict) -> None:
        """非状态事件统一记录口（search / 拒绝 / failure 等零状态变更事件）。

        append-only，与 :meth:`apply` / :meth:`check_permission` 共用同一 ``_seq``
        计数器（评分器 earliest_blocker 依赖全局统一 seq）。``event_type`` 须在
        :data:`EVENT_TYPES` 词表内；``payload`` 不得含保留键 ``type`` / ``seq``
        （显式 ``ValueError``，防覆写事件类型与序号）。
        """
        if event_type not in EVENT_TYPES:
            raise ValueError(f"未知事件类型 event_type: {event_type!r}")
        reserved = {"type", "seq"} & payload.keys()
        if reserved:
            raise ValueError(f"事件 payload 不得含保留键: {sorted(reserved)!r}")
        self._seq += 1
        self._events.append(
            {"type": event_type, "seq": self._seq, **copy.deepcopy(payload)}
        )

    def check_permission(self, permission: str) -> None:
        """权限闸：不在授权集 → 先记 ``permission_denied`` 事件再抛专用异常；有权限静默。"""
        if permission not in self._permissions:
            self.record_event("permission_denied", {"permission": permission})
            raise SandboxPermissionError(
                f"越权调用：权限 {permission!r} 不在本 episode 授权集内"
            )

    def apply(self, kind: str, payload: dict) -> None:
        """状态变更唯一入口：payload 条目写入谓词对应集合并经记录口记事件。

        ``kind`` 须为四状态谓词之一（先做谓词合法性校验，非法 ``ValueError``），
        随后经 :meth:`record_event` 记事件——词表/保留键校验失败时零状态变更、
        零事件。seq 从 1 单调递增。
        """
        state_key = STATE_KEY_BY_PREDICATE.get(kind)
        if state_key is None:
            raise ValueError(f"未知状态谓词 kind: {kind!r}")
        self.record_event(kind, payload)
        self._state[state_key].append(copy.deepcopy(payload))

    def events(self) -> tuple[dict, ...]:
        """append-only 事件日志的只读视图（逐条拷贝，外部 mutate 不影响内部）。"""
        return tuple(copy.deepcopy(event) for event in self._events)

    def diff_from_initial(self) -> dict[str, dict[str, list[dict]]]:
        """初态 vs 当前态逐 key 差异；仅含差异 key，removed 字段恒保留。

        差异按条目相等比较（multiset 口径）——初态种子不参与，供 E3 谓词
        「只对 episode 新增条目判定」消费。
        """
        diff: dict[str, dict[str, list[dict]]] = {}
        for key, after in self._state.items():
            before = self._initial[key]
            added = [entry for entry in after if entry not in before]
            removed = [entry for entry in before if entry not in after]
            if added or removed:
                diff[key] = {
                    "added": copy.deepcopy(added),
                    "removed": copy.deepcopy(removed),
                }
        return diff

    def state_snapshot(self, key: str) -> list[dict]:
        """指定集合当前条目列表的拷贝（只读视图——外部 mutate 不影响沙箱内部）。"""
        return copy.deepcopy(self._state[key])
