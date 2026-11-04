"""E1 sandbox 测试：episode 隔离（深拷贝零共享）、事件 append-only、权限观测面。

「事件先记再抛」的顺序用 try/except 在异常传播路径上观测 events() 钉住。
"""

import pytest

from experiments.agent_e2e.sandbox import (
    EVENT_TYPES,
    KNOWN_PERMISSIONS,
    PERMISSION_BY_STATE_KEY,
    STATE_KEY_BY_PREDICATE,
    Sandbox,
    SandboxPermissionError,
)


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
        "emails": [
            {
                "id": "eml-1",
                "from": "liu@corp.example.com",
                "subject": "周报",
                "snippet": "本周迭代已完成。",
            }
        ],
        "documents": [
            {"doc_id": "doc-1", "title": "员工手册", "updated_at": "2026-09-01"}
        ],
        "reports": [],
        "sent_messages": [],
    }


FULL = frozenset(KNOWN_PERMISSIONS)
EMAIL = {
    "to": "wang@corp.example.com",
    "subject": "Atlas 会议通知",
    "body": "正文",
    "sent_at": "2026-11-10T09:00:00+08:00",
}
REPORT = {
    "topic": "Q3 销售季度报告",
    "facts_source": "user-provided",
    "created_at": "2026-11-10T09:00:00+08:00",
}


def test_from_world_two_instances_isolated():
    world = _world()
    s1 = Sandbox.from_world(world, FULL)
    s2 = Sandbox.from_world(world, FULL)

    s1.check_permission("email.send")
    s1.apply("email_sent", dict(EMAIL))

    # 另一实例零可见（episode 隔离）
    assert s2.diff_from_initial() == {}
    assert s2.state_snapshot("sent_messages") == []
    assert s2.events() == ()
    # 原 world 零污染（深拷贝语义）
    assert world["sent_messages"] == []
    # 外部 mutate world 不渗入沙箱，初态种子也不被判为 diff
    world["calendar_events"][0]["project"] = "MUTATED"
    assert s1.state_snapshot("calendar_events")[0]["project"] == "Atlas"
    assert "calendar_events" not in s1.diff_from_initial()


def test_apply_records_added_entries_in_diff():
    s = Sandbox.from_world(_world(), FULL)
    s.check_permission("email.send")
    s.apply("email_sent", dict(EMAIL))

    diff = s.diff_from_initial()
    assert set(diff) == {"sent_messages"}
    assert diff["sent_messages"]["added"] == [EMAIL]
    assert diff["sent_messages"]["removed"] == []

    s.check_permission("report.write")
    s.apply("report_exists", dict(REPORT))
    diff = s.diff_from_initial()
    assert set(diff) == {"sent_messages", "reports"}
    assert diff["reports"]["added"] == [REPORT]


def test_events_seq_monotonic_from_one():
    s = Sandbox.from_world(_world(), FULL)
    s.check_permission("email.send")
    s.apply("email_sent", dict(EMAIL))
    s.check_permission("report.write")
    s.apply("report_exists", dict(REPORT))

    events = s.events()
    assert [event["seq"] for event in events] == [1, 2]
    assert events[0]["type"] == "email_sent"
    assert events[0]["to"] == EMAIL["to"]
    assert events[1]["type"] == "report_exists"
    assert events[1]["topic"] == REPORT["topic"]


def test_permission_denied_event_recorded_before_raise():
    s = Sandbox.from_world(_world(), frozenset({"calendar.read"}))
    observed: tuple[dict, ...] = ()
    with pytest.raises(SandboxPermissionError):
        try:
            s.check_permission("calendar.write")
        finally:
            observed = s.events()  # 异常抛出时刻事件已入日志（先记再抛）
    assert observed == (
        {"type": "permission_denied", "permission": "calendar.write", "seq": 1},
    )
    # 越权观测面不产生任何状态变更
    assert s.diff_from_initial() == {}


def test_check_permission_within_grants_is_silent():
    s = Sandbox.from_world(_world(), FULL)
    assert s.check_permission("calendar.write") is None
    assert s.events() == ()


def test_snapshots_events_and_diff_return_copies():
    s = Sandbox.from_world(_world(), FULL)
    s.check_permission("email.send")
    payload = dict(EMAIL)
    s.apply("email_sent", payload)

    events = s.events()
    events[0]["type"] = "MUTATED"
    assert s.events()[0]["type"] == "email_sent"

    snapshot = s.state_snapshot("sent_messages")
    snapshot.append({"to": "hack", "subject": "hack", "body": "hack", "sent_at": "x"})
    assert s.state_snapshot("sent_messages") == [EMAIL]

    payload["subject"] = "HACK"  # apply 入参别名不渗入内部
    assert s.state_snapshot("sent_messages")[0]["subject"] == EMAIL["subject"]

    diff = s.diff_from_initial()
    diff["sent_messages"]["added"][0]["subject"] = "HACK"
    assert s.diff_from_initial()["sent_messages"]["added"][0]["subject"] == EMAIL["subject"]


def test_apply_rejects_unknown_predicate_kind():
    s = Sandbox.from_world(_world(), FULL)
    with pytest.raises(ValueError):
        s.apply("fax_sent", {"to": "x"})
    assert s.events() == ()
    assert s.diff_from_initial() == {}


def test_record_event_rejects_unknown_event_type():
    s = Sandbox.from_world(_world(), FULL)
    with pytest.raises(ValueError):
        s.record_event("teleported", {"note": "x"})
    assert s.events() == ()


def test_record_event_seq_continuous_mixed_with_apply():
    s = Sandbox.from_world(_world(), FULL)
    s.check_permission("email.send")
    s.apply("email_sent", dict(EMAIL))
    s.record_event("search", {"query": "会议安排", "shown": ["calendar-create"]})
    s.record_event("failure", {"resource_id": "email-send", "message": "域名无法投递"})
    s.check_permission("report.write")
    s.apply("report_exists", dict(REPORT))

    events = s.events()
    assert [event["seq"] for event in events] == [1, 2, 3, 4]
    assert [event["type"] for event in events] == [
        "email_sent",
        "search",
        "failure",
        "report_exists",
    ]
    assert events[1]["query"] == "会议安排"
    assert events[2]["resource_id"] == "email-send"


def test_record_event_rejects_reserved_payload_keys():
    s = Sandbox.from_world(_world(), FULL)
    with pytest.raises(ValueError):
        s.record_event("search", {"type": "search"})
    with pytest.raises(ValueError):
        s.record_event("search", {"seq": 9})
    assert s.events() == ()


def test_apply_rejects_reserved_payload_keys_atomically():
    s = Sandbox.from_world(_world(), FULL)
    with pytest.raises(ValueError):
        s.apply("email_sent", {"to": "x", "seq": 1})
    assert s.events() == ()
    assert s.diff_from_initial() == {}  # 保留键校验先于状态变更：零事件零副作用


def test_event_types_vocabulary_matches_contract():
    assert EVENT_TYPES == frozenset(
        {
            "search",
            "permission_denied",
            "detail_denied",
            "skill_denied",
            "invoke_rejected",
            "failure",
            "event_exists",
            "email_sent",
            "report_exists",
            "document_updated",
        }
    )


def test_permission_and_mapping_constants_match_contract():
    assert KNOWN_PERMISSIONS == frozenset(
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
    assert STATE_KEY_BY_PREDICATE == {
        "event_exists": "calendar_events",
        "email_sent": "sent_messages",
        "report_exists": "reports",
        "document_updated": "documents",
    }
    assert PERMISSION_BY_STATE_KEY == {
        "calendar_events": "calendar.write",
        "sent_messages": "email.send",
        "reports": "report.write",
        "documents": "document.write",
    }


def test_diff_removed_field_present_even_when_no_removal():
    s = Sandbox.from_world(_world(), FULL)
    assert s.diff_from_initial() == {}

    s.check_permission("document.write")
    s.apply(
        "document_updated",
        {"doc_id": "doc-1", "title": "员工手册 v2", "updated_at": "2026-11-10"},
    )
    diff = s.diff_from_initial()
    assert diff["documents"]["removed"] == []  # removed 字段保留以备将来（冒烟只增不删）
    assert diff["documents"]["added"] == [
        {"doc_id": "doc-1", "title": "员工手册 v2", "updated_at": "2026-11-10"}
    ]
