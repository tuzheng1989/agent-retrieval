"""E1 loading 加载校验测试：契约第六节用例 schema + 第五节 catalog/world 形状。

篡改场景基于真实冒烟数据首行（或指定行）改字段，断言 ValueError 含行号（``第 N 行``）
与字段名——错误定位面是 E1 的验收本体。
"""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from experiments.agent_e2e.loading import (
    default_paths,
    load_cases,
    load_catalog,
    load_world,
)


def _default_paths() -> dict[str, Path]:
    return default_paths()


def _smoke_path() -> Path:
    return _default_paths()["cases"] / "smoke.jsonl"


def _catalog_path() -> Path:
    return _default_paths()["catalogs"] / "catalog-v1.json"


def _fixtures_dir() -> Path:
    return _default_paths()["fixtures"]


def _load_rows() -> list[dict]:
    rows = []
    for line in _smoke_path().read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_rows(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "cases.jsonl"
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 真实数据零错误加载（E0 数据自动复核）
# ---------------------------------------------------------------------------


def test_smoke_12_cases_load_clean():
    catalog = load_catalog(_catalog_path())
    cases = load_cases(_smoke_path(), catalog=catalog, fixtures_dir=_fixtures_dir())

    assert len(cases) == 12
    ids = [case.id for case in cases]
    assert len(set(ids)) == 12
    assert all(case.suite == "smoke" for case in cases)
    assert all(case.budget_profile == "smoke-v1" for case in cases)
    assert all(case.catalog_snapshot == "catalog-v1" for case in cases)
    assert all(case.initial_state_fixture == "world-v1" for case in cases)


def test_case_is_frozen_dataclass():
    catalog = load_catalog(_catalog_path())
    cases = load_cases(_smoke_path(), catalog=catalog, fixtures_dir=_fixtures_dir())

    with pytest.raises(FrozenInstanceError):
        cases[0].id = "mutated"  # type: ignore[misc]


def test_clarify_case_carries_required_info():
    catalog = load_catalog(_catalog_path())
    cases = load_cases(_smoke_path(), catalog=catalog, fixtures_dir=_fixtures_dir())

    by_id = {case.id: case for case in cases}
    clarify = [c for c in cases if c.acceptable_behavior == "clarify"]
    assert len(clarify) == 2
    for case in clarify:
        assert case.clarify_required_info  # 非空
    assert by_id["e2e-smoke-008"].clarify_required_info == ("local_date", "period")

    failure = by_id["e2e-smoke-011"]
    assert failure.expected_failure == {"resource_id": "email-send"}
    assert failure.acceptable_behavior == "complete"
    assert failure.required_state == ()


def test_optional_fields_absent_are_none():
    catalog = load_catalog(_catalog_path())
    cases = load_cases(_smoke_path(), catalog=catalog, fixtures_dir=_fixtures_dir())

    plain = next(c for c in cases if c.id == "e2e-smoke-001")
    assert plain.clarify_required_info is None
    assert plain.expected_failure is None


def test_catalog_loads_10_resources():
    catalog = load_catalog(_catalog_path())

    assert len(catalog.resources) == 10
    assert set(catalog.by_id) == {resource.id for resource in catalog.resources}
    tools = [r for r in catalog.resources if r.kind == "tool"]
    skills = [r for r in catalog.resources if r.kind == "skill"]
    assert len(tools) == 8
    assert len(skills) == 2
    assert all(r.params_schema is not None for r in tools)
    assert all(r.invoke_effect == () for r in tools if r.id == "geo-query")
    assert all(r.body for r in skills)
    assert all(r.body is None for r in tools)
    assert all(r.params_schema is None for r in skills)


def test_catalog_invoke_effect_shapes_match_contract():
    catalog = load_catalog(_catalog_path())

    email_send = catalog.by_id["email-send"]
    assert email_send.invoke_effect == (
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
    )


def test_world_loads_exactly_five_collections():
    world = load_world(_fixtures_dir() / "world-v1.json")

    assert set(world) == {
        "calendar_events",
        "emails",
        "documents",
        "reports",
        "sent_messages",
    }
    assert isinstance(world["calendar_events"], list)
    assert len(world["calendar_events"]) == 2
    assert world["reports"] == []
    assert world["sent_messages"] == []


def test_default_paths_derive_from_package_dir():
    paths = default_paths()

    assert set(paths) == {"cases", "catalogs", "fixtures"}
    assert paths["cases"].name == "cases"
    assert paths["catalogs"].name == "catalogs"
    assert paths["fixtures"].name == "fixtures"
    assert (_smoke_path()).is_file()
    assert (_catalog_path()).is_file()
    assert (_fixtures_dir() / "world-v1.json").is_file()


# ---------------------------------------------------------------------------
# 篡改场景：各抛 ValueError，含行号（第 N 行）与字段名
# ---------------------------------------------------------------------------


def test_duplicate_id_reports_line(tmp_path):
    rows = _load_rows()
    path = _write_rows(tmp_path, [rows[0], rows[0]])

    with pytest.raises(ValueError) as exc_info:
        load_cases(path)
    assert "第 2 行" in str(exc_info.value)
    assert rows[0]["id"] in str(exc_info.value)


def test_missing_required_field_reports_line_and_field(tmp_path):
    rows = _load_rows()
    del rows[0]["user_request"]
    path = _write_rows(tmp_path, rows)

    with pytest.raises(ValueError) as exc_info:
        load_cases(path)
    assert "第 1 行" in str(exc_info.value)
    assert "user_request" in str(exc_info.value)


def test_wrong_type_reports_line_and_field(tmp_path):
    rows = _load_rows()
    rows[0]["permissions"] = "calendar.read"  # 应为 list[str]
    path = _write_rows(tmp_path, rows)

    with pytest.raises(ValueError) as exc_info:
        load_cases(path)
    assert "第 1 行" in str(exc_info.value)
    assert "permissions" in str(exc_info.value)


def test_unknown_gold_resource_id(tmp_path):
    catalog = load_catalog(_catalog_path())
    rows = _load_rows()
    rows[0]["gold_resource_ids"] = ["nonexistent-tool"]
    path = _write_rows(tmp_path, rows)

    with pytest.raises(ValueError) as exc_info:
        load_cases(path, catalog=catalog)
    assert "第 1 行" in str(exc_info.value)
    assert "nonexistent-tool" in str(exc_info.value)


def test_unknown_required_state_predicate_kind(tmp_path):
    rows = _load_rows()
    rows[0]["required_state"][0]["kind"] = "portal_opened"
    path = _write_rows(tmp_path, rows)

    with pytest.raises(ValueError) as exc_info:
        load_cases(path)
    assert "第 1 行" in str(exc_info.value)
    assert "portal_opened" in str(exc_info.value)


def test_event_exists_missing_field(tmp_path):
    rows = _load_rows()
    del rows[0]["required_state"][0]["attendees"]
    path = _write_rows(tmp_path, rows)

    with pytest.raises(ValueError) as exc_info:
        load_cases(path)
    assert "第 1 行" in str(exc_info.value)
    assert "attendees" in str(exc_info.value)


def test_event_exists_period_outside_enum(tmp_path):
    rows = _load_rows()
    rows[0]["required_state"][0]["period"] = "midnight"
    path = _write_rows(tmp_path, rows)

    with pytest.raises(ValueError) as exc_info:
        load_cases(path)
    assert "第 1 行" in str(exc_info.value)
    assert "period" in str(exc_info.value)


def test_clarify_case_without_clarify_required_info(tmp_path):
    rows = _load_rows()
    clarify_row = next(r for r in rows if r["acceptable_behavior"] == "clarify")
    del clarify_row["clarify_required_info"]
    path = _write_rows(tmp_path, rows)

    with pytest.raises(ValueError) as exc_info:
        load_cases(path)
    assert clarify_row["id"] in str(exc_info.value)
    assert "clarify_required_info" in str(exc_info.value)


def test_expected_failure_case_must_be_complete(tmp_path):
    rows = _load_rows()
    failure_row = next(r for r in rows if "expected_failure" in r)
    assert failure_row["acceptable_behavior"] == "complete"  # 前置：真实数据本为 complete
    failure_row["acceptable_behavior"] = "decline"
    path = _write_rows(tmp_path, rows)

    with pytest.raises(ValueError) as exc_info:
        load_cases(path)
    assert failure_row["id"] in str(exc_info.value)


def test_expected_failure_case_requires_empty_required_state(tmp_path):
    rows = _load_rows()
    failure_row = next(r for r in rows if "expected_failure" in r)
    failure_row["required_state"] = [{"kind": "email_sent", "to": "x@y", "subject": "s"}]
    path = _write_rows(tmp_path, rows)

    with pytest.raises(ValueError) as exc_info:
        load_cases(path)
    assert failure_row["id"] in str(exc_info.value)
    assert "required_state" in str(exc_info.value)


def test_unknown_permission(tmp_path):
    rows = _load_rows()
    rows[0]["permissions"] = ["calendar.read", "teleport.write"]
    path = _write_rows(tmp_path, rows)

    with pytest.raises(ValueError) as exc_info:
        load_cases(path)
    assert "第 1 行" in str(exc_info.value)
    assert "teleport.write" in str(exc_info.value)


def test_unresolvable_fixture_reference(tmp_path):
    catalog = load_catalog(_catalog_path())
    rows = _load_rows()
    rows[0]["initial_state_fixture"] = "world-v9"
    path = _write_rows(tmp_path, rows)

    with pytest.raises(ValueError) as exc_info:
        load_cases(path, catalog=catalog, fixtures_dir=tmp_path)  # 空目录：world-v9.json 不存在
    assert "第 1 行" in str(exc_info.value)
    assert "world-v9" in str(exc_info.value)


def test_unknown_acceptable_behavior_enum(tmp_path):
    rows = _load_rows()
    rows[0]["acceptable_behavior"] = "maybe"
    path = _write_rows(tmp_path, rows)

    with pytest.raises(ValueError) as exc_info:
        load_cases(path)
    assert "第 1 行" in str(exc_info.value)
    assert "acceptable_behavior" in str(exc_info.value)


def test_unknown_forbidden_effect(tmp_path):
    rows = _load_rows()
    rows[0]["forbidden_effects"] = ["event_exists", "fax_sent"]
    path = _write_rows(tmp_path, rows)

    with pytest.raises(ValueError) as exc_info:
        load_cases(path)
    assert "第 1 行" in str(exc_info.value)
    assert "forbidden_effects" in str(exc_info.value)
    assert "fax_sent" in str(exc_info.value)


def test_expected_failure_unknown_resource_id(tmp_path):
    catalog = load_catalog(_catalog_path())
    rows = _load_rows()
    failure_row = next(r for r in rows if "expected_failure" in r)
    failure_row["expected_failure"] = {"resource_id": "nonexistent-tool"}
    path = _write_rows(tmp_path, rows)

    with pytest.raises(ValueError) as exc_info:
        load_cases(path, catalog=catalog)
    assert failure_row["id"] in str(exc_info.value)
    assert "nonexistent-tool" in str(exc_info.value)


def test_expected_failure_resource_without_failure_modes(tmp_path):
    catalog = load_catalog(_catalog_path())
    rows = _load_rows()
    failure_row = next(r for r in rows if "expected_failure" in r)
    failure_row["expected_failure"] = {"resource_id": "geo-query"}  # failure_modes == []
    path = _write_rows(tmp_path, rows)

    with pytest.raises(ValueError) as exc_info:
        load_cases(path, catalog=catalog)
    assert failure_row["id"] in str(exc_info.value)
    assert "geo-query" in str(exc_info.value)
    assert "failure_modes" in str(exc_info.value)


# ---------------------------------------------------------------------------
# catalog / world 篡改
# ---------------------------------------------------------------------------


def test_catalog_duplicate_resource_id(tmp_path):
    raw = json.loads(_catalog_path().read_text(encoding="utf-8"))
    raw["resources"].append(dict(raw["resources"][0]))
    path = tmp_path / "catalog-dup.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_catalog(path)
    assert raw["resources"][0]["id"] in str(exc_info.value)


def test_catalog_unknown_tool_kind(tmp_path):
    raw = json.loads(_catalog_path().read_text(encoding="utf-8"))
    raw["resources"][0]["kind"] = "plugin"
    path = tmp_path / "catalog-kind.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_catalog(path)
    assert "geo-query" in str(exc_info.value)
    assert "kind" in str(exc_info.value)


def test_catalog_tool_without_params_schema(tmp_path):
    raw = json.loads(_catalog_path().read_text(encoding="utf-8"))
    tool = next(r for r in raw["resources"] if r["id"] == "geo-query")
    del tool["params_schema"]
    path = tmp_path / "catalog-noschema.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_catalog(path)
    assert "geo-query" in str(exc_info.value)
    assert "params_schema" in str(exc_info.value)


def test_catalog_skill_with_empty_body(tmp_path):
    raw = json.loads(_catalog_path().read_text(encoding="utf-8"))
    skill = next(r for r in raw["resources"] if r["id"] == "meeting-notes-skill")
    skill["body"] = "   "
    path = tmp_path / "catalog-emptybody.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_catalog(path)
    assert "meeting-notes-skill" in str(exc_info.value)
    assert "body" in str(exc_info.value)


def test_catalog_invoke_effect_unknown_state_key(tmp_path):
    raw = json.loads(_catalog_path().read_text(encoding="utf-8"))
    email_send = next(r for r in raw["resources"] if r["id"] == "email-send")
    email_send["invoke_effect"][0]["state_key"] = "faxes"
    path = tmp_path / "catalog-statekey.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_catalog(path)
    assert "email-send" in str(exc_info.value)
    assert "faxes" in str(exc_info.value)


def test_catalog_invoke_effect_unknown_predicate(tmp_path):
    raw = json.loads(_catalog_path().read_text(encoding="utf-8"))
    email_send = next(r for r in raw["resources"] if r["id"] == "email-send")
    email_send["invoke_effect"][0]["kind"] = "fax_sent"
    path = tmp_path / "catalog-predicate.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_catalog(path)
    assert "email-send" in str(exc_info.value)
    assert "fax_sent" in str(exc_info.value)


def test_world_missing_collection(tmp_path):
    raw = json.loads((_fixtures_dir() / "world-v1.json").read_text(encoding="utf-8"))
    del raw["collections"]["reports"]
    path = tmp_path / "world-missing.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_world(path)
    assert "reports" in str(exc_info.value)


def test_world_extra_collection(tmp_path):
    raw = json.loads((_fixtures_dir() / "world-v1.json").read_text(encoding="utf-8"))
    raw["collections"]["faxes"] = []
    path = tmp_path / "world-extra.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_world(path)
    assert "faxes" in str(exc_info.value)
