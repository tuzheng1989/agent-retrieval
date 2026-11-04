# -*- coding: utf-8 -*-
"""E0 批次自检脚本：校验 contract 产物（catalog-v1 / world-v1 / smoke.jsonl）。

只读校验，不修改任何产物文件。退出码 0 = 全过；1 = 存在失败项。
"""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\tuzhe\runspace\agent-retrieval")
CATALOG = ROOT / "experiments/agent_e2e/catalogs/catalog-v1.json"
WORLD = ROOT / "experiments/agent_e2e/fixtures/world-v1.json"
CASES = ROOT / "experiments/agent_e2e/cases/smoke.jsonl"
JEV = ROOT / "experiments/jev_cases.json"

REQUIRED_FIELDS = [
    "id", "suite", "slice", "user_request", "clock", "catalog_snapshot",
    "initial_state_fixture", "permissions", "gold_resource_ids",
    "required_state", "forbidden_effects", "acceptable_behavior", "budget_profile",
]
PREDICATE_FIELDS = {
    "event_exists": {"project", "attendees", "local_date", "period"},
    "email_sent": {"to", "subject"},
    "report_exists": {"topic"},
    "document_updated": set(),  # doc_id 可选
}
ALL_PREDICATES = set(PREDICATE_FIELDS)
COLLECTION_KEYS = {
    "calendar_events": {"id", "project", "attendees", "local_date", "period"},
    "emails": {"id", "from", "subject", "snippet"},
    "documents": {"doc_id", "title", "updated_at"},
    "reports": {"topic", "facts_source", "created_at"},
    "sent_messages": {"to", "subject", "body", "sent_at"},
}
COLLECTION_SEED_COUNTS = {"calendar_events": 2, "emails": 2, "documents": 1, "reports": 0, "sent_messages": 0}
FULL_PERMISSIONS = {"calendar.read", "calendar.write", "email.read", "email.send",
                    "report.write", "document.read", "document.write"}
TOOL_IDS_FROM_JEV = 8

failures: list[str] = []
checks = 0


def check(cond: bool, msg: str) -> None:
    global checks
    checks += 1
    if not cond:
        failures.append(msg)


def classify(case: dict) -> str:
    """五类归类：可完成 / 无匹配 / 信息不足 / 无权限 / 工具故障。"""
    if "expected_failure" in case:
        return "工具故障"
    if case["acceptable_behavior"] == "clarify":
        return "信息不足"
    if case["acceptable_behavior"] == "decline":
        return "无权限" if case["gold_resource_ids"] else "无匹配"
    return "可完成"


catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
world = json.loads(WORLD.read_text(encoding="utf-8"))
jev = json.loads(JEV.read_text(encoding="utf-8"))
lines = CASES.read_text(encoding="utf-8").splitlines()

# ---------- catalog-v1 ----------
resources = catalog.get("resources", [])
ids = [r["id"] for r in resources]
check(catalog.get("catalog_version") == "catalog-v1", "catalog_version != catalog-v1")
check(len(resources) == 10, f"资源数 {len(resources)} != 10")
check(len(ids) == len(set(ids)), f"catalog id 重复: {[i for i in ids if ids.count(i) > 1]}")

jev_by_id = {r["id"]: r for r in jev["resources"]}
tools = [r for r in resources if r.get("kind") == "tool"]
skills = [r for r in resources if r.get("kind") == "skill"]
check(len(tools) == 8, f"tool 资源数 {len(tools)} != 8")
check(len(skills) == 2, f"skill 资源数 {len(skills)} != 2")
check(len(tools) + len(skills) == len(resources), "存在 kind 既非 tool 又非 skill 的资源")

for r in tools:
    rid = r["id"]
    src = jev_by_id.get(rid)
    check(src is not None, f"{rid}: 不在 jev_cases.json 声明面中")
    if src:
        for field in ("id", "name", "description", "use_when", "avoid_when"):
            check(r.get(field) == src.get(field),
                  f"{rid}: 声明面字段 {field} 与 jev_cases.json 不一致")
    ps = r.get("params_schema")
    check(isinstance(ps, dict) and len(ps) > 0, f"{rid}: params_schema 缺失或为空")
    for pname, spec in (ps or {}).items():
        check(isinstance(spec, dict) and "type" in spec,
              f"{rid}: params_schema.{pname} 缺少 type")
    for eff in r.get("invoke_effect", []):
        check(eff.get("kind") in ALL_PREDICATES, f"{rid}: invoke_effect.kind 非法 {eff.get('kind')}")
        skey = eff.get("state_key")
        check(skey in COLLECTION_KEYS, f"{rid}: invoke_effect.state_key 非法 {skey}")
        if skey in COLLECTION_KEYS:
            check(set(eff.get("fields", {})) <= COLLECTION_KEYS[skey],
                  f"{rid}: invoke_effect.fields 与 {skey} 元素形状不一致")
    fm = r.get("failure_modes", [])
    if rid == "email-send":
        check(len(fm) == 1, "email-send: failure_modes 应恰一条")
        check("condition" in fm[0] and "message" in fm[0],
              "email-send: failure_modes 元素缺 condition/message")
    else:
        check(fm == [], f"{rid}: 仅 email-send 允许定义 failure_modes，此工具应为空数组")

for r in skills:
    check(isinstance(r.get("body"), str) and len(r["body"]) > 0, f"{r['id']}: skill body 非空缺失")
    for field in ("id", "name", "description", "use_when", "avoid_when", "kind", "body"):
        check(field in r, f"{r['id']}: skill 资源缺字段 {field}")

# ---------- world-v1 ----------
colls = world.get("collections", {})
check(world.get("world_version") == "world-v1", "world_version != world-v1")
check(set(colls) == set(COLLECTION_KEYS), f"world 集合不是恰五个: {sorted(colls)}")
for key, expected in COLLECTION_SEED_COUNTS.items():
    check(len(colls.get(key, [])) == expected,
          f"{key} 种子数 {len(colls.get(key, []))} != {expected}")
for key, elements in colls.items():
    for i, el in enumerate(elements):
        check(set(el) <= COLLECTION_KEYS[key],
              f"{key}[{i}] 元素键 {sorted(set(el) - COLLECTION_KEYS[key])} 超出形状约定")
for key in ("calendar_events", "documents", "reports"):
    id_field = "id" if key == "calendar_events" else "doc_id" if key == "documents" else "topic"
    vals = [el.get(id_field) for el in colls.get(key, [])]
    check(len(vals) == len(set(vals)), f"{key} 种子 {id_field} 重复")

# ---------- smoke.jsonl ----------
cases = []
for lineno, line in enumerate(lines, start=1):
    if not line.strip():
        continue
    try:
        cases.append((lineno, json.loads(line)))
    except json.JSONDecodeError as exc:
        failures.append(f"smoke.jsonl 第 {lineno} 行非法 JSON: {exc}")
        checks += 1

case_ids = [c["id"] for _, c in cases]
check(len(cases) == 12, f"用例数 {len(cases)} != 12")
check(len(case_ids) == len(set(case_ids)), f"用例 id 重复: {sorted(i for i in case_ids if case_ids.count(i) > 1)}")

consumed_predicates: set[str] = set()
for lineno, c in cases:
    cid = c.get("id", f"<第{lineno}行>")
    for f in REQUIRED_FIELDS:
        check(f in c, f"{cid}: 缺必填字段 {f}")
    check(c.get("suite") == "smoke", f"{cid}: suite != smoke")
    check(c.get("clock") == "2026-11-10T09:00:00+08:00", f"{cid}: clock 非约定固定值")
    check(c.get("catalog_snapshot") == "catalog-v1", f"{cid}: catalog_snapshot != catalog-v1")
    check(c.get("initial_state_fixture") == "world-v1", f"{cid}: initial_state_fixture != world-v1")
    check(c.get("budget_profile") == "smoke-v1", f"{cid}: budget_profile != smoke-v1")
    check(c.get("acceptable_behavior") in {"complete", "decline", "clarify"},
          f"{cid}: acceptable_behavior 非法 {c.get('acceptable_behavior')}")
    check(isinstance(c.get("slice"), list) and len(c["slice"]) > 0, f"{cid}: slice 应为非空数组")
    perms = set(c.get("permissions", []))
    check(perms <= FULL_PERMISSIONS, f"{cid}: permissions 含词表外项 {perms - FULL_PERMISSIONS}")

    for gid in c.get("gold_resource_ids", []):
        check(gid in ids, f"{cid}: gold_resource_ids 引用未知资源 {gid}")

    for pred in c.get("required_state", []):
        kind = pred.get("kind")
        check(kind in ALL_PREDICATES, f"{cid}: required_state 谓词 kind 非法 {kind}")
        if kind in PREDICATE_FIELDS:
            consumed_predicates.add(kind)
            given = set(pred) - {"kind"}
            allowed = PREDICATE_FIELDS[kind] | ({"doc_id"} if kind == "document_updated" else set())
            check(given <= allowed, f"{cid}: 谓词 {kind} 字段 {sorted(given - allowed)} 超出口径")
            if kind != "document_updated":
                check(given == PREDICATE_FIELDS[kind],
                      f"{cid}: 谓词 {kind} 缺字段 {sorted(PREDICATE_FIELDS[kind] - given)}")

    for fmode in c.get("forbidden_effects", []):
        check(fmode in ALL_PREDICATES, f"{cid}: forbidden_effects 冒烟口径外取值 {fmode}")

    # 可选字段口径
    if c.get("acceptable_behavior") == "clarify":
        info = c.get("clarify_required_info")
        check(isinstance(info, list) and len(info) > 0, f"{cid}: clarify 例缺 clarify_required_info")
    else:
        check("clarify_required_info" not in c, f"{cid}: 非 clarify 例不应带 clarify_required_info")
    if "expected_failure" in c:
        check(c["expected_failure"].get("resource_id") in ids,
              f"{cid}: expected_failure.resource_id 非法")
    check(isinstance(c.get("user_request"), str) and len(c["user_request"]) > 0, f"{cid}: user_request 为空")

    # 权限设计：无权限例缺 calendar.write，其余全量
    if c.get("acceptable_behavior") == "decline" and c.get("gold_resource_ids"):
        check("calendar.write" not in perms, f"{cid}: 无权限例应缺 calendar.write")
    else:
        check(perms == FULL_PERMISSIONS, f"{cid}: 非无权限例应给全量权限集")

check(len(consumed_predicates) == 4,
      f"四种谓词须各被 ≥1 条用例消费，实际消费 {sorted(consumed_predicates)}")
cases_with_predicates = sum(1 for _, c in cases if c.get("required_state"))
check(cases_with_predicates >= 4, f"谓词消费用例数 {cases_with_predicates} < 4")

# ---------- 五类计数 ----------
counts: dict[str, int] = {"可完成": 0, "无匹配": 0, "信息不足": 0, "无权限": 0, "工具故障": 0}
for _, c in cases:
    counts[classify(c)] += 1
completable_positive = sum(1 for _, c in cases
                           if classify(c) == "可完成" and c.get("required_state"))
print("五类用例计数：")
for name, n in counts.items():
    print(f"  {name}: {n}")
print(f"  可完成类中带 required_state 的正例: {completable_positive}")
check(counts["可完成"] >= 2, "可完成类 < 2")
check(completable_positive >= 4, "可完成类正例 < 4")
check(counts["无匹配"] >= 2, "无匹配类 < 2")
check(counts["信息不足"] >= 2, "信息不足类 < 2")

# ---------- 结果 ----------
print()
print(f"校验项总数: {checks}，失败: {len(failures)}")
if failures:
    for f in failures:
        print(f"FAIL: {f}")
    sys.exit(1)
print("E0 自检全部通过")
sys.exit(0)
