"""e2e 数据加载校验器（E1）：契约 `contract.md` 第五节/第六节的机械落地。

三入口（均接受显式 ``Path``，不硬编码仓库路径；runner/smoke 用 :func:`default_paths`
组装默认位置）：

- :func:`load_cases` —— 冒烟用例 JSONL → ``list[Case]``。形状校验（契约第六节
  必填 13 字段 + 两个可选字段）与语义校验（id 唯一、枚举合法、引用可解析、
  谓词口径、clarify/expected_failure 专用约束）。
- :func:`load_catalog` —— catalog-v1 JSON → :class:`Catalog`（id 唯一、kind 枚举、
  tool 必有 params_schema、skill 必有非空 body、invoke_effect 谓词与 state_key 合法）。
- :func:`load_world` —— world-v1 JSON → 五集合 ``dict``（恰五集合，元素形状校验留给
  E3 谓词比对面）。

schema 校验失败一律 ``ValueError``，消息含行号（``第 N 行``）与字段名。
stdlib-only（json / dataclasses / pathlib / copy / typing）。
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.agent_e2e.sandbox import (
    EVENT_TYPES,
    KNOWN_PERMISSIONS,
    STATE_KEY_BY_PREDICATE,
)

# world-v1 恰五个 keyed 集合（契约第五节 world-v1 表）
WORLD_COLLECTIONS: frozenset[str] = frozenset(
    {"calendar_events", "emails", "documents", "reports", "sent_messages"}
)

# required_state / invoke_effect 的状态谓词枚举（契约第四节，与 sandbox 写映射同源）
KNOWN_PREDICATES: frozenset[str] = frozenset(STATE_KEY_BY_PREDICATE)

ACCEPTABLE_BEHAVIORS: frozenset[str] = frozenset({"complete", "decline", "clarify"})
PERIODS: frozenset[str] = frozenset({"morning", "afternoon", "evening"})

# 每个谓词的必填字段（契约第四节谓词语义表）；document_updated 的 doc_id 可选
REQUIRED_PREDICATE_FIELDS: dict[str, tuple[str, ...]] = {
    "event_exists": ("project", "attendees", "local_date", "period"),
    "email_sent": ("to", "subject"),
    "report_exists": ("topic",),
    "document_updated": (),
}

_CASE_FIELDS: dict[str, type] = {
    "id": str,
    "suite": str,
    "user_request": str,
    "clock": str,
    "catalog_snapshot": str,
    "initial_state_fixture": str,
    "acceptable_behavior": str,
    "budget_profile": str,
}
_CASE_LIST_FIELDS: dict[str, type] = {
    "slice": str,
    "permissions": str,
    "gold_resource_ids": str,
    "required_state": dict,
    "forbidden_effects": str,
}


@dataclass(frozen=True)
class Case:
    """冒烟用例（契约第六节 smoke-v1）；slice 等列表字段以 tuple 落地保不可变。"""

    id: str
    suite: str
    slice: tuple[str, ...]
    user_request: str
    clock: str
    catalog_snapshot: str
    initial_state_fixture: str
    permissions: tuple[str, ...]
    gold_resource_ids: tuple[str, ...]
    required_state: tuple[dict[str, Any], ...]
    forbidden_effects: tuple[str, ...]
    acceptable_behavior: str
    budget_profile: str
    clarify_required_info: tuple[str, ...] | None = None
    expected_failure: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResourceSpec:
    """catalog 资源规格（契约第五节 catalog-v1）；tool/skill 字段按形态互补。"""

    id: str
    name: str
    description: str
    use_when: str
    avoid_when: str
    kind: str
    params_schema: dict[str, Any] | None = None
    invoke_effect: tuple[dict[str, Any], ...] = ()
    failure_modes: tuple[dict[str, Any], ...] = ()
    body: str | None = None


@dataclass(frozen=True)
class Catalog:
    """catalog-v1 整体：resources 元组 + by_id 映射。"""

    resources: tuple[ResourceSpec, ...]
    by_id: dict[str, ResourceSpec] = field(default_factory=dict, compare=False)


def _fail(line_no: int, case_id: str, message: str) -> ValueError:
    prefix = f"第 {line_no} 行"
    if case_id:
        prefix += f" [{case_id}]"
    return ValueError(f"{prefix} {message}")


def _require_str(line_no: int, row_id: str, row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise _fail(line_no, row_id, f"字段 {key} 必须为 str，实际为 {type(value).__name__}")
    return value


def _require_str_list(
    line_no: int, row_id: str, row: dict[str, Any], key: str
) -> list[str]:
    value = row.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise _fail(line_no, row_id, f"字段 {key} 必须为 list[str]")
    return value


def _validate_required_state(
    line_no: int, row_id: str, required_state: list[Any]
) -> None:
    for index, predicate in enumerate(required_state):
        label = f"required_state[{index}]"
        if not isinstance(predicate, dict):
            raise _fail(line_no, row_id, f"{label} 必须为 dict")
        kind = predicate.get("kind")
        if kind not in KNOWN_PREDICATES:
            raise _fail(line_no, row_id, f"{label} 未知谓词 kind: {kind!r}")
        for field_name in REQUIRED_PREDICATE_FIELDS[kind]:
            if field_name not in predicate:
                raise _fail(line_no, row_id, f"{label} 谓词 {kind} 缺字段 {field_name}")
        if kind == "event_exists":
            if not isinstance(predicate["attendees"], list) or not all(
                isinstance(v, str) for v in predicate["attendees"]
            ):
                raise _fail(line_no, row_id, f"{label} 谓词 {kind} 字段 attendees 必须为 list[str]")
            if predicate["period"] not in PERIODS:
                raise _fail(
                    line_no, row_id, f"{label} 谓词 {kind} 字段 period 非法: {predicate['period']!r}"
                )
        else:
            str_fields = REQUIRED_PREDICATE_FIELDS[kind]
            for field_name in str_fields:
                if not isinstance(predicate[field_name], str):
                    raise _fail(
                        line_no, row_id, f"{label} 谓词 {kind} 字段 {field_name} 必须为 str"
                    )
            if kind == "document_updated" and "doc_id" in predicate and not isinstance(
                predicate["doc_id"], str
            ):
                raise _fail(line_no, row_id, f"{label} 谓词 {kind} 字段 doc_id 必须为 str")


def _validate_semantics(
    line_no: int,
    row: dict[str, Any],
    row_id: str,
    catalog: Catalog | None,
    fixtures_dir: Path | None,
) -> None:
    acceptable = row["acceptable_behavior"]
    if acceptable not in ACCEPTABLE_BEHAVIORS:
        raise _fail(line_no, row_id, f"字段 acceptable_behavior 非法: {acceptable!r}")

    permissions = row["permissions"]
    unknown_permissions = [p for p in permissions if p not in KNOWN_PERMISSIONS]
    if unknown_permissions:
        raise _fail(
            line_no, row_id, f"字段 permissions 含未知权限: {unknown_permissions!r}"
        )

    unknown_forbidden = [f for f in row["forbidden_effects"] if f not in EVENT_TYPES]
    if unknown_forbidden:
        raise _fail(
            line_no,
            row_id,
            f"字段 forbidden_effects 含未知事件类型: {unknown_forbidden!r}",
        )

    if catalog is not None:
        unknown_gold = [
            g for g in row["gold_resource_ids"] if g not in catalog.by_id
        ]
        if unknown_gold:
            raise _fail(
                line_no,
                row_id,
                f"字段 gold_resource_ids 引用不在 catalog 中: {unknown_gold!r}",
            )

    if fixtures_dir is not None:
        fixture_ref = row["initial_state_fixture"]
        if not (fixtures_dir / f"{fixture_ref}.json").is_file():
            raise _fail(
                line_no, row_id, f"字段 initial_state_fixture 引用不可解析: {fixture_ref!r}"
            )

    _validate_required_state(line_no, row_id, row["required_state"])

    clarify_info = row.get("clarify_required_info")
    if acceptable == "clarify":
        if not isinstance(clarify_info, list) or not clarify_info or not all(
            isinstance(v, str) for v in clarify_info
        ):
            raise _fail(
                line_no,
                row_id,
                "acceptable_behavior=clarify 必须携带非空 list[str] 字段 clarify_required_info",
            )

    expected_failure = row.get("expected_failure")
    if expected_failure is not None:
        if acceptable != "complete":
            raise _fail(
                line_no,
                row_id,
                f"带 expected_failure 的用例 acceptable_behavior 必须为 complete，实际 {acceptable!r}",
            )
        if row["required_state"]:
            raise _fail(
                line_no,
                row_id,
                "带 expected_failure 的用例（工具故障例）required_state 必须为空数组",
            )
        if not isinstance(expected_failure, dict) or not isinstance(
            expected_failure.get("resource_id"), str
        ):
            raise _fail(
                line_no, row_id, "字段 expected_failure 形状必须为 {\"resource_id\": str}"
            )
        if catalog is not None:
            failure_resource_id = expected_failure["resource_id"]
            failure_target = catalog.by_id.get(failure_resource_id)
            if failure_target is None:
                raise _fail(
                    line_no,
                    row_id,
                    f"expected_failure.resource_id 不在 catalog 中: {failure_resource_id!r}",
                )
            if not failure_target.failure_modes:
                raise _fail(
                    line_no,
                    row_id,
                    f"expected_failure 指向无 failure_modes 的资源: {failure_resource_id!r}",
                )


def _build_case(line_no: int, row_id: str, row: dict[str, Any]) -> Case:
    for key in _CASE_FIELDS:
        _require_str(line_no, row_id, row, key)
    for key, element_type in _CASE_LIST_FIELDS.items():
        value = row.get(key)
        if not isinstance(value, list) or not all(
            isinstance(v, element_type) for v in value
        ):
            expected = "dict" if element_type is dict else "str"
            raise _fail(line_no, row_id, f"字段 {key} 必须为 list[{expected}]")

    clarify_info = row.get("clarify_required_info")
    expected_failure = row.get("expected_failure")
    return Case(
        id=row["id"],
        suite=row["suite"],
        slice=tuple(row["slice"]),
        user_request=row["user_request"],
        clock=row["clock"],
        catalog_snapshot=row["catalog_snapshot"],
        initial_state_fixture=row["initial_state_fixture"],
        permissions=tuple(row["permissions"]),
        gold_resource_ids=tuple(row["gold_resource_ids"]),
        required_state=tuple(copy.deepcopy(p) for p in row["required_state"]),
        forbidden_effects=tuple(row["forbidden_effects"]),
        acceptable_behavior=row["acceptable_behavior"],
        budget_profile=row["budget_profile"],
        clarify_required_info=(
            tuple(clarify_info) if clarify_info is not None else None
        ),
        expected_failure=(
            copy.deepcopy(expected_failure) if expected_failure is not None else None
        ),
    )


def default_paths() -> dict[str, Path]:
    """agent_e2e 数据目录默认位置（相对本文件推导，供 runner/smoke 组装）。"""
    package_dir = Path(__file__).resolve().parent
    return {
        "cases": package_dir / "cases",
        "catalogs": package_dir / "catalogs",
        "fixtures": package_dir / "fixtures",
    }


def load_cases(
    path: Path,
    catalog: Catalog | None = None,
    fixtures_dir: Path | None = None,
) -> list[Case]:
    """解析冒烟用例 JSONL，逐行按契约第六节校验。

    引用校验采用配套参数口径（本函数不硬编码仓库路径）：
    - ``catalog`` 给定 → 校验 ``gold_resource_ids`` 元素 ⊆ catalog 资源 id 集；
    - ``fixtures_dir`` 给定 → 校验 ``initial_state_fixture`` 引用可解析为
      ``<fixtures_dir>/<值>.json`` 存在的 fixture 文件。
    两者缺省时跳过对应校验（枚举、谓词与专用约束恒校验）。
    """
    cases: list[Case] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise _fail(line_no, "", f"JSON 解析失败: {error}") from error
        if not isinstance(row, dict):
            raise _fail(line_no, "", f"每行必须为 JSON object，实际为 {type(row).__name__}")

        row_id = row.get("id")
        row_id = row_id if isinstance(row_id, str) else ""
        if row_id in seen_ids:
            raise _fail(line_no, row_id, f"用例 id 重复: {row_id!r}")
        seen_ids.add(row_id)

        missing = [
            key for key in (*_CASE_FIELDS, *_CASE_LIST_FIELDS) if key not in row
        ]
        if missing:
            raise _fail(line_no, row_id, f"缺少必填字段: {missing}")

        _validate_semantics(line_no, row, row_id, catalog, fixtures_dir)
        cases.append(_build_case(line_no, row_id, row))
    return cases


def load_catalog(path: Path) -> Catalog:
    """解析 catalog-v1 JSON，按契约第五节校验（id 唯一、kind 枚举、形态字段齐备）。"""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"catalog JSON 解析失败: {error}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("resources"), list):
        raise ValueError("catalog 顶层必须为 {\"catalog_version\": str, \"resources\": list}")

    resources: list[ResourceSpec] = []
    by_id: dict[str, ResourceSpec] = {}
    for index, item in enumerate(raw["resources"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"catalog 第 {index} 个资源必须为 JSON object")
        resource_id = item.get("id")
        resource_id = resource_id if isinstance(resource_id, str) else ""
        for key in ("id", "name", "description", "use_when", "avoid_when", "kind"):
            if not isinstance(item.get(key), str):
                raise ValueError(
                    f"catalog 资源 {resource_id!r} 缺字段或类型错: {key} 必须为 str"
                )
        if resource_id in by_id:
            raise ValueError(f"catalog 资源 id 重复: {resource_id!r}")

        kind = item["kind"]
        if kind == "tool":
            params_schema = item.get("params_schema")
            if not isinstance(params_schema, dict):
                raise ValueError(
                    f"catalog 工具资源 {resource_id!r} 必须携带 dict 字段 params_schema"
                )
            invoke_effect = tuple(
                copy.deepcopy(effect) for effect in item.get("invoke_effect", ())
            )
            for effect in invoke_effect:
                if effect.get("kind") not in KNOWN_PREDICATES:
                    raise ValueError(
                        f"catalog 资源 {resource_id!r} invoke_effect 未知谓词 kind: "
                        f"{effect.get('kind')!r}"
                    )
                if effect.get("state_key") not in WORLD_COLLECTIONS:
                    raise ValueError(
                        f"catalog 资源 {resource_id!r} invoke_effect 未知 state_key: "
                        f"{effect.get('state_key')!r}"
                    )
            failure_modes = tuple(
                copy.deepcopy(mode) for mode in item.get("failure_modes", ())
            )
            resources.append(
                ResourceSpec(
                    id=resource_id,
                    name=item["name"],
                    description=item["description"],
                    use_when=item["use_when"],
                    avoid_when=item["avoid_when"],
                    kind=kind,
                    params_schema=copy.deepcopy(params_schema),
                    invoke_effect=invoke_effect,
                    failure_modes=failure_modes,
                )
            )
        elif kind == "skill":
            body = item.get("body")
            if not isinstance(body, str) or not body.strip():
                raise ValueError(
                    f"catalog 技能资源 {resource_id!r} 必须携带非空 str 字段 body"
                )
            resources.append(
                ResourceSpec(
                    id=resource_id,
                    name=item["name"],
                    description=item["description"],
                    use_when=item["use_when"],
                    avoid_when=item["avoid_when"],
                    kind=kind,
                    body=body,
                )
            )
        else:
            raise ValueError(f"catalog 资源 {resource_id!r} kind 非法: {kind!r}（须 tool/skill）")
        by_id[resource_id] = resources[-1]

    return Catalog(resources=tuple(resources), by_id=by_id)


def load_world(path: Path) -> dict[str, list[dict[str, Any]]]:
    """解析 world-v1 JSON：恰五集合（多、少均 ValueError），返回集合本体 dict。

    只做形状校验（顶层 collections + 恰五 key + 各为 list）；条目形状由 E3 谓词
    比对面消费，深拷贝语义由调用方 Sandbox 负责。
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"world JSON 解析失败: {error}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("collections"), dict):
        raise ValueError("world 顶层必须为 {\"world_version\": str, \"collections\": dict}")

    collections = raw["collections"]
    keys = set(collections)
    if keys != set(WORLD_COLLECTIONS):
        missing = sorted(set(WORLD_COLLECTIONS) - keys)
        extra = sorted(keys - set(WORLD_COLLECTIONS))
        raise ValueError(
            f"world collections 必须恰为五集合 {sorted(WORLD_COLLECTIONS)}；"
            f"缺 {missing}，多 {extra}"
        )
    for key, value in collections.items():
        if not isinstance(value, list):
            raise ValueError(f"world 集合 {key} 必须为 list，实际为 {type(value).__name__}")
    return {key: list(value) for key, value in collections.items()}
