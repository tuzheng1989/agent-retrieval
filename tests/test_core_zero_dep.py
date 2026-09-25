"""core 零依赖守卫：core 子包不得引入任何第三方模块（CI 另有裸 venv 环境级守卫）。

进程内守卫检查 ``sys.modules``：导入 core 全部模块后，不得出现 stdlib 与本包之外的
模块（``pytest`` 等测试自身依赖在断言时豁免——只看 import 期新增的第三方项）。
"""

import importlib
import sys
import types

CORE_MODULES = [
    "agent_retrieval.core.bm25",
    "agent_retrieval.core.fusion",
    "agent_retrieval.core.ports",
    "agent_retrieval.core.resource_index",
    "agent_retrieval.core.ranking",
]


def _stdlib_module_names() -> set[str]:
    return set(sys.stdlib_module_names)


def test_core_imports_no_third_party_modules():
    stdlib = _stdlib_module_names()
    before = set(sys.modules)

    for module_name in CORE_MODULES:
        importlib.import_module(module_name)

    new_modules = set(sys.modules) - before
    offenders = []
    for name in new_modules:
        root = name.split(".")[0]
        if root in stdlib or root == "agent_retrieval" or root in {"pytest", "_pytest"}:
            continue
        if isinstance(sys.modules.get(name), types.ModuleType) and getattr(
            sys.modules[name], "__spec__", None,
        ) is not None and getattr(sys.modules[name].__spec__, "origin", "") not in (None, "built-in"):
            offenders.append(name)

    assert not offenders, f"core 引入了第三方模块: {sorted(offenders)}"
