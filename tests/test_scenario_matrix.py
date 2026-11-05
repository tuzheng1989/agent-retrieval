"""scenario-matrix 离线 8 格参数化验证（SDK P2 · M2）。

格 id 权威载体是 ``docs/scenario-matrix.md`` §3 主表；本文件的测试 id 与主表格 id
一一对应，主表验证命令 ``python -m pytest tests/test_scenario_matrix.py -k <格 id>``
能选中且只选中该格的测试（每格恰一个参数化用例）。

语料指针（与主表 §3.2/§3.3 语料指针列逐格一致，§3.6 语料口径）：

- ``core_only_en_tool`` / ``api_embed_en_tool`` → ``experiments/jev_cases.json``
  （英文声明面，英文 query 检索）；
- 其余 6 格 → ``experiments/agent_e2e/catalogs/catalog-v1.json``（双语声明面；
  zh 格用中文 query，en-skill 格用英文 query；经 E1 收口组件 ``load_catalog``
  消费，不另造加载器）。

判定标准（S-D4，无阈值断言——质量红线归金版本 gate）：

1. 候选非空；
2. top-1 命中检索词目标资源（query/target 组合经实跑验证后钉定）；
3. 组件参与存在性：core_only 格断言 ``bm25_score`` 非 None（BM25 路参与）；
   api_embed 格额外断言 ``cosine`` 非 None（向量路参与融合）。

MockEmbedder 语义注意：hash 向量嵌入质量为零，cosine 排序对检索质量无意义——
api_embed 格的 top-1 断言仍锚 BM25 路（fused top-1 即检索目标资源），
``cosine`` 字段存在性只证明向量路参与了融合，不断言 cosine 排序的 top-1。

零 extras 纪律（对照 ``tests/test_core_zero_dep.py`` 的 before/after 差集纪律）：
core_only 格不得引入 embedders/stores/run/expansion/judges——以本模块 import 期
``sys.modules`` 为基线，断言**基线之后新增**的模块无这些前缀（「core_only 测试
路径自身不引入 extras」，进程内既有 extras 不背在本格头上）。因此
``EmbeddingConfig``（embedders 惰性导出）只在 api_embed 测试路径内导入；
core_only 测试定义在 api_embed 之前（pytest 按定义顺序执行，守护断言先于本文件
引入 embedders）。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from agent_retrieval import CandidateHit, MockEmbedder, rank_candidates
from experiments.agent_e2e.loading import ResourceSpec, default_paths, load_catalog

if TYPE_CHECKING:
    # 运行期不导入（embedders 属 extras——core_only 格的 sys.modules 断言要求它缺席）：
    # 与 core/ports.py 对 CandidateHit 的同款纪律，注解经 ``from __future__ import
    # annotations`` 惰性求值。
    from agent_retrieval import EmbeddingConfig

# 本文件 import 期基线（对照 test_core_zero_dep 的 before/after 差集纪律）：零 extras
# 断言只看「基线之后新出现的 extras 模块」。全量运行时，字母序更早的兄弟测试文件的
# 顶层 extras 导入在收集期先于本模块发生，已落在基线内、不构成本文件路径的违规；
# 而本文件自身的顶层误导入（Round 2 场景）发生在基线之后，仍会被咬住。
_MODULES_AT_IMPORT = frozenset(sys.modules)

_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class JevResource:
    """jev_cases.json 资源行（照 catalog-v1 ResourceSpec 的声明面形状）。"""

    id: str
    name: str
    description: str
    use_when: str
    avoid_when: str


@dataclass(frozen=True)
class CellSpec:
    """单个矩阵格的验证参数（格 id 与 docs/scenario-matrix.md §3 主表逐字一致）。"""

    #: 格 id（主表验证命令 ``-k <格 id>`` 即按此串过滤）
    cell_id: str
    #: 语料池键（映射见 _POOLS，键名对齐主表语料指针）
    pool: str
    #: 检索词（语言随格：en 格英文 query、zh 格中文 query）
    query: str
    #: 期望 top-1 的检索目标资源 id
    target_id: str


Resource = JevResource | ResourceSpec

def _jev_resources() -> tuple[JevResource, ...]:
    # jev_cases 直接 json 解析为本地资源 dataclass（声明面形状照 catalog-v1 ResourceSpec）。
    raw = json.loads(
        (_REPO_ROOT / "experiments" / "jev_cases.json").read_text(encoding="utf-8")
    )
    return tuple(JevResource(**row) for row in raw["resources"])


# catalog-v1 经 E1 收口组件 load_catalog 消费（路径由 default_paths 推导，不硬编码仓库路径）。
_CATALOG = load_catalog(default_paths()["catalogs"] / "catalog-v1.json")

# 语料池（加载替身；键名对齐主表语料指针列）：
# - "jev_cases" → experiments/jev_cases.json（en-tool 格，8 个英文 tool）
# - "catalog_tools" / "catalog_skills" → catalog-v1.json 按 kind 过滤（其余格）
_POOLS: dict[str, tuple[Resource, ...]] = {
    "jev_cases": _jev_resources(),
    "catalog_tools": tuple(r for r in _CATALOG.resources if r.kind == "tool"),
    "catalog_skills": tuple(r for r in _CATALOG.resources if r.kind == "skill"),
}

# core_only 4 格：core-only × {en,zh} × {tool,skill}
CORE_ONLY_CELLS = (
    CellSpec("core_only_en_tool", "jev_cases", "calendar create meeting", "calendar-create"),
    CellSpec("core_only_zh_tool", "catalog_tools", "创建会议日程", "calendar-create"),
    CellSpec("core_only_en_skill", "catalog_skills", "schedule meetings", "meeting-notes-skill"),
    CellSpec("core_only_zh_skill", "catalog_skills", "会议安排与通知", "meeting-notes-skill"),
)

# api_embed 4 格：api-embed × {en,zh} × {tool,skill}（语料与 query 同对应 core_only 格）
API_EMBED_CELLS = (
    CellSpec("api_embed_en_tool", "jev_cases", "calendar create meeting", "calendar-create"),
    CellSpec("api_embed_zh_tool", "catalog_tools", "创建会议日程", "calendar-create"),
    CellSpec("api_embed_en_skill", "catalog_skills", "schedule meetings", "meeting-notes-skill"),
    CellSpec("api_embed_zh_skill", "catalog_skills", "会议安排与通知", "meeting-notes-skill"),
)

# 零 extras 守护的模块前缀（S-D1 姿态①：core 仅标准库，extras 均不装配）
_EXTRAS_ROOTS = (
    "agent_retrieval.embedders",
    "agent_retrieval.stores",
    "agent_retrieval.run",
    "agent_retrieval.expansion",
    "agent_retrieval.judges",
)


def _item_id(resource: Resource) -> str:
    return resource.id


def _item_name(resource: Resource) -> str:
    return resource.name


def _corpus_text(resource: Resource) -> str:
    # 声明面 = id + name + description（matrix §2.1 装配片段口径；检索面与判定面同语料）。
    return f"{resource.id} {resource.name} {resource.description}"


def _assert_zero_extras(cell_id: str) -> None:
    # 差集语义：「core_only 测试路径自身不引入 extras」，非「进程内无 extras」——
    # 全量运行的收集/运行期污染（兄弟测试文件带来的 extras）不背在本格头上。
    newly_imported = sorted(
        name
        for name in sys.modules
        if name not in _MODULES_AT_IMPORT and name.startswith(_EXTRAS_ROOTS)
    )
    assert not newly_imported, (
        f"[{cell_id}] core_only 路径引入了 extras 模块: {newly_imported}"
    )


@pytest.mark.parametrize("cell", list(CORE_ONLY_CELLS), ids=lambda cell: cell.cell_id)
def test_core_only_cell(cell: CellSpec) -> None:
    """core_only 姿态格：rank_candidates 直调（不传 vector，纯 BM25 单路）。"""
    hits: list[CandidateHit[Resource]] = rank_candidates(
        _POOLS[cell.pool],
        cell.query,
        corpus_text=_corpus_text,
        item_id=_item_id,
        item_name=_item_name,
    )
    _assert_zero_extras(cell.cell_id)

    assert hits, f"[{cell.cell_id}] 候选为空——语料替身或装配未生效"
    top = hits[0]
    assert top.item.id == cell.target_id, (
        f"[{cell.cell_id}] top-1 应为检索目标 {cell.target_id}，实际 {top.item.id}"
    )
    assert top.bm25_score is not None, (
        f"[{cell.cell_id}] top-1 的 bm25_score 缺席——BM25 路未参与（S-D4 组件存在性）"
    )
    assert top.cosine is None, (
        f"[{cell.cell_id}] 未注入向量路的单路形态，cosine 应缺席，实际 {top.cosine}"
    )


def _embedding_config() -> EmbeddingConfig:
    # §2.2 装配片段的 api 形态配置（mock 端点凭据——本格零 API 依赖，配置只做装配证据）。
    # 顶层惰性导出（PEP 562）：在此处取值才触发 embedders 导入——core_only 格的
    # sys.modules 零 extras 断言因此不受影响（pytest 按定义顺序先跑 core_only 格）。
    from agent_retrieval import EmbeddingConfig

    return EmbeddingConfig(
        kind="api",
        provider="mock-scenario",
        model_name="mock-embedding",
        base_url="https://mock.invalid/api",
        api_key="offline-matrix",
        dimensions=8,
    )


@pytest.mark.parametrize("cell", list(API_EMBED_CELLS), ids=lambda cell: cell.cell_id)
def test_api_embed_cell(cell: CellSpec) -> None:
    """api_embed 姿态格：EmbeddingConfig 构造 + MockEmbedder 注入向量路（双路 RRF）。

    MockEmbedder 嵌入质量为零：top-1 断言锚 BM25 路；cosine 字段存在性只证明向量路
    参与了融合（S-D4），不断言 cosine 排序的 top-1。
    """
    config = _embedding_config()
    hits: list[CandidateHit[Resource]] = rank_candidates(
        _POOLS[cell.pool],
        cell.query,
        corpus_text=_corpus_text,
        item_id=_item_id,
        item_name=_item_name,
        vector=MockEmbedder(dimensions=config.dimensions),
    )

    assert hits, f"[{cell.cell_id}] 候选为空——语料替身或装配未生效"
    top = hits[0]
    assert top.item.id == cell.target_id, (
        f"[{cell.cell_id}] top-1 应为检索目标 {cell.target_id}，实际 {top.item.id}"
    )
    assert top.bm25_score is not None, (
        f"[{cell.cell_id}] top-1 的 bm25_score 缺席——top-1 断言必须锚 BM25 路"
    )
    assert top.cosine is not None, (
        f"[{cell.cell_id}] top-1 的 cosine 缺席——向量路未参与融合（S-D4 组件存在性）"
    )
