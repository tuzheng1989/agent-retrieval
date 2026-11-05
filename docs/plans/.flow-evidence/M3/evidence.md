# M3 证据包（sdk-p2-plan）

> 审计输入：本文件 + 方案 `docs/plans/sdk-p2-plan.md` M3 节 + 矩阵主表 `docs/scenario-matrix.md`（§2 装配片段 + §3 主表）+ 工作区差异。
> 实施人：Executor。抽查人：主控。

## 1. 差异范围（边界内 10 文件）

`experiments/scenario_checks/`：`__init__.py`（空）+ `_common.py`（403 行装配唯一出处）+ 8 格脚本（各 29-32 行 ≤60：格 id docstring → require_env → runner → check_grid → exit）。pyproject 零改动。

## 2. 验收清单（方案 M3 节 3 条）

| # | 标准 | 自报 | 证据 |
|---|---|---|---|
| 1 | 每脚本独立可跑 | PASS | 8 脚本 `python -m` 加载执行至 require_env；首轮 GridSpec 缺字段 TypeError 修复后 8/8 绿（checks.log） |
| 2 | 缺依赖 exit 4 指路含变量名 | PASS | 8/8 exit 4；主控抽查 `full_stack_zh_skill` 复跑复现（stderr 缺失清单 + REDIS_URL 示例值 + 嵌入网关口径指路） |
| 3 | 断言面非空；_common.py 装配唯一出处 | PASS | check_grid：redis 4 项 / full_stack 7 项组件存在性字段全真 + hits 非空 + top-1 存在；装配逻辑 100% 收 _common.py |

## 3. 主控抽查（可采信标注）

- 独立复跑 `full_stack_zh_skill` 缺配置路径：exit 4 + 指路文本（含本机 Redis 示例命令）复现 ✅。
- ruff 全过；10 文件恰为边界内容。
- 偏差 1 核实：`DocGenConfig`/`build_docgen` 确不在顶层 `_LAZY_EXPORTS`/`__all__`（`agent_retrieval.expansion` 是公开出口）——matrix §2.4 片段 import 行有误，**M4 回写时修正文档**（Executor 建议采纳）。

## 4. Executor 偏差（主控裁决）

1. matrix §2.4 import 行错误 → **M4 修正文档**（已列 M4 待办）。
2. 快照 meta dim 写实际向量维度（非 config.dimensions）→ 采纳（端点自报维度为 0 时无信息量）。
3. MockEmbedder 分支不可达（嵌入变量必需）→ 采纳（live 语义一致；断言面 S-D4 不卡质量）。
4. top-1 收窄为「存在」+ gold 观察项 → **裁决采纳**（S-D4 组件存在性口径；gold 硬断言会混入质量红线违反职责分离）。
5. jev 语料 owner_version 常量 `jev-cases-v1` → 采纳（宿主版本标签）。
6. 不读 .env（保 `env -u` 证据可复现性）→ 采纳（M4 主控注入进程环境变量）。

## 5. 建议抽查命令

```bash
env -u REDIS_URL -u AGENT_RETRIEVAL_EMBEDDING_BASE_URL -u AGENT_RETRIEVAL_EMBEDDING_API_KEY -u AGENT_RETRIEVAL_EMBEDDING_MODEL -u AGENT_RETRIEVAL_E2E_BASE_URL -u AGENT_RETRIEVAL_E2E_API_KEY -u AGENT_RETRIEVAL_E2E_MODEL .venv/Scripts/python.exe -m experiments.scenario_checks.redis_snapshot_en_tool
.venv/Scripts/python.exe -m ruff check experiments/scenario_checks/
grep -n "DocGenConfig" docs/scenario-matrix.md | head -3
```
