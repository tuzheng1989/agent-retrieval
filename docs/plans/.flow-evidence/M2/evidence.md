# M2 证据包（sdk-p2-plan）

> 审计输入：本文件 + 方案 `docs/plans/sdk-p2-plan.md` M2 节 + 矩阵主表 `docs/scenario-matrix.md`（M1 收口，格清单权威载体）+ 工作区差异。
> 实施人：Executor（TDD 按姿态两轮红绿）。抽查人：主控。

## 1. 差异范围

`tests/test_scenario_matrix.py`（216 行，新建）——离线 8 格参数化（格 id 即测试名组成部分，与主表 `-k` 验证命令一一对应）。

## 2. 验收清单（方案 M2 节 3 条）

| # | 标准 | 自报 | 证据 |
|---|---|---|---|
| 1 | 8 格全绿 | PASS | 全文件 8 passed；逐格 `-k` 抽查 8 条命令各 `1 selected / 7 deselected` PASSED |
| 2 | 失败报错含格 id | PASS | 断言消息 `[<格 id>]` 前缀；红态实证（Round 2 守护咬合，四格各自定位） |
| 3 | core-only 零 extras 导入 | PASS | 4 格断言 sys.modules 无 embedders/stores/run/expansion/judges 前缀；Round 2 红态实际咬合（EmbeddingConfig 初稿顶层导入 → 4 格全红 → 修复为惰性导入） |

## 3. Executor 命令回报摘要

8 格全绿 + 逐格 -k 抽查 8 条 + 回归 41 passed + ruff 全过。TDD 按姿态两轮红绿（委派单允许），红态落盘。

## 4. 主控抽查（可采信标注）

- 独立复跑：全文件 8 passed；`-k core_only_en_tool` / `-k api_embed_zh_skill` 两条抽查 `1 selected / 7 deselected`（选中性机械证据）。
- 边界：恰为 test_scenario_matrix.py 新建。
- Pyright 告警甄别：EmbeddingConfig "not defined" 为 TYPE_CHECKING 惰性导入设计（core_only 零 extras 纪律的实现手段）；could not be resolved 为陈旧快照模式。

## 5. Executor 偏差（主控核可）

1. 红绿按姿态两轮非逐格 8 轮——委派单允许，采纳。
2. 池规模断言不重复（E1 已钉）——职责分离，采纳。
3. MockEmbedder 校准发现（en-skill cosine 高于标靶但 fused top-1 锚 BM25）→ 印证 S-D4 断言纪律（top-1 锚 BM25、cosine 只断参与），已入 docstring。

## 6. 建议抽查命令

```bash
.venv/Scripts/python.exe -m pytest tests/test_scenario_matrix.py -q
.venv/Scripts/python.exe -m pytest tests/test_scenario_matrix.py -k core_only_zh_skill -v
.venv/Scripts/python.exe -m ruff check tests/test_scenario_matrix.py
```
