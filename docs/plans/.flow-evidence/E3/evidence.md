# E3 证据包（e2e-phase01）

> 审计输入：本文件 + 方案 `docs/plans/e2e-phase01-plan.md` E3 节 + 契约 `experiments/agent_e2e/contract.md` 第四节（成功规则唯一权威）+ 工作区差异。
> 实施人：Executor（TDD，红态落盘；红态根因单一：测试助手键位换算错误，被测模块零改动）。抽查人：主控（复跑通过）。

## 1. 差异范围（边界内 5 文件）

| 文件 | 内容 |
|---|---|
| `experiments/agent_e2e/scorer.py` | EpisodeResult frozen + ATTRIBUTION_LABELS（10 项冻结）+ check_required_state（只对 added 判定、document_updated 按 added 匹配不假设替换）/check_forbidden/score_episode（评分顺序 forbidden→预算→required→终止行为）|
| `experiments/agent_e2e/runner.py` | Trace frozen + run_episode 通用循环（tools_builder/agent_step 注入、LLM 解耦；ToolLimitError 按类名识别终止、其他工具异常进 step 记录续跑）|
| `experiments/agent_e2e/report.py` | summarize（分项/归因分布/逐例表）+ write_report |
| `tests/test_e2e_scoring.py` | 21 条（谓词正反例/forbidden 优先/五类口径/「回答正确但没执行」反例/document_updated 追加语义） |
| `tests/test_e2e_runner.py` | 7 条（重放逐位一致/步数耗尽/ToolLimitError/final 提前终止/异常续跑 + L2 离线编排 2 例用替身工具不经 E2） |

## 2. 验收清单（方案 E3 节 5 条）

Executor 自报全 PASS（测试名见其回报）：反例 state_mismatch / forbidden 优先 / 重放逐位一致 / 步数耗尽 budget_exhausted / summarize 12 条混合计数。

## 3. Executor 命令回报

红态（11 failed，测试助手键位换算错，被测零改动）→ 绿态 **74 passed**（新增 28 + E1 回归 46）→ ruff 五文件全过。

## 4. 主控抽查与裁决（可采信标注）

- 独立复跑：**74 passed in 0.30s** → 证据可采信。
- **clarify 关键词口径裁决**：维持契约「每个字段名关键词」（契约唯一权威；委派单内联「之一」为笔误）——Executor 按契约实现正确，测试钉住。
- 以下偏差采纳记档：① runner 逐 step 事件 seq 未实现（委派单签名不暴露沙箱；事件面由 scorer 经 events() 全量消费，评分与重放不受影响——跨批接缝不值当）；② ToolLimitError 按类名识别不导入 E2（W3 并行的正确解法，E4 集成天然对齐）；③ forbidden 归因 = ATTRIBUTION_LABELS ∪ 命中事件类型（主控裁决照录，docstring 载明）；④ expected_failure 三条件（比委派单多「零副作用变更」——契约 complete 第三子条合理展开）；⑤ 终止行为失败无冻结归因不造标签（failure_reasons 照实记录）。
- Pyright test_e2e_runner 容器属性告警：测试类型推断宽度问题，运行时正确、ruff 过，不阻塞。

## 5. 建议抽查命令

```bash
.venv/Scripts/python.exe -m pytest tests/test_e2e_scoring.py tests/test_e2e_runner.py tests/test_e2e_loading.py tests/test_e2e_sandbox.py -q
.venv/Scripts/python.exe -m ruff check experiments/agent_e2e/scorer.py experiments/agent_e2e/runner.py experiments/agent_e2e/report.py
```
