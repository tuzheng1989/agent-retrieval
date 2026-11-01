# B5 执行与主控验证证据（stats 收编）

## 改动文件（边界内）

1. `src/agent_retrieval/stats.py`（新建）——`paired_bootstrap`（源 26-51 逐字平移，种子 20261022）/`load_arm_cases`（源 73-77）/`paired_report`（源 80-117，五参签名不变）；纯标准库；不收 chance_baseline/hit_rate_report/power_report/CLI。
2. `experiments/benchmarks/ablation_stats.py`——三函数改薄壳转发（`# noqa: F401` 保既有导入路径契约），`import random` 随下沉移除；其余原样。`git status` 核验 `tests/test_ablation_stats.py` 零改动。
3. `tests/test_stats_pkg.py`（新建）——种子逐位回归（期望值 = 改动前旧实现实跑记录，精确相等断言）+ 包侧端到端（共享/缺失警告/不交 ValueError）。

## Executor 命令记录

| 命令 | 退出码 | 结果 |
|---|---|---|
| `pytest tests/test_stats_pkg.py tests/test_ablation_stats.py -v` | 0 | 18 passed |
| `ruff check stats.py ablation_stats.py test_stats_pkg.py` | 0 | All checks passed! |
| `ablation_stats --help` | 0 | 薄壳 CLI 完好 |
| CLI pair 冒烟（真实 runner 报告 `jev-rerank-skillret-300-w20.json`） | 0 | 完整 JSON（cli_smoke.json 留档）；原委派命令对 golden fixture 不可行系该 fixture 为构建语料而非报告形状（既有行为，非平移引入） |

## 主控机械验证（T1 收口，独立复跑）

| 命令 | 退出码 | 结果 |
|---|---|---|
| `pytest tests/test_stats_pkg.py tests/test_ablation_stats.py -q` | 0 | 18 passed |
| 独立复算 `paired_bootstrap([0.1, -0.1, 0.3])` | 0 | `(0.09999999999999999, -0.10000000000000002, 0.3)` 与冻结字面量逐位一致 |
| CLI `pair --a jev-rerank-skillret-300-w20.json --arm-a baseline --arm-b jev --metrics mrr` | 0 | 正常产出配对 JSON |
| `ruff check` 三文件 | 0 | All checks passed! |

附查：Pyright 报 `ablation_stats.py:43 random 未定义`——grep 证实全文仅 :35 docstring 内 "uniformly random groups" 字样，无代码引用，误报。

## 验收结论

三条验收全 PASS（种子逐位 / 端到端含缺失警告 / 既有测试零改动全绿）。T1 主控机械验证通过，无审计代理（定级流程如此）。
