# B1 执行证据（Judge 端口 + 判停辅助）

Executor 回执要点（完整内容见会话记录；本文件为主控代写归档）。

## 改动文件（边界内）

1. `src/agent_retrieval/core/ports.py`——文件尾追加 `JudgedCandidate`（frozen 五字段）、`Judgment`（frozen，tokens 可空）、`Judge` Protocol（runtime_checkable，`judge(query, candidates, *, instruction="")`）、`select_with_judge(hits, judgment, *, minimum_probability, item_id)`；导入区增 `Callable`/`TYPE_CHECKING`/`TypeVar`。
2. `src/agent_retrieval/__init__.py`——直导入区与 `__all__` 增四名（stdlib-only，未走 lazy exports）。
3. `tests/test_judge_port.py`——新增 12 用例。

## 关键实现决策

- **TYPE_CHECKING 防环**：ports 与 ranking 互相 import；`select_with_judge` 的 `CandidateHit` 注解走 `if TYPE_CHECKING`（`from __future__ import annotations` 下运行期不求值）。双向导入顺序已验证。
- 一次测试侧修正（实现零改动）：组合用例初版候选语料与查询零重叠，被 fuse 零分契约滤除——测试前提错，修正候选语料后全绿。

## 命令与退出码

| 命令 | 退出码 | 结果 |
|---|---|---|
| `pytest tests/test_judge_port.py -q`（实现前，TDD 红） | 非0 | ImportError: cannot import name 'Judge' |
| `pytest tests/test_judge_port.py -q`（绿） | 0 | 12 passed |
| `pytest tests/test_ranking.py tests/test_core_zero_dep.py -q` | 0 | 6 passed（零依赖守卫仍绿） |
| `pytest tests/test_judge_port.py tests/test_ranking.py tests/test_core_zero_dep.py -q -v` | 0 | 18 passed（完整日志 pytest.log） |
| `ruff check src/ tests/test_judge_port.py` | 0 | All checks passed! |
| fresh 进程先 `import agent_retrieval.core.ranking` | 0 | 循环导入验证 OK |

## 验收清单

exact 编排判停（计数 0）✓ / 并列保序 ✓ / abstain ✓ / 缺键 ✓ / 1.2 越界 ✓ / NaN ✓ / [a,a] 重复 ✓ / 空 shortlist ✓ / 真实 rank_candidates 组合 ✓ / instruction 透传 ✓ / isinstance ✓ / minimum_probability 漏传 TypeError ✓——12/12 PASS。
