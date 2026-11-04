# C1 证据包（sdk-p1-plan）

> 审计输入：本文件 + 方案 `docs/plans/sdk-p1-plan.md` C1 节（唯一需求出处）+ 工作区实际差异。
> 实施人：Executor（TDD 垂直切片）。抽查人：主控（复跑通过后标注可采信）。

## 1. 差异范围（边界内 4 文件）

| 文件 | 类型 | 内容 |
|---|---|---|
| `src/agent_retrieval/core/projection.py` | 新建 | `CorpusProjection`（frozen dataclass：`fields`/`separator`，`__post_init__` 校验）+ `to_callable()`（getattr 取值、`if part` 过滤、join）；docstring 含双形态条款 |
| `src/agent_retrieval/__init__.py` | 修改 | ports 导入块后加单行 `from agent_retrieval.core.projection import CorpusProjection`；`__all__` 大写区插 `"CorpusProjection"`（"CandidateHit" 后） |
| `tests/test_projection.py` | 新建 | 12 项测试（3 构造校验 + 8 投影行为 + 1 rank_candidates 组合），真值为测试内手写 `_plain_text` |
| `tests/test_core_zero_dep.py` | 修改（主控 T0） | `CORE_MODULES` 补 `"agent_retrieval.core.projection"`（Executor 范围外发现：新模块未被零依赖守卫扫描；守卫清单完整性属本批契约面） |

内核文件（`ranking.py`/`resource_index.py`）零改动——方案 C1 评审要点第一项。

## 2. 验收清单与测试映射（方案 C1 节 6 条）

| # | 验收标准 | Executor 自报 | 测试 |
|---|---|---|---|
| 1 | 四字段投影与 `gate._plain_text` 逐位一致 | PASS | `test_four_field_projection_matches_plain_text_truth_bitwise`（5 种资源形态：完整/空 tags/空 description/None description/全空） |
| 2 | 三字段组合可构造 | PASS | `test_three_field_combination_constructs` |
| 3 | 空 description 无连续空格；全空 → 空串不抛 | PASS | `test_empty_description_produces_no_double_space` + `test_all_empty_fields_return_empty_string` |
| 4 | 非法构造（空元组/含空串）ValueError | PASS | `test_empty_fields_raises` + `test_empty_string_field_raises` |
| 5 | 经 `rank_candidates` 的 hits 与手写投影逐位一致 | PASS | `test_hits_match_handwritten_projection_bitwise`（item 序列 + 全部注解字段展平比较） |
| 6 | `fields` 可 JSON 序列化（D5） | PASS | `test_fields_are_json_serializable` |

附加：frozen / 未知字段 AttributeError / 非默认 separator 各有专测。

## 3. Executor 命令回报摘要

- TDD 六切片（①红 ImportError → ①绿 3 passed → ②红 7F4P → ②绿 11 passed → ③红 1F → ③绿 12 passed）——红绿循环全程留痕。
- 定向收口：`pytest tests/test_projection.py tests/test_core_zero_dep.py tests/test_ranking.py -q` → **18 passed**，PIPESTATUS[0]=0。
- ruff 三改动文件 → All checks passed。
- `git status --short` 恰为边界内文件 + 证据目录。

## 4. 主控抽查（可采信标注）

- 复跑定向映射：**18 passed in 0.10s**（主控独立执行，与 Executor 回报一致）→ Executor 测试证据**可采信**。
- diff 抽查：`__init__.py` 改动恰两处（导入 + `__all__`），无无关格式化。
- Pyright 中间态诊断甄别：`Callable`/`TypeVar` 未定义、`description` 缺参均为编辑过程陈旧快照，当前文件已核实无此问题；`__all__` DunderAll 告警为 lazy exports 既有模式噪音。

## 5. TDD 过程修正声明（Executor 自报，主控核可）

- 组合测试初版误断言「全量召回 3 hits」，修正为 top-1 强命中——依据 `ranking.py` 既有契约（BM25 零分候选不进 hits），非实现缺陷。
- 两个用例初版漏传 Resource 必填参数（TypeError），补参修正。

## 6. 建议抽查命令

```bash
.venv/Scripts/python.exe -m pytest tests/test_projection.py tests/test_core_zero_dep.py tests/test_ranking.py -q
.venv/Scripts/python.exe -m ruff check src/agent_retrieval/core/projection.py tests/test_projection.py
git diff src/agent_retrieval/__init__.py tests/test_core_zero_dep.py
```
