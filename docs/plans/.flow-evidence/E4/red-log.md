# E4 红态摘要

日期：2026-11-05 · 执行：executor（E4 批次）

## 红态 1：步进适配冒烟 —— ToolLimitError 炸穿 graph（1 failed, 12 passed）

命令：`.venv/Scripts/python.exe -m pytest tests/test_e2e_agent.py -q`（第一轮）
退出码：1（pytest）

- 失败用例：`test_smoke_step_adapter_recursion_exhaustion_maps_budget`
- 关键错误：`experiments.agent_e2e.tools.ToolLimitError: 搜索次数超限：每 episode 最多 2 次（契约 smoke-v1）`，
  抛出点 `langgraph/prebuilt/tool_node.py:399 _default_handle_tool_errors → raise e`
- 根因：langgraph ToolNode 对非预期异常（非 ToolException 家族）直接 re-raise，
  不转 ToolMessage——`ToolLimitError`（RuntimeError 子类）从工具闭包炸穿 graph，
  `GraphRecursionError` 转译通道根本没机会触发；runner 的工具异常 try/except
  只包 runner 自己的 `tools[...](...)` 调用，包不到 langgraph ToolNode 内的工具执行
- 修复（不动 E1-E3 文件，改动限于 smoke.py 适配层）：`agent_step` 内增捕
  `ToolLimitError` → `_SearchBudgetExhausted` → `run_case` 构造
  `Trace(error="ToolLimitError", exhausted=False)`——与 runner 同形状，
  scorer 按既有子串通道映射 `budget_exhausted`（契约第三节语义：搜索超限
  终止 episode + 归因 budget_exhausted）；原测试改为 `recursion_limit=2`
  真正触发 GraphRecursionError，另增
  `test_smoke_step_adapter_search_limit_maps_tool_limit_error` 覆盖新转译路径
- 回归：`pytest tests/test_e2e_agent.py -q` → 14 passed（EXIT=0）

## 红态 2：ruff F401 × 3（tests/test_e2e_agent.py 未使用导入）

命令：`ruff check experiments/agent_e2e/agent.py experiments/agent_e2e/smoke.py tests/test_e2e_agent.py`
退出码：1（ruff 本体；管道显示值被 tee 掩盖）

- `GraphRecursionError` / `run_episode` / `score_episode` 三处 F401（初版测试
  拟直连 runner/scorer，后统一走 `run_case` 适配层）
- 修复：`ruff check --fix`（3 fixed, 0 remaining）；复跑 pytest 仍 14 passed

## revision 2：live exit 3 修复后的离线红态（1 failed, 15 passed）

背景：主控 live 冒烟 exit 3，根因探针坐实——pydantic v2 对 `@tool` 装饰函数的参数名
`args` 有字段 mangle（实测 schema 字段变 `v__args`），LLM 按 schema 发参 → 闭包
`args` 不识别 → TypeError。离线闭环（decline 用例）不触发 tool call 路径，未抓到
（委派单盲区，主控承认）。

本侧执行主控三项修复：

1. `agent.py _wrap_tools`：invoke_resource 暴露名改 `tool_args`，内部转发原闭包
   （tools.py 零改动）；名字/docstring 仍继承闭包（冻结面同源）。
   - 探针先行复现：`langchain_tool(闭包)` → schema props `['id', 'v__args']` ✓；
     修复后 `['id', 'tool_args']` + 调用转发 OK。
2. `contract.md` 第二节注记（主控授权，语义不变）：tool_args 与契约 args 语义一致。
3. `tests/test_e2e_agent.py` 补 tool call 主路径两条（`_ScriptedChatModel` 脚本化
   假模型：search → invoke tool call → 文本）：
   - 001 多步：ToolNode 真调工具 → diff 恰两条新增、事件面（2×search +
     event_exists + email_sent）、schema 参数名断言（mangle 修复回归面）、
     required_state 全满足 → success=True；
   - 011 故障：.invalid 命中 failure_modes → 零状态变更 + failure 事件 +
     按预期报告失败算完成（complete 第三子条全链路）。

红态：`test_smoke_tool_call_main_path_invokes_tools_and_scores_success` 首轮挂——
单轮 search("email send") 的 top-5 未展示 calendar-create，第二轮 invoke 被拒
（`资源未展示` ValueError，shown 集合机制正常工作，脚本未先展示目标资源）。
修复：脚本改两轮 search（各自展示 calendar-create / email-send，恰好用满
契约 max_searches=2）。复跑 16 passed。

收尾（revision 2 日志已追加 E4/pytest.log）：
- 定向回归 7 文件 → **130 passed**（E4 16 + E1-E3 114），exit=0
- ruff 三文件 → All checks passed, exit=0

## revision 3：live 9/12 三条失败定性修复（无离线红态，一次全绿）

背景：live 第二跑 9/12（exit 1），管线全通（12 Trace + report 落盘），主控三条定性：

1. **失败 1（判定面缺陷 → 修 scorer）**：008 Agent 中文列出缺失项（日期/时段），
   评分器只找英文字段名判败。修复：scorer.py 增 `CLARIFY_KEYWORD_ALIASES`
   冻结别名组（8 字段），clarify 判定改组内或/组间且（契约「每个字段」口径不变，
   词表外字段回落仅字段名原文）；contract.md 第四节补口径注记。
2. **失败 2（用例设计缺陷 → 修 E0 数据）**：001/005 中文请求 → 中文 query → 英文
   声明面 BM25 无匹配 → Agent 正确拒绝（行为无错，判败符合语义；005 另暴露
   Agent 自选不存在 kind="dataset"）。修复：两条 user_request 改中英混合
   （保留全部参数断言面——001 的 Atlas/[Mei,Chen]/2026-11-12/afternoon/
   wang@corp.example.com/'Atlas 会议通知'，005 的 'Q3 销售季度报告'/
   quarterly-sales-2026 + 顺带 document index 引导 document_updated 效果）；
   005 required_state 谓词字段与 invoke_effect 一致性自检通过（report_exists
   topic=<args.topic> ✓，document_updated 无 doc_id vs generated:report-doc-id ✓）；
   跨语言检索留给场景矩阵的 docgen/向量路格（冒烟不做，主控裁决）。
3. **失败 3**：见主控回报（非本批次改动面）。

测试：test_e2e_scoring.py +3（冻结词表机械对照 / 中文澄清成功 / 部分中文命中
仍败——现有英文关键词测试照旧过，组内含英文原文不破坏旧口径）；
test_e2e_agent.py +1（008 中文澄清步进闭环：假模型中文澄清 → run_case →
success=True + 零副作用）。

收尾（revision 3 日志已追加 E4/pytest.log）：
- JSONL 自检：12 条干净加载、id 唯一、001/005 required_state 断言面未动，exit=0
- 定向回归 7 文件 → **134 passed**（E4 17 + E1-E3 117），exit=0
- ruff 五文件（含 scorer.py 与 test_e2e_scoring.py 本轮改动面）→ All checks passed, exit=0
