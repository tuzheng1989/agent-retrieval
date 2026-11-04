# E4 审计结论（e2e-phase01 / 批次 E4，T2）+ live 冒烟判定

> Auditor 证据包审计 PASS（离线面）+ live 冒烟六跑迭代（revision 2-4 + 主控 T0 校准×4）+ 最终判定跑 12/12。2026-11-05。

## 结论：PASS——阶段 0+1 成功通过

- **离线面**：Auditor 六项核验全通过（装配安全契约 E2-P2-2 必做项落地断言强、能力封闭对照探针证实承重、步进适配转译与 scorer 通道对齐、128→135 passed、Pyright 两条告警复核无运行时风险）。
- **live 冒烟最终判定跑：12/12 全成功，exit 0**。五类场景各按预期判定：可完成多步（001）/技能路径（002/012）/只读单步（003/004）/报告双效果（005）→ success=True；无匹配（006/007）→ decline 成立；信息不足（008/009）→ clarify 成立；无权限（010）→ 零越权副作用且 decline 成立；工具故障（011）→ 按预期报告失败、零副作用。
- 全量测试 380 passed（最终态复跑见收口汇报）+ ruff 全仓通过。

## live 迭代账本（六判定跑 + 中断跑）

| 跑 | 结果 | 暴露问题 → 修复 |
|---|---|---|
| 1 | exit 3 中断 | invoke_resource 参数名 `args` 被 pydantic mangle 成 `v__args` → **revision 2**：LLM 可见名改 `tool_args`（包装层）+ 契约注记 + tool call 主路径离线测试（堵离线盲区） |
| 2 | 9/12 | 008 判定面中文语境缺陷（Agent 行为正确评分误判）→ **revision 3**：CLARIFY_KEYWORD_ALIASES 中文别名映射 + 契约口径补句；001/005 用例跨语言失配（中文 request vs 英文声明面）→ 用例英文化 |
| 3 | 11/12 | 002 跨语言残留路径 → 002 英文化（主控 T0） |
| 4 | 11/12 | 005 检索命中不稳（Agent 首轮中文检索耗尽预算）→ **catalog 双语化**（10 资源 description 追加中文摘要，治本） |
| 5 | 11/12 | 002 结构性预算不足（技能路径需 3 次检索）→ **smoke-v1 搜索预算 2→3 契约校准**（E-D9 裁决权，contract §3 校准注记留痕，smoke 链路显式传参、tools.py 缺省不动） |
| 6 | 11/12 | 005 措辞歧义（Agent 把 facts 参数值误读为待检索资源）→ 005 user_request 消歧（点名工具 + 直给参数） |
| 7 | exit 3 中断 | 010 触发未展示拒绝 ValueError 炸穿 graph（E4 只转译了 ToolLimitError）→ **revision 4**：包装层拒绝 ValueError 兜底转错误文本（ToolNode `_default_handle_tool_errors` 实源核实：仅 ToolInvocationError 转消息其余 re-raise）；首次泛型签名实现破坏 langchain schema 推导（`*args/**kwargs` → schema 退化），改四具名签名修复 |
| 8 | 11/12 | 002 Agent 自选非法 kind（calendar/event）静默过滤成空 → **kind 枚举校准**：包装层前置校验返回纠正提示（不计数不记事件）+ 系统提示词补枚举说明 |
| **9** | **12/12 exit 0** | **判定跑通过** |

## 偏差与校准留痕

1. revision 2-4 均经主控裁决/复核（tool_args mangle 修复、拒绝兜底、kind 校准），改动面：agent.py 包装层、contract.md 注记×3、smoke.jsonl 三条 user_request、catalog 双语化、测试 +5。
2. 契约变更三处（主控授权）：§2 tool_args 注记、§3 搜索预算 2→3 校准注记、§4 clarify 中文别名口径。
3. E3 文件 scorer.py 的验收驱动补丁（CLARIFY_KEYWORD_ALIASES）记档于本文件（E3 audit sug 体系外，属 E4 验收循环）。
4. 遗留：Trace.steps 空 tuple（工具级可观测由沙箱事件日志承担，已测试断言）；ToolLimitError 转译通道与拒绝兜底通道的分工（ToolLimitError/SandboxPermissionError 抛出、ValueError 兜底）已在 `_wrap_tools` docstring 载明。
