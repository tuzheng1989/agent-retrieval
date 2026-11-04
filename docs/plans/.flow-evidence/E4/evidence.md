# E4 证据包（e2e-phase01）

> 审计输入：本文件 + 方案 `docs/plans/e2e-phase01-plan.md` E4 节 + 契约 `experiments/agent_e2e/contract.md` + 工作区差异。
> 实施人：Executor（含 deepagents 实源核对 7 项，实施前实跑）。抽查人：主控。

## 1. 差异范围（边界内 3 文件）

| 文件 | 内容 |
|---|---|
| `experiments/agent_e2e/agent.py` | build_model（E2E_* 三变量缺失指路）/ AGENT_SYSTEM_PROMPT 冻结 / **allowed_ids_for（E2 审计 P2-2 必做：权限→可见资源集，只读恒可见、写资源全权限才可见）** / build_episode（子 catalog 装配 + case_clock 必传）/ build_reference_agent（HarnessProfile 排除 9 内置工具 + 四契约工具） |
| `experiments/agent_e2e/smoke.py` | live-only 入口：退出码 0/1/2/3/4；单 invoke 步进适配（recursion_limit=20 预算）；outputs/ 落盘 |
| `tests/test_e2e_agent.py` | 14 条全离线（缺配置指路 / allowed_ids_for / **010 装配断言：无 calendar.write 则 search 不可见 calendar-create + case_clock 非空** / checkpointer 与闭包双新建 / 工具名恰四名 / 假模型全链路闭环） |

## 2. deepagents 实源核对（Executor 实施前实跑，7 项）

create_deep_agent 签名（keyword-only backend/checkpointer）/ StateBackend 无参 / GraphRecursionError ← langgraph.errors（pregel 路径不可用）/ InMemorySaver + tool 装饰器 OK / **默认工具注入行为：`tools=` 是叠加非替换——默认注入 8 内置工具 + general-purpose subagent**（方案未覆盖的实源行为）/ HarnessProfile 排除后恰四名 / recursion_limit 触发实证。

## 3. 验收与偏差

- 离线验收 3 条全 PASS（测试名见 Executor 回报）；**128 passed**（E4 新增 14 + E1-E3 全回归）+ ruff 全过 + 缺配置 exit 4 真实验证（主控复跑）。
- 五条偏差（详见 Executor 回报）：① build_reference_agent 加 tools 参数（E-D7 每 episode 新建闭包的入参通道）；② HarnessProfile 排除内置工具（deepagents beta API，版本锁定 0.7.32，升级复核点）；③ smoke 增捕 ToolNode 炸穿的 ToolLimitError 转译（红态修复，E1-E3 零改动）；④ run_case 加 recursion_limit 参数（离线测试注小预算）；⑤ 导入 deepagents._models.get_model_provider 私有模块（provider 键推导唯一入口，升级复核点）。
- 步进适配：单 invoke + recursion_limit（12+8 缓冲）；Trace.steps 空 tuple，工具级可观测由沙箱事件日志承担（含 over_limit 超限事件，测试断言）。

## 4. 主控抽查（可采信标注）

- 独立复跑：**128 passed in 3.19s**；缺配置路径 exit 4 + 三变量名指路清晰（主控亲跑）。
- E2 审计 P2-2 必做项落地确认：allowed_ids_for + 010 装配断言在测试面（test_e2e_agent.py）。
- Pyright 告警甄别：could not be resolved 为陈旧快照模式；SecretStr / register key 两条类型宽度告警运行时无影响（ruff 过、测试过）——留 Auditor 复核。

## 5. 建议抽查命令

```bash
.venv/Scripts/python.exe -m pytest tests/test_e2e_agent.py tests/test_e2e_tools.py tests/test_e2e_scoring.py tests/test_e2e_runner.py tests/test_e2e_loading.py tests/test_e2e_sandbox.py tests/test_e2e_retrieval_adapter.py -q
env -u AGENT_RETRIEVAL_E2E_MODEL -u AGENT_RETRIEVAL_E2E_BASE_URL -u AGENT_RETRIEVAL_E2E_API_KEY .venv/Scripts/python.exe -m experiments.agent_e2e.smoke
grep -n "HarnessProfile\|allowed_ids_for" experiments/agent_e2e/agent.py | head
```
