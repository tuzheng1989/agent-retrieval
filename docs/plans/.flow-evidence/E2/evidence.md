# E2 证据包（e2e-phase01）

> 审计输入：本文件 + 方案 `docs/plans/e2e-phase01-plan.md` E2 节 + 契约 `experiments/agent_e2e/contract.md`（含 §5 failure_modes match 变更）+ 工作区差异。
> 实施人：Executor（TDD，红态落盘）。抽查人：主控（复跑通过）。

## 1. 差异范围（边界内 4 文件 + 主控 T0 一处）

| 文件 | 内容 |
|---|---|
| `experiments/agent_e2e/retrieval_adapter.py` | SearchResult frozen + RetrievalAdapter（CorpusProjection(("id","name","description")) 投影、先过滤后 rank_candidates、evidence 全量 / shown 前 k） |
| `experiments/agent_e2e/tools.py` | ToolLimitError + match_failure_mode 纯函数（endswith/equals/contains 三 op）+ build_tools 四工具闭包（shown 集合/搜索计数/generated 序号闭包持有；权限单一持有在 sandbox；generated:clock ← case_clock 参数、generated:其他 ← {resource_id}-{调用序号} 确定性派生） |
| `tests/test_e2e_retrieval_adapter.py` | 12 条（资格过滤/kind/k 截断/投影对照/空查询） |
| `tests/test_e2e_tools.py` | 28 条（授权四象限/invoke 全链路/generated 确定性/ToolLimitError/保留键） |
| `experiments/agent_e2e/contract.md` | §5 failure_modes 形状变更（主控预授权：人读 condition + 机器可读 match{arg,op,value} + message） |
| `experiments/agent_e2e/catalogs/catalog-v1.json` | **主控 T0**：email-send failure_modes[0] 补 `"match": {"arg":"to","op":"endswith","value":".invalid"}`（E0 存量数据适配契约变更，value 取 .invalid 与 condition「以 .invalid 结尾」语义一致，比 executor 建议的 example.invalid 覆盖面更贴合 condition 描述） |

## 2. 验收清单（方案 E2 节 6 条）

Executor 自报全 PASS（测试名见其回报）：权限外资源全链路不可见 / 未展示拒绝四象限闭合 + SandboxPermissionError 穿透零副作用 / invoke diff 恰含预期 + seq 单调 / failure 命中零状态变更 / CorpusProjection 投影对照逐位一致 / 附加面（L2 编排、ToolLimitError、generated 跨闭包逐位一致）。

## 3. Executor 命令回报

红态（收集期 ModuleNotFoundError）→ 绿态 40 passed → E1 回归连带 86 passed（tools 28 / adapter 12 / loading 26 / sandbox 20）→ ruff 首轮 2 F401 修复后全过。

## 4. 主控抽查（可采信标注）

- 独立复跑 E2+E1 连带：**86 passed in 0.43s**（catalog T0 修改后复跑零回归）。
- `_reject` 必 raise（tools.py:175-177）——Pyright "parsed possibly unbound" 为跨函数 raise 推断局限，非缺陷。
- 契约变更 diff 单行、catalog 修改单元素——无越界。

## 5. Executor 偏差与主控处置

1. catalog 存量缺 match 字段 → **主控 T0 已补**（上表）。
2. 超限搜索事件口径（复用 "search" + "over_limit": True 标记）→ **采纳记档**：E3 评分器 budget 归因不依赖事件计数（来自 trace.exhausted/error），无消费冲突；E4 冒烟与审计注意该标记语义。
3. generated 确定性测试双口径（跨闭包逐位一致 + 同闭包序号递增）→ 采纳（契约 §5 语义的正确展开）。
4. 权限检查先于 args 校验 → 采纳（委派单罗列顺序）。

## 5b. 审计补记（Auditor PASS 后处置）

- **P2-1**：build_tools 返回单一 dict（方案原文 `tuple[已展示集合闭包, dict]`）——已展示集合观测面经事件日志 `search.shown` 达成、E3 无消费需求，功能无损，**记档收窄**（E1 audit P2-2 同性质先例）。
- **P2-2（转 E4 必做项）**：「权限过滤在装配层」最后一公里在 E4——委派单必做：装配时 permissions→allowed 资源集（构造子 catalog 或注入）+ `case.clock`→`case_clock`（禁默认空串静默），各配一条装配断言（权限外资源不进 shown；generated:clock 非空）。
- sug-1（loading 补 failure_modes match 形状校验）/ sug-2（`<args.X>` 引用校验）→ **留档**：冒烟数据固定，阶段 3 数据冻结前随 loading 强化一并处理。
- sug-3（invoke_seq 仅成功路径递增）→ **记档口径**：「同资源成功调用序号」，E4 判定面引用此表述。

## 6. 建议抽查命令

```bash
.venv/Scripts/python.exe -m pytest tests/test_e2e_tools.py tests/test_e2e_retrieval_adapter.py tests/test_e2e_sandbox.py tests/test_e2e_loading.py -q
.venv/Scripts/python.exe -m ruff check experiments/agent_e2e/tools.py experiments/agent_e2e/retrieval_adapter.py
git diff experiments/agent_e2e/contract.md experiments/agent_e2e/catalogs/catalog-v1.json
```
