# e2e 阶段 0+1 实施方案（分批优化计划）

> 依据：`experiments/agent-e2e-evaluation-design.md`（设计稿，2026-09-28 + 2026-11-04 三条修订）§2/§3/§4/§5/§7/§8 阶段 0-2（§4 窗口预算、§7 轨迹字段为 runner/适配器的口径出处）+ `experiments/agent-e2e-evaluation-plan.md`。
> 范围：设计稿 §8 阶段 0（契约冻结）+ 阶段 1（运行底座）+ 阶段 2（冒烟验证，作为阶段 0+1 的验收手段）。A2/A3 组与校准集/留出集不在本方案（阶段 3 起另立，双人标注 HITL 未到位）。
> 宿主纪律不变：包不读环境变量；experiments 是宿主（e2e 运行器读环境变量是宿主权利）。
> 评审状态：定稿（2026-11-05）。自审一轮（auditor 六项机械核查）：无 P1；3 条 P2（ToolLimitError 归属定于工具闭包层 / E3 L2 替身工具口径 / 归因标签集跨阶段冻结面补齐）+ 9 条 sug，已全部修复。经裁决直接定稿开工。
> 实施状态：**已完成并通过**（2026-11-05）。E0-E4 五批逐批流程（T1 主控验证 / T3 审计+Advisor / T2×3 审计）全 PASS；**live 冒烟判定跑 12/12 全成功 exit 0**，五类场景各按预期判定。live 迭代六跑暴露并修复：tool_args mangle（revision 2）、clarify 中文别名 + 用例跨语言适配（revision 3 + catalog 双语化）、smoke-v1 搜索预算 2→3 契约校准、拒绝 ValueError 炸穿 ToolNode 兜底（revision 4）、kind 枚举校准——全部有离线回归测试钉住（135 e2e passed）。证据：`docs/plans/.flow-evidence/{E0,E1,E2,E3,E4}/`（E4/audit.md 含 live 迭代账本）。遗留：A1/A2 臂与校准集属阶段 4；双人标注 HITL 属阶段 3（issue #11）。

## 环境前提（已核实，2026-11-05）

- `deepagents==0.7.32` 已装（`.venv`）：`create_deep_agent(model: str | BaseChatModel | None, tools: Sequence, *, system_prompt, middleware=(), …, backend: BackendProtocol | None, checkpointer: BaseCheckpointSaver | None, …)` 实源核实（中段 subagents/skills/memory/permissions 参数省略）；`deepagents.backends.StateBackend`、`langgraph.checkpoint.memory.InMemorySaver` 可用。
- `langchain-openai` 已装：`ChatOpenAI(base_url, api_key, model)` 可实例化——智谱 coding 网关（`https://open.bigmodel.cn/api/coding/paas/v4`）+ `glm-5.3-flash` 接入（C2 冒烟同网关已验证）。
- 包侧可复用资产：`agent_retrieval.rank_candidates`（`core/ranking.py:57`）、`CorpusProjection`（`core/projection.py`，P1 收编）、`CandidateHit`。Jev/ChatJudge 本方案不接（A1/A2 组属阶段 4 校准后）。

## 设计裁决（前提假设，逐条可挑战）

| # | 裁决 | 理由与取舍 |
|---|---|---|
| E-D1 | **冒烟目录复用 `experiments/jev_cases.json` 的 8 工具资源为雏形，扩执行语义**（`params_schema`/`invoke_effect`/`failure_modes`），另加 2 个 skill 资源（带 `body`）凑齐 tool/skill 两形态 | 设计稿 §2.1 要求 read_skill 有技能正文可读；8 资源的判定语义已被 Law 1 实验验证，声明面稳定 |
| E-D2 | **沙箱是进程内虚拟状态机**（dict 状态 + fixture 快照复制 + append-only 事件日志），不做真实外部系统 | 设计稿 §2.2「外部写入一律用可回放模拟器」；冒烟阶段 12 用例的终态谓词全部可表达为 dict 条件 |
| E-D3 | **检索适配器固定 A0 形态**：资格过滤（权限 + kind）→ `rank_candidates` → 展示窗口 k=5，全量排序证据记录在适配器侧；每 episode 最大搜索次数 2 | 设计稿 §4 冒烟建议 W=20/k=5；冒烟只验管线（A0），Jev 臂 W=20 留阶段 4 |
| E-D4 | **投影用 `CorpusProjection(("id", "name", "description"))`**：catalog-v1 无 tags 字段（jev_cases 雏形资源无 tags；检索语料 = 声明面身份字段） | D7（P1 方案）已确立声明式投影是场景配置；e2e 目录即「那一格」的宿主 |
| E-D5 | **「已展示」集合由工具闭包持有**：Agent 只能 `get_resource_details`/`read_skill`/`invoke_resource` 已 search 展示过的 id；未展示 id 一律拒绝并记事件 | 设计稿 §2.1「只返回已展示、已授权资源的公开规格；不能用 gold ID 绕开搜索」 |
| E-D6 | **评分器确定性优先**：冒烟 12 条全部走确定性终态检查，零 LLM 裁判（裁判环节是留出集开放回答才引入的） | 设计稿 §5「机器无法判定的部分才用盲化 LLM 裁判」；冒烟用例全部可机器判定 |
| E-D7 | **DeepAgents 配置首版固定**：`middleware` 空序列（不启用 subagents/skills 自动加载/memory）、`backend=StateBackend()`、`checkpointer` 每 episode 新建 `InMemorySaver` | 设计稿 §2「未用于真实任务的能力不开放」；冒烟只测「搜索 → 读取 → 应用」路径 |
| E-D8 | **冒烟运行器串行执行**（无并发）：12 用例 × A0 单组 × 1 次重复，逐例落盘轨迹 | 设计稿阶段 2 是工程验证非配对评测（那是阶段 5，每例 3 次）；串行避免沙箱间干扰 |
| E-D9 | **预算 profile `smoke-v1`**：每 episode 最多 12 个 Agent 步、最多 2 次搜索；步数超限由 runner 截断记 `budget_exhausted`；搜索超限由工具闭包抛 `ToolLimitError` 终止 episode 并记同归因 | 数值是冒烟起点（设计稿 §4 预注册精神）；步数上限防 live 失控循环。两类限额的主体不同：步数归 runner（执行循环层），搜索计数归工具闭包（与已展示集合同层持有） |
| E-D10 | **e2e 运行时模型配置独立命名**：`AGENT_RETRIEVAL_E2E_{MODEL,BASE_URL,API_KEY}`，无回落链（被测 Agent 与判定组件语义不同） | C2 冒烟教训的反面应用：环境前提显式报错指路（缺配置 exit 4 同款口径） |

## 路线图

| 批次 | 内容 | T 级 | 依赖 |
|---|---|---|---|
| E0 | 契约冻结文档 + catalog-v1 + 冒烟集 12 用例 + 状态 fixtures | T1 | — |
| E1 | 数据加载校验（loading.py）+ 状态沙箱（sandbox.py） | **T3** | E0 |
| E2 | 受控四工具（tools.py）+ 检索适配器（retrieval_adapter.py） | T2 | E1 |
| E3 | 评分器（scorer.py）+ 运行器（runner.py）+ 汇报（report.py） | T2 | E1 |
| E4 | DeepAgents 参考装配（agent.py）+ 冒烟入口（smoke.py）+ live 冒烟 | T2 | E2、E3 |

**波次**：W1 = E0；W2 = E1；W3 = E2 ∥ E3（文件集互斥：E2 动 tools.py/retrieval_adapter.py，E3 动 scorer.py/runner.py/report.py）；W4 = E4 + 末波收口（全量测试）。
依赖图：E0 → E1 → {E2, E3} → E4，无环。
**红线回归**：本方案包（`src/agent_retrieval`）零改动，各批验收不含包回归；W4 末波收口全量测试覆盖包红线（golden gate 零退化 + 零依赖守卫）作为收口义务。

## 详案

### E0 契约冻结 + catalog-v1 + 冒烟集（T1）

**如何改**（全部新建，无代码改动）：
- `experiments/agent_e2e/__init__.py`（空，包标记）与 `contract.md`：契约文档，六节——① Agent 能力表（四工具能力与约束，照设计稿 §2.1 四条）；② 工具 schema（四工具参数/返回 JSON 形状，与 E2 详案签名同源起草；**contract.md 落盘后即为唯一权威**，E2 实施以落盘契约为准）；③ 预算 profile `smoke-v1`（E-D9 数值）；④ 成功规则（评分顺序照设计稿 §5 + `safe_success` 定义照 §1）；⑤ 目录版本 `catalog-v1`（资源清单语义，照 E-D1）；⑥ 冒烟用例 schema（设计稿 §3 JSON 形状的冒烟子集：必填 id/suite/slice/user_request/clock/catalog_snapshot/initial_state_fixture/permissions/gold_resource_ids/required_state/forbidden_effects/acceptable_behavior/budget_profile，负例字段口径照设计稿 §3 说明）。
- `experiments/agent_e2e/catalogs/catalog-v1.json`：10 资源——8 工具（geo-query/email-send/calendar-create/chart-render/report-write/doc-search/inbox-search/weather-forecast，继承 jev_cases 声明面 + 新增 `kind:"tool"`/`params_schema`/`invoke_effect`/`failure_modes`）+ 2 技能（`meeting-notes-skill`/`data-summary-skill`，`kind:"skill"` + `body` 多段文本）。
- `experiments/agent_e2e/fixtures/world-v1.json`：初始状态世界——`calendar_events`/`emails`/`documents`/`reports`/`sent_messages` 五个 keyed 集合，含少量种子实体（供多步用例操作）。
- `experiments/agent_e2e/cases/smoke.jsonl`：12 条冒烟用例（JSONL 每行一条），五类覆盖——可完成（多步：calendar-create + email-send 组合；单步：geo-query；读技能：meeting-notes-skill）、无匹配（decline）、信息不足（clarify）、无权限（越权即败）、工具故障（invoke 触发 failure_mode）。每条 `required_state` 谓词只用 world-v1 状态 kinds（`event_exists`/`email_sent`/`report_exists`/`document_updated`）；`gold_resource_ids` 仅诊断。

**如何验证**：人工逐条对照 schema 清单（无代码可跑）；用例 id 唯一、fixture 引用存在、gold id ∈ catalog、谓词 kind ∈ world-v1 定义——E1 的加载校验器落地后自动复核（E1 验收含「12 条冒烟用例零错误加载」）。
**验收标准**：
- 契约六节齐全，四工具 schema 签名与 E2 详案逐字一致
- catalog-v1 恰 10 资源、id 唯一、tool/skill 两形态齐备、每工具资源有 params_schema + invoke_effect
- smoke.jsonl 恰 12 条、五类分布 6/2/2/1/1（可完成 6 / 无匹配 2 / 信息不足 2 / 无权限 1 / 工具故障 1——实施勘误：原「五类各 ≥2」与 12 条硬约束矛盾，按明细分布执行）、id 唯一、引用完整（fixture/gold/谓词 kind 全部可解析）
- world-v1 五集合齐备，四种谓词 kind（event_exists/email_sent/report_exists/document_updated）各被 ≥1 条用例消费、合计被 ≥4 条用例消费
**评审要点**：契约是后续四批的需求出处——schema 字段名与 E1 loading 校验器、E2 工具签名、E3 评分谓词必须逐字可对齐（单一出处在本文件与 contract.md，批次详案引用不复制）。
**定级依据**：纯新增文档与数据文件、零代码、无契约邻近——T1。

### E1 数据加载校验 + 状态沙箱（T3）

> 合并依据：`loading.load_world` 的输出即 `Sandbox.from_world` 的输入（同一接缝的两半程），同 stdlib 零依赖 L1 测试面，E0 数据复核闭环依赖 loading——连续信号（合）。

**如何改**：
- 新建 `experiments/agent_e2e/loading.py`：`load_cases(path) -> list[Case]`、`load_catalog(path) -> Catalog`、`load_world(path) -> dict`——frozen dataclass（`Case`/`ResourceSpec`/`Catalog`），schema 校验失败 `ValueError` 指明行号与字段；校验面 = E0 契约第六节清单（id 唯一、枚举合法、引用可解析、谓词 kind 已知）。stdlib-only（json + dataclasses + pathlib）。
- 新建 `experiments/agent_e2e/sandbox.py`：
  - `SandboxPermissionError(RuntimeError)`：越权调用专用异常。
  - `Sandbox`：`from_world(world: dict, permissions: frozenset[str]) -> Sandbox`（深拷贝初始状态——episode 间零共享）；`grant() -> frozenset` 只读；`state(key) -> dict`（只读视图拷贝）；`check_permission(permission: str)`（缺 → 抛 `SandboxPermissionError` 且事件日志记录 `permission_denied`）；`apply(kind: str, payload: dict)`（状态变更唯一入口：写状态 + append 事件 `{"type": kind, ...payload, "seq": n}`）；`events() -> tuple[dict, ...]`（append-only 只读）；`diff_from_initial() -> dict`（初态 vs 当前逐 key 差异——终态检查器的输入）。
  - 事件日志是副作用的唯一观测面（设计稿红线「禁止副作用可被检出」的机制前提）。
- 新增 `tests/test_e2e_loading.py`：合法 12 用例全过；id 重复、未知 fixture 引用、未知 gold、未知谓词 kind、缺必填字段各抛 `ValueError` 且含行号。
- 新增 `tests/test_e2e_sandbox.py`：同 fixture 两次 `from_world` 状态互不干扰（ episode 隔离）；`apply` 后 `diff_from_initial` 恰含变更 key；`events` 序号单调；越权抛专用异常且事件含 `permission_denied`；权限内调用不抛；`state()` 返回拷贝（外部改不影响沙箱）。

**如何验证**：L1 全单元（替身零依赖——loading/sandbox 纯 stdlib）；E0 数据自动复核（12 用例零错误加载即「评测作者依据契约生成用例」证据）。
**验收标准**：
- `load_cases("experiments/agent_e2e/cases/smoke.jsonl")` 返回 12 条 `Case`；篡改任意一行（复制一条改 id 重复）→ `ValueError` 含行号
- catalog 10 资源加载为 `Catalog`；skill 资源 `body` 非空
- 两个 `Sandbox`（同 world）分别 `apply` 不同 kind → 互不可见、`diff_from_initial` 各自正确
- 无权限集合调 `check_permission("calendar.write")` → `SandboxPermissionError` + 事件记录；有权限 → 静默
- 事件 seq 从 1 单调递增；`events()` 修改返回值不影响沙箱内部
**评审要点**：深拷贝隔离是 episode 可重放的根基（重置 = 重新 from_world）；事件日志 append-only 不给删除口；状态变更只有 `apply` 一个入口。
**定级依据**：e2e 结论正确性的根基面——episode 隔离、重放一致、副作用可观测是设计稿 §8 阶段 1 通过条件本体，根基缺陷直接污染后续全部实验结论；全新路径无外部参照。E2/E3/E4 虽同为全新文件但属消费层（错误可被 E1 根基面 + 各自 L1/L2 定位修复，爆炸半径限于单文件），仓库先例（sdk-p0-plan B1/B2/B4 全新文件均 T2）同口径——E1 差异化定 T3、后三批 T2。宿主能力映射：T3 = Executor → Auditor（证据包）→ **Advisor 独立意见**（Claude Code 宿主无 Codex 第二意见角色，Advisor 独立上下文替代，意见原文落盘证据目录）。

### E2 受控四工具 + 检索适配器（T2）

**如何改**：
- 新建 `experiments/agent_e2e/retrieval_adapter.py`：`RetrievalAdapter`（构造持 catalog + `CorpusProjection(("id","name","description"))`（E-D4，P1 组件））；`search(query: str, allowed_ids: frozenset[str], kind: str | None, k: int = 5) -> SearchResult`——先资格过滤（allowed_ids ∩ kind 匹配）再 `rank_candidates`（全量排序），返回 `SearchResult(shown: list[ResourceSpec]（前 k）, evidence: dict（全量 id 序 + 分数 + matched_terms）)`。权限对检索不可见者不进候选（出处：`src/agent_retrieval/core/ranking.py` 模块 docstring「候选集由调用方先过滤后融合」纪律，设计稿 §2.1 同语义表述）。
- 新建 `experiments/agent_e2e/tools.py`：`ToolBundle`——`build_tools(sandbox, adapter) -> tuple[已展示集合闭包, dict[str, Callable]]`，四工具签名照契约第二节：
  - `search_resources(query: str, kind: str | None = None) -> list[dict]`（展示字段 id/name/declared_description；结果写入已展示集合；每次搜索事件 `search`；每 episode 搜索上限的**计数与限额都由工具闭包持有**（`build_tools(..., max_searches: int = 2)`，与已展示集合同层）——超限抛 `ToolLimitError`（`tools.py` 定义），步进回调捕获后 episode 终止、`trace.error` 记录、评分器映射 `budget_exhausted` 归因）
  - `get_resource_details(id: str) -> dict`（未展示 → 拒绝事件 + `ValueError`；返回 params_schema 公开规格）
  - `read_skill(id: str) -> str`（未展示拒绝；仅 kind="skill"；返回 body）
  - `invoke_resource(id: str, args: str) -> str`（未展示/无权限/schema 不符拒绝；按 `invoke_effect` 调 `sandbox.apply`；命中 `failure_modes` 条件时返回失败并记事件；返回人类可读结果文本）
  - 工具函数 docstring 用中文写清行为契约（DeepAgents 会把 docstring 作为工具描述喂给模型——这是工具描述冻结面，契约第二节与其逐字一致）。
- 新增 `tests/test_e2e_tools.py` + `tests/test_e2e_retrieval_adapter.py`。
**如何验证**：L1 单元（adapter 资格过滤/窗口/证据字段；四工具授权矩阵——已展示/未展示 × 有权限/无权限 四象限；invoke_effect 状态变更与事件；failure_mode 触发返回失败文本；搜索上限抛 `ToolLimitError`）；L2 编排（adapter+tools+sandbox 装配后模拟一次完整「搜索 → 详情 → 调用」序列断言状态与事件）。
**验收标准**：
- 权限外资源从不出现于 search 结果（构造一条 allowed_ids 排除场景断言全量 evidence 也不含）
- 未展示 id 调 `get_resource_details`/`read_skill`/`invoke_resource` → 拒绝 + 事件；已展示 + 无权限 invoke → `SandboxPermissionError` 穿透（工具不吞）
- invoke 合法调用后 `diff_from_initial` 恰含预期变更、事件 seq 递增
- failure_mode 命中 → 返回含失败说明的文本、状态零变更、事件含 failure
- `rank_candidates` 消费 `CorpusProjection` 投影（替换手写函数 hits 逐位一致的对照测试）
**评审要点**：工具不读 gold 标签；检索全量证据进 evidence 不进展示；docstring 是冻结工具描述面。
**定级依据**：多文件新增、消费 P1 既有组件（`rank_candidates`/`CorpusProjection` 有既有测试覆盖）、不触碰包内核——T2。

### E3 评分器 + 运行器 + 汇报（T2）

**如何改**：
- 新建 `experiments/agent_e2e/scorer.py`：
  - `check_required_state(required: list[dict], diff: dict, events: tuple) -> list[str]`——返回未满足谓词列表；谓词 kind 分派到 world-v1 状态 kinds 的确定性检查（`event_exists`: 日期/时段/参会人字段比对；`email_sent`: 收件人+主题；`report_exists`/`document_updated`: key 存在性）。
  - `check_forbidden(forbidden: list[str], events: tuple) -> list[str]`——事件日志中命中 forbidden 模式即列出。
  - `score_episode(case, trace, sandbox) -> EpisodeResult`（frozen：`success: bool`/`failure_reasons: tuple[str, ...]`/`attributions: frozenset[str]`（可重叠标签）/`earliest_blocker: str | None`（互斥）/`budget_exhausted: bool`）——评分顺序照契约第四节：forbidden → 预算 → required_state → 终止行为（`acceptable_behavior` ∈ complete/decline/clarify 的判定面）→ 二值成功。
  - 归因标签集（可重叠，跨阶段冻结面）：`candidate_absent`/`outside_window`/`within_window_not_shown`（「进 W 未进 k」——设计稿 §5 九项的漏项补齐）/`shown_not_selected`/`detail_read_failure`/`wrong_arguments`/`tool_failure`/`state_mismatch`/`timeout_or_service`/`budget_exhausted`（清单外有意新增，E-D9 支撑）。**冒烟 A0 无 W 窗口（E-D3 只有 k=5）：`outside_window`/`within_window_not_shown` 本阶段不产出**，其判定口径随阶段 4 A1 臂（W=20）启用时定义；`earliest_blocker` 取归因中最早发生者（按事件 seq）。
- 新建 `experiments/agent_e2e/runner.py`：`run_episode(case, tools_builder, step_limit: int) -> Trace`（frozen：`case_id`/`steps: tuple[dict, ...]`/`final_output: str | None`/`exhausted: bool`/`error: str | None`）——通用执行循环：接受 `tools_builder() -> dict[str, Callable]`（每 episode 新建沙箱+工具闭包，重放 = 同输入再调一次）与 `agent_step(tools, history) -> dict` 步进回调（LLM 无关——冒烟用 DeepAgents 步进器，离线测试用脚本化步进器回放）；每步记录工具调用/结果/事件序号；超步数即截断。
- 新建 `experiments/agent_e2e/report.py`：`summarize(results: list[EpisodeResult]) -> dict`（成功计数/失败分项/归因分布/逐例表）+ `write_report(report: dict, path) -> None`（JSON 落盘）。
- 新增 `tests/test_e2e_scoring.py` + `tests/test_e2e_runner.py`。
**如何验证**：L1 单元（谓词检查正反例；forbidden 命中；评分顺序——forbidden 存在时即使 required 全满足也败；归因可重叠与 earliest 互斥；脚本化步进器回放重放一致——同一 builder 两次 run 的 trace 与结果逐位一致；步数耗尽截断）。L2：端到端离线编排（E0 用例 + 脚本化 Agent 步进器——成功路径脚本与「只说完成不执行」反例脚本各一；**步进器与 tools_builder 均用测试内替身工具**（直接调 `sandbox.apply` 的脚本工具），不经 E2 的 tools.py——E3 声明面与验证面均不依赖 E2，W3 并行成立）。
**验收标准**：
- 「回答正确但没执行」反例：脚本化 Agent 输出完成文案但零状态变更 → `success=False`、归因含 `state_mismatch`（设计稿阶段 2 通过条件的机制面）
- forbidden 命中（越权 email_sent）→ `success=False` 且 failure_reasons 含该项，即使 required_state 全满足
- 同一 case + 同一脚本步进器连跑两次 → Trace 与 EpisodeResult 逐位一致（可重放证据）
- 步进器步数超 limit → `exhausted=True`、归因 `budget_exhausted`、不再调用步进器
- `summarize` 对 12 条混合结果输出分项与归因分布计数正确
**评审要点**：评分器只读 trace 与沙箱终态、不改状态；runner 与 LLM 解耦（步进回调注入）；归因不硬塞唯一原因。
**定级依据**：多文件新增、LLM 解耦设计使全部逻辑可离线替身测试、不触碰包内核——T2。

### E4 DeepAgents 参考装配 + 冒烟入口（T2）

**如何改**：
- 新建 `experiments/agent_e2e/agent.py`：
  - `build_model() -> ChatOpenAI`：从 `AGENT_RETRIEVAL_E2E_{MODEL,BASE_URL,API_KEY}` 读（缺任一 → `RuntimeError` 指路，E-D10）；`temperature=0.0`。
  - `AGENT_SYSTEM_PROMPT`：冻结冒烟系统提示词（模块常量——职责：可用工具说明、先搜索后调用、无匹配时明确拒绝、信息不足时澄清、不虚构执行结果）。
  - `build_reference_agent(model) -> CompiledStateGraph`：`create_deep_agent(model=model, tools=[四工具（langchain `tool` 装饰器包装，docstring 即契约描述）], system_prompt=AGENT_SYSTEM_PROMPT, backend=StateBackend(), checkpointer=InMemorySaver())`——**checkpointer 与工具闭包每 episode 新建**（E-D7）。
- 新建 `experiments/agent_e2e/smoke.py`（live-only 入口，CI 不跑）：读契约 → 逐用例构造沙箱 + 工具闭包 + DeepAgents 步进器（`agent.astream`/`invoke` 按锁定版本 API 调用，实施时以实源为准）→ `run_episode` → `score_episode` → 汇总 `report.json` 落盘 `experiments/agent_e2e/outputs/smoke-report.json` + 逐例轨迹 `outputs/traces/*.json`。退出码：0 全过 / 1 有用例评分失败 / 2 形状异常 / 3 调用失败 / 4 缺配置。
- 新增 `tests/test_e2e_agent.py`：离线装配面——`build_model` 缺变量报错指路；`build_reference_agent` 用脚本化假模型（不触网）构造成功、返回对象可编译、四工具名绑定正确、checkpointer 为新实例（两次构造不共享）。
**如何验证**：L1 离线装配（假模型替身）；L3 live 冒烟一次（glm-5.3-flash，12 用例串行）。
**验收标准**：
- 离线：缺配置报错含变量名；两次 `build_reference_agent` 的 checkpointer 与工具闭包互不共享；工具名恰为契约四名
- live：12 用例全部产出 Trace + EpisodeResult；`smoke-report.json` 生成且逐例字段齐；**E0 五类场景各按预期判定**（可完成多步 → required_state 全满足 success=True；无匹配 → decline 成立；信息不足 → clarify 成立；越权 → success=False；工具故障 → 归因 `tool_failure`）
- live 重放：任选一条多步用例用固定步进器回放（同 episode 输入）→ 评分结果一致
**评审要点**：装配不启用未开放能力（middleware 空）；系统提示词与工具 docstring 是冻结面（进契约）；冒烟是工程验证非效果结论。
**定级依据**：装配面全部可离线替身测试（假模型不触网），live 冒烟是验收手段非实现路径；消费 deepagents 公共 API（实源已核实）——T2。

## 与既有轨的边界

- A1（Jev 重排臂）/A2（语义准入）不进本方案：接入属阶段 4 校准（W/k/τ 冻结后），届时 `JevJudge`/`ChatJudge`/`select_with_judge`（P0/P1 组件）作为判定面候选。
- 阶段 3（双人标注 + κ）/阶段 4-6 不进本方案：HITL 依赖（issue #11 已记）。
- 包（`src/agent_retrieval`）零改动：e2e 全部在 `experiments/agent_e2e/` 消费包公共面。
