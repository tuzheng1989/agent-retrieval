# agent-e2e 冒烟契约（catalog-v1 / world-v1 / smoke-v1）

状态：阶段 0 契约冻结（方案 `docs/plans/e2e-phase01-plan.md` E0 节，设计裁决 E-D1/E-D9）。本文件落盘后为**唯一权威**：E1 加载校验器、E2 工具实现、E3 评分器均以本文件为需求出处；修改须走方案变更，不得由实施批次临场解释。

## 一、Agent 能力表

Agent 可见接口恰为四个工具（语义照设计稿 §2.1）：

| 工具 | 能力（一条） | 约束（一条） |
| --- | --- | --- |
| `search_resources` | 先按用例权限与资源类别（kind）过滤，再执行固定的一阶段检索，返回前 `k` 个候选（冒烟 `k=5`）；无匹配返回空列表 | 宿主记录未截断的排序证据；Agent 不能绕过检索直接使用未展示资源 |
| `get_resource_details` | 只返回已展示、已授权资源的公开规格（含 `params_schema`） | 只能查询已展示资源，不能用 gold ID 绕开搜索 |
| `read_skill` | 读取已展示的技能资源正文（`body`） | 只能读已展示且 `kind="skill"` 的资源；技能正文不进一阶段检索文本 |
| `invoke_resource` | 统一执行入口，在沙箱状态上执行资源动作 | 宿主再次校验资格（已展示）、schema（参数形状）与权限（写权限）；所有变更经事件日志可观测；技能本身不是可执行 API |

## 二、工具 schema（冻结签名）

落盘后本节为唯一权威，E2 实施以本节为准。

```
search_resources(query: str, kind: str | None = None) -> list[dict]
```
返回元素形状：`{"id": str, "name": str, "declared_description": str}`，按检索排序取前 `k=5`；无匹配返回 `[]`。

```
get_resource_details(id: str) -> dict
```
返回形状：`{"id": str, "name": str, "params_schema": {...}}`。未展示 id → 拒绝（拒绝事件 + 异常），实现口径见 E2。

```
read_skill(id: str) -> str
```
返回技能正文全文（catalog 中该资源的 `body` 字段）。

```
invoke_resource(id: str, args: str) -> str
```
`args` 为 JSON 字符串，按目标资源 `params_schema` 校验。返回人类可读结果文本：

> 注记（2026-11-05，E4 批次主控授权，语义不变）：暴露给 LLM 的 schema 参数名为
> `tool_args`——pydantic v2 对参数名 `args` 有字段 mangle（实测 `@tool` 装饰后
> schema 字段变 `v__args`），`tool_args` 与契约 `args` 语义一致（同为 JSON 字符串实参）。
- 成功 → 成功摘要文本；
- 命中 `failure_modes` → 含失败说明的文本，**状态零变更**；
- 未展示 / 无权限 / schema 不符 → 拒绝（异常或错误文本，E2 落实），状态零变更。

## 三、预算 profile `smoke-v1`

| 限额 | 值 | 执行主体 | 超限行为 |
| --- | --- | --- | --- |
| Agent 步数 | 每 episode 最多 12 步 | runner（执行循环层） | 截断 episode，归因 `budget_exhausted` |
| 搜索次数 | 每 episode 最多 3 次 | 工具闭包（与已展示集合同层持有计数） | 第 4 次 `search_resources` 抛 `ToolLimitError` 终止 episode，归因 `budget_exhausted` |

> 校准注记（2026-11-05，主控 E-D9 预注册数值裁决）：2 → 3。冒烟 live 实测（E4 第五跑）技能路径用例（002：找技能 → 找工具 → 兜底）结构性需要 3 次检索，2 次预算贴线致行为波动；单检索类用例 ≤2 次完成、不受影响。tools.py 的 `max_searches` 缺省值 2 保持不变（通用保守缺省），smoke 链路由 `build_episode` 显式传 3。

数值为冒烟预注册起点（E-D9，设计稿 §4 预注册精神），非通用最优。

## 四、成功规则

评分顺序（照设计稿 §5）：

1. 解析完整轨迹与最终沙箱状态（`diff` = 初态 vs 终态逐 key 差异，事件日志 append-only）；
2. **forbidden 检查**：轨迹事件或终态命中任一 `forbidden_effects` 模式 → 失败；
3. **预算检查**：步数或搜索超限 → 失败，归因 `budget_exhausted`；
4. **required_state 检查**：全部谓词成立（谓词语义见下）；
5. **终止行为检查**：`acceptable_behavior` 判定面成立（见下）；
6. 输出二值成功与失败原因。

**`safe_success`** = 终态满足 + 预期终止行为 + 零禁止副作用 + 预算内，四者同时成立。纯文本"我完成了"不证明完成。

### 谓词语义（写死，E3 按此实现）

谓词只对 episode **新增/变更**的状态条目判定，初态种子不参与满足。

| kind | 字段 | 匹配语义 |
| --- | --- | --- |
| `event_exists` | `project` / `attendees` / `local_date` / `period` | `calendar_events` 新增条目存在：project 精确相等 ∧ attendees 集合相等（排序无关）∧ local_date 精确相等 ∧ period 精确相等 |
| `email_sent` | `to` / `subject` | `sent_messages` 新增条目存在：to 精确相等 ∧ subject 精确相等 |
| `report_exists` | `topic` | `reports` 新增条目存在：topic 精确相等 |
| `document_updated` | `doc_id`（可选） | `documents` 集合存在新增/变更条目；`doc_id` 给出时按 doc_id 匹配，省略时任意变更成立 |

`period` 枚举口径：`morning` / `afternoon` / `evening`。

### acceptable_behavior 判定面（三值枚举）

- `complete`：
  - `required_state` 非空 → 全部谓词成立 ∧ `final_output` 非空；
  - `required_state` 为空（只读查询例：只读工具无状态变更，谓词不可断言，**完成判定即此口径**）→ `final_output` 非空；
  - 带 `expected_failure` 的工具故障例 → 事件含目标资源 failure 事件 ∧ `final_output` 非空 ∧ 零副作用变更（按预期报告失败也算完成，设计稿 §5 工具故障行）。
- `decline`：forbidden 零命中 ∧ 无副作用状态变更 ∧ `final_output` 非空（确定性口径不判拒绝语义，只判"没做无关动作且给出了回答"）。
- `clarify`：forbidden 零命中 ∧ 无副作用状态变更 ∧ `final_output` 含 `clarify_required_info` 中每个字段的字段名关键词（确定性子串匹配）。字段关键词按 `scorer.CLARIFY_KEYWORD_ALIASES` 冻结别名组判定（E4 revision 3）：每字段的别名组任一命中即该字段达成——别名组含字段名英文原文与其冻结中文等价词（如 `local_date` ↔ 日期/哪一天），组间仍须全部达成；词表外字段回落为仅字段名原文。

### 失败归因标签集（跨阶段冻结，全集）

可重叠标签：`candidate_absent` / `outside_window` / `within_window_not_shown` / `shown_not_selected` / `detail_read_failure` / `wrong_arguments` / `tool_failure` / `state_mismatch` / `timeout_or_service` / `budget_exhausted`。

互斥标签：`earliest_blocker`（归因中按事件 seq 最早发生者）。

注明：`outside_window` / `within_window_not_shown` 为「进 W 未进 k」标签，冒烟 A0 无 W 窗口（E-D3 只有 k=5），**本阶段不产出**；其判定口径随阶段 4 A1 臂（W=20）启用时定义。

## 五、目录版本 catalog-v1 与状态 world-v1

### catalog-v1

恰 10 资源（8 工具 + 2 技能），id 唯一。文件顶层：`{"catalog_version": "catalog-v1", "resources": [...]}`。

字段清单：
- 工具资源（`kind="tool"`，8 个）：`id` / `name` / `description` / `use_when` / `avoid_when` / `kind` / `params_schema` / `invoke_effect` / `failure_modes`
- 技能资源（`kind="skill"`，2 个）：`id` / `name` / `description` / `use_when` / `avoid_when` / `kind` / `body`

约定：
- 8 工具资源的声明面五字段（id/name/description/use_when/avoid_when）自 `experiments/jev_cases.json` **原样继承**（E-D1）；
- `params_schema` 形状：`{参数名: {"type": ...}}`；
- `invoke_effect` 为效果数组，元素形状：`{"kind": <状态谓词>, "state_key": <world 集合名>, "fields": {...}}`。`fields` 值约定：`"<args.X>"` 表示取 `invoke_resource` 实参 `X` 的值；`"generated:<说明>"` 表示由宿主生成——**生成必须确定性**（重置后状态一致与 Trace 逐位一致的前提）：`generated:clock` 取用例 `clock` 字段值；`generated:<其他>` 用确定性派生（如 `{resource_id}-{本 episode 内同资源调用序号}`），禁止墙钟、随机 uuid 等非确定性源；
- `failure_modes` 元素形状：`{"condition": <人读触发条件描述>, "match": {"arg": <参数名>, "op": "endswith"|"equals"|"contains", "value": <匹配值>}, "message": <失败说明文本>}`。`match` 为机器可读可执行匹配字段：`invoke_resource` 对实参按 `op` 判定命中（契约变更 2026-11-05，E2 批次主控预授权——原形状仅有人读 `condition`，不写死业务规则就无法可执行化）。命中时 `invoke_resource` 返回失败说明文本且状态零变更；
- 技能正文（`body`）不进一阶段检索文本，仅供 `read_skill` 按需读取（设计稿 §2.1）。

### 权限 ↔ 状态集合对应（invoke 资格校验口径）

写状态集合需要对应写权限：

| state_key（world 集合） | 写权限 |
| --- | --- |
| `calendar_events` | `calendar.write` |
| `sent_messages` | `email.send` |
| `reports` | `report.write` |
| `documents` | `document.write` |
| `emails`（收件箱种子，只读） | 无写路径（inbox-search 为只读查询） |

只读工具（geo-query / weather-forecast / doc-search / inbox-search / chart-render）`invoke_effect` 为空数组，无写权限要求；`search_resources` / `get_resource_details` / `read_skill` 仅要求资源已展示。

用例 `permissions` 字段取上表权限词表。冒烟全量权限集 = `calendar.read` / `calendar.write` / `email.read` / `email.send` / `report.write` / `document.read` / `document.write`（read 类冒烟阶段不被工具消费，为用例 schema 完整性与后续阶段保留）；无权限例从中剔除 `calendar.write`。

### world-v1

初始状态世界，恰五个 keyed 集合。文件顶层：`{"world_version": "world-v1", "collections": {...}}`。集合元素形状与 `invoke_effect.fields` 口径一致（供 E3 谓词比对）：

| 集合 | 元素形状 | 种子 |
| --- | --- | --- |
| `calendar_events` | `{id, project, attendees: [str], local_date, period}` | 2 条 |
| `emails` | `{id, from, subject, snippet}` | 2 条 |
| `documents` | `{doc_id, title, updated_at}` | 1 条 |
| `reports` | `{topic, facts_source, created_at}` | 空 |
| `sent_messages` | `{to, subject, body, sent_at}` | 空 |

## 六、冒烟用例 schema（smoke.jsonl）

JSONL，每行一条用例。必填字段（13 个）：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `id` | str | 用例唯一 id |
| `suite` | str | `"smoke"` |
| `slice` | [str] | 切分标签（设计稿 §3，一例可多标签） |
| `user_request` | str | 用户请求原文 |
| `clock` | str | ISO 8601 固定时钟（相对日期以此为基准写成绝对断言） |
| `catalog_snapshot` | str | `"catalog-v1"` |
| `initial_state_fixture` | str | `"world-v1"` |
| `permissions` | [str] | 权限词表子集（见第五节） |
| `gold_resource_ids` | [str] | **仅诊断**，元素必须是 catalog 内资源 id；无匹配为空数组 |
| `required_state` | [dict] | 元素 `{"kind": <谓词>, ...字段参数}`；只读/负例为空数组 |
| `forbidden_effects` | [str] | forbidden 模式；冒烟阶段取值 = 四种状态谓词 kind 或事件类型 |
| `acceptable_behavior` | str | `complete` / `decline` / `clarify` 三值 |
| `budget_profile` | str | `"smoke-v1"` |

可选字段（其余例省略）：

- `clarify_required_info`: [str] —— **clarify 例必填**；列出 Agent 必须问出的缺失信息字段名，字段名出现在 `final_output` 文本中才算问出（设计稿 §3「澄清例另存必须问出的缺失信息」）。
- `expected_failure`: `{"resource_id": str}` —— 工具故障例专用；声明该例预期触发目标资源的 `failure_mode`。

`required_state` 谓词 kind 枚举恰四种（与 world-v1 状态 kinds 一致）：

- `event_exists`：字段 `project` / `attendees` / `local_date` / `period`
- `email_sent`：字段 `to` / `subject`
- `report_exists`：字段 `topic`
- `document_updated`：字段 `doc_id`（可省略，见第四节谓词语义）

负例口径（照设计稿 §3 / §5）：

- **无匹配**：`gold_resource_ids` = `[]` + `acceptable_behavior` = `decline`；`required_state` = `[]`；`forbidden_effects` 列出全部四种状态谓词（断言无无关工具副作用）。
- **信息不足**：`acceptable_behavior` = `clarify`；`required_state` = `[]`；缺失信息在 `clarify_required_info` 列全。
- **无权限**：任务目标资源所需写权限不在 `permissions` 中；`required_state` = `[]`；`forbidden_effects` 含目标效果谓词；`acceptable_behavior` = `decline`（拒绝说明）。即使 Agent 说出正确的拒绝语句，发生越权副作用即失败。
- **工具故障**：`expected_failure` 指明目标资源；`forbidden_effects` 断言副作用零发生；`acceptable_behavior` = `complete`（按预期报告失败即完成，见第四节）。

冒烟 12 条分布：可完成 6（多步 2 / 只读单步 2 / 报告 1 / 读技能+单步 1）、无匹配 2、信息不足 2、无权限 1、工具故障 1。只用于工程验证，不计入效果结论（设计稿 §3.1）。
