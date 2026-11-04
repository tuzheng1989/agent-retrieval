# SDK P1 实施方案（分批优化计划）

> 依据：`docs/roadmap.md` 立项清单第 3 步 + `docs/adr/0001-sdk-integration-boundaries.md`（P1 = ChatJudge + CorpusProjection；语料投影双形态条款）。
> 范围：CorpusProjection 声明式投影（零依赖层）、ChatJudge 裸 HTTP chat 端点判定实现（复用 `[judge]` extras）、`judge_state` 投影逻辑提取共享、live 冒烟脚本。
> 宿主纪律不变：包不读环境变量、不读配置文件——一切经构造参数；experiments 是宿主。
> 评审状态：定稿（2026-11-04）。自审一轮（auditor 六项机械核查）：4 条 P2 + 5 条 sug，已全部修复（行号锚点、测试计数 16、`--skip-live` 口径、D3 冒烟定位与验收对齐）；无 P1 硬伤。经裁决跳过对抗评审直接定稿。

## 设计裁决（本方案的前提假设，逐条可挑战）

| # | 裁决 | 理由与取舍 |
|---|---|---|
| D1 | **ChatJudge 是判定语义的新载体，不是 JevJudge 的替代**：System One wire 形状（Noul 问题）与 chat 自由文本是两种线协议；场景矩阵中只有 chat 端点可用的格需要 ChatJudge | ADR-0001「线协议分支不构成设 Protocol 的理由」同理——两者是并列实现，不互相包装；`Judge` 端口（B1）已抽象掉差异 |
| D2 | **ChatJudge 一次调用判全部候选**（单 prompt 输出 JSON 概率映射），不逐候选一问 | 与 JevJudge 调用形状对齐（一次判定一个 `Judgment`）；逐候选一问 token 成本线性放大且无证据支持 |
| D3 | **ChatJudge 不与 JevJudge 做数值对拍**：判定载体不同（chat 模型 vs System One），概率数值不可比。语义保留锚定在**代码级复用**（候选投影共用同一函数）+ 结构测试 | B3 数值对拍的前提是同 payload 双臂传输，此处不成立；ChatJudge 的系统性判定质量评估归 P2 scenario-matrix——live 冒烟只验管线、输出形状与最弱语义合理性（明显正例概率高于明显负例），后者是冒烟 sanity 而非质量结论 |
| D4 | **`judge_state` 的候选投影逻辑提取为共享纯函数**，`judge_state` 与 ChatJudge 共用——提取后 `judge_state` 输出逐位不变 | 单一出处：候选投影口径（name + description 截断 2000）漂移即两类 judge 语义分叉；提取是纯重构，由既有 16 项测试 + B3 判定一离线复跑背书 |
| D5 | **CorpusProjection 不提供 JSON 序列化 API**：`fields` 是 `tuple[str, ...]`，天然可 JSON（宿主 `json.dumps(list(...))`） | 极简原则：`tuple[str]` 无序列化歧义，包内 API 是冗余面；ADR「可 JSON」由数据形态直接满足 |
| D6 | **投影身份不进 `EmbeddingConfig.identity()`**（那是嵌入器语义身份），由宿主把投影描述串加进 `vector_snapshot_id` 组件序列；`corpus_hash` 天然覆盖投影变化（文本变哈希变） | 与 ADR-0001「投影进 `identity()`/`corpus_hash」半句对照：ADR 所指的 identity 是快照寻址的组件序列（`vector_snapshot_id` 的 `identity: Sequence[str]` 参数），不是 `EmbeddingConfig.identity()`——本裁决即该语义的落实而非偏离。内容寻址已闭环：投影变 → 语料文本变 → corpus_hash 变、向量重嵌；声明式投影的 fields 就是那一格配置文档的投影半节 |
| D7 | **experiments 层三处现有投影函数不迁移**（`skillret.py:112`、`toolret.py:143`、`golden/gate.py:77`） | P0 教训：六方零改动。声明式投影服务 P2 scenario-matrix 新格；既有基准的投影字段组合（含 id 与否、过滤空与否）各有已验证语义，换写法无收益 |
| D8 | **`JevJudgeConfig` 不补 `identity()`** | P0 定稿后无消费方需要它；避免无消费场景的契约面扩张（YAGNI）。`ChatJudgeConfig` 按 ADR 四件套模子配 `identity()`——新组件照模子 |
| D9 | **api_style 双分支（chat/responses）不在 P1**：ChatJudge 只走 `/chat/completions` | docgen 双分支的真实需求来自 responses-only 网关实测；judge 格尚未遇到该约束，出现时随场景矩阵格补 ~30 行分支（模式已在 `expansion/api.py` 验证） |
| D10 | **temperature 进 `ChatJudgeConfig` 且进 `identity()`**，默认 0.0 | 判定是分类任务，低温是判定质量的组成部分；采样参数影响输出语义，身份串不含它会掩盖复现差异 |

## 路线图

| 批次 | 内容 | T 级 | 依赖 |
|---|---|---|---|
| C1 | CorpusProjection 声明式投影（零依赖层） | T2 | — |
| C2 | ChatJudge + 投影提取共享 + live 冒烟 | T2 | C1（仅文件集顺序，无接口依赖） |

**波次**：C1 与 C2 无代码依赖，但都要动顶层 `__init__.py`（C1 直导入区 `:27-37` 的 ports 导入块、C2 `_LAZY_EXPORTS` `:53-56` judges 区）——文件集冲突，串行执行：**W1 = C1；W2 = C2 + 末波收口**（全量测试）。
依赖图：C1 → C2（仅文件集顺序约束，无接口依赖）。

## 详案

### C1 CorpusProjection 声明式投影（T2）

**如何改**：
- 新建 `src/agent_retrieval/core/projection.py`：
  - `CorpusProjection`（frozen dataclass）：`fields: tuple[str, ...]`、`separator: str = " "`。构造即校验：`fields` 非空、逐项非空串（`ValueError`）——宿主资源有哪些字段是宿主的事（内核零建模，D6/D7 同口径），字段缺失由 `getattr` 在调用时天然抛 `AttributeError`，不做白名单校验。
  - `to_callable() -> Callable[[T], str]`：闭包逐字段 `getattr` 取值；**取值链 `None` 或空串的 part 丢弃**（照 `experiments/golden/gate.py:77-80` 的 `if part` 过滤语义——字段缺席不产生双空格）；`separator.join` 输出。全字段皆空返回空串（调用方保证查询非空的既有纪律不变，`ranking.py:68`）。
  - 模块 docstring 写明双形态条款（ADR）：声明式为默认（可进场景矩阵配置文档），callable 逃生口保留（`rank_candidates` 的 `corpus_text` 参数签名不变——本批**零改动内核**）。
- `src/agent_retrieval/__init__.py`：直导入区 `:27-37` 的 ports 导入块之后新增 `core.projection` 导入块（`CorpusProjection`）；`__all__` 同步。stdlib-only 走直导入（照 `core.ranking` 导入块 `:44` 模式）。
- 新增 `tests/test_projection.py`。

**如何验证**：L1 单元：fields 校验（空元组抛、含空串抛）；`to_callable()` 用真实 `experiments` 形状的资源 dataclass——id/name/description/tags 四字段投影与 `gate._plain_text` 逐位一致（同输入同输出）；含 `use_when` 五字段扩展组合；空 description 资源不产生双空格；全空字段返回空串；未知字段抛 `AttributeError`；frozen 不可变；`rank_candidates` 组合用例——`CorpusProjection(...).to_callable()` 传入 `corpus_text=` 与手写函数产出相同 hits（排序逐位一致）。
**验收标准**：
- `CorpusProjection(("id","name","description","tags")).to_callable()(resource)` 与 `gate._plain_text(resource)` 对同一资源逐位一致
- `("name","description","tags")` 三字段组合可构造（与 `skillret.py:112` 字段集同构，验证多形态可行）
- description 为空串的候选 → 语料无连续空格；全空候选 → 空串不抛
- `CorpusProjection(())` 与 `CorpusProjection(("name",""))` → 构造即 ValueError
- 经 `rank_candidates(corpus_text=...)` 的 hits 与手写投影函数的 hits 逐位一致
- `json.dumps(list(CorpusProjection(...).fields))` 可序列化（D5 证据）
**评审要点**：内核文件零改动（`ranking.py`/`resource_index.py` 不进改动面）；过滤语义与 `gate._plain_text` 对齐而非 `skillret`（D7 已裁决不迁移，取更稳语义为默认）。
**定级依据**：新公共导出面（顶层 `__init__.py`）+ core 层新文件，加法式、不触碰既有契约——T2。

### C2 ChatJudge + 投影提取共享 + live 冒烟（T2）

**如何改**：
- `src/agent_retrieval/judges/api.py`：提取共享投影纯函数 `projected_candidates(candidates: Sequence[JudgedCandidate]) -> dict[str, dict[str, str]]`（现 `:82-93` 内联逻辑——`c{index}` → `{"name": ..., "description": description[:JEV_DESCRIPTION_MAX_CHARS]}`）；`judge_state` 改调它，**函数签名与输出 dict 逐位不变**。`JEV_DESCRIPTION_MAX_CHARS` 常量留在本文件（包内单一出处，`chat.py` 导入；`experiments/benchmarks/jev_rerank.py:55` 的同名 legacy 常量是判定一对照前提，按 D7 有意不收敛）。
- 新建 `src/agent_retrieval/judges/chat.py`：
  - `CHAT_JUDGE_PROMPT_VERSION = "1"`（模块常量：prompt 措辞变更即升版本，进 identity）。
  - `ChatJudgeConfig`（frozen：`base_url/api_key/model_name/temperature: float = 0.0/timeout_seconds: float = 60.0`）；`identity() -> tuple[str, str, str]` = `(model_name, CHAT_JUDGE_PROMPT_VERSION, str(temperature))`（供宿主记录与展示，docstring 口径照 `DocGenConfig.identity`）。
  - `_chat_judge_prompt(task: str, instruction: str, candidates: Sequence[JudgedCandidate]) -> str`：纯函数（判定 prompt 唯一构造点）。语义继承 `judge_state` 判定语义（`judges/api.py:24-28` 注释锚点：task / retrieval_instruction / 逐候选「具体步骤」判定，判声明能力不判共享关键词——`_CRITERIA_TRUE/FALSE` 语义翻译进 prompt 正文）；候选清单用 `projected_candidates` 输出逐行渲染（`c{i}: {name} — {description}`）；`instruction` 非空时作为 retrieval_instruction 节并列（D1：与 `Judge` 端口 `instruction` 逐次调用参数语义一致）；输出格式约定段：单一 JSON 对象、键恰为 `c0..c{n-1}`、值为 0-1 浮点、无其他文本。
  - `ChatJudge`（实现 `Judge` 端口，`judge(query, candidates, *, instruction="") -> Judgment`）：`_chat_judge_prompt` 构造 → POST `{base_url}/chat/completions`（Bearer 鉴权，`json=` 含 `model/temperature/max_tokens=4096/messages/stream=False`，`messages` 单 user 消息）→ 解析 `choices[0].message.content` → JSON 提取（先直接 `json.loads`，失败剥 markdown 围栏 ```json ... ``` 再 loads，仍失败按解析异常进重试循环）→ 键集校验（与候选 id 一一对应：缺键/多键/键非 `c{i}` 形态 → `JudgeResponseError` 不重试）→ 概率 `float()` + `[0,1]` 越界抛 `JudgeResponseError`（不 clamp——模型输出越界是失控信号）→ `Judgment(probabilities, model=body["model"], latency_seconds, input_tokens/output_tokens 从 body.get("usage") 透传缺席 None)`。
  - HTTP 纪律照 `JevJudge._post`（`judges/api.py:130-176`）平移：退避 `(0, 2.0, 5.0, 10.0)`、401/403 单发即抛不重试、其余 4xx 记 `last_error` 重试 3 次、网络/解析异常同循环；requests 惰性导入缺失报「先安装 agent-retrieval[judge]」（同 `:138-142`）——**extras 复用 `[judge]`，pyproject 零改动**。
- `src/agent_retrieval/judges/__init__.py`：导出 `ChatJudge`、`ChatJudgeConfig`（`JudgeResponseError` 已在 `api.py` 定义，不重复导出）。
- `src/agent_retrieval/__init__.py`：`_LAZY_EXPORTS` judges 区 `:53-56` 加 `"ChatJudge"`、`"ChatJudgeConfig"` 两项；`__all__` 同步。
- 新增 `tests/test_api_chat_judge.py`（替身 HTTP，照 `tests/test_api_judge.py` 的 `judge_wire` fixture 模式）。
- 新增 `experiments/benchmarks/judge_chat_smoke.py`：live-only 冒烟（CI 不跑，配置从环境变量读——experiments 是宿主，读环境变量是宿主权利）。候选组合固定取 `experiments/jev_cases.json` 资源池 4 项，任务文本 `"Research the population of California in 2024, then email a summary to Mei"`：正例 `geo-query`（查人口——步骤 1）与 `email-send`（发摘要——步骤 2）；负例 `chart-render`（话题相关——数据可视化沾边但任务无图表步骤）；干扰项 `weather-forecast`（无任务语义）。断言：max(p(geo-query), p(email-send)) > p(chart-render)、全部概率 ∈ [0,1]、键集恰为 4 候选 id。输出逐候选概率表到 stdout。

**如何验证**：L1 替身单元：`projected_candidates` 投影口径（与 `judge_state` 既有断言同构）+ 截断 2000；`_chat_judge_prompt` 含 task/instruction（非空与默认空两态）/候选 name、**不含** use_when/avoid_when（D4 投影口径）；`judge` 请求体断言（endpoint、Bearer、model、temperature、messages 单条）；解析三态（纯 JSON / 围栏包裹 / 不可解析进重试）；键集缺失不重试即抛；概率越界抛；usage 缺席透传 None；`isinstance(chat_judge, Judge)` 真；frozen；identity 三元组分化（temperature 变 → identity 变）。L2 编排：`ChatJudge` + `select_with_judge` 组合（替身 200 响应 → 判停消费）。L3：judge_parity 判定一离线复跑（`--skip-live`）+ live 冒烟（有 key 时）。
**验收标准**：
- `judge_state` 输出与提取前逐位一致：`tests/test_api_judge.py` 16 项零改动全绿 + `python -m experiments.benchmarks.judge_parity --skip-live` 判定一 PASS（离线复跑口径——不带 `--skip-live` 缺 key 退出码 4 不可断言；冻结 diff 清单不变，恰为已登记 tags 项）
- 替身 200（纯 JSON 响应）→ `Judgment.probabilities` 键与候选 id 一一对应、值与替身响应逐位一致
- 替身 200（```json 围栏包裹）→ 解析成功同上
- 替身 400→200 → 重试成功；401 → 单发即抛；响应缺 `c1` 键 → 单发即抛不重试；概率 1.2 → 抛且异常类型 `JudgeResponseError`
- `instruction="仅限本地文件操作"` 出现在 prompt 正文；默认调用 prompt 无 instruction 节
- 无 requests → 报错含 `agent-retrieval[judge]`
- live 冒烟（有 key 时）：固定 4 候选概率表输出、max(p(geo-query), p(email-send)) > p(chart-render)、键集完整
**评审要点**：`judge_state` 是 B3 对拍冻结锚点——提取为纯重构的语义保留证据链（测试零改动 + 判定一复跑）必须齐全；ChatJudge 判定质量不在本批验收范围（D3，归 P2 场景矩阵），冒烟只验管线。
**定级依据**：多文件 + 公共导出面（judges/`__init__` + 顶层 lazy exports）+ 邻近判定契约（`Judge` 端口实现、`judge_state` 对拍锚点重构）但输出契约本体不变、有既有测试覆盖——T2（不到 T3：无外部状态变更、无契约签名变化、迁移清单为空）。

## 与 e2e 立项的边界

e2e 评测是独立轨（`docs/roadmap.md` §三），不进本方案批次——其立项动作（设计稿回填 06 票三条修订 + GitHub 立项票）与本方案并行推进，产出物见 `experiments/agent-e2e-evaluation-design.md` 的修订提交与立项票。
