# SDK P0 实施方案（分批优化计划）

> 依据：`docs/roadmap.md` 立项清单第 1 步 + `docs/adr/0001-sdk-integration-boundaries.md`。
> 范围：Judge 端口、JevJudge 裸 HTTP（含对拍）、docgen 收编、stats 收编、pooling/前缀下沉。
> 宿主纪律不变：包不读环境变量、不读配置文件——一切经构造参数；experiments 是宿主。
> 评审状态：R1（Codex 独立评审 5.5 分）九条发现全部实质响应，见文末「评审响应记录」。

## 路线图

| 批次 | 内容 | T 级 | 依赖 |
|---|---|---|---|
| B1 | Judge 端口 + 判停辅助（零依赖层） | T2 | — |
| B2 | JevJudge 裸 HTTP 参考实现 + extras `[judge]` | T2 | B1 |
| B3 | 对拍脚本（裸 HTTP vs typesafe-sdk，live） | T1 | B2 |
| B4 | docgen 收编（`expansion/` + extras `[docgen]`） | T2 | — |
| B5 | stats 收编（配对 bootstrap 平移） | T1 | — |
| B6 | pooling/前缀下沉 + Run 缓存入口分键（`identity()` 契约扩展） | **T3** | — |

**波次**（文件集互斥裁决）：W1 = B1 ∥ B4 ∥ B5；W2 = B2 ∥ B6；W3 = B3 + 末波收口。
依赖图：B1→B2→B3 链；B4/B5/B6 互不依赖、不依赖 B1 链。
**W2 文件集修正（R2 前提）**：B6 新增 `run/cache.py` 入口分键改动——W2 变为 {`judges/*`, `pyproject.toml`, `__init__.py`} ∥ {`embedders/config.py`, `embedders/local.py`, `embedders/factory.py`, `run/cache.py`}，仍互斥（B6 不动 pyproject 与 `__init__.py`）。

## 详案

### B1 Judge 端口 + 判停辅助（T2）

**如何改**：
- `src/agent_retrieval/core/ports.py:110`（文件尾追加）：`JudgedCandidate`（frozen dataclass：`id/name/description/use_when=""`/`avoid_when=""`——声明面最小结构，与 `experiments/jev_cases.json` 资源池同构）；`Judgment`（frozen dataclass：`probabilities: dict[str, float]`、`model: str`、`latency_seconds: float`、`input_tokens: int | None`、`output_tokens: int | None`）；`Judge` Protocol（`runtime_checkable`，方法 `judge(query: str, candidates: Sequence[JudgedCandidate], *, instruction: str = "") -> Judgment`——**instruction 是逐次调用参数**：ToolRet is_inst 协议把它与 query 并列传入 state（`jev_rerank.py:71-74` 用 `case.instruction`），预合并进 query 会改变已验证语义；默认空 = 无指令场景，与 ADR「窗口与 τ 不进端口」不冲突）。
- 同文件追加 `select_with_judge(hits: Sequence[CandidateHit[T]], judgment: Judgment, *, minimum_probability: float, item_id: Callable[[T], str]) -> str | None`：泛化自 `experiments/jev_retrieval.py:139-152` 的 `select_with_jev`。**ID 从 `hit.item` 提取（R3 修正）**：内核 `CandidateHit`（`core/ranking.py:35-46`）持有泛型 `item`，宿主数据在 `hit.item` 内、`item_id` 回调接收的是宿主数据 `T`（与 `rank_candidates` 的 `item_id(item)` 同层），提取式为 `item_id(hit.item)`——不是对 hit 本身调用回调。**契约前提是 Judgment 已存在**（exact 短路在调用方编排层，`jev_retrieval.py:196-199` 在调 judge 之前检查）；概率域校验失败抛 `ValueError`；`max()` 并列保检索序；最佳概率 < `minimum_probability` 返回 `None`（abstain）。`minimum_probability` 无默认值（照 `resource_index.py:109` 风格）。
- `src/agent_retrieval/__init__.py`：直导入区（:17-40）追加 `Judge`、`JudgedCandidate`、`Judgment`、`select_with_judge`；`__all__`（:63-103）同步。stdlib-only，不走 lazy exports。
- 新增 `tests/test_judge_port.py`。
- 边界契约（R2 修订）：候选 id 唯一性**不能用纯集合比较落实**（`[a,a]` vs `{a}` 会漏）——先取 `[item_id(hit.item) for hit in hits]` 断言 `len(ids) == len(set(ids))`（重复 → `ValueError`），再对 `set(probabilities) != set(ids)` 抛缺键/多键 `ValueError`；NaN/inf 概率按越界处理（`not 0 <= p <= 1` 对 NaN 为真，天然拦截，测试钉住该行为）；空 shortlist → 返回 `None`（不抛）。

**如何验证**：L1 单元：Protocol `isinstance`；`select_with_judge` 用例——exact 命中、并列保序、τ 下 abstain、概率缺键抛、越界/NaN 抛、**重复候选 id（[a,a]）抛**、空列表返 None、instruction 透传可见性（StubJudge 记录入参）；**组合用例（R3）**：真实 `rank_candidates`（非对象宿主数据元组 + `item_id=lambda x: x[0]`）产出 hits 后连同同一回调传入 `select_with_judge`。L2 编排（判停断言在此）：带调用计数替身，exact 命中时 `judge.judge` 调用次数为 0（照 `jev_retrieval.py:196-208` 模式）。
**验收标准**：
- 给 exact 命中编排 → StubJudge 调用计数 0（判停在编排层）
- 概率 {a:0.9, b:0.9}、门槛 0.7 → 返回检索序在前者
- 最高概率 0.6、门槛 0.7 → 返回 None
- probabilities 缺候选 id 或含 1.2 或 NaN → ValueError；候选 [a,a] → ValueError；空 shortlist → None
- `rank_candidates` 真实输出 + 非对象宿主数据回调 → 正确取 ID 与返回
- StubJudge 收到 `instruction="xxx"` 透传
**评审要点**：`minimum_probability` 无默认值；Judgment tokens 可空（02 票实测部分端点不计量）。
**定级依据**：加法式，但 `ports.py` 是内核契约文件、`__init__.py` 是公共导出面——T2。

### B2 JevJudge 裸 HTTP 参考实现（T2，依赖 B1）

**如何改**：
- **实现前调研子步（不脑补）**：读 `.venv/Lib/site-packages/typesafe_sdk/_core/questions.py` 的 `normalize_questions` 与 `_core/endpoints.py`，核实 Noul 问题 JSON 序列化形状与 `/v1/systemone` payload 全貌；形状写进实现注释。
- **判定语义继承（R1-P1 裁决；R3-P1 修正投影口径）**：`judge_state` 继承 **`jev_rerank.py:71-94` 版**（task / retrieval_instruction / 每候选"具体步骤"判定——Law 1 实验、窗口 20 协议与 findings 全部数字的来源；`jev_retrieval.py:92-119` 的 user_goal 直执行判定**不继承**）。**候选投影：state 中每候选只含 `{name, description}` 两键**（R3-P1：旧版候选键恰为 name/description/tags 三键，五字段声明面中的 `use_when/avoid_when` 旧版本就未进 state，保持不投影——端口五字段是候选完整声明面，投影是各 judge 实现自己的事）。对拍差异因此**恰为 tags 键删除一项**；描述截断（旧版 :67-68 的 2000 字符）随投影继承。
- 新建 `src/agent_retrieval/judges/__init__.py` + `judges/api.py`：`judge_state(task, retrieval_instruction, candidates) -> dict`（纯函数，B3 两臂唯一构造点）；`JevJudgeConfig`（frozen：`base_url/api_key/model_name/timeout_seconds=60.0`）；`JevJudge`（实现 Judge：`judge_state` 构造 + 模型包装 + HTTP 传输）；`JudgeResponseError`。HTTP 纪律以 `experiments/benchmarks/docgen.py:87-146` 的 `_chat_completion` **实现**为准：401/403 不重试；其余 4xx 记入 `last_error` 退避重试 3 次（0/2/5/10s）；网络/解析异常同循环。⚠ `reranker.py:86-87` 注释声称"除鉴权外都重试"但其代码（:108-116）非鉴权 4xx 直接 raise——注释与实现不一致，**以 docgen 实现为准**。requests 惰性导入缺失时报「先安装 agent-retrieval[judge]」（照 `embedders/api.py:55-60`）。
- `pyproject.toml:31-36`：extras 增 `judge = ["requests>=2.28"]`；`all` 改 `"agent-retrieval[api,redis,docgen,judge]"`（与 B4 协同，W2 收口核对终值）。
- `src/agent_retrieval/__init__.py`：`_LAZY_EXPORTS`（:42-61）增三项；`__all__`（:63-103）同步。
- 新增 `tests/test_api_judge.py`（替身 HTTP，照 `tests/test_api_embedder.py` 模式）。

**如何验证**：L1 替身单元（payload = `judge_state` 输出 + instruction 非空用例、重试、401 单发、usage 透传、协议满足）；L2 编排。
**验收标准**：
- 替身 200 → Judgment，probabilities 键与候选 id 一一对应；`instruction="..."` 出现在 state 的 retrieval_instruction
- 401 → 单次请求即抛；400→200 → 重试成功
- 无 requests → 报错含 `agent-retrieval[judge]`
- `isinstance(jev_judge, Judge)` 真
**评审要点**：对拍缓解挂钩 B3 落地；`judge_state` 是唯一构造点。
**定级依据**：多文件 + extras 契约 + 第三方线协议实现，无外部状态变更——T2。

### B3 对拍脚本（T1，依赖 B2）

**如何改**：
- 新建 `experiments/benchmarks/judge_parity.py`，**双判定设计（R1-P1 修复）**：
  - **判定一（构造层语义保留，替身/离线）**：`judge_state` 输出与旧实现 `jev_rerank.py:71-94` 在同 case + 同候选下对照，**冻结并比较完整请求**（R2 建议；R3-sug 扩展）：state 全字段 + 每个 Noul 问题的 instructions 与 criteria 措辞 + 候选顺序 + **响应键到资源 id 的映射（与 payload 分开记录、与候选顺序一起核验）**——相同部分逐位一致，**差异仅限已登记的有意变更**（恰为候选 tags 键删除一项；投影口径见 B2——use_when/avoid_when 两侧都不在 state，无差异），差异清单写入对拍报告。冻结参照：旧实现先跑一遍，完整请求 payload 与响应映射分存 `judge-parity-legacy.json` 两节。
  - **判定二（传输层等价，live）**：SDK 臂（typesafe-sdk `TypeSafeClient`）与 JevJudge 臂对**同一 payload**（`judge_state` 生成）各判一遍。请求体对照取"捕获后 JSON 解析、结构逐字段一致"（不做字节对照——SDK 序列化器不受我方控制）；概率判定 `|Δp| ≤ 1e-9`。
  - 数据源 `experiments/jev_cases.json`（五字段与 `JudgedCandidate` 同构；R3-P1：样本须含**非空 use_when/avoid_when** 的候选——证实其正确地不进 state。R4-P2：该文件描述均为短句、无超 2000 字符样本——**对拍脚本内构造**一条超长 description 候选（同一文本同时适配旧 `benchmarks.Resource` 与新 `JudgedCandidate` 两型），断言两侧截断后的 description 逐位一致）；指令场景加一例非空 instruction（ToolRet 形态拼接）。
- 配置从 `TYPESAFE_API_KEY`/`TYPESAFE_BASE_URL` 读（宿主纪律）。live-only，CI 不跑。
- 输出 `experiments/judge-parity-report.json`：冻结样本 diff 清单 + 双臂判定 + 逐 case 概率差。

**如何验证**：L3 端到端（live 一次）+ 判定一可离线复跑。
**验收标准**：
- 判定一：字段级 diff 只含已登记差异项 → PASS；出现未登记差异 → FAIL 并列出
- 判定二：结构一致后 `|Δp| ≤ 1e-9` 全过 → PASS；任一超差 → FAIL 非零退出、打印两侧原始响应片段
- 缺 key → 报错指路
**评审要点**：判定一是「现有语义保留」的直接证据（R1-P1）；两臂不得各自内联构造。
**定级依据**：experiments 层单文件 live 脚本——T1。

### B4 docgen 收编（T2）

**如何改**：
- 新建 `src/agent_retrieval/expansion/__init__.py` + `expansion/api.py`：`DocGenConfig`（frozen：`base_url/api_key/model_name/api_style: Literal["chat","responses"]="chat"/questions_per_resource: int=5/timeout_seconds=60.0`）+ `identity()`（`(model_name, api_style, questions_per_resource)`）。与 `corpus_hash` 的连接**不经代码**（`redis_store.py:49-52` 只哈希传入文本）：**生成后的语料文本实际变化时才产生新哈希**（配置变化不一定改变生成输出；identity 本身也未接入该哈希）——嵌入层按内容寻址自然重嵌，此连接方式写入组件文档。
- `DocGenClient`：**`complete(prompt: str) -> str` 为通用单轮传输基元（R1-P1 修复）**——groupdesc/summarize 的自由文本生成走它；`generate(name, description, tags) -> list[str]` = PROMPT_TEMPLATE + complete + parse（docgen 专用，叠在基元上）。`build_docgen(config) -> DocGenClient | None`（缺凭据返 None，照 `embedders/factory.py:19-43`）。`docgen_corpus_text` 平移（docgen.py:197-207）。
- HTTP 纪律照 `_chat_completion`（docgen.py:87-146）平移；`api_style` 双分支差异（payload/解析/关思考参数/token 字段名）约 30 行。
- **包不收**：`resolve_docgen_config`（docgen.py:48-68 留 experiments）、`load_docgen`/jsonl IO（留）、`main()`（留）。
- `pyproject.toml:31-36`：extras 增 `docgen = ["requests>=2.28"]`；`all` 补 docgen。
- `experiments/benchmarks/docgen.py` 改薄壳：转发 `DocGenError/DocGenConfig/build_docgen/docgen_corpus_text/parse_generated_queries`；保留 `resolve_docgen_config/load_docgen/main()`；删 `_chat_completion/PROMPT_TEMPLATE/generate_queries` 旧实现；**新增两个兼容入口（R2-P1 修订——保持原签名含原参数名）**：`chat_text(config, prompt) -> str`（转发 client.complete，供自由文本生成）；`generate_queries(resource_name, resource_description, resource_tags, *, base_url, api_key, model, k=5) -> list[str]` **保留原签名与原参数名**（R3-P2：原形参名可按关键字传入，改名即破坏关键字调用；内部组装 DocGenConfig → client.generate）——`main()`（docgen.py:254-256）零改动成立的前提正在于此。
- **调用方处置**：六方零改动（`golden/build_fixtures.py`、`run_benchmark.py`、`jev_rerank.py`、`golden/gate.py`、`two_tier.py`、`tests/test_docgen.py`——只用转发名）。`groupdesc.py:55-64`、`summarize.py:119-124` 列入改动面：各自保留自己的提示词与自由文本消费，传输调用改 `chat_text(...)`（config 由 `resolve_docgen_config` 组装）；`_DATA_DIR` 改本地常量。执行代理落地前先读两文件 `_chat_completion` 实参用法取最小改法。
- 新增 `tests/test_docgen_pkg.py`（替身 HTTP：两分支 payload、complete/generate 分层、identity、None 语义、装配）。

**如何验证**：L1 替身单元（两分支 + complete 基元独立用例 + **CLI 与消费方实际生成路径（R3-P2）**：`generate_queries` 原签名原参数名经替身走通 + `main()` 冒烟 + **`groupdesc`/`summarize` 各自的生成函数经替身直接调用**——断言提示词内容出自其自身文件、config 正确传递、自由文本返回）；L3：`run_benchmark --corpus docgen` smoke 一档（扩展数与既有一致）+ `tests/test_summarize.py` 全绿。
**验收标准**：
- chat 分支替身 → `messages`+`thinking disabled`+`max_tokens`；responses 分支替身 → `input`+`reasoning minimal`+`max_output_tokens`；都产出 k 条去重 query
- `complete("任意提示词")` 经替身返回自由文本（通用性证据）
- `generate_queries(resource_name=…, resource_description=…, resource_tags=…, base_url=…, api_key=…, model=…, k=…)` 关键字调用经替身走通（main() 零改动成立）
- `groupdesc`/`summarize` 的生成函数替身直调 → 提示词与改前逐位一致、返回为自由文本
- 缺 api_key → `build_docgen` 返 None；`identity()` 任一元变更值变
- 六方 + `tests/test_docgen.py` 零改动全绿；groupdesc/summarize 改造后既有测试全绿且**提示词仍在其自身文件内**（grep 证实）
**评审要点**：薄壳不残留第二份模板或 HTTP 代码；`api_style` 进 identity 不进提示词。
**定级依据**：多文件 + extras 契约 + runner 与两分析脚本调用面——T2。

### B5 stats 收编（T1）

**如何改**：
- 新建 `src/agent_retrieval/stats.py`：`paired_bootstrap`（ablation_stats.py:26-51 平移，种子 `20261022` 逐位不变）+ `load_arm_cases`（:73-77 **一并平移**——`paired_report` 的直接依赖，读调用方传入路径的 JSON，纯标准库无环境依赖）+ `paired_report`（:80-117 平移，签名不变）。**不收** `chance_baseline`/`hit_rate_report`/`power_report`（two_tier 专用）与 CLI（留 experiments）。
- `experiments/benchmarks/ablation_stats.py` 改薄壳：三函数转发，`main()` 与其余分析函数原样；`tests/test_ablation_stats.py` 零改动。
- 不进 `__init__.py`（B1 持有本波次该文件）；子模块导入。

**如何验证**：L1：既有测试全过 = 行为逐位对照；新增包侧用例——`paired_report` 从包导入并读两份真实报告 JSON 的端到端小样本。
**验收标准**：
- `paired_bootstrap([0.1, -0.1, 0.3])` 与平移前逐位一致
- 包侧 `paired_report(a.json, "x", b.json, "y", ["mrr"])`（metrics 为第五个必需参数，签名照 `ablation_stats.py:80-86` 平移不变）正确产出共享 case 配对（含缺失 case 警告路径）
- 既有测试文件零改动全绿
**评审要点**：平移禁改数值逻辑（含 `1e-12` 浮点残渣防护，:191）。
**定级依据**：纯标准库平移 + 薄壳——T1。

### B6 pooling/前缀下沉 + Run 缓存入口分键与身份隔离（T3）

**如何改**：
- `embedders/config.py:29` 增三字段：`pooling: str = "mean"`、`query_prefix: str = ""`、`corpus_prefix: str = ""`；构造即校验 `pooling ∈ {"mean","cls","last_token"}`（非法值 `ValueError`）。`identity()` 返回 `(provider, model_name, version, flags)`——`flags` 用**无歧义规范序列化**：`""`（全默认）或 `json.dumps([pooling, query_prefix, corpus_prefix], ensure_ascii=False, separators=(",", ":"))`（R1：`: ` 分隔可碰撞——`mean:a:b:c` 双解；JSON 数组消除）。**兼容性明示（R1-P2；R2-P2 修订验收口径）**：公开返回从三元变四元属 0.x 阶段可见变化，`redis_store.py` 同批适配、experiments 展示面逐一核对，写进组件文档；**受契约变化影响的既有测试随契约更新**（`tests/test_api_embedder.py:31-32` 的三元断言、`tests/test_run_cache.py` 的旧键形状用例），更新清单写入本批 resolution，其余回归断言零改动。
- **前缀是 EmbeddingConfig 级能力，两路都生效（R2-P1 裁决）**：`ApiEmbedder`（`embedders/api.py:25-35`）构造增 `query_prefix`/`corpus_prefix`（默认空），`embed_query`/`embed_corpus` 请求前拼接——语义统一（同一 config 两路行为可预期），拒绝式设计（API 用非默认前缀即抛）会让 config 契约出现"合法但不可用"态。`factory.py:25-41`：api 与 local 两分支均透传 pooling（仅 local 消费）与双前缀（两路消费）。
- `stores/redis_store.py:34-46`：`vector_snapshot_id` 签名 `Sequence[str]`；解包改容错。**快照 id 哈希输入零碰撞化（R5-P1 修订）**：flags 为空时哈希输入沿用既有 `|` join 串（逐位不变，既有 write-once 快照零重嵌）；flags 非空时哈希输入 = 旧 join 串 + `"\x00v2\x00"` + flags 的 JSON 规范序列化（`\x00` 不可能出现在任何 legacy 组件中，跨格式无碰撞；flags 的 JSON 形态自身无歧义）。
- **实验侧持久化缓存随行（R5-P1）**：`experiments/benchmarks/embedder_cache.py`（:66-70 查询/语料共用 `_embed`、:75-77 键只含原文、:85-86 未命中恒走 `inner.embed_corpus`——前缀非空时查询入口会被语料前缀污染）同步三处：键形状增入口与身份维度（`(identity_digest, entry, sha256(text))`，identity_digest 取宿主装配传入的 `identity()` 序列化摘要，装配点 `run_benchmark.py:397-400`、`jev_rerank.py`、`rerank_paired.py`、`embedding_paired.py`、`golden/build_fixtures.py` 全部改传 `config.identity()` 序列化）；未命中分支按入口调 `inner.embed_query`/`embed_corpus`（前缀各归其位）；`run_benchmark.py` 等装配点身份串同步。实验缓存是 sqlite 无 TTL——旧键留存无害（新键形状自然分离），无迁移动作。
- `run/cache.py` **入口分键 + 嵌入器身份隔离（R1+R2+R3-P1 修复）**：现状 `RunCachedEmbedder`（:181-226）对两入口用相同 `run_id`+原文作键，且键不含嵌入器身份——同 Run 内换嵌入器配置即互相污染。修复三层：
  - **公开装配入口随行（R3-P1）**：`query_embedder(*, embedder, run_id, backend=None, embedder_identity=None)`（:240-261 现签名无身份参）增 `embedder_identity: Sequence[str] | None = None`——**类型是身份元组**（如 `config.identity()` 直接传入），不是摘要串：摘要由包装器内部以**规范 JSON 序列化**（`json.dumps(list(identity), ensure_ascii=False, separators=(",", ":"))`）后取 sha256 前 12 位计算，单一规范化点（R4-P1：`"|".join` 对含 `|` 的字段可碰撞——`("a|b","c","v","")` 与 `("a","b|c","v","")` 同串）。**None = 禁向量缓存语义**（安全侧默认）；非空元组即进共享缓存命名空间，自定义 Embedder 的宿主必须提供能唯一代表其向量语义的元组。
  - `RunCachedEmbedder` 构造增 `embedder_identity: Sequence[str] | None`：非空 → 进共享缓存（键含身份摘要段）；None → **bypass 模式仅跳过向量缓存读写（R4-P1 边界厘清）**——两条入口对 `QueryEmbeddingError` 的捕获 → `_freeze`（写降级标记 + 计数）→ 重抛**原样保留**（:194-198/:217-220/:230-233 的首败冻结语义与身份模式正交；`run/cache.py` 的降级标记键不含身份，冻结是 Run 级全域语义）。**向量缓存**与**降级标记**是两套键、两种生命周期，bypass 只作用于前者。禁用固定哨兵串（R3-P1）——要共享就必须拿出唯一身份元组，拿不出就明示禁缓存。
  - 键形状显式修订（`run/cache.py:34-40` 声明的键域契约）：`{prefix}:run:{run_id}:{identity_digest}:{entry}:{sha256(text)}`，`entry ∈ {"q","c"}`；进程内一级缓存键同形状复合隔离。
  - **调用面全量枚举（R2-P2）**：`query_key`（:47-49）签名变更，内部调用点 `get_vector`（:54-63）/`set_vector`（:65-77）与 `RunCachedEmbedder` 两入口全部随行；`degraded_key` 不变。
  - **后端命中回写一级缓存（R2-P2）**：`embed_corpus`（:213-226）后端命中即回写一级缓存（现状只跳过重嵌、未回写，随后按一级缓存取值落空）。
  - 迁移口径：旧 Redis 键随 TTL 自然过期，无迁移动作；docstring 记录新形状与 bypass 语义（只免缓存、不免冻结）。
- `embedders/local.py:16`：构造增 `pooling`；`_embed`（:39-56）三分支——`mean`（现状）、`cls`（**按 attention_mask 取每行首个非 pad 索引**，左填充时 `[:,0]` 取到 pad）、`last_token`（每行最后非 pad 索引）。local 侧前缀包装同款两入口。
- 新增 `tests/test_local_pooling.py`（替身 session）：三 pooling 数学、前缀拼接、**左/右 padding 双向用例**、非法 pooling 值、默认回归；`tests/test_run_cache.py` 增用例：同 run 同入口同文本**不同嵌入器身份**互不污染、**两个不同自定义嵌入器同 run 同文本（含清空一级缓存后再访问）**各自语义正确、`embedder_identity=None` 的 bypass 模式零向量缓存读写**且失败冻结语义完好（R4-P1：两入口失败后标记/计数/再装配阻断逐项断言）**、跨字段竖线碰撞（`("a|b","c","v","")` vs `("a","b|c","v","")`）经 `query_embedder` 隔离（清一级缓存前后均成立）、清空一级缓存后后端命中回写、q/c 分键。
- **迁移清单（R3-P2 补全）**：契约受影响的既有用例随契约更新并记录——`tests/test_api_embedder.py:31-32`（identity 三元断言）、`tests/test_run_cache.py:176-187`（**跨入口共享缓存断言与 q/c 分键相反**）、`tests/test_run_cache.py:125/:203` 等旧 `query_key` 签名调用、同文件其他经 `query_embedder` 的装配调用（新签名 `embedder_identity` 缺省 = bypass，缓存断言类用例需显式传身份元组）；文档迁移：`README.md:125` 的装配示例。其余回归断言零改动。

**如何验证**：L1 替身单元（pooling 数学 + identity 分化矩阵含冒号/竖线/空串/Unicode 前缀碰撞例 + 快照 id 兼容 + 缓存键入口与身份区分 + 回写路径）；L2：`build_embedder` api/local 两分支编排（api 前缀拼接用替身 HTTP 断言请求文本已带前缀）；L3：全量测试 + `tests/test_redis_store.py` + `tests/test_run_cache.py`（含更新后的契约用例）。
**验收标准**：
- mean+空前缀 → 输出与改动前逐位一致（api 与 local 两路）
- api 路径非空前缀 → 替身收到的 query/corpus 文本已带各自前缀
- cls/last_token 输出确定、互异、L2 归一化；左右 padding 双例一致；pooling="bogus" → 构造即 ValueError
- `identity()` flags：`("mean","a:b","c")` ≠ `("mean","a","b:c")`；默认 config flags == "" 且前三元与改动前逐位一致
- `vector_snapshot_id(默认 identity)` 与改动前逐位一致；显式 cls identity 产出不同快照 id
- Run 缓存：同 run 同文本 q/c 两入口互不污染；同 run 同入口同文本不同嵌入器身份互不污染；跨字段竖线碰撞对 `("a|b","c","v","")` vs `("a","b|c","v","")` 产生不同摘要且隔离（清一级缓存前后均成立）；两个不同自定义嵌入器各自语义正确（含清空一级缓存后再访问）；`embedder_identity=None` bypass 零**向量缓存**读写**且失败冻结语义完好**（失败 → 降级标记写入 + 计数 + 同 Run 再装配被阻断）；后端命中回写一级缓存（清空一级后再取仍命中）
- 既有测试：非契约受影响用例零改动全绿；契约受影响用例（迁移清单：`test_api_embedder` identity 断言、`test_run_cache` 键形状/跨入口共享/旧签名装配用例、`README.md:125` 示例）随契约更新并在 resolution 记录
**评审要点**：identity 是内容寻址正确性根（ADR-0001），flags 空值语义使默认不参与哈希、显式必分化；Run 缓存键的 entry 与身份两维缺一即污染、身份拿不出即明示 bypass（bypass 只免缓存、不免冻结）；padding 方向陷阱双覆盖。
**定级依据**：改 `identity()` 契约本体 + `run/cache.py` 键形状契约与 `query_embedder` 公共装配签名 + `ApiEmbedder` 公共构造签名 + `redis_store.py` 快照消费方适配 + 新增推理路径无既有覆盖——**T3**。

## 评审响应记录

| R1 发现 | 响应 |
|---|---|
| P1 Run 缓存混用查询/语料向量 | 采纳——B6 增 `run/cache.py` 入口分键 + 端到端污染用例 |
| P1 flags 分隔符碰撞 | 采纳——改 JSON 数组规范序列化 + 四类碰撞测试 |
| P1 docgen 兼容入口不覆盖自由文本生成 | 采纳——包增 `complete` 基元，薄壳 `chat_text` 兼容入口，groupdesc/summarize 保留自有提示词；`generate_queries` 兼容入口保 `main()` 零改动 |
| P1 paired_report 文件依赖遗漏 | 采纳——`load_arm_cases` 一并平移 + 包侧端到端小样本用例 |
| P1 Judge 端口缺 retrieval_instruction | 采纳——裁决为逐次调用参数（`instruction: str = ""`），贯穿端口/实现/对拍非空用例；ADR「τ 不进端口」不涉此 |
| P1 新构造器无法证明语义保留 | 采纳——B3 双判定：判定一（离线）与 `jev_rerank` 旧构造逐字段对照、差异显式登记（继承 jev_rerank 语义、不继承 jev_retrieval 版，tags 不进端口）；判定二（live）传输等价 |
| P2 CLS 左填充取到 pad | 采纳——cls 按 mask 取首非 pad |
| P2 exact 短路表述不一致 | 采纳——短路归属调用方编排，L2 调用计数断言 |
| P2 identity 返回形状变化 | 采纳——0.x 兼容性明示入组件文档，公开返回与快照 id 分测 |

| R2 发现 | 响应 |
|---|---|
| P1 Run 缓存缺嵌入器身份隔离 | 采纳——键形状增 `identity_digest` 段贯穿三层；自定义 Embedder 契约写明；同 run 同入口同文本不同身份的隔离用例 |
| P1 docgen 兼容入口与 CLI 签名不匹配 | 采纳——`generate_queries` 保留原签名（内部组装 config），`main()` 零改动成立；替身覆盖 CLI 生成路径 |
| P1 判停辅助无法读排序 ID | 采纳——`select_with_judge` 增 `item_id` 回调参数（照 `rank_candidates` 风格）+ 非对象型宿主数据用例 |
| P1 前缀未覆盖 API 路径 | 采纳——裁决为 config 级能力两路生效：`ApiEmbedder` 构造增双前缀并拼接，替身断言请求文本带前缀 |
| P2 缓存分键调用面与回写遗漏 | 采纳——`query_key/get_vector/set_vector` 及全部调用点随行；后端命中回写一级缓存 + 用例 |
| P2 重复 ID 集合比较失效 | 采纳——先断言 id 列表长度==去重数（重复即 ValueError），NaN 归入越界拦截并测试钉住 |
| R2 发现 | 响应 |
|---|---|
| P1 Run 缓存缺嵌入器身份隔离 | 采纳——键形状增 `identity_digest` 段贯穿三层；自定义 Embedder 契约写明；同 run 同入口同文本不同身份的隔离用例 |
| P1 docgen 兼容入口与 CLI 签名不匹配 | 采纳——`generate_queries` 保留原签名（内部组装 config），`main()` 零改动成立；替身覆盖 CLI 生成路径 |
| P1 判停辅助无法读排序 ID | 采纳——`select_with_judge` 增 `item_id` 回调参数（照 `rank_candidates` 风格）+ 非对象型宿主数据用例 |
| P1 前缀未覆盖 API 路径 | 采纳——裁决为 config 级能力两路生效：`ApiEmbedder` 构造增双前缀并拼接，替身断言请求文本带前缀 |
| P2 缓存分键调用面与回写遗漏 | 采纳——`query_key/get_vector/set_vector` 及全部调用点随行；后端命中回写一级缓存 + 用例 |
| P2 重复 ID 集合比较失效 | 采纳——先断言 id 列表长度==去重数（重复即 ValueError），NaN 归入越界拦截并测试钉住 |
| P2 契约变更与零改动验收冲突 | 采纳——验收口径改为「非契约受影响用例零改动；契约受影响用例（清单列出）随契约更新并记录」 |
| P2 paired_report 示例缺 metrics 参 | 采纳——示例补第五必需参，签名照原平移不变 |
| sug 判定一冻结范围 | 采纳——冻结完整请求（state + 每问题 instructions/criteria + 候选顺序 + 响应映射），tags 差异单独列报 |
| sug corpus_hash 表述 | 采纳——改为「生成后语料文本变化时才产生新哈希；identity 未接入该哈希」 |

| R3 发现 | 响应 |
|---|---|
| P1 item_id 回调作用于错误数据层 | 采纳——签名显式 `hits: Sequence[CandidateHit[T]]`、`item_id: Callable[[T], str]`、提取式 `item_id(hit.item)`；组合用例经真实 `rank_candidates` |
| P1 固定串 "custom" 无禁共享语义 | 采纳——废除哨兵串，改 `embedder_identity: str \| None`：None = bypass 模式（穿透两级缓存零读写），非空才进共享命名空间；两个自定义嵌入器 + bypass 用例 |
| P1 身份未穿透公开装配入口 | 采纳——`query_embedder` 增 `embedder_identity: str \| None = None`（缺省 bypass，安全侧）；`README.md:125` 与装配类测试列入迁移清单 |
| P1 五字段候选 vs 三键旧请求矛盾 | 采纳——投影口径定稿：state 每候选只 `{name, description}`（继承旧版），`use_when/avoid_when` 不投影（旧版本来没有，无差异）；对拍差异恰为 tags 删除一项；样本加非空 use_when/avoid_when 与超 2000 字符 description |
| P2 迁移清单不完整 | 采纳——补 `test_run_cache.py:176-187` 跨入口共享断言（与分键相反）、旧 `query_key` 签名调用、其他装配调用、`README.md:125` |
| P2 docgen 参数名漂移 + 消费方生成路径未验收 | 采纳——兼容入口保留原参数名（`resource_name` 等三名均可关键字传入）；验收增两消费方生成函数替身直调（提示词逐位一致 + 自由文本返回） |
| sug 冻结样本含响应映射 | 采纳——payload 与响应键→id 映射分存两节，映射与候选顺序一起核验 |

| R4 发现 | 响应 |
|---|---|
| P1 bypass 绕过首败冻结 | 采纳——bypass 边界厘清：仅跳过**向量缓存**读写，`QueryEmbeddingError` 捕获 → `_freeze` → 重抛原样保留（冻结是 Run 级全域语义，与身份模式正交；向量缓存与降级标记是两套键两种生命周期）；验收增不传身份的两入口失败冻结逐项断言 |
| P1 身份摘要 join 可碰撞 | 采纳——公开入口改为接收**身份元组**（`Sequence[str]`，如 `config.identity()` 直传），摘要由包装器内部以规范 JSON 序列化统一计算（单一规范化点）；跨字段竖线碰撞用例（清一级缓存前后均隔离） |
| P2 数据源无超长描述样本 | 采纳——对拍脚本内构造 >2000 字符候选（同文本适配旧 `Resource` 与新 `JudgedCandidate` 两型），断言两侧截断逐位一致；非空 use_when/avoid_when 覆盖保留 |

| R5 发现（迭代循环五轮封顶后补录，修订直接落方案） | 响应 |
|---|---|
| P1 实验侧持久化缓存未穿透前缀与身份 | 采纳——B6 增 `embedder_cache.py` 键形状（入口+身份维度）与未命中分支按入口分发；五个装配点身份串同步 |
| P1 `vector_snapshot_id` join 可碰撞 | 采纳——flags 空沿用旧 join（零迁移），flags 非空切 `\x00v2\x00` + JSON 规范序列化的新哈希输入域 |
