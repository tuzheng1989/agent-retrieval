# SDK P2 实施方案（分批优化计划）：scenario-matrix 铺开

> 依据：`docs/roadmap.md` 立项清单第 5 步 + `docs/adr/0001-sdk-integration-boundaries.md`（P2 = scenario-matrix 铺开；验收实体 `docs/scenario-matrix.md`：4 部署姿态 × 2 语言 × 3 资源形态，每格 extras + 配置片段 + 验证命令；无载体格显式标「未验证」，不留空白）。
> 范围：部署姿态定义、24 格清单、每格验证载体（可执行）、非卡点格铺开与验证、zh-plot 格占位接线。
> 宿主纪律不变：包零改动（P2 是组合化验证与文档，不是组件收编——P0/P1 已把组件备齐）。
> 评审状态：定稿（2026-11-05）。S-D1/S-D2 经用户裁决采纳；自审一轮（1 P1 + 3 P2 + 4 sug）已全部修复（格 id 下划线化、S-D1 ④锚点声称收窄、判定格归属写死、M2-M4 补如何验证要素、zh-plot 分工/载体/中间态口径）。
> 实施状态：**已完成**（2026-11-05）。M1-M4 四批（M2 审计 P1 退回修复 revision 2：守护改基线差集 + 全量 396 实证；M3 审计 PASS；M4 主控逐格执行）全收口。**验收实体达成**：`docs/scenario-matrix.md` 24 格状态齐（16 ✅ 已验证 2026-11-05 / 8 `pending.zh_plot`，零空白）——离线 8 格 CI 常驻、live 8 格逐格退出码证据（`docs/plans/.flow-evidence/M4/*.log`，含 ChatJudge live 判定与 Run 缓存/快照/双路融合参与断言全真）。全量 396 passed + ruff 全仓。遗留：zh_plot 8 格随 zh-plot-v1 接线；M2 sug-1 语料版本耦合注记随 M4 回写已留（matrix §5 判定示例节）。

## 设计裁决（前提假设，逐条可挑战）

| # | 裁决 | 理由与取舍 |
|---|---|---|
| S-D1 | **四种部署姿态按 README Quickstart 四节正式定义**（事实锚点已存在，非新造）：① `core-only` 进程内最小（纯 BM25 内核直调，零 extras）；② `api-embed` 进程内 API 嵌入（`[api]`，融合检索）；③ `redis-snapshot` 生产快照（`[api,redis]`，write-once 快照 + 消费）；④ `full-stack` 全链路（`[all]`，Run 缓存 + 判定面 + docgen 语料扩展——①②③直接锚定 README 同名节，④ 的 Run 缓存锚定第 4 节、判定面与 docgen 组合是本方案裁决（依据 P0/P1 组件事实与 README Quality benchmarks 段），非 README 原文）。已裁决采纳 | ADR 只给结构未给定义；README 四节是被文档验证过的渐进组合，映射自然；候选替代（HTTP 服务化姿态）被否——服务化是宿主形态不是包形态，包的服务化包装属宿主示例不属于包验收 |
| S-D2 | **中文格载体用 `experiments/agent_e2e/catalogs/catalog-v1.json`（P2 前已双语化）**：tool/skill 形态的中文格以它为语料（10 资源中文声明面齐备、判定语义经冒烟验证）；zh-plot 形态 8 格标 `pending.zh_plot`（金版本已有同款占位语义），随 zh-plot-v1 构建接线 | 中文格「不留空白」要求有真载体——agent_e2e catalog 是现成的中文（双语）资源池，判定面（Jev/ChatJudge）在冒烟已实测；中文格判定输入按 gate.py declared 口径消费 name+description（use_when/avoid_when 为英文，不入判定输入）；zh-plot 数据集未建（HITL） |
| S-D3 | **验证载体混合制**：离线格（零外部依赖）进 `tests/test_scenario_matrix.py`（pytest 参数化，CI 常驻）；live 格（需嵌入/判定网关）为 `experiments/scenario_checks/` 独立脚本（`python -m experiments.scenario_checks.<格 id>`（格 id 即模块名，下划线连接），退出码语义与冒烟同构 0/4），CI 不跑 | 与既有纪律对齐：金版本 gate 是 pytest 离线常驻；live 脚本（judge_parity、judge_chat_smoke）是宿主侧手动验证。矩阵格的可复现性由「命令 + 断言」双写保证 |
| S-D4 | **格「已验证」判定标准**：该格验证命令 exit 0 且断言最小检索行为（非空候选 + top-1 合理 + 融合/判定组件确实参与了——通过组件存在性断言而非阈值断言，避免把质量红线混入矩阵） | 质量红线归金版本 gate（既有职责分离）；矩阵验证的是「组合可达性 + 装配正确性」 |
| S-D5 | **scenario-matrix.md 每格四件**：extras 安装命令、配置片段（可复制）、验证命令（可执行）、验证状态（✅ 已验证含日期 / `pending.zh_plot` / `未验证`+缺载体原因） | ADR 原文四件套口径；「不留空白」由状态枚举强制 |
| S-D6 | **包零改动**：矩阵验证若暴露包缺口（如某姿态缺工厂函数），缺口记档开新票，不在本方案内顺手改包 | P2 是验收与文档批；包变更走独立立项（收编判据照 ADR） |

## 路线图

| 批次 | 内容 | T 级 | 依赖 |
|---|---|---|---|
| M1 | 24 格清单冻结 + `docs/scenario-matrix.md` 骨架（全格四件初稿，状态待验证） | T1 | — |
| M2 | 离线验证机制：`tests/test_scenario_matrix.py` 参数化骨架（core-only/api-embed 姿态 × 中英 × tool/skill 共 8 格——用 MockEmbedder/catalog-v1 替身） | T2 | M1 |
| M3 | live 验证脚本：`experiments/scenario_checks/`（redis-snapshot/full-stack 姿态 8 格 + live 判定格） | T2 | M1 |
| M4 | 逐格执行验证 + `scenario-matrix.md` 状态回写（✅/日期）+ zh-plot 8 格占位接线（接金版本 `pending.zh_plot` 语义）+ 末波收口 | T1 | M2、M3 |

**波次**：W1 = M1；W2 = M2 ∥ M3（文件集互斥：M2 动 tests/，M3 动 experiments/scenario_checks/）；W3 = M4 + 末波收口。
依赖图：M1 → {M2, M3} → M4，无环。

## 详案

### M1 格清单冻结 + 矩阵骨架（T1）

**如何改**：
- 新建 `docs/scenario-matrix.md`：头部（目的/判定标准 S-D4/与金版本 gate 的职责分工）+ 四姿态定义（S-D1 正式化：姿态名、extras、适用场景、对应 README 节）+ 24 格主表（行=姿态，列=语言×形态；每格四件初稿（状态列初稿为「待验证」中间态，M4 收口归一为三枚举））+ zh-plot 占位节（接 `pending.zh_plot` 语义）。
- 格 id 命名：`<姿态>_<语言>_<形态>` 下划线连接（如 `api_embed_zh_tool`——`python -m` 模块名不允许连字符，文档/脚本/模块名三方共用同一串）——脚本与文档共用此 id（权威载体：scenario-matrix.md 主表）。
- 中文格语料指针：tool/skill 中文格一律指 `experiments/agent_e2e/catalogs/catalog-v1.json`（S-D2）。
**如何验证**：人工对照 ADR 结构（4×2×3 全覆盖、无空白）+ 格 id 唯一性脚本自检。
**验收标准**：24 格齐、每格四件初稿（状态列初稿为「待验证」中间态，M4 收口归一为三枚举）齐、格 id 全仓唯一、姿态定义与 README 四节一一对应、zh-plot 8 格状态直接标 `pending.zh_plot`（初稿终态，M4 仅补票链接）。
**评审要点**：这是后续三批的需求出处（单一出处纪律同 contract.md 模式）。
**定级依据**：纯文档，T1。

### M2 离线验证机制（T2）

**如何改**：
- 新建 `tests/test_scenario_matrix.py`：8 格参数化（core-only × {en,zh} × {tool,skill} 用 jev_cases/agent_e2e 语料替身；api-embed × {en,zh} × {tool,skill} 用 MockEmbedder + catalog-v1）——每格断言：装配成功、检索非空、top-1 合理（检索词命中目标资源）、组件参与存在性（S-D4）。
- 替身边界：零外部依赖（MockEmbedder 既有组件 + 既有 fixtures 语料），CI 常驻。
**如何验证**：L1——8 格参数化测试全绿即验证本体（测试即被验物）；失败报错含格 id 可定位。L2——对照 `test_core_zero_dep` 纪律跑零 extras 导入检查。
**验收标准**：8 格全绿；每格失败时报错含格 id；`core-only` 格零 extras 导入（对照 `test_core_zero_dep` 纪律）。
**评审要点**：替身语料与真实语料的映射关系写进测试 docstring；不复制金版本 gate 的职责（不设质量阈值）。
**定级依据**：新测试文件 + 邻近 CI 契约（pytest 收集面）但纯新增——T2。

### M3 live 验证脚本（T2）

**如何改**：
- 新建 `experiments/scenario_checks/__init__.py` + 逐格脚本（`redis_snapshot_en_tool.py` 等 8+ 个：redis-snapshot × {en,zh} × {tool,skill} 4 格 + full-stack × 4 格；full-stack 4 格的验证脚本内含判定组件参与断言（判定格属 24 格内，非矩阵外附加项））。每个脚本：装配该格 extras + 配置 → 执行一次检索/判定 → 断言 → 退出码 0/4（缺配置指路，照冒烟同构）。Redis 依赖用真实连接（宿主环境变量 `REDIS_URL`），无 Redis 时 exit 4 指路（不假装通过）。
- 共享装配工具函数放 `experiments/scenario_checks/_common.py`（格 id → 装配函数映射，单一出处）。
**如何验证**：L1——逐脚本 `python -m` 干跑（缺配置路径 exit 4 断言）；L2——有配置时逐脚本真跑 exit 0（M4 执行验证统一做，本批只验机制）。
**验收标准**：每脚本独立可跑、缺依赖 exit 4 指路含变量名、断言面非空；`_common.py` 是装配唯一出处。
**评审要点**：脚本不复制包内逻辑（只装配消费）；Redis 不可达的显式失败优于跳过。
**定级依据**：多脚本 + experiments 层 live 纪律——T2。

### M4 执行验证 + 状态回写 + 占位接线 + 收口（T1）

**如何改**：
- 逐格执行：离线 8 格跑 pytest（CI 已常驻）；live 格由主控带环境变量逐格跑（Redis URL / 网关配置注入），每格状态回写 `scenario-matrix.md`（✅ + 日期）。
- zh-plot 8 格补金版本占位与 zh-plot-v1 票的链接（状态 M1 已标）。
- 无载体格（若有——如某姿态在某语言缺语料）标 `未验证` + 缺载体原因，不留空白。
- 末波收口：全量测试 + ruff。
**如何验证**：L3——逐格执行验证命令留退出码证据；全量测试 + ruff 收口；文档三枚举零空白机械检查。
**验收标准**：24 格状态齐（✅/pending.zh_plot/未验证+原因 三枚举之一，零空白）；离线 8 格 CI 绿；live 格各留退出码证据；文档与脚本格 id 一致。
**评审要点**：状态回写是验收实体的最终形态——ADR「无载体格显式标未验证，不留空白」在此闭环。
**定级依据**：文档回写 + 逐格执行，无代码——T1。

## 与既有轨的边界

- zh-plot-v1 构建（清单 #2）不进本方案：8 格占位随 zh-plot-v1 接线，届时另立。
- 质量红线归金版本 gate；本方案的格验证不含阈值断言（S-D4）。
- HTTP 服务化姿态不在四姿态内（S-D1 否决理由）：宿主示例若需要，属消费方文档，另立。
