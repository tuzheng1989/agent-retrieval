# Scenario Matrix（场景矩阵）

> 组合化验收实体：4 部署姿态 × 2 语言 × 3 资源形态 = 24 格，每格四件（extras 安装命令、配置片段、验证命令、验证状态），无载体格显式标注、不留空白。
> 需求出处：`docs/plans/sdk-p2-plan.md`（S-D1～S-D6 设计裁决）；结构出处：`docs/adr/0001-sdk-integration-boundaries.md`。
> **本主表是格 id 的权威载体**：文档、`tests/test_scenario_matrix.py`、`experiments/scenario_checks/` 三方共用同一串格 id，以本表为单一出处。

## 1. 目的、判定标准与状态枚举

### 1.1 目的

验证「包 + extras + 语料 + 形态」的组合可达性与装配正确性：任何一种部署姿态下，任何一种语言与资源形态的语料，都能按本文档装配出可工作的检索组合。

### 1.2 判定标准（S-D4）

一格「已验证」当且仅当：

1. 该格验证命令 **exit 0**；
2. 断言**最小检索行为**：候选非空 + top-1 合理（检索词命中目标资源）+ 融合/判定组件确实参与（**组件存在性断言**，非阈值断言）。

不设质量阈值。质量红线归金版本 gate（`experiments/golden/gate.py`）——矩阵与 gate 职责分离：矩阵验「组合可达 + 装配正确」，gate 验「检索质量不退化」。

### 1.3 格 id 规则

`<姿态>_<语言>_<形态>`，全部下划线连接（如 `api_embed_zh_tool`）。

- 姿态段：`core_only` / `api_embed` / `redis_snapshot` / `full_stack`（§2）；
- 语言段：`en` / `zh`；
- 形态段：`tool` / `skill` / `zh_plot`。

下划线连接的原因：`python -m experiments.scenario_checks.<格 id>` 的模块名不允许连字符，文档、pytest `-k` 过滤、live 脚本模块名三方共用同一串。**格 id 权威载体 = 本文 §3 主表**；新增/改名格先改主表，再同步脚本。

### 1.4 状态三枚举

| 状态 | 语义 |
|---|---|
| `✅ 已验证（YYYY-MM-DD）` | 验证命令 exit 0 且 S-D4 断言通过，日期 = 执行日 |
| `pending.zh_plot` | zh-plot 形态占位：语料载体 zh-plot-v1 数据集未建（§4） |
| `未验证 + 缺载体原因` | 语料或依赖载体缺失，显式写明原因，不留空白 |

**M1 中间态（已归一）**：M1 冻结时 16 格标「待验证」；M4 逐格执行验证后于 2026-11-05 统一回写为 ✅（离线 8 格 CI 证据、live 8 格逐格退出码证据见 `docs/plans/.flow-evidence/M4/`），「待验证」字样已清零。

## 2. 四姿态定义（S-D1）

四种部署姿态按 README Quickstart 四节正式定义（①②③直接锚定 README 同名节；④ 的 Run 缓存锚定 §4，判定面与 docgen 组合是方案裁决，依据 P0/P1 组件事实与 README Quality benchmarks 段，非 README 原文）。

### 2.1 core_only —— 进程内最小（README §1）

- **extras 安装命令**：`pip install agent-retrieval`（零 extras，core 仅标准库）
- **适用场景**：嵌入式进程内检索、零依赖约束环境、BM25 单路已满足召回的场景
- **README 锚定节**：§1「1. Rank candidates with BM25 (+ optional vector path)」
- **装配片段**（`rank_candidates` 直调，纯 BM25；向量路缺省即单路）：

```python
from dataclasses import dataclass

from agent_retrieval import rank_candidates

@dataclass(frozen=True)
class Tool:
    id: str
    name: str
    description: str

tools = [
    Tool("geo-query", "Geo Query", "query geospatial data by region and time range"),
    Tool("doc-search", "Doc Search", "full-text search over uploaded documents"),
]

def corpus_text(t: Tool) -> str:      # 声明面：BM25 索引与判定证据同语料
    return f"{t.id} {t.name} {t.description}"

hits = rank_candidates(
    tools,
    "find population data for California in 2024",
    corpus_text=corpus_text,
    item_id=lambda t: t.id,
    item_name=lambda t: t.name,
)                                     # 不传 vector：纯 BM25 单路（零 extras）
for hit in hits:
    print(hit.fused_score, hit.item.name, hit.matched_terms)
```

### 2.2 api_embed —— 进程内 API 嵌入（README §2）

- **extras 安装命令**：`pip install "agent-retrieval[api]"`
- **适用场景**：进程内融合检索（BM25 + 向量双路 RRF），嵌入走 OpenAI 兼容端点，无需 Redis
- **README 锚定节**：§2「2. Real embeddings」（装配）+ §1（`rank_candidates` 注入 `vector=` 消费）
- **装配片段**（`EmbeddingConfig` / `build_embedder` 形态；凭据由宿主配置体系解析后传入，包不读环境变量）：

```python
from agent_retrieval import EmbeddingConfig, build_embedder, rank_candidates, vector_available

config = EmbeddingConfig(
    kind="api", provider="zhipu", model_name="embedding-3",
    base_url="https://open.bigmodel.cn/api/paas", api_key="...",
    dimensions=2048,
)
embedder = build_embedder(config)     # None = 向量路缺席，BM25 现状继续（不是异常）
print(vector_available(config))       # 部署形态三态诊断："api" | "local" | "none"

hits = rank_candidates(
    tools, "按区域与年份查询人口统计数据",
    corpus_text=corpus_text,          # 同 §2.1：声明面双路共用
    item_id=lambda t: t.id,
    item_name=lambda t: t.name,
    vector=embedder,                  # 注入后双路 RRF 融合
)
```

### 2.3 redis_snapshot —— 生产快照（README §3）

- **extras 安装命令**：`pip install "agent-retrieval[api,redis]"`
- **适用场景**：生产部署的 write-once 向量快照——发布侧全量嵌入一次，消费侧按 `owner_version` 读取子集；快照身份内容寻址，语料/模型不变即零重嵌
- **README 锚定节**：§3「3. Write-once vector snapshots (Redis)」
- **装配片段**（`RedisVectorStore` / `vector_snapshot_id` / `corpus_hash` 形态；省略 `async def main()` 包装，`asyncio.run` 进入）：

```python
from agent_retrieval import (
    EmbeddingConfig,
    RedisVectorStore,
    build_embedder,
    corpus_hash,
    vector_snapshot_id,
)

config = EmbeddingConfig(...)         # 同 §2.2；嵌入靠 [api]，快照靠 [redis]
embedder = build_embedder(config)

store = RedisVectorStore(redis_url="redis://localhost:6379/0", key_prefix="myapp")

# 语料文本映射 {"tool:geo-query": "geo-query Geo Query ..."}（声明面同语料）
corpus_texts = {f"tool:{t.id}": corpus_text(t) for t in tools}

snapshot_id = vector_snapshot_id(
    owner_version="catalog-v1",              # 候选集版本（宿主资源池版本号）
    identity=config.identity(),              # 嵌入器身份四元组进快照 id
    corpus_hash=corpus_hash(corpus_texts),   # 语料内容 hash（键序固定）
)

if await store.read_meta(snapshot_id) is None and await store.acquire_lock(snapshot_id):
    # 锁内发布：语料侧全量嵌入 + write-once 写入（向量 + meta + owner 指针）
    vectors = dict(zip(
        corpus_texts.keys(),
        embedder.embed_corpus(list(corpus_texts.values())),
    ))
    await store.publish_vectors(
        snapshot_id=snapshot_id,
        owner_version="catalog-v1",
        vectors=vectors,
        meta={"provider": config.provider, "dim": str(config.dimensions)},
        retention_seconds=7 * 24 * 3600,
    )
    await store.release_lock(snapshot_id)
else:
    # 同 id 重发布 = write-once 预检命中：只续 TTL，零重嵌
    await store.extend_ttl(snapshot_id, "catalog-v1", 7 * 24 * 3600)

# 消费面：按 owner_version 取子集向量；{} = 快照缺席 → 向量路缺席降级 BM25
vectors = await store.load_snapshot_vectors("catalog-v1", ["tool:geo-query"])
await store.aclose()
```

### 2.4 full_stack —— 全链路（README §4 + 判定/docgen 组合裁决）

- **extras 安装命令**：`pip install "agent-retrieval[all]"`（= `[api,redis,docgen,judge]`）
- **适用场景**：完整生产形态——Redis 快照消费 + Run 级查询缓存 + docgen 语料扩展（发布流程离线消费）+ 判定面（检索后裁决）
- **README 锚定节**：§4「4. Per-run query caching with degrade freezing」（Run 缓存锚）；判定面与 docgen 组合为方案裁决（P0/P1 组件 + README「Quality benchmarks」段指向的 `docs/retrieval-config-guide.md` Jev-rerank 选型）
- **装配片段**（在 §2.3 redis_snapshot 基础上加 `query_embedder` + `build_docgen` + 判定面）：

```python
from agent_retrieval import (
    ChatJudge,
    ChatJudgeConfig,
    JudgedCandidate,
    build_embedder,
    clear_run_cache,
    make_backend,
    query_embedder,
    rank_candidates,
)
from agent_retrieval.expansion import DocGenConfig, build_docgen  # 公开出口是 expansion 子模块（M3 勘误）

# ① Run 级查询缓存（README §4）：查询向量每 Run 只嵌一次；首败冻结整 Run 向量路
embedder = query_embedder(
    embedder=build_embedder(config),         # api 形态嵌入器（同 §2.2）
    run_id="run-123",                        # 宿主执行域 id（显式注入）
    backend=make_backend(redis_url="redis://localhost:6379/0", key_prefix="myapp"),
    embedder_identity=config.identity(),     # 身份四元组命名缓存键域；None = bypass 不缓存
)
hits = rank_candidates(
    tools, "按区域与年份查询人口统计数据",
    corpus_text=corpus_text,
    item_id=lambda t: t.id,
    item_name=lambda t: t.name,
    vector=embedder,
)

# ② 语料扩展（doc2query）：发布流程离线消费；None = 未配置 docgen 的正常形态
docgen = build_docgen(DocGenConfig(
    base_url="https://open.bigmodel.cn/api/paas", api_key="...",
    model_name="glm-4.6",                    # api_style="chat"（缺省），questions_per_resource=5
))
if docgen is not None:
    queries = docgen.generate(
        resource_name=t.name,
        resource_description=t.description,
        resource_tags="geo, census",
    )                                        # -> list[str]

# ③ 判定面：检索候选声明能力裁决（判定输入按 declared 口径只投影 name+description；
#    System One 端点换 JevJudge + JevJudgeConfig，judge 端口语义一致）
judge = ChatJudge(ChatJudgeConfig(
    base_url="https://open.bigmodel.cn/api/paas", api_key="...", model_name="glm-4.6",
))
judgment = judge.judge("按区域与年份查询人口统计数据", [
    JudgedCandidate(id=t.id, name=t.name, description=t.description)
    for t in (hit.item for hit in hits[:5])
])
# judgment.probabilities: {候选 id: 概率}

clear_run_cache()                            # Run 收尾：清进程内一级缓存与降级计数
```

## 3. 24 格主表（冻结清单）

### 3.1 总览

行 = 姿态，列 = 语言 × 形态；格内为格 id。

| 姿态 \ 格 | en · tool | en · skill | en · zh_plot | zh · tool | zh · skill | zh · zh_plot |
|---|---|---|---|---|---|---|
| core_only | `core_only_en_tool` | `core_only_en_skill` | `core_only_en_zh_plot` | `core_only_zh_tool` | `core_only_zh_skill` | `core_only_zh_zh_plot` |
| api_embed | `api_embed_en_tool` | `api_embed_en_skill` | `api_embed_en_zh_plot` | `api_embed_zh_tool` | `api_embed_zh_skill` | `api_embed_zh_zh_plot` |
| redis_snapshot | `redis_snapshot_en_tool` | `redis_snapshot_en_skill` | `redis_snapshot_en_zh_plot` | `redis_snapshot_zh_tool` | `redis_snapshot_zh_skill` | `redis_snapshot_zh_zh_plot` |
| full_stack | `full_stack_en_tool` | `full_stack_en_skill` | `full_stack_en_zh_plot` | `full_stack_zh_tool` | `full_stack_zh_skill` | `full_stack_zh_zh_plot` |

### 3.2 core_only 姿态（6 格）

配置片段一律见 §2.1 装配片段（单一出处）。离线格进 `tests/test_scenario_matrix.py`（M2 建，CI 常驻）。

| 格 id | extras 安装命令 | 配置片段 | 验证命令 | 状态 | 语料指针 |
|---|---|---|---|---|---|
| `core_only_en_tool` | `pip install agent-retrieval` | 见 core_only 节装配片段 | `python -m pytest tests/test_scenario_matrix.py -k core_only_en_tool` | ✅ 已验证（2026-11-05） | `experiments/jev_cases.json` |
| `core_only_en_skill` | `pip install agent-retrieval` | 见 core_only 节装配片段 | `python -m pytest tests/test_scenario_matrix.py -k core_only_en_skill` | ✅ 已验证（2026-11-05） | `experiments/agent_e2e/catalogs/catalog-v1.json` |
| `core_only_en_zh_plot` | `pip install agent-retrieval` | 见 core_only 节装配片段 | 待 zh-plot-v1 接线（无验证命令） | `pending.zh_plot` | zh-plot-v1 数据集（未建，见 §4） |
| `core_only_zh_tool` | `pip install agent-retrieval` | 见 core_only 节装配片段 | `python -m pytest tests/test_scenario_matrix.py -k core_only_zh_tool` | ✅ 已验证（2026-11-05） | `experiments/agent_e2e/catalogs/catalog-v1.json` |
| `core_only_zh_skill` | `pip install agent-retrieval` | 见 core_only 节装配片段 | `python -m pytest tests/test_scenario_matrix.py -k core_only_zh_skill` | ✅ 已验证（2026-11-05） | `experiments/agent_e2e/catalogs/catalog-v1.json` |
| `core_only_zh_zh_plot` | `pip install agent-retrieval` | 见 core_only 节装配片段 | 待 zh-plot-v1 接线（无验证命令） | `pending.zh_plot` | zh-plot-v1 数据集（未建，见 §4） |

### 3.3 api_embed 姿态（6 格）

配置片段一律见 §2.2 装配片段。离线格用 MockEmbedder/语料替身（M2 建，零外部依赖，CI 常驻）。

| 格 id | extras 安装命令 | 配置片段 | 验证命令 | 状态 | 语料指针 |
|---|---|---|---|---|---|
| `api_embed_en_tool` | `pip install "agent-retrieval[api]"` | 见 api_embed 节装配片段 | `python -m pytest tests/test_scenario_matrix.py -k api_embed_en_tool` | ✅ 已验证（2026-11-05） | `experiments/jev_cases.json` |
| `api_embed_en_skill` | `pip install "agent-retrieval[api]"` | 见 api_embed 节装配片段 | `python -m pytest tests/test_scenario_matrix.py -k api_embed_en_skill` | ✅ 已验证（2026-11-05） | `experiments/agent_e2e/catalogs/catalog-v1.json` |
| `api_embed_en_zh_plot` | `pip install "agent-retrieval[api]"` | 见 api_embed 节装配片段 | 待 zh-plot-v1 接线（无验证命令） | `pending.zh_plot` | zh-plot-v1 数据集（未建，见 §4） |
| `api_embed_zh_tool` | `pip install "agent-retrieval[api]"` | 见 api_embed 节装配片段 | `python -m pytest tests/test_scenario_matrix.py -k api_embed_zh_tool` | ✅ 已验证（2026-11-05） | `experiments/agent_e2e/catalogs/catalog-v1.json` |
| `api_embed_zh_skill` | `pip install "agent-retrieval[api]"` | 见 api_embed 节装配片段 | `python -m pytest tests/test_scenario_matrix.py -k api_embed_zh_skill` | ✅ 已验证（2026-11-05） | `experiments/agent_e2e/catalogs/catalog-v1.json` |
| `api_embed_zh_zh_plot` | `pip install "agent-retrieval[api]"` | 见 api_embed 节装配片段 | 待 zh-plot-v1 接线（无验证命令） | `pending.zh_plot` | zh-plot-v1 数据集（未建，见 §4） |

### 3.4 redis_snapshot 姿态（6 格）

配置片段一律见 §2.3 装配片段。live 格为 `experiments/scenario_checks/` 独立脚本（M3 建；Redis 依赖真实连接，环境变量 `REDIS_URL`，缺配置 exit 4 指路；CI 不跑）。中文格判定输入按 declared 口径 name+description（`projected_candidates` 投影；catalog 的 use_when/avoid_when 为英文，不入判定输入）。

| 格 id | extras 安装命令 | 配置片段 | 验证命令 | 状态 | 语料指针 |
|---|---|---|---|---|---|
| `redis_snapshot_en_tool` | `pip install "agent-retrieval[api,redis]"` | 见 redis_snapshot 节装配片段 | `python -m experiments.scenario_checks.redis_snapshot_en_tool` | ✅ 已验证（2026-11-05） | `experiments/jev_cases.json` |
| `redis_snapshot_en_skill` | `pip install "agent-retrieval[api,redis]"` | 见 redis_snapshot 节装配片段 | `python -m experiments.scenario_checks.redis_snapshot_en_skill` | ✅ 已验证（2026-11-05） | `experiments/agent_e2e/catalogs/catalog-v1.json` |
| `redis_snapshot_en_zh_plot` | `pip install "agent-retrieval[api,redis]"` | 见 redis_snapshot 节装配片段 | 待 zh-plot-v1 接线（无验证命令） | `pending.zh_plot` | zh-plot-v1 数据集（未建，见 §4） |
| `redis_snapshot_zh_tool` | `pip install "agent-retrieval[api,redis]"` | 见 redis_snapshot 节装配片段 | `python -m experiments.scenario_checks.redis_snapshot_zh_tool` | ✅ 已验证（2026-11-05） | `experiments/agent_e2e/catalogs/catalog-v1.json` |
| `redis_snapshot_zh_skill` | `pip install "agent-retrieval[api,redis]"` | 见 redis_snapshot 节装配片段 | `python -m experiments.scenario_checks.redis_snapshot_zh_skill` | ✅ 已验证（2026-11-05） | `experiments/agent_e2e/catalogs/catalog-v1.json` |
| `redis_snapshot_zh_zh_plot` | `pip install "agent-retrieval[api,redis]"` | 见 redis_snapshot 节装配片段 | 待 zh-plot-v1 接线（无验证命令） | `pending.zh_plot` | zh-plot-v1 数据集（未建，见 §4） |

### 3.5 full_stack 姿态（6 格）

配置片段一律见 §2.4 装配片段。live 格同 §3.4 纪律；验证脚本内含判定组件参与断言（判定格属 24 格内，非矩阵外附加项）。

| 格 id | extras 安装命令 | 配置片段 | 验证命令 | 状态 | 语料指针 |
|---|---|---|---|---|---|
| `full_stack_en_tool` | `pip install "agent-retrieval[all]"` | 见 full_stack 节装配片段 | `python -m experiments.scenario_checks.full_stack_en_tool` | ✅ 已验证（2026-11-05） | `experiments/jev_cases.json` |
| `full_stack_en_skill` | `pip install "agent-retrieval[all]"` | 见 full_stack 节装配片段 | `python -m experiments.scenario_checks.full_stack_en_skill` | ✅ 已验证（2026-11-05） | `experiments/agent_e2e/catalogs/catalog-v1.json` |
| `full_stack_en_zh_plot` | `pip install "agent-retrieval[all]"` | 见 full_stack 节装配片段 | 待 zh-plot-v1 接线（无验证命令） | `pending.zh_plot` | zh-plot-v1 数据集（未建，见 §4） |
| `full_stack_zh_tool` | `pip install "agent-retrieval[all]"` | 见 full_stack 节装配片段 | `python -m experiments.scenario_checks.full_stack_zh_tool` | ✅ 已验证（2026-11-05） | `experiments/agent_e2e/catalogs/catalog-v1.json` |
| `full_stack_zh_skill` | `pip install "agent-retrieval[all]"` | 见 full_stack 节装配片段 | `python -m experiments.scenario_checks.full_stack_zh_skill` | ✅ 已验证（2026-11-05） | `experiments/agent_e2e/catalogs/catalog-v1.json` |
| `full_stack_zh_zh_plot` | `pip install "agent-retrieval[all]"` | 见 full_stack 节装配片段 | 待 zh-plot-v1 接线（无验证命令） | `pending.zh_plot` | zh-plot-v1 数据集（未建，见 §4） |

### 3.6 语料口径说明

- **en-tool 格**（4 格）：`experiments/jev_cases.json`——英文 tool 资源池，判定语义经 Jev 基准实测。
- **zh-tool / zh-skill / en-skill 格**（12 格）：`experiments/agent_e2e/catalogs/catalog-v1.json`——双语资源池（10 资源：8 tool + 2 skill，中文声明面齐备、判定语义经冒烟验证）。zh 格取中文声明面，en-skill 格取英文声明面；判定输入按 declared 口径只消费 name+description（`agent_retrieval.judges.api.projected_candidates` 投影口径，tags/use_when/avoid_when 不进判定输入）。
- **zh_plot 格**（8 格）：zh-plot-v1 数据集未建，状态 `pending.zh_plot`（§4）。

## 4. zh-plot 占位节

zh_plot 形态列的 8 格（4 姿态 × en/zh 两语言段）统一占位，状态 `pending.zh_plot`——这是初稿终态，M4 仅补票链接、不改状态语义。

- **金版本占位语义**：`experiments/golden/gate.py` 的 `_PENDING = {"zh_plot": "待 zh-plot-v1 构建立项后接线（03 票冻结方案）"}`（gate.py:40）；gate 报告以 `{"pending": _PENDING, ...}` 形态显式携带该占位（`run_gate()`）。矩阵的 `pending.zh_plot` 与金版本占位同义：中文绘图场景评测数据集缺席，格不可验证，显式占位而非留空白。
- **zh-plot-v1 构建票语义**：zh-plot-v1 方案已冻结、构建立项票已关闭（issue #4，closed）——当前状态为**待 zh-plot-v1 数据集构建接线**。数据集（HITL 构建）落成后，本节 8 格换装真实语料指针与验证命令，状态归一三枚举；接线属独立立项（`docs/plans/sdk-p2-plan.md` 「与既有轨的边界」），不在 P2 范围内。
- **语言段说明**：zh_plot 格带 en/zh 两语言段是 4×2×3 网格的结构完备性要求；en 语言段的消费形态（英文查询是否进 zh-plot 数据集）随 zh-plot-v1 数据集设计定形，当前不做假设。

## 5. 判定示例：`api_embed_zh_tool` 「已验证」终态样例

M4 状态回写的定形样例——2026-11-05 回写完成后，本文 16 格已照此结构展开为终态（四件齐 + 日期）；zh_plot 8 格保持 `pending.zh_plot` 占位（§4）。

| 格 id | extras 安装命令 | 配置片段 | 验证命令 | 状态 | 语料指针 |
|---|---|---|---|---|---|
| `api_embed_zh_tool` | `pip install "agent-retrieval[api]"` | 见 api_embed 节装配片段 | `python -m pytest tests/test_scenario_matrix.py -k api_embed_zh_tool` | ✅ 已验证（YYYY-MM-DD） | `experiments/agent_e2e/catalogs/catalog-v1.json` |

终态判定记录（M4 回写时随行附注）：

- 验证命令 exit 0；
- 断言通过：候选非空 + top-1 命中检索目标资源 + MockEmbedder 向量路参与存在性（S-D4 组件存在性断言，无阈值）；
- 日期回填实际执行日；质量红线未在此格校验（归金版本 gate）。
