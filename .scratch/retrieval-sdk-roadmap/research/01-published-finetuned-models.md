# 票 01：已发布领域微调模型盘点（零训练姿态）

- 日期：2026-10-31
- 研究问题：盘点**他人已发布**的 tool/skill 检索领域微调嵌入模型，评估零训练姿态（本仓不训练，本机无 GPU）下直接接入做配对对比的可行性。
- 方法：全部结论溯源到一手来源（HF 模型卡 / HF API 元数据 / 论文 HTML 全文 / 官方 GitHub README），未采用二手转述。网络工具：mcp web-search-prime + web-reader。
- 本仓现状锚点（来自仓库源码与票面）：embedder 是 GLM-Embedding-3（API，OpenAI 兼容端点）；ToolRet 一阶段覆盖弱（41% gold 在 top-50 外）；最优配置 fusion docgen + Jev@20 = NDCG@10 0.388；SkillRet 语料面 = name+description+taxonomy tags（6,006 技能），ToolRet 语料面 = documentation JSON 原文（~44k 工具）。

## 1. 仓库接入面（决定接入成本的事实）

来自 `src/agent_retrieval/core/ports.py`、`embedders/{api,local,factory,config}.py`、`core/ranking.py`、`experiments/benchmarks/run_benchmark.py`：

1. **`Embedder` Protocol 只有 `embed_corpus(texts)` / `embed_query(text)` 两入口**，向量按惯例归一化（cosine = 点积），失败抛 `QueryEmbeddingError` 自动退化 BM25 单路。
2. **API 形态**：`ApiEmbedder` 走 OpenAI 兼容 `/embeddings`，配置全由 `embedder_from_env()` 从环境变量组装（`AGENT_RETRIEVAL_EMBEDDING_BASE_URL/MODEL/API_KEY/DIMENSIONS`）。**换任何 OpenAI 兼容托管的嵌入模型 = 改环境变量，零代码改动**。`EmbeddingConfig.identity()`（provider/model/version）自动进向量快照 id，换模型即换快照。
3. **本地形态**：`LocalEmbedder(model_dir)` = ONNX Runtime（CPUExecutionProvider）+ transformers tokenizer，**固定 mean-pooling + L2 归一化，max_length 默认 512**。两个硬约束：
   - CLS pooling 系（bge 系、SkillRet-Edge-109M）与 last-token pooling 系（Qwen3 系、harrier 系）都不能直接用现有 mean-pooling 跑对，需要给 LocalEmbedder 加 pooling 模式参数（估计 30–80 行改动，属库内核小扩展）；
   - 512 默认长度对 ToolRet（平均工具文档 174.56 token）基本够用，对长尾文档需调参。
4. **查询侧指令前缀**：`rank_candidates` 把 query 原文直接喂 `embed_query`；ToolRet 的 `--query-mode instruction` 只拼基准自身 instruction，不含模型自身的 `Instruct: ...\nQuery:` 模板。用指令敏感模型（Qwen3/harrier/SkillRet 系）需要一层 query 前缀包装（experiments 层 ~20–50 行，不动库内核）。

## 2. 两个基准的领域微调证据（论文一手数值）

### SkillRet（arXiv 2605.05726，表 3，NDCG@10）

| 模型 | 参数 | NDCG@10 | 备注 |
| --- | --- | --- | --- |
| BM25 | – | 48.86 | 稀疏基线 |
| bge-small-en-v1.5 | 33M | 51.68 | encoder 上限组 |
| bge-large-en-v1.5 | 335M | 55.82 | encoder-only 最强 |
| NV-Embed-v1 | 7B | 53.12 | 规模倒挂例 1 |
| KaLM-Gemma3-12B | 12B | 55.38 | 规模倒挂例 2（MTEB 75.66 第一） |
| Qwen3-Embedding-0.6B | 0.6B | 58.35 | SkillRet 微调基座 |
| harrier-oss-v1-0.6b | 0.6B | 66.55 | **现成（未针对本域微调）最强** |
| SkillRouter-Embedding-0.6B | 0.6B | 70.38 | 前作领域微调（公开 checkpoint） |
| **SkillRet-Embedding-0.6B** | 0.6B | **78.03** | 基准作者发布 |
| **SkillRet-Embedding-8B** | 8B | **83.45** | 基准作者发布 |

- 摘要口径：任务内微调比最强前作检索器 +13.1、比最强现成检索器 +16.9 NDCG@10；MTEB 分与 SkillRet 分仅 Spearman ρ=0.71，倒挂常见——「微调压倒规模」即本表结论。
- 8B 模型卡在公开切分（4,392 查询/6,006 技能，与仓库 SkillRet 评测同口径）上自报 NDCG@10 **0.8644**；0.6B 教师在同切分 78.48（Edge-109M 卡转述）。论文表 3 的 78.03/83.45 与卡面数字的切分口径差异见 §6 不确定点。

### ToolRet（arXiv 2503.01763，ACL 2025 Findings）

- 基准：7,615 任务 / 43,215 工具（平均工具文档 174.56 token）；训练集 ToolRet-Training-20w：20 万+ 实例、平均工具文档 163.52 token、每查询 5 个负例。
- 零样本（论文表 5，w/ inst.，平均 NDCG@10）：**gte-Qwen2-1.5B-instruct 45.96** > NV-Embed-v1 42.71 > GritLM-7B 41.13 > bge-large-en-v1.5 38.49 > BM25s 36.46 > bge-base 34.77；w/o inst 下最佳嵌入 NV-Embed-v1 33.83。重排器：jina-reranker-v2-base 47.17、bge-reranker-v2-gemma 47.52（初排来自 NV-Embed-v1）。
- **官方微调增益（论文附录表 15，ToolRet 全集 NDCG@10，w/o inst 口径）**：
  - bge-large-en-v1.5：23.03 → **32.95**（+43.1%）
  - bge-base-en-v1.5：22.06 → 29.35（+33.1%）
  - e5-large-v2：20.78 → 30.18（+45.2%）
  - e5-base-v2：18.97 → 25.81（+36.1%）
  - 对应 checkpoint 已发布（§3），但绝对值**低于仓内现状 0.388**（含 Jev 精排）与 gte-Qwen2 零样本——微调小 encoder 并未追上大 decoder 指令模型。
- Tool-COLT contriever（第三方工具域微调 contriever）在 ToolRet 平均 N@10 仅 **26.75**（w/ inst），低于 BM25s 36.46——**证据性排除**。

## 3. 候选清单

缩写：L=许可证；P=参数量；DL=HF 下载体量（截至 2026-10-31）；部署=CPU ONNX 可行性/现成 API；契约=query/passage 前缀、pooling、最大长度；成本=接入改动量级（A=仅 env；B=+query 前缀包装 20–50 行；C=+LocalEmbedder pooling/长度扩展 30–80 行；D=需自建 GPU 服务）。

| # | 模型（HF id） | L | P / DL | 部署形态 | 输入契约 | 语料面适配 | 成本 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ThakiCloud/SKILLRET-Embedding-0.6B | Apache-2.0 | 595.8M BF16 / 1.20GB | CPU：decoder 0.6B 可 ONNX int4/GGUF（0.6B 级）；TEI 兼容标签在卡；无公开托管 API（未核实到） | 查询侧 `Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery:`，文档侧裸；last-token pooling；max 8192 | SkillRet 面同域 SOTA（78.03/78.48）；ToolRet 面域外，预期差 | B 或 C（本地） |
| 2 | ThakiCloud/SKILLRET-Embedding-8B | Apache-2.0 | 7.57B BF16 / 15.1GB | CPU 不可行；TEI 可自建但需 GPU | 同 #1 | SkillRet 面最强（83.45/86.44）；零 GPU 姿态下仅当出现托管 API 才可行 | D（暂缓） |
| 3 | ThakiCloud/SKILLRET-Edge-109M-int4（fp16 版同库） | Apache-2.0 | 109.5M / 82MB packed（fp16 safetensors 219MB，卡声明可被 transformers 直接加载） | **CPU 原生定位**（卡：run on a CPU next to the agent）；int4 真整数核未测（卡自述） | **查询侧裸编码**（query_prefix.json: resolved ""）；CLS pooling；max 256（评测口径） | SkillRet 面 79.21（±0.42，与教师平）；**域外崩塌**（NFCorpus 7.31），ToolRet 面禁用 | C（CLS）+注意 max_length 256 |
| 4 | pipizhao/SkillRouter-Embedding-0.6B | Apache-2.0 | 0.6B（Qwen3-Embedding-0.6B 微调，arXiv 2603.22455） | 同 #1 形态；另有 mradermacher GGUF 衍生 | 查询侧 `Instruct: Given a task description, retrieve the most relevant skill document...\nQuery:`；文档 `name \| description \| body`；last-token pooling、**左 padding** | SkillRet 面 70.38（前作微调，被 #1 显著超越）；对仓库「SkillRet 语料面=短拼接」是合理次级对照 | B 或 C |
| 5 | mangopy/ToolRet-trained-bge-large-en-v1.5（另有 bge-base/e5-base/e5-large 三兄弟） | **卡面无 license 标签**（基座 bge-en-v1.5 系 MIT） | 335M（BERT） | CPU ONNX 成熟（BERT 类，optimum 一条命令）；无托管 API | bge 契约：查询侧 `Represent this sentence for searching relevant passages: `，文档侧裸；CLS pooling + L2；512 | ToolRet 面官方微调（23.03→32.95）；SkillRet 面无证据 | C（CLS）+B（前缀） |
| 6 | Tool-COLT/contriever-base-msmarco-v1-ToolBenchG3 | **无 license 标签**；仅 pytorch_model.bin（无 safetensors） | ~110M（contriever-base，仓库 876MB） | CPU ONNX 无障碍（mean pooling 与 LocalEmbedder 天然匹配、无前缀、max 350） | 无前缀；mean pooling | **证据性排除**：ToolRet 平均 N@10 26.75 < BM25 36.46；训练域 ToolBench 子集已被 ToolRet 语料吸收 | A/技术上零障碍，但无接入价值 |
| 7 | microsoft/harrier-oss-v1-0.6b（另有 270m/27b） | MIT | 0.6B（同构 Qwen3-0.6B，595.8M 级）/20.1 万月下载 | onnx-community 官方移植（-ONNX）+ 大量 GGUF；无已知公共托管 API | 查询侧必须带一句话任务指令（`Instruct: {task}\nQuery: {q}`，预置 web_search_query prompt），文档侧裸；last-token pooling；max 32768；多语言（含中文） | SkillRet 面现成最强 66.55；ToolRet 面属「指令型大嵌入」家族（该家族 gte-Qwen2 零样本 45.96 最佳） | C（last-token）+B（前缀） |
| 8 | BAAI/bge-m3 | MIT | ~0.56B（XLM-R-large 底座，XLMRobertaModel）/ 仓库 13.7GB（含 fp32 权重+ONNX+colbert/sparse 头） | **仓库自带 onnx/model.onnx**；主流平台常见托管（未逐一核实） | 无强制前缀（对称编码）；CLS（dense 路用 [CLS]，sentence-transformers 配置 mean）+ L2；8192 ctx；dense 1024 维 | 两面均无基准内证据强于上者；多语言/长文档兜底 | C（pooling 需对齐导出版） |
| 9 | BAAI/bge-large-zh-v1.5 | MIT | 326M（BERT） | CPU ONNX 成熟 | 查询侧中文指令前缀（`为这个句子生成表示以用于检索相关文章：`）；CLS；512 | 两个基准语料均英文，中文模型不对口（列入仅为完整性） | C+B |
| 10 | BAAI/bge-reranker-v2-m3 | Apache-2.0 | 567.8M（XLM-R，F32 safetensors 2.27GB） | CPU 可跑但 cross-encoder 对 44k×top-100 窗口成本高；主流平台有托管 | query-doc 对输入，无嵌入语义；不是 Embedder 端口形态 | 属精排位（与 Jev 配对对比的候选），非一阶段嵌入；票 01 范围外，登记备查 | 独立精排 arm |

参考锚点：Qwen/Qwen3-Embedding-0.6B（Apache-2.0，609 万下载）= #1/#4/#7 共同基座家族；gte-Qwen2-1.5B-instruct（Apache-2.0）= ToolRet 零样本王，但无领域微调版、1.5B CPU 边缘。

## 4. 推荐：进入配对实验的 3 个

1. **ThakiCloud/SKILLRET-Embedding-0.6B**（SkillRet 面）
   - 预期增益最确定：基准作者发布的同域 SOTA checkpoint，单路 NDCG@10 78.03（论文）/78.48（公开切分），对仓内 BM25 单路 48.86 与 GLM-Embedding-3 向量路都是大幅升级；与 RRF 融合、docgen、Jev 精排的全部既有 arm 直接正交可配对。
   - 接入：Apache-2.0 无法务障碍；0.6B decoder 在 CPU 上量化可行（Q4 约 400–500MB），或本机 TEI CPU 模式服务化后走现有 ApiEmbedder（TEI 兼容标签在卡）；需要 query Instruct 前缀包装（B 级）+ 若走 LocalEmbedder 需 last-token pooling（C 级）。
   - 它直接检验票面命题「领域微调压倒规模」在本仓融合架构下是否复利，还是被 BM25+RRF 吃掉增益。
2. **mangopy/ToolRet-trained-bge-large-en-v1.5**（ToolRet 面官方微调锚点）
   - 这是 ToolRet 作者唯一公开的微调 checkpoint 家族中的最强者（32.95），给「ToolRet 领域微调嵌入进 RRF 融合」一个干净对照：仓内 41% top-50 外的一阶段覆盖弱问题，官方微调嵌入在纯向量路下能否改善召回，是与 GLM-Embedding-3 完全同协议的配对实验。
   - 接入：335M BERT，CLS+查询前缀，CPU ONNX 成熟（C+B 级）；绝对值低于仓内 0.388 属预期（该值含 Jev 精排与 docgen），实验目的是分解「嵌入域适配」单因子。
   - 风险：checkpoint 无 license 标签（基座 BAAI bge MIT，衍生未声明）——对外发布结果前需向作者确认，内部实验风险低。
3. **microsoft/harrier-oss-v1-0.6b**（双面通用强基线）
   - 零训练姿态下的「换更强通用模型」对照臂：MIT、微软官方、SkillRet 现成最强 66.55、多语言、生态最好（onnx-community ONNX + GGUF 现成，270m 版更轻）。与 #1/#2 构成三角：领域微调（#1/#2）vs 通用强指令嵌入（#3）vs 现有 API 嵌入（GLM），一次回答「增益来自域适配还是模型代差」。
   - 接入：last-token pooling + 查询一句话指令（C+B 级，与 #1 共用同一套改动）；ToolRet 面上它所属的指令型大嵌入家族有零样本最佳证据（gte-Qwen2 45.96）。
   - 备选替换：若 CPU 预算紧，SkillRet 面可用 **SkillRet-Edge-109M-int4**（82MB、79.21、Apache-2.0）替代 #1 做第一轮快速验证，CLS+max256 契约更简单；但它域外崩塌，只能服务 SkillRet 面。

## 5. 明确不推荐（附证据）

- **Tool-COLT contriever**：ToolRet 平均 N@10 26.75 < BM25 36.46（论文表 5）；无许可证标注；仅 .bin 权重。
- **SkillRet-Embedding-8B / bge-reranker-v2-gemma 等 ≥7B**：本机无 GPU，CPU 不可行，且无公共托管 API（未核实到）。
- **bge-large-zh-v1.5**：中文模型对两个英文基准不对口。
- **ToolRet-trained-e5-base-v2**（最弱微调版 25.81）：被同族 bge-large 32.95 支配。

## 6. 关键不确定点（网上未能完全核实的事实）

1. **ToolRet-trained 四个 checkpoint 均无 license 标签**，卡面未声明许可；基座（BAAI bge MIT / intfloat e5-v2 MIT）允许衍生，但严格口径需向作者（mangopy / ToolRet 团队）确认。
2. **公共 API 托管情况未核实**：SiliconFlow/DashScope 等是否已上架 Qwen3-Embedding 系、bge-m3 常见但有；SkillRet-Embedding 与 harrier-oss 未找到公共托管（两者卡面均带 text-embeddings-inference 兼容标签，可自建）。
3. **ToolRet 排行榜逐模型数值未核**：HF space 数据文件为 xlsx，本机沙箱无法下载二进制；论文 HTML 数值已覆盖本报告引用的全部行。
4. **harrier-oss-v1-0.6b 精确参数/存储**未单独抓 API（按同构 Qwen3-Embedding-0.6B = 595.8M BF16 推断）。
5. **SkillRet 数字口径差异**：0.6B 卡自报切分 4,997/6,660（NDCG@10 78.03），而 Edge 卡证实公开发布数据集实为 4,392/6,006 且两者不可换算；8B 卡在公开切分自报 86.44，高于论文表 83.45（卡 2026-09-25 更新，疑似复训版重评，未核对训练日志）。仓库 SkillRet 评测即 6,006 口径，配对时应以公开切分数字（78.48/79.21/86.44）为预期锚。
6. **bge-m3 的 8192 ctx / 1024 维**取自模型卡通行描述，本次仅经 API 核实架构（XLMRobertaModel）与 MIT 许可，未逐字复读卡面。
7. **CPU 推理吞吐全部为推断值**（参数量/量化体量 + Edge 卡的 CPU 定位声明），未实测；44k 工具语料一次性嵌入的墙钟时间需在配对实验前用 100 条抽样试跑标定。

## 7. 来源清单（全部 2026-10-31 访问）

- SkillRet 论文（arXiv 2605.05726 HTML v1）：https://arxiv.org/html/2605.05726v1 （表 3/4/5/6、附录 F/G/表 13/15）
- SkillRet-Embedding-0.6B 卡 + HF API：https://huggingface.co/ThakiCloud/SKILLRET-Embedding-0.6B 、https://huggingface.co/api/models/ThakiCloud/SKILLRET-Embedding-0.6B （apache-2.0、595,776,512 BF16、1.20GB、TEI 标签、query prompt）
- SkillRet-Embedding-8B 卡 + API：https://huggingface.co/ThakiCloud/SKILLRET-Embedding-8B 、https://huggingface.co/api/models/ThakiCloud/SKILLRET-Embedding-8B （apache-2.0、7,567,295,488 BF16、15.1GB、4,392/6,006 切分自评 0.8644）
- SkillRet-Edge-109M-int4 卡 + API：https://huggingface.co/ThakiCloud/SKILLRET-Edge-109M-int4 （79.21、82MB、CLS、裸查询、域外崩塌、切分口径澄清）
- SkillRouter-Embedding-0.6B 卡 + API：https://huggingface.co/pipizhao/SkillRouter-Embedding-0.6B （apache-2.0、契约代码、arXiv 2603.22455）
- ToolRet 论文（arXiv 2503.01763，ar5iv 全文）：https://ar5iv.labs.arxiv.org/html/2503.01763 （表 4/5/7/14/15、语料统计）
- ToolRet 官方仓库 README：https://github.com/mangopy/tool-retrieval-benchmark （评测模型协议、FlagEmbedding 1.3.3、训练集链接）
- ToolRet-trained-bge-large-en-v1.5 卡：https://huggingface.co/mangopy/ToolRet-trained-bge-large-en-v1.5 （存在性与训练数据说明）
- ToolRet-trained 家族发现：https://huggingface.co/api/models?search=ToolRet （bge-base/large、e5-base/large 四个官方微调 checkpoint，均无 license 标签）
- Tool-COLT contriever 卡 + API：https://huggingface.co/Tool-COLT/contriever-base-msmarco-v1-ToolBenchG3 （mean pooling、max 350、无 license 标签、仅 .bin）
- harrier-oss-v1-0.6b 卡：https://huggingface.co/microsoft/harrier-oss-v1-0.6b （last-token pooling、指令必加、max 32768、多语言）
- harrier 家族与 ONNX 移植：https://huggingface.co/api/models?search=harrier-oss-v1 （microsoft MIT 三档 + onnx-community/GGUF 生态）
- bge-m3 API：https://huggingface.co/api/models/BAAI/bge-m3 （MIT、XLMRobertaModel、仓库含 onnx/model.onnx）
- bge-large-zh-v1.5 API：https://huggingface.co/api/models/BAAI/bge-large-zh-v1.5 （MIT、BertModel、zh）
- bge-reranker-v2-m3 API：https://huggingface.co/api/models/BAAI/bge-reranker-v2-m3 （Apache-2.0、567,755,777 F32、XLMRobertaForSequenceClassification）
- Qwen3-Embedding-0.6B API：https://huggingface.co/api/models/Qwen/Qwen3-Embedding-0.6B （Apache-2.0、595.8M BF16、Qwen3ForCausalLM）
- 仓库源码：`src/agent_retrieval/core/ports.py`、`src/agent_retrieval/embedders/{api,local,factory,config}.py`、`src/agent_retrieval/core/ranking.py`、`experiments/benchmarks/run_benchmark.py`
