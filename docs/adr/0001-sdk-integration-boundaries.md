# SDK 接入点协议与组件收编边界

零依赖内核不动的约束下，experiments/ 里验证过的杠杆要成为 SDK「一等组件」才可被组合化拼装；一张票（issue #5，2026-11-01 grilling 冻结）定了准入判据、协议形状与依赖划分，此后新增组件一律对照本 ADR 的模子。

**判据：场景矩阵某格需要才收编一等。** 选型指南决策表（docs/retrieval-config-guide.md）至少一行用到该组件，否则留在 experiments——一等组件是维护承诺，矩阵无格的组件只增加面积不增加场景覆盖。据此：Judge 精排、docgen 管线、配对统计（`agent_retrieval.stats`）收编；rerank 客户端不收编（02 票三档显著有害，杠杆已证伪）；评测 runner 不收编（数据集下载与环境耦合是实验场属性）。

**协议模子照 embedder 四件套**：端口 Protocol（零依赖层）+ Config（`identity()` 内容寻址）+ 工厂（`build_xxx → None` = 正常缺席）+ extras（按组件命名，依赖重叠不归并）。三条组件特有语义：

- Judge 端口输入是 `JudgedCandidate` 最小声明结构（id/name/description/use_when/avoid_when），窗口与判停 τ 是宿主配置不进端口；`select_with_judge`（exact 短路 + τ + 并列保序）作为辅助收编——判停语义每宿主重写一遍必漂移。
- **Judge 失败 fail-fast 唯一语义，不做降级**：Jev 挂了回退检索原序会把"要不要绑定"的判停标准静默换掉。与 embedder 的 fail-open（退化 BM25）刻意不对称——降档安全才 fail-open，语义会漂移就 fail-fast。
- `EmbeddingConfig` 增 `pooling`（mean/cls/last_token）与 `query_prefix`/`corpus_prefix` 且全部进 `identity()`：这不是期权而是正确性强制——pooling 改变向量语义，身份串不含它则不同 pooling 的向量混入同一份 write-once 快照，内容寻址被击穿。

## Considered Options

- **收编 reranker（拒）**：客户端已生产化（含 usage 透传）但杠杆被 02 票证伪；"代码质量好"不是收编理由，场景需要才是。未来某域消融翻转结论时随该域票再议。
- **走官方 typesafe-sdk（拒）**：TypeSafe 是纯 HTTP 服务（`POST /v1/systemone` 三键 JSON + Bearer），仓内惯例本就是裸 requests 客户端（ApiEmbedder 不用 openai-sdk、ApiReranker 不用任何官方 SDK）；JevJudge 裸 HTTP 化后 `[judge]` extras 依赖只剩 requests，typesafe-sdk 整个消除。代价（API 变更自己跟）以落地对拍缓解：与 SDK 版同 case 输出逐位一致后卸 SDK。
- **docgen 只支持 chat completions（拒）**：chat 与 Responses API 是两种线协议（payload、输出解析、关思考参数、token 字段名都不同），responses-only 网关真实存在；单类 + `api_style: chat(默认)/responses` 双分支，约 30 行——线协议分支不构成设 Protocol 的理由。
- **extras 按依赖归并（拒）**：`[api]`/`[docgen]`/`[judge]` 三者同为 requests 仍分开命名——extras 名是组件边界语义，依赖重叠是巧合。

## Consequences

- extras 最终形态：`[api]` `[docgen]` `[judge]`（requests）、`[redis]`、`[local]`（重依赖单独装，不进 all）、`[all]=[api,redis,docgen,judge]`。
- 语料投影双形态：声明式 `CorpusProjection`（可 JSON、进 `identity()`/`corpus_hash`）为默认，callable 逃生口保留——场景矩阵每格要能写成配置文档，投影是那一格的一半内容。
- 08 票（#9）experiments 层的 pooling/前缀实现（50–130 行）下沉 `embedders/`，实验与生产同源。
- 场景矩阵验收实体 `docs/scenario-matrix.md`：4 部署姿态 × 2 语言 × 3 资源形态（tool / skill / zh-plot），每格 extras + 配置片段 + 验证命令；无载体格显式标"未验证"，不留空白。
- 实现优先级：P0 = Judge 端口 + JevJudge 裸 HTTP（含对拍）+ docgen + stats + pooling 下沉（解绑 08）；P1 = ChatJudge + CorpusProjection；P2 = scenario-matrix 铺开。
