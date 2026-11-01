# agent-retrieval

确定性 BM25 + 向量 RRF 融合检索内核，为 agent 资源发现（tool / skill / flow / 文档等）提供排序证据。

## Language

**提升路线图（Locked Roadmap）**:
本次规划的目的地产物：按序排列的检索质量杠杆清单，每条附采纳门槛，后续各自立项执行。
_Avoid_: 优化计划、任务列表

**零训练姿态（Zero-training Posture）**:
领域适配只通过「直接使用他人已发布的微调模型」与现成 API 杠杆实现，自己不训练模型。
_Avoid_: 领域微调（一词两义：自训 vs 用现成微调模型，本仓取后者）

**采纳门槛（Adoption Gate）**:
仓内既定的杠杆采纳/否决协议：配对样本 + bootstrap 95% 置信区间 + 预注册停止门槛 + 逐指标零退化。
_Avoid_: 显著性检验（口径不完整）、A/B（缺预注册与零退化约束）

**组合化（Composition）**:
内核零依赖不动；SDK 为外部组件（embedder、reranker、LLM judge、存储等）定义显式接入协议，应用侧可把已验证杠杆（语料投影、docgen、向量路、重排、窗口）与第三方组件自由拼装。
_Avoid_: 框架化（隐式配置驱动，本仓明确不取）

**场景（Scenario）**:
部署姿态 × 语言 × 资源类型的组合。部署姿态取选型指南决策表的行（均衡 / 延迟受限 / 覆盖优先 / 零 LLM）。SDK 系统化的验收 = 每个场景组合都能用文档化配置拼出且有对应验证流程跑通；验收实体是 `docs/scenario-matrix.md`（无验证载体的格显式标"未验证"）。

**一等组件（First-class Component）**:
收编进 SDK（`src/agent_retrieval`）的已验证杠杆，遵循端口 Protocol + Config（`identity()` 内容寻址）+ 工厂（`None` = 正常缺席）+ extras 四件套。准入判据：场景矩阵某格需要（见 ADR-0001）；不满足者留在 `experiments/` 作研究散件。
_Avoid_: SDK 插件（暗示运行时注册机制）、官方集成（暗示服务商关系）

**自建评测集（Owned Eval Set）**:
为自有域构造的检索评测集（首个：智能标绘 `zh-plot-v1`）：五阶段构造（生成 / 逆频 / 防泄漏 / 交叉审查 / 人工标注）+ manifest 与 git tag 冻结；改动升新版本，新旧结果永不混比；与公开基准分别报、不混算。
_Avoid_: 私有基准（不强调冻结与配对口径）、内部测试集（无构造与标注协议约束）
