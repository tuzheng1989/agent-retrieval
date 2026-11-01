# 检索质量提升路线图与 SDK 组合化蓝图

> Wayfinder 地图「retrieval-sdk-roadmap」终点产物（2026-11-01 锁定）。
> 决策细节一律索引到来源票据/文档，本文不复制内容（索引不重复存储）。

## 一、检索质量杠杆排序

| # | 杠杆 | 收益证据 | 采纳门槛 | 成本 | 排位 |
|---|---|---|---|---|---|
| 1 | **docgen 语料扩展** | Law 2：SkillRet R@5 **+51%**、ToolRet +3.8%（词汇贫乏域大赢） | 已过（双基准配对 CI） | 一次性 LLM 批生成 | **起步杠杆**（短描述/行话域首选；词汇丰富域看 R@50 定夺） |
| 2 | **Jev@20 精排** | Law 1：弱检索面增益大（ToolRet +24% R@5）；强面 MRR 微负 | 已过（双基准） | 每查询 ~5.3k tok、~1s | 一阶段 R@5 < 0.5 时启用（Law 1 判据）；窗口 20 为工作点 |
| 3 | **向量路 RRF 融合** | Law 3：ToolRet +16% R@5；SkillRet docgen 后冗余 | 已过（域消融） | 语料嵌入一次 + 每查询 | 异构/参数语义域启用；纯短声明域可关 |
| 4 | **Edge-109M 域微调换嵌入** | 08 票：SkillRet 面七指标 CI 全绿（NDCG@10 +0.073 [0.052, 0.094]，R@5 +9.7pp，gold-in-window@20 0.967） | **SkillRet 基准已过；自有域（zh-plot-v1）待验证**——基准面胜利不自动迁移 | 84 min CPU 一次性 + 82MB 权重，零 API 费 | **备选杠杆之一**；zh-plot 验证通过后方可在自有域升位 |
| 5 | 窗口 N | w20→w50：+1pp R@5 换 2.3× tokens、MRR −0.016 | 已过 | token 线性 | w20 默认 |

**已证伪（不再试）**：API reranker（02 票，三档显著有害）、字段级多路 RRF、Jev 触发迭代检索、两级 scope-before-rank、state 摘要替代截断。
**条件项**：ToolRet 面域微调嵌入——纯 CPU 不可行（bge 实测 94h 外推；0.6B decoder 族 40 天）；托管 API 或 GPU 出现时重开对照。

## 二、SDK 组合化蓝图（定稿）

照 [ADR-0001](adr/0001-sdk-integration-boundaries.md) 原样冻结：判据（场景矩阵某格需要才收编）、协议模子（端口 / Config+identity / 工厂 / extras 四件套）、收编清单（Judge、docgen、stats；rerank 与评测 runner 留 experiments）、judge fail-fast、pooling/前缀进 identity、JevJudge 裸 HTTP 消 typesafe-sdk、`api_style` 双分支。

**实施序列**：**P0** = Judge 端口 + JevJudge 裸 HTTP（含对拍）+ docgen + stats + pooling 下沉 → **P1** = ChatJudge + CorpusProjection → **P2** = scenario-matrix 铺开。
extras：`[api][docgen][judge]`(requests) + `[redis]` + `[local]` + `[all]=前四`。

## 三、评测体系排位

- **金版本门控**（已落地，05 票）：四臂零退化红线常驻 CI，升级走 `--update` 显式固化；中文格占位待 zh-plot 接线。
- **zh-plot-v1 自建集构建**：SDK P1 后、P2 前立项（P2 中文格的前置载体）。流程照 03 票冻结案；双人标注需第二名标注者到位（HITL 依赖）。
- **zh-plot 首轮消融**（构建后 immediate）：Law 1/2/3 在自有域复核 + **Edge 臂验证**（杠杆 #4 的升位判据）。
- **e2e 评测**：独立轨，SDK P0 落地后可并行立项；06 票 3 条修订（裁判入冻结清单、Cohen's κ、统计前提显式化）随立项回填设计稿。

## 四、雾区去向（移交「后续方向」）

| 项 | 现状与触发 | 去向 |
|---|---|---|
| 反馈闭环（读写对称） | 路线图已锁定，形态可展开 | 执行期立项 |
| ToolRet 激进方案 | 08 票改写为条件项：托管域微调嵌入 API 出现即触发新对照 | 条件项，随条件触发立项 |
| 多资源组合检索评测 | Completeness@k 之外的组合语义未知 | 执行期研究方向 |
| 场景矩阵逐格验证 | 哪些格实测、哪些插值 | P2 执行内容 |
| 技能文档同步 | SDK P0/P1 落地后 | 执行期文档任务 |
| 纵向评估 | 金版本之上的版本演进曲线 | 长期，不排期 |

## 五、立项清单（顺序）

1. **SDK P0**（Judge/JevJudge 对拍/docgen/stats/pooling 下沉）
2. **zh-plot-v1 构建**（可与之并行；双人标注需第二标注者）
3. **SDK P1**（ChatJudge + CorpusProjection）
4. **zh-plot 首轮消融**（Law 复核 + Edge 升位判据）
5. **SDK P2**（scenario-matrix，中文格以 zh-plot 为载体）
6. **e2e 立项**（独立轨）

## 证据索引

- 杠杆证据：`experiments/jev-benchmark-findings.md` + `docs/retrieval-config-guide.md`（三定律）
- 采纳协议：`GLOSSARY.md`「采纳门槛」；配对统计 `experiments/benchmarks/ablation_stats.py`
- 嵌入臂实验：issue #9 resolution + findings「Domain-tuned embedding arm」节
- reranker 否决：issue #3 resolution + findings「valid endpoint, harmful direction」节
- SDK 边界：`docs/adr/0001-sdk-integration-boundaries.md`
- 金版本门控：`experiments/golden/` + `tests/test_golden_gate.py`
- 自建集方案：issue #4 resolution
