# Handoff：agent-retrieval 技能更新（agent-retrieval → agent-skills 维护）

> 交接日期：2026-11-08。本文档自包含：维护方可不读 agent-retrieval 仓库其他文档直接执行；证据出处以括号标注，供深挖回查。
> 更新对象：`agent-skills` 插件的 `agent-retrieval` 技能（SKILL.md + references/ 六份文档；本 handoff 基于 1.3.1 版撰写）。
> 差距来源：库 0.2.0（tag `v0.2.0`，2026-11-08 发布）新增 SDK 组件 + Jev 题型实验线终局（issues #13/#14/#15，2026-11-07/08 结案）。

## 一、差距总览

技能文档冻结于 0.1.0 时代（核心检索 + 嵌入 + Redis + 缓存）。0.2.0 后库多出一整层**判定面组件**与**实验统计工具**，且 L4（Jev 精排）层的配置语义发生实质变化（题型两档并存）。技能的现状：L4 只讲 noul 一种题型，判定端口组件零覆盖，安装矩阵缺两个 extras。

| # | 差距 | 影响的技能文件 | 优先级 |
|---|---|---|---|
| 1 | Judge 端口组件族未覆盖 | 新增 references/judges.md + SKILL.md 路由表 | **高** |
| 2 | L4 题型并存终局未同步 | SKILL.md + references/config-selection.md | **高** |
| 3 | CorpusProjection 未覆盖 | SKILL.md 场景 A 步骤 3 + references/ranking-api.md | 中 |
| 4 | docgen 客户端（expansion）未覆盖 | SKILL.md L0.5 行 + references/config-selection.md | 中 |
| 5 | agent_retrieval.stats 未覆盖 | SKILL.md 场景 C + references/config-selection.md 验证流程 | 中 |
| 6 | 安装矩阵缺 `[judge]`/`[docgen]` extras | SKILL.md 安装矩阵节 | **高**（一行） |

## 二、逐项更新内容

### 1. Judge 端口组件族（新增一节或独立 references/judges.md）

0.2.0 新导出（`src/agent_retrieval/judges/`）：

```python
from agent_retrieval import (
    Judge,               # Protocol: judge(query, candidates, *, instruction="") -> Judgment
    JudgedCandidate,     # id/name/description + use_when/avoid_when 五字段声明面
    Judgment,            # probabilities: dict[str, float] + model/latency/tokens 元数据
    select_with_judge,   # 消费 Judgment：argmax + minimum_probability 门槛（无默认值，弃权返回 None）
    JevJudge, JevJudgeConfig,      # [judge] extra：System One wire 裸 HTTP（不依赖 typesafe-sdk）
    ChatJudge, ChatJudgeConfig,    # [judge] extra：任意 chat completions 端点判定
    JudgeResponseError,
)
from agent_retrieval.judges import judge_state  # System One 请求体唯一构造点
```

要点（写给技能读者）：
- **编排模式**：exact 命中短路在调用方（先查 `hit.exact` 再决定是否调 judge），`select_with_judge` 只消费结果（`experiments/jev_retrieval.py` 的编排是范例）。
- **两种判定的取舍**：JevJudge（System One 线协议，结构化概率）vs ChatJudge（任意 chat 端点，冻结 prompt 版本 "1"，temperature 0.0 进 identity）——ADR-0001 定性为并列实现不互相包装，`Judge` 端口抽象掉差异（`docs/adr/0001-sdk-integration-boundaries.md`）。
- **候选投影单一出处**：`judge_state` / `projected_candidates`——每候选只投影 name + description（截 2000 字符），tags/use_when/avoid_when 不进判定输入（R3 定案）。防投影口径漂移的口径要在技能里讲清。
- 配置经 `JevJudgeConfig`/`ChatJudgeConfig` 显式构造（库纪律：不读环境变量）。

### 2. L4 Jev 精排：题型两档并存（SKILL.md + config-selection.md）

三票预注册实验的终局（#13/#14/#15，`experiments/jev-benchmark-findings.md` 2026-11-07/08 两小节）：

- **noul（默认）**："能否执行任务所需的一个具体步骤"，二元概率。硬 recall 下限场景维持。
- **score 量规 + P(3) 折叠**：4 档量规（0 无关 / 1 仅主题 / 2 部分覆盖 / 3 直接覆盖），排序标量 = P(第 3 档)。分级排序消费处（MRR/NDCG）选用——三独立样本 NDCG@10 +0.028/+0.030/+0.042（CI 下界全 >0），成本 ~1.4× input tokens。
- **机制结论**（技能里值得写，帮读者避免踩坑）：量规的排序信息**几乎全部在第 3 档**——含第 2 档的折叠从稀释（期望分）到主动有害（P(≥2)：−0.05，CI 全负）。
- **准入红线**：任何判定读数（noul 概率或 P(3)）**不得单独作准入闸**——端点对纯问候语稳定给 P(3)=0.96（三连复现）；准入配 `clearly_related` 声明面证据。
- 量化噪声背景：端点概率 0.01 量化、同臂重跑抖动 0.02–0.05（judge_parity 实测）——判定阈值校准与"微差异"判读都要以此为底座。
- 源头文档：`docs/retrieval-config-guide.md` Jev re-rank 节（已更新，可直接镜像）+ `docs/roadmap.md` 杠杆 2 行。

### 3. CorpusProjection（ranking-api.md 补一节）

```python
from agent_retrieval import CorpusProjection  # 声明式语料投影（字段 include/exclude）
```

场景 A 步骤 3 现在有两条路：手写 `corpus_text` lambda（现状）或声明式 `CorpusProjection`。要点：doc2query 扩展只进**检索面**、不进判定/准入证据（doc2query-- 模式），CorpusProjection 是该分离的钩子。

### 4. docgen 客户端（L0.5 行升级）

```python
from agent_retrieval.expansion import (
    DocGenClient, DocGenConfig, DocGenError, build_docgen,
    docgen_corpus_text, parse_generated_queries,
)
```

技能 L0.5 行现在只讲"离线生成口语查询拼进检索面"概念；0.2.0 起库内有一等客户端（`[docgen]` extra）。补：`build_docgen` 工厂、`docgen_corpus_text` 拼接、生成结果持久化与语料变更重嵌入由内容寻址快照处理。坑（config-guide 已记，技能应同步）：推理型模型（GLM-4.5/5.3）会把 max_tokens 花在隐藏思考上——须关思考 token 并给 ≥2,000 预算，否则 content 静默为空（finish_reason=length）。

### 5. agent_retrieval.stats（场景 C 验证流程的工具化）

```python
from agent_retrieval.stats import paired_bootstrap, load_arm_cases, paired_report
```

场景 C 第 3 步"为自己的域复跑消融"现在有库内统计出口：runner 报告 JSON（`{"cases": [{"case_id", "<臂名>": {指标: float}}]}` 形状）直接喂 `paired_report`。参数冻结在仓规约：10,000 次重采样、种子 20261022、CI95。CLI 侧 `python -m experiments.benchmarks.ablation_stats pair/power` 仍在 experiments（不上库）。

### 6. 安装矩阵（一行补齐）

```bash
pip install "agent-retrieval[judge]"       # + JevJudge / ChatJudge（requests）
pip install "agent-retrieval[docgen]"      # + DocGenClient（requests）
```

`all` extras 已含两者。

## 三、不需要动的部分

- 核心 8 条契约、`rank_candidates` 签名、`CandidateHit` 字段、BM25 内核、判定闸——0.2.0 未动核心，速查表照旧。
- L1–L3（embedders / redis-snapshots / run-cache）三层文档——无变更。
- 场景 B 排查路由——新组件的排查模式（`JudgeResponseError` 不重试 vs 网络退避重试）可并入 judges 文档，不单开场景。

## 四、验证清单（更新完成后自查）

1. SKILL.md 的 L4 行与安装矩阵与 `docs/retrieval-config-guide.md`（commit `a2341c2` 后状态）逐字对得上；
2. 新 judges 文档的代码样例在 0.2.0 上可运行（`pip install agent-retrieval[judge]==0.2.0` 后 import 全部成功）；
3. 技能触发词（description frontmatter）酌情补 "Judge 端口"、"ChatJudge"、"P(3)"、"题型"——非必须，按技能维护惯例定。

## 五、证据出处速查

| 主题 | 出处 |
|---|---|
| 0.2.0 组件面与 extras | `CHANGELOG.md` 0.2.0 节、`pyproject.toml`、`src/agent_retrieval/judges/` |
| 题型实验三票 | issues #13/#14/#15（全部结案）；`experiments/jev-benchmark-findings.md` 2026-11-07/08 三小节 |
| 配置面终局 | `docs/retrieval-config-guide.md` Jev re-rank 节、`docs/roadmap.md` 杠杆 2 |
| 组件边界纪律 | `docs/adr/0001-sdk-integration-boundaries.md` |
| 实验基建（折叠/fresh 采样） | `experiments/benchmarks/jev_questions.py`、`jev_question_paired.py`（--fold/--exclude-report） |
