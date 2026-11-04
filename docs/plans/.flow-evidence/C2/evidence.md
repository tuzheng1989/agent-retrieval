# C2 证据包（sdk-p1-plan）

> 审计输入：本文件 + 方案 `docs/plans/sdk-p1-plan.md` C2 节（唯一需求出处）+ 工作区实际差异。
> 实施人：Executor（重构安全网 + TDD 八切片）。抽查人：主控（复跑通过后标注可采信）。
> 注：工作区含 C1 已收口改动（projection/test_projection/test_core_zero_dep/C1 证据目录），审计时与 C2 改动（本文件第 1 节）区分。

## 1. 差异范围（边界内 6 文件）

| 文件 | 类型 | 内容 |
|---|---|---|
| `src/agent_retrieval/judges/api.py` | 修改 | 提取 `projected_candidates` 共享纯函数（docstring 注明单一出处理由），`judge_state` 改调；签名与输出逐位不变 |
| `src/agent_retrieval/judges/chat.py` | 新建 | `CHAT_JUDGE_PROMPT_VERSION="1"`、`ChatJudgeConfig`（frozen + identity 三元组）、`_chat_judge_prompt` 纯函数、`_extract_json` 三态解析、`ChatJudge`（HTTP 纪律照 JevJudge._post 平移） |
| `src/agent_retrieval/judges/__init__.py` | 修改 | 导出 ChatJudge/ChatJudgeConfig，docstring 更新（并列实现 + D1） |
| `src/agent_retrieval/__init__.py` | 修改 | `_LAZY_EXPORTS` judges 区加 2 项（指向 judges.chat）；`__all__` 同步（本批恰 5 行追加：1 注释 + 2 lazy + 2 `__all__`；C1 行为会话前已有） |
| `tests/test_api_chat_judge.py` | 新建 | 25 项替身测试（judge_wire fixture 模式） |
| `experiments/benchmarks/judge_chat_smoke.py` | 新建 | live-only 冒烟：固定任务文本 + jev_cases 4 候选（geo-query/email-send 正例、chart-render 负例、weather-forecast 干扰）；退出码 0/1/2/3/4 分级 |

## 2. 语义保留证据链（D4，本批最高风险项）

1. 重构前基线：`tests/test_api_judge.py` 16 passed（锁定）。
2. 重构后零改动全绿：同命令 16 passed（测试文件零 diff）。
3. `judge_parity --skip-live` 判定一 PASS（exit 0，4 cases × 9 candidates）。
4. **冻结快照 git 零 diff**：`experiments/judge-parity-legacy.json` 与 `judge-parity-report.json` 均无改动（主控已独立核实）——重构后判定一输出与 B3 冻结快照逐位一致。

## 3. 验收清单（方案 C2 节 7 条）

| # | 标准 | 自报 | 证据 |
|---|---|---|---|
| 1 | judge_state 逐位一致：16 项零改动 + 判定一复跑 PASS | PASS | 第 2 节四重证据 |
| 2 | 替身 200 纯 JSON → 键集对应、值逐位一致 | PASS | test_api_chat_judge 请求体/解析组 |
| 3 | 围栏 ```json 包裹 → 解析成功 | PASS | `_FENCE_RE` 剥围栏分支专测 |
| 4 | 400→200 重试成功；401 单发即抛；缺键单发即抛；1.2 → JudgeResponseError | PASS | 另覆盖 403、多键 c2、content 不可解析进退避循环 |
| 5 | instruction 两态（进 prompt / 默认无节） | PASS | 单元 + 端到端 user 消息两态 |
| 6 | 无 requests → 报错含 `agent-retrieval[judge]` | PASS | sys.modules 注入 |
| 7 | live 冒烟 | **PASS（2026-11-05 live 补验）** | 配置链：`AGENT_RETRIEVAL_CHAT_JUDGE_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4` + `AGENT_RETRIEVAL_CHAT_JUDGE_MODEL=glm-5.3-flash`（端点与模型名经用户核定）+ 既有 `ZHIPUAI_API_KEY` 回落；exit 0，p(geo-query)=1.000 / p(email-send)=1.000 / p(chart-render)=0.050 / p(weather-forecast)=0.000，max(正例)>负例 ✅、全概率 ∈[0,1] ✅、键集恰 4 id ✅。日志：同目录 `smoke-live.log` |

## 4. Executor 命令回报摘要

重构前 16 passed → 重构后 16 passed → parity exit 0 → 定向收口 55 passed（chat 25 + judge 16 + port 14）→ ruff 六文件全过 → py_compile OK → import 冒烟 4 候选五字段齐 → 冒烟缺配置 exit 4。TDD 八循环红态**未存档**（目录日志仅含绿态输出）——红态以「25 项测试全部针对新符号 + 终态全绿」背书（审计 P2-1 修正口径）。完整日志：同目录 `pytest.log`/`ruff.log`/`step0-*.log`/`step1-targeted.log`。

## 5. 主控抽查（可采信标注）

- 独立复跑定向映射：**55 passed in 0.26s** → Executor 测试证据可采信。
- 冻结快照零 diff 独立核实（git diff --stat 为空）。
- `api.py` 重构 diff 抽查：恰为函数提取 + 调用替换，注释随迁，无其他改动。
- Pyright 中间态诊断甄别：`_extract_json is not defined`（实际定义于 chat.py:99）、`could not be resolved`、`is not accessed` 均为编辑中间态陈旧快照；`__all__` DunderAll 为 lazy exports 既有噪音。
- 边界核验：git status 恰为 C1+C2 边界内文件。

## 6. 实施裁量声明（Executor 自报，主控核可）

- 冒烟脚本环境变量回落链（`AGENT_RETRIEVAL_CHAT_JUDGE_*` → DOCGEN_* → ZHIPUAI/EMBEDDING_*）：委派单授权自定，与 `resolve_docgen_config` 同构，合理。
- 概率越界/键集错误不重试即抛：委派单明文；温度 0.0 下重试同输入无意义，语义自洽。

## 7. 建议抽查命令

```bash
.venv/Scripts/python.exe -m pytest tests/test_api_chat_judge.py tests/test_api_judge.py tests/test_judge_port.py -q
.venv/Scripts/python.exe -m experiments.benchmarks.judge_parity --skip-live
git diff src/agent_retrieval/judges/api.py
git diff --stat experiments/judge-parity-legacy.json experiments/judge-parity-report.json
```
