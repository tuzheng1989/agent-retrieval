# C2 审计结论（sdk-p1-plan / 批次 C2）

> Auditor 独立审计（证据包模式），2026-11-04。全部命令亲跑（.venv），不默认采信证据包陈述。

## 结论：PASS 可收口

- 差异边界：C2 恰 6 文件，C1/C2 改动可清晰分离，无越界
- 语义保留证据链（D4）亲跑复核：16 passed 零改动 + parity exit 0 + 冻结快照 git 零 diff + diff 逐行核对纯重构
- ChatJudge 契约逐条对照方案 C2 详案无弱化：PROMPT_VERSION / Config+identity / prompt 语义继承 / 请求体 / 解析三态 / 键集校验 / 越界不 clamp / usage 透传 / HTTP 纪律同构 / 异常复用
- 测试质量：25 项断言均为具体值（精确 URL/payload 五键/请求计数/异常类型+match），失败路径充分，非 happy-path-only；L2 含 abstain 分支
- 独立复跑：55 passed exit 0（25+16+14）；ruff 六文件全过
- 冒烟脚本：4 候选与 jev_cases.json 实存条目核对无误，断言齐备，退出码分级，不触网路径亲跑 exit 4 指路正确
- 导出面：恰 2+2 项，lazy 指向正确，可达性有专测
- 零依赖与 extras：pyproject 零 diff（复用 [judge]），requests 惰性导入

## 问题处置

| 级 | 发现 | 处置 |
|---|---|---|
| P2-1 | 证据包「TDD 八循环红态均留痕」超出实物（目录日志仅绿态） | **已修正**——evidence.md 第 4 节改为「红态未存档，以 25 项新符号测试 + 终态全绿背书」；后续批次红态日志落盘为惯例 |
| sug-1 | 顶层 `__init__.py` 行数计数不精确（7 行 → 实为 5 行） | **已修正** evidence.md 第 1 节 |
| sug-2 | `judge_chat_smoke.py:138` 错误文案括号失衡 | **已修**（主控 T0 一行），fail 文案改为 `max(p(a), p(b))=` 正确括号 |

## 补验（2026-11-05）

- **验收第 7 条 live 冒烟 PASS**：Executor 实施时误判「本机无 key」采信为环境事实，主控未独立核实即标注留待——用户指出 `TYPESAFE_API_KEY` 已配置后核实：chat completions 链经 `AGENT_RETRIEVAL_CHAT_JUDGE_*` 注入智谱 coding 网关（`https://open.bigmodel.cn/api/coding/paas/v4`，端点与模型名 `glm-5.3-flash` 经用户核定）+ 既有 `ZHIPUAI_API_KEY` 回落即通。exit 0，p(geo-query)=1.000 / p(email-send)=1.000 / p(chart-render)=0.050 / p(weather-forecast)=0.000，请求/响应 model 一致。日志 `smoke-live.log`。教训：环境前提类断言主控必须亲自核实，不采信转述；网关端点与模型名以环境主人核定为准。
- 首跑曾用 `/api/paas/v4` + `glm-4.6`（exit 0 但非用户环境的正确端点/模型；响应 model 字段透传 `glm-5.3-flash` 已提示真实路由）——已按用户核定配置重跑并覆盖日志。

