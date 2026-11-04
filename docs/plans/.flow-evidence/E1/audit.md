# E1 审计结论（e2e-phase01 / 批次 E1，T3）

> Auditor 证据包审计 PASS + Advisor 独立意见 + revision 2 退回补齐 + 主控复验，2026-11-05。

## 结论：PASS 可收口（revision 2）

- **Auditor**：根基面四项机制独立探针实证（deepcopy 双份隔离 / apply 唯一写入口 / 事件 append-only / 先记再抛）；契约逐字零漂移（第六节校验面 + 第五节三常量）；39→47 测试复跑一致；红绿留痕自洽。结论 PASS。
- **Advisor**：根基面可支撑 E2-E4 开工；两条接缝口径（事件汇统一 seq、generated 确定性）须 E2 动工前钉死——已按裁决处理（见下）。
- **revision 2 退回补齐**（主控裁决，四项全落地）：`record_event` 统一非状态事件口（共用 seq、EVENT_TYPES 十项词表冻结、payload 保留键防覆写、apply 原子性）+ loading 补 forbidden_effects 词表校验与 expected_failure.resource_id 引用/故障模式校验 + grant() 收窄留痕 + 8 项随行测试。主控复验 47 passed in 0.45s。

## 问题处置总表

| 来源 | 发现 | 处置 |
|---|---|---|
| Auditor P2-1 | expected_failure.resource_id 无引用校验 | **revision 2 已补**（∈ catalog 且 failure_modes 非空） |
| Auditor P2-2 | 方案列 grant() 未实施且偏差未留痕 | **收窄裁决 + 留痕补记**（无消费方，episode 内无动态授权场景） |
| Advisor 高-1 | 非状态事件无记录通道，seq 空间分裂风险 | **revision 2 已补** record_event 统一事件汇 + 词表冻结 |
| Advisor 高-2 | 未授权 attempt vs effect 口径漂移（设计稿 §5 vs 契约 decline 判定） | **记档，阶段 3 数据冻结前主控裁决**；冒烟维持契约现状（副作用即败）；归因标签 unauthorized_attempt 消费方式届时随裁决定 |
| Advisor 中-3 | generated 字段生成源未钉死 | **契约 §5 已补确定性口径**（主控方案变更：generated:clock ← case.clock；其余确定性派生，禁墙钟/uuid）；E2 委派单引用 |
| Advisor 中-4 | forbidden_effects 未校验枚举 | **revision 2 已补**（∈ EVENT_TYPES） |
| Advisor 低-5 | permissions 双持有 / apply 保留键 / E3 替换语义假设 | 保留键已防覆写（revision 2）；permissions 单一持有（sandbox 权威）与 document_updated 按 added 匹配写进 E2/E3 委派单 |

## 遗留（不阻塞）

1. 未授权 attempt/effect 口径裁决——阶段 3 前（挂 issue #11 备注）。
2. 冒烟数据兜底的枚举项（suite/budget_profile 值域）——多版本时再冻结。
