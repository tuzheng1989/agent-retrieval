# W3 执行证据（B3 对拍 + R5-P1 实验侧随行）

## 判定一：PASS（构造层语义保留）
4 case × 9 候选，36 条登记差异恰为 `tags 键删除`（legacy "" → new null），其余逐位一致：state.task/retrieval_instruction（含非空指令通道）、每问题 type/instructions/criteria、候选顺序与响应映射、长描述（3040 字符）两侧截断逐位一致。冻结参照 `judge-parity-legacy.json`（payload + 响应映射两节）。

## 判定二：FAIL（exit 3）→ 主控裁决修订口径
- 实测：请求体逐字段一致 4/4；跨臂 max|Δp| 0.03；**同臂自比 max 0.02（4/4 case 超差）**——端点概率按 0.01 量化、采样跨档抖动，非实现偏差。
- **主控裁决**：判定二口径修订为「请求体逐字段一致（硬门）+ |Δp| ≤ 0.05（端点量化步长 0.01 的 5 倍，覆盖实测跨臂 0.03 与同臂 0.02 噪声界）」；同臂抖动基线诊断常驻脚本（超差时自动运行）；「卸 SDK」证据链表述改为「请求体结构等价 + 概率差在端点采样噪声界内」——不再声称逐位一致。已回同 Executor 修订并复跑 live。

## R5-P1：完成
键形状 `(identity_str, "{entry}:{sha256}")` 复合主键零迁移；miss 按入口分发（q 逐条 / c 批量，降级路径同步）；六处装配点 identity_str 改 `"|".join(config.identity())`（四元含 flags）；ST 路径 `local-st|{repo}|{dims}` 与 LIKE 同步；golden gate 零影响（206s 实证）。
- 口径变化如实记录：新 identity_str 不含 dimensions（同 model 异 dims 共享缓存行；dimensions 由端点按模型决定，风险低）
- 范围外登记：`iterative.py:187` 第六处旧格式构造（行隔离自洽，后续批次随行收编）
- 契约更新：`test_benchmarks.py` 1 用例（q/c 共享语义 → 分键语义，按既定口径）

## 命令记录
| 命令 | 退出码 | 结果 |
|---|---|---|
| `pytest tests/test_judge_parity.py -q` | 0 | 10 passed |
| 全量 `pytest tests/ -q` | 0 | 215 passed（225s） |
| `pytest tests/test_golden_gate.py -q` | 0 | 1 passed（206s） |
| `judge_parity --skip-live` | 0 | 判定一 PASS |
| `judge_parity`（live） | 3 | 判定二 FAIL（端点非确定性，见裁决） |
| `env -u TYPESAFE_API_KEY judge_parity` | 4 | 缺 key 报错指路 |
| ruff experiments/benchmarks/ | 0 | All checks passed |
