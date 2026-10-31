# 2026-10-24 基准验证运行归档

本目录是 2026-10-24 在 rhetra-harness 会话中对本仓库（agent-retrieval）SkillRet / ToolRet 基准做的一批验证运行原始产物，2026-10-31 自 rhetra-harness/.scratch/ 迁入归档。

## 内容

- `skillret-e2e-first-run/`、`skillret-e2e-live-flash/`——SkillRet e2e 两轮运行（live-flash 为最终态；summary.json 含数据集 SHA256、采样口径与指标）
- `toolret-node-e2e-first-run/`——ToolRet node e2e 首跑（pytest/ + tests.log）
- `retrieval-check-<hash>/` ×3——分项核查运行
- `retrieval-final-<hash>/`——终局运行（retrieval-cases.json、independent-audit.json、summary.json、逐案 q-* 目录）
- `retrieval-live-kernel-<hash>/`——live 内核核查
- `retrieval-first-check.log`——本批首查日志

## 说明

- 产物当时落在 rhetra-harness 的工作目录（错位），未按 run ID 收录进 experiments/ 正式结果；迁此留档原始证据。
- 数据集源：`experiments/benchmarks/data`（skillret summary 内 `source_dir` 有原始绝对路径与 SHA256）。
