# M2 审计结论（sdk-p2-plan / 批次 M2，T2）

> Auditor 证据包审计 + P1 退回修复（revision 2）+ 主控复验，2026-11-05。

## 结论：PASS 可收口（revision 2）

- 审计六项核验：5 通过 + 1 项实现通过但发现 **P1-1**（`_assert_zero_extras` sys.modules 绝对状态判定——全量套件收集期污染必假红，审计亲证 `pytest tests/test_api_embedder.py tests/test_scenario_matrix.py -q` → 4 failed）
- 审计其余确认：格覆盖与主表逐格一致、`-k` 选中性亲跑、S-D4 无阈值断言全文扫描零匹配、top-1 锚 BM25 的必要性被独立实跑证实（cosine 第一名非标靶但融合 top-1 正确）、惰性导入三重核实、红态自洽

## revision 2 修复（审计处方方案 a：基线差集）

- `_MODULES_AT_IMPORT = frozenset(sys.modules)` 模块级基线 + `_assert_zero_extras` 改差集断言（序无关抗污染；本文件顶层误导入仍被兜住）
- 验证三链：① 单文件 8 passed；② **审计假红复现命令 `24 passed`**（主控复验通过）；③ 回归 40 passed
- **超出处方**：运行期污染探针 46 passed + **全量实跑 396 passed**（P1 根因形态 CI 全量实证全绿）+ 全仓静态核查（函数级 extras 导入 4 处全部幂等重导）
- 主控复验：24 passed + 8 passed ✅

## sug 处置

| # | 建议 | 处置 |
|---|---|---|
| 1 | 钉定断言与 catalog 内容耦合（刀锋 tie 场景） | **M4 回写时留语料版本耦合注记** |
| 2 | skill 池 2 资源区分度弱 | 记档——S-D4 允许，扩容自然增强 |
| 3 | Pyright 告警解释未独立验证 | 记档——非验收项 |
