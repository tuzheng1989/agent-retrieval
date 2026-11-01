# B4 执行证据（docgen 收编）

Executor 回执要点归档（完整内容见会话记录；pytest.log 已由 Executor 落盘）。

## 改动文件（边界内，7 个）

1. `src/agent_retrieval/expansion/api.py`（新建）——`DocGenConfig`（frozen，identity 三元组 `(model_name, api_style, questions_per_resource)`，corpus_hash 连接口径入 docstring）、`DocGenError`、`DocGenClient`（`complete` 通用基元 + `generate` 封装，chat/responses 双分支）、`build_docgen`（缺凭据返 None）、`parse_generated_queries`、`docgen_corpus_text`、`PROMPT_TEMPLATE`。HTTP 纪律照 `_chat_completion` 平移（惰性导入 + 退避 + 401/403 不重试 + 4xx 记 last_error 重试 + 超时显式）。
2. `src/agent_retrieval/expansion/__init__.py`（新建）——导出公共名（含 `__all__`）。
3. `pyproject.toml`——extras `[docgen]`；`all = api,redis,docgen`（judge 留 W2）。
4. `experiments/benchmarks/docgen.py`——薄壳化：转发五名 + 保留 `resolve_docgen_config/load_docgen/main()/_DATA_DIR` + 兼容入口 `chat_text` 与 `generate_queries`（原签名原参数名）；删旧实现三件。
5. `groupdesc.py` / `summarize.py`——生成函数公开签名一字未动，函数体内传输调用换 `chat_text(config, prompt).strip()`；自身 PROMPT_TEMPLATE 原样；`_DATA_DIR` 改本地常量。
6. `tests/test_docgen_pkg.py`（新建 16 用例，替身 HTTP）。

## 执行前实读摘要（groupdesc/summarize `_chat_completion` 用法）

两处同形：`_chat_completion(prompt, base_url=…, api_key=…, model=…, label=…)` 返回 `.strip()` 自由文本；prompt 由各自模块级模板 format 组装。所取最小改法 = 公开签名不动、仅传输调用一处替换。

## 命令与退出码

| 命令 | 退出码 | 结果 |
|---|---|---|
| TDD 红1 `pytest tests/test_docgen_pkg.py -q` | 非0 | ModuleNotFoundError: expansion（预期红起点） |
| TDD 红2（包后薄壳兼容 3 例） | 非0 | chat_text/reexports 未存在（预期二段红） |
| 绿 `pytest tests/test_docgen_pkg.py -q` | 0 | 14→16 passed |
| 定向 `pytest test_docgen_pkg+test_docgen+test_summarize -q` | 0 | 26 passed |
| 消费方回归 `test_two_tier+test_golden_gate+test_benchmarks -q` | 0 | 40 passed（216s） |
| smoke 薄壳导入 | 0 | shell ok |
| ruff 五文件 | 0 | All checks passed! |

## 验收清单（8 条全 PASS）

双分支 payload 与解析（含 chat 键不漏进 responses）✓ / complete 通用基元 ✓ / generate_queries 原签名关键字走通 ✓ / groupdesc+summarize 替身直调提示词逐位一致 ✓ / 缺凭据 None + identity 三元 ✓ / 六方零改动 + 40 消费方回归 ✓ / 提示词 grep 留在自身文件 ✓ / DocGenError 转发 ✓

## 主控抽查（独立复跑）

定向 26 passed ✓ / 薄壳导入面 `shell ok` ✓ / `PROMPT_TEMPLATE =` 全仓 grep 除 groupdesc/summarize/expansion 外零残留 ✓

## 遗留注意

- responses payload 对称带 `temperature: 0.2`、`stream: False`（批生成纪律，非委派单列出但语义正确）
- 薄壳错误 label 统一为 "docgen"（原 groupdesc/summarize 各自 label 不再区分——失败计数逻辑零影响）
- L3 `run_benchmark --corpus docgen` smoke 未跑（需真实数据集+网关凭据，主控裁量项；消费方回归 40 passed 已覆盖扩展装配路径）
