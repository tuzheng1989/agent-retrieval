# B2 执行证据（JevJudge 裸 HTTP 参考实现）

Executor 回执要点归档（完整内容见会话记录；pytest.log 已由 Executor 落盘）。

## 改动文件（边界内，5 个）

1. `src/agent_retrieval/judges/api.py`（新建）——`judge_state` 纯函数（B3 两臂唯一构造点；投影 `{name, description}` 两键、截断 2000、问题措辞继承 jev_rerank "concrete step" 版）、`JevJudgeConfig`（frozen 三必填 + timeout）、`JevJudge`（实现 Judge，instruction 逐次参）、`JudgeResponseError`、`JEV_DESCRIPTION_MAX_CHARS=2000`。
2. `src/agent_retrieval/judges/__init__.py`（新建）——导出四名。
3. `pyproject.toml`——extras `[judge]`；`all` 终态四项（api,redis,docgen,judge）。
4. `src/agent_retrieval/__init__.py`——`_LAZY_EXPORTS` 增三项；`__all__` 同步。
5. `tests/test_api_judge.py`（新建 16 用例，替身 HTTP）。

## 调研子步核实（已进模块 docstring）

- Noul wire：`{"type": "noul", "instructions": str, "criteria": {"true": str, "false": str}}`
- `/v1/systemone` 顶层恰三键 `state/model/questions`；鉴权 `Authorization: Bearer` + `Content-Type: application/json`（SDK 自标识头不复刻）
- 响应：`answers.c{i}.noul`（float 0-1 非 bool）+ `model` + `usage{input_tokens, output_tokens}`

## 关键实现决策

- `judge_state` 返回 `{"state":…, "questions":…}`（不含 model）——B3 SDK 臂拆键传 `system_one`、裸 HTTP 臂加 model 键，两臂共用同一构造点
- `JevJudgeConfig` 三字段必填无默认（凭据必填才有意义；与 DocGenConfig 空串默认不同，有意）
- 响应缺候选键 → `JudgeResponseError` 经 `except: raise` 穿透重试循环（实现错误不重试，与鉴权同享语义）
- 空候选列表不设防（System One 至少一问 + B1 编排层空 shortlist 不调 judge）
- `_post` 解析一次返回三元组（避免重复解析）

## 命令记录

| 命令 | 退出码 | 结果 |
|---|---|---|
| TDD 红（ImportError: JevJudge 不存在） | 1 | 预期红 |
| 绿 `pytest tests/test_api_judge.py -q` | 0 | 16 passed |
| `pytest test_judge_port+test_core_zero_dep -q` | 0 | 15 passed |
| 三文件 -v | 0 | 31 passed |
| 全量 `pytest tests/ -q` | 0 | 189 passed（230s） |
| extras tomllib 验证 | 0 | 恰四项 |
| ruff 五文件 | 0 | All checks passed |
| 顶层惰性导出冒烟 | 0 | 三名可达；judge_state 不在顶层（范围符合） |

## 主控抽查（独立复跑）

`pytest tests/test_api_judge.py tests/test_judge_port.py -q` → 30 passed ✓；extras 终值恰四项 ✓；顶层三导出可达 ✓

## 遗留注意（交审计）

- Pyright 报 `judges/api.py:166` 返回类型不一致（dict vs tuple）——实现已重构 `_post` 返回三元组，疑为注解残留，需审计核实是否真实缺陷
- live 对拍（B3）未做——B2 验收挂钩 B3
