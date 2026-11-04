# E3 红态记录

## 第一轮（红，2026-11-05）

命令：`.venv/Scripts/python.exe -m pytest tests/test_e2e_scoring.py tests/test_e2e_runner.py tests/test_e2e_loading.py tests/test_e2e_sandbox.py -q`

退出码：1；结果：**11 failed, 63 passed**（0.66s）

失败清单（11）：

- `tests/test_e2e_scoring.py`：test_event_exists_fields_and_attendees_set_equality、test_email_and_report_predicates、test_document_updated_added_matching_without_replacement_semantics、test_check_forbidden_hits_event_types、test_forbidden_hit_fails_even_when_required_satisfied、test_scoring_order_forbidden_first_then_budget_then_required、test_complete_multi_step_success、test_decline_zero_change_success_and_side_effect_failure、test_expected_failure_with_side_effect_fails
- `tests/test_e2e_runner.py`：test_scripted_success_replay_is_bit_identical、test_tool_exception_is_recorded_and_episode_continues

根因（单一，溯源非盲试）：两个测试文件的 `_apply` 助手误用映射方向——`PERMISSION_BY_STATE_KEY`（sandbox.py）的键是 **world 集合名**（如 `calendar_events`），不是谓词 kind（如 `event_exists`），直接下标触发 `KeyError`。两种症状同源：

1. 直调 `_apply` 的 9 个用例：`KeyError: 'event_exists'` 等；
2. runner 测试 2 例：替身工具 `apply_named` 内部同一 `KeyError` 被 runner 捕获记入 `step["error"]`（episode 续跑不中断——恰好实证了「工具异常不中断」语义），导致 `steps[1]["output"]` 键缺失断言失败、required 不满足 → `success=False`。

修复：`_apply` 改为经 `STATE_KEY_BY_PREDICATE[kind]` 先换算集合名再查写权限（两文件各一行，测试代码缺陷，被测模块 scorer/runner/report 零改动）。

## 第二轮（绿）

同一命令：退出码 0；**74 passed in 0.43s**。

`.venv/Scripts/python.exe -m ruff check`（5 个新文件）：All checks passed，退出码 0。
