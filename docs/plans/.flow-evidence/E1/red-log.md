# E1 红态日志（TDD 先红后绿）

## R1：loading 切片（tests/test_e2e_loading.py，26 条测试）

- 命令：`.venv/Scripts/python.exe -m pytest tests/test_e2e_loading.py -q`
- 输出（`red-loading.log` 全文同目录）：
  ```
  E   ModuleNotFoundError: No module named 'experiments.agent_e2e.loading'
  ERROR tests/test_e2e_loading.py
  !!! Interrupted: 1 error during collection !!!
  1 error in 0.40s
  ```
- 红态性质：loading.py 未实现，收集期 ImportError——26 条测试（真实数据零错误加载 + 篡改校验面）全部未执行。

## R2：sandbox 切片（tests/test_e2e_sandbox.py，9 条测试）

- 命令：`.venv/Scripts/python.exe -m pytest tests/test_e2e_sandbox.py -q`
- 输出（`red-sandbox.log` 全文同目录）：
  ```
  E   ModuleNotFoundError: No module named 'experiments.agent_e2e.sandbox'
  ERROR tests/test_e2e_sandbox.py
  !!! Interrupted: 1 error during collection !!!
  ```
- 红态性质：sandbox.py 未实现，收集期 ImportError。

## 红转绿

- R1 → 实现 `experiments/agent_e2e/loading.py` 后全绿（见 pytest.log）。
- R2 → 实现 `experiments/agent_e2e/sandbox.py` 后全绿（见 pytest.log）。

## revision 2 时序说明（2026-11-05）

主控退回的四项补齐（record_event / EVENT_TYPES 词表 / loading 两处校验 / 随行测试）为
指定实现的课程纠偏：实现细节与测试面均由主控逐条给定，按指定实现直接落地后以既有全量
定向面回归验证（47 passed，见 pytest.log revision 2 节），未再单独立红态轮次。
R1/R2 的先红后绿纪律针对本批初始垂直切片，不受影响。
