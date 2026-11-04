# E2 红态日志（TDD 先红后绿，2026-11-05）

$ .venv/Scripts/python.exe -m pytest tests/test_e2e_tools.py tests/test_e2e_retrieval_adapter.py -q

```
tests\test_e2e_retrieval_adapter.py:10: in <module>
    from experiments.agent_e2e.retrieval_adapter import RetrievalAdapter
E   ModuleNotFoundError: No module named 'experiments.agent_e2e.retrieval_adapter'
=========================== short test summary info ===========================
ERROR tests/test_e2e_tools.py
ERROR tests/test_e2e_retrieval_adapter.py
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!
2 errors in 0.41s
```

- 退出码：pytest 收集失败退出码 2（本文件首版误记 `exit=0`——当时 `$?` 取到了
  管道 tail 的退出码；实现前两模块不存在，真实 pytest 退出码非 0，红态成立）。
- 红态性质：retrieval_adapter.py / tools.py 未实现，两文件 40 条测试全部收集期
  ImportError 未执行。
