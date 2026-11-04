# E1 证据包（e2e-phase01）

> 审计输入：本文件 + 方案 `docs/plans/e2e-phase01-plan.md` E1 节 + 数据契约 `experiments/agent_e2e/contract.md`（E0 落盘唯一权威）+ 工作区实际差异。
> 实施人：Executor（TDD 垂直切片，红态落盘）。抽查人：主控（复跑通过后标注可采信）。

## 1. 差异范围（边界内 4 文件）

| 文件 | 内容 |
|---|---|
| `experiments/agent_e2e/loading.py` | `load_cases`/`load_catalog`/`load_world`/`default_paths` + frozen dataclass（Case/ResourceSpec/Catalog）；引用校验采用可选参数口径（catalog/fixtures_dir 给定才校验对应引用） |
| `experiments/agent_e2e/sandbox.py` | `Sandbox`（from_world 深拷贝双份/check_permission 先记事件再抛/apply 唯一入口 payload 深拷贝/events append-only/diff_from_initial/state_snapshot 拷贝）+ 三常量单一出处 |
| `tests/test_e2e_loading.py` | 29 条（12 用例干净加载 + 13+ 篡改面含行号断言） |
| `tests/test_e2e_sandbox.py` | 9 条（隔离/diff/seq/越权事件先于异常/拷贝视图/非法谓词/常量与契约逐项一致） |

## 2. 验收清单（方案 E1 节 10 条）

Executor 自报全 PASS（逐条测试名见其回报）；主控复核要点：第 1 条含 E0 数据自动复核（12 用例零错误加载）；第 9 条常量与契约映射逐项一致有专测。

## 3. Executor 命令回报

- 定向：39 passed exit 0（loading 29 + sandbox 9 + 零依赖守卫 1）；ruff 四文件全过。
- TDD 红态落盘（C2 审计 P2-1 教训落实）：`red-log.md` + `red-loading.log`（R1 26 条收集期全红）+ `red-sandbox.log`（R2）；`default_paths()` 遗漏由测试抓出补齐——红绿循环有效证据。

## 4. 主控抽查（可采信标注）

- 独立复跑定向映射：**39 passed in 0.43s** → Executor 测试证据可采信。
- 边界核验：git status 恰为 4 文件 + 证据目录，包 `src/agent_retrieval` 零改动。
- sandbox 关键实现抽查：`from_world` 对 world 与 initial 各 deepcopy 一份（隔离根基）；`check_permission` 事件先记再抛（:78-82）；`apply` 对 payload 先 deepcopy 再写事件与状态（防别名渗入）。

## 5. Executor 偏差声明（主控核可）

1. `load_cases` 引用校验可选参数口径——委派单明示两选项之一，采纳合理（不硬编码仓库路径）。
2. `state_snapshot` 返回 `list[dict]`——委派单签名 `-> dict` 为笔误，按内容语义（「条目列表的拷贝」）实现，顺应正确。
3. 常量分布：三映射常量在 sandbox.py、`WORLD_COLLECTIONS` 在 loading.py——依赖方向 sandbox←loading 单向无环，符合「单一出处」且未造环。

## 6. 建议抽查命令

```bash
.venv/Scripts/python.exe -m pytest tests/test_e2e_loading.py tests/test_e2e_sandbox.py tests/test_core_zero_dep.py -q
.venv/Scripts/python.exe -m ruff check experiments/agent_e2e/loading.py experiments/agent_e2e/sandbox.py
git status --short
```
