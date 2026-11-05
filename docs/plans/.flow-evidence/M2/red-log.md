# M2 红态记录（tests/test_scenario_matrix.py 垂直切片）

TDD 纪律：每姿态一轮红绿。红/绿态命令均为 `.venv/Scripts/python.exe -m pytest tests/test_scenario_matrix.py -q`（仓库根运行，与主表验证命令同形态）。

## Round 1 — core_only 姿态 4 格

- **红态落位**：测试函数与 4 格 CellSpec 表先写，语料加载替身（`_POOLS`）留空未接线（装配未实现）。
- **红态结果**：`4 failed`——`core_only_en_tool` / `core_only_zh_tool` / `core_only_en_skill` / `core_only_zh_skill` 四格各自失败，失败用例 id 均含格 id。
- **失败摘要**：`KeyError: 'jev_cases'` / `KeyError: 'catalog_tools'` / `KeyError: 'catalog_skills'`（语料替身池未接线，先于断言触发）。
- **绿态接线**：`_jev_resources()`（`experiments/jev_cases.json` → 本地 `JevResource` frozen dataclass）+ `load_catalog(default_paths()["catalogs"] / "catalog-v1.json")` 按 `kind` 过滤出 `catalog_tools` / `catalog_skills` 两个池 → `4 passed`。

## Round 2 — api_embed 姿态 4 格

- **红态落位**：新增 api_embed 4 格与 `EmbeddingConfig` + `MockEmbedder` 融合路；初稿按 matrix §2.2 装配片段的直观写法把 `EmbeddingConfig` 放在**模块级顶层导入**。
- **红态结果**：`4 failed, 4 passed`——core_only 4 格全部红，api_embed 4 格自身绿。
- **失败摘要**（四格一致，含格 id）：

  ```
  AssertionError: [<格 id>] core_only 格引入了 extras 模块: ['agent_retrieval.embedders', 'agent_retrieval.embedders.config']
  ```

- **该红的价值**：直接验证 M2 验收第 3 条的守护本体——embedders 在收集期被顶层导入拉入 `sys.modules`，core_only 格的零 extras 断言（对照 `test_core_zero_dep` 纪律）咬合，且报错含格 id 与违规模块名可定位。
- **绿态接线**：`EmbeddingConfig` 改为 `_embedding_config()` 内**惰性导入**（PEP 562 顶层惰性导出——取值才触发 embedders 导入；core_only 测试按定义顺序先于 api_embed 运行，守护断言先于 embedders 首次导入）。返回注解引用按 `core/ports.py` 同款 TYPE_CHECKING 纪律惰性化（ruff F821 修复）→ `8 passed`。
