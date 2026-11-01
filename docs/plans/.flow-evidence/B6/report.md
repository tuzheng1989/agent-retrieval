# B6 执行证据（pooling/前缀下沉 + Run 缓存入口分键与身份隔离）

Executor 回执要点归档（完整内容含 17 条验收×具名测试对照表、迁移清单 7 条逐条记录、两处接缝推断，见会话记录；pytest.log / pytest-full.log / ruff.log 已由 Executor 落盘）。

## 改动文件（边界内：源码 6 + 测试 4 + 文档 1）

源码：`embedders/config.py`（三字段 + `__post_init__` 校验 + identity 四元 flags JSON）、`embedders/api.py`（双前缀）、`embedders/local.py`（pooling 三分支 + 双前缀 + 校验前置）、`embedders/factory.py`（两分支透传）、`stores/redis_store.py`（Sequence[str] + 分域哈希：flags 空→legacy join 逐位不变，非空→`\x00v2\x00`+flags）、`run/cache.py`（`embedder_identity_digest` 单一规范化点、键形状五段、RunCachedEmbedder 必传身份 None=bypass+冻结正交、后端命中回写 L1、query_embedder 缺省 bypass）。
测试：`tests/test_local_pooling.py`（新建 7）、`tests/test_run_cache.py`（迁移 4 + 新增 6）、`tests/test_api_embedder.py`（迁移 1 + 新增 4）、`tests/test_redis_store.py`（新增 2，既有 9 零改动）。
文档：`README.md` 装配示例。

## 命令记录

| 命令 | 退出码 | 结果 |
|---|---|---|
| 定向四文件 -q | 0 | 49 passed |
| 全量 `pytest tests/ -q` | 0 | 205 passed（220s） |
| `ruff check src/ tests/` | 0 | All checks passed（2 处 UP012 按建议修复） |

## 关键实现决策（Executor 披露，交审计核实）

1. `query_key` 新签名 `(run_id, identity_digest, entry, text)`——键段序与契约形状一一对应（委派单笔误写 `query_key(run_id, text, entry)`，按键形状契约裁决）
2. 摘要计算 = 模块级公开函数 `embedder_identity_digest`（可观察接缝，测试与实现共用防漂移）
3. `RunCachedEmbedder.embedder_identity` 必传无默认（grep 证实全仓唯一构造点是 query_embedder）
4. LocalEmbedder pooling 校验放构造最前（不依赖 local extras）
5. `POOLINGS` 常量单点定义（config 导出、local 导入）
6. `tests/test_redis_store.py` 增 2 用例承接方案验收断言（该文件不在委派清单但为 vector_snapshot_id 验收的自然落点；既有 9 用例零改动）

## 范围外核对（Executor 侧，零破坏）

identity() 四元化后五处 experiments 消费面（embedding_paired:268 / rerank_paired:73 / jev_rerank:216 / build_fixtures:124 / run_benchmark:472,488）全部无三元解包点；`run_benchmark.py:472` 展示面多 flags 字符串 = R2-P2 已登记变化。

## 主控抽查（独立复跑，见会话命令记录）

默认回归字面量 / 快照 legacy 逐字节锚定 / bypass 冻结两入口 / padding 双向 / 竖线碰撞隔离——五类关键测试独立复跑通过。
