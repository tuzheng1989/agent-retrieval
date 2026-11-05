# 姿态-README 节对应关系（人工对照表）

> 机械核对：README 标题存在性 + 文档引用一致性由 self_check.py 断言；
> 映射语义（姿态是否恰当地锚定该节）由执行者人工核对 README 实文后确认。

| 姿态 | 文档节 | README 锚定节 | 机械核对 | 人工确认 |
|---|---|---|---|---|
| core_only | §2.1 | `### 1. Rank candidates with BM25 (+ optional vector path)` | 通过 | 确认：§1 即 `rank_candidates` 直调形态，片段取材该节并去掉 `vector=MockEmbedder()`（姿态定义 = 纯 BM25 零 extras） |
| api_embed | §2.2 | `### 2. Real embeddings` | 通过 | 确认：§2 即 `EmbeddingConfig`/`build_embedder` 装配；片段补 §1 的 `vector=` 注入行（消费面在 §1，S-D1 ② 锚定关系如此） |
| redis_snapshot | §2.3 | `### 3. Write-once vector snapshots (Redis)` | 通过 | 确认：§3 即 `RedisVectorStore`/`vector_snapshot_id`/`corpus_hash` write-once 流程，片段忠实取材该节 |
| full_stack | §2.4 | `### 4. Per-run query caching with degrade freezing` | 通过 | 确认：Run 缓存锚 §4 如实；判定面（ChatJudge/JevJudge）与 docgen 组合在 README 无对应节——按 S-D1 为本方案裁决（P0/P1 组件事实 + README Quality benchmarks 段），文档 §2.4 已显式标注该差异 |

人工核对执行者：executor（M1）。核对方式：实读 README.md 第 42/76/90/120 行四节标题与片段原文、文档 §2.1–2.4 逐姿态比对；机械核对由 self_check.py 同轮断言（见 self_check.log）。
