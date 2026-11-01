# B6 独立第二意见（Advisor，全文归档）

总结论：**实现可放行**——包内六文件五维度核验无阻断性缺陷。

1. [P2 需主控裁决] R5-P1「实验侧 embedder_cache 随行」方案归 B6 但委派排除——方案-实现矛盾（当前不可触发：embedder_from_env 不解析 pooling/前缀；触发条件 = 宿主 replace(config, query_prefix=...) 惯例已在 embedding_paired.py:263 存在）。主控裁决：列入 W3 工作项（不销账）。
2. golden fixtures embedder_identity 字段：gate 不比较该字段，冻结件无恙。
3. bypass/共享同 Run 混用：is_degraded 先于包装器构造（冻结 Run 级全域）、degraded_key 不含身份、一级缓存键含 digest——无键交叠无漏网。
4. pooling 退化输入：全 pad 行 argmax 返 0 静默取 pad 位、mean clip 防除零——不崩溃无 NaN，左右 padding 已钉。
5. [观察] API 前缀推高 token → 400 → 整批失败冻结全 Run 向量路 = 前缀固有代价叠加既定首败冻结语义（非 B6 引入）；[P3] README.md:99 注释仍写三元描述——随下一文档触点一行修正。
