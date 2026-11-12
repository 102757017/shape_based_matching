# 二轮性能优化报告（指针）

> 本报告正文已迁移至 **docs/OPTIMIZATION_REPORT.md**，请从那里阅读。

本指针在以下情形时引导你打开正文：

- 需要性能优化结论与加速比（fusion 线程 / 粗层搜索 / 缓冲池 / ICP 场景缓存等）
- 排查已修复缺陷的背景：case3 漏检、粗层阈值语义、`fusion_thread_num()` 堆破坏、训练 padding 过度分配
- 使用 bench_json 基准工具或 `SBM_*` 环境变量（`SBM_FUSION_THREADS` / `SBM_COARSE_MARGIN` / `SBM_MAX_CAND_PER_TPL` / `SBM_ROI_RESPONSE_MAP`）

基准配置已集中到 `bench/` 目录（`bench/bench.json` 等）。
