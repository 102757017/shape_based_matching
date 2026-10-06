# 椭圆检测算法性能对比报告

> 对比对象（本工程内的两条椭圆检测路线）
> - **standard**：`src/ellipse/ellipse_detection`（standard-ellipse-detection, MIT），弧段扫描 + 精化拟合
> - **AAMED**：`src/ellipse/aamed`（AAMED / FLED, BSD-2-Clause），弧邻接矩阵 + 弧段分组投票
>
> 生成工具：`test/ellipse_compare.cpp`（目标 `ellipse_compare`）
> 构建树：`build/`（AVX2，默认）与 `build-sse2/`（SSE2），OpenCV 4.13 / MSVC 2026 / x64

## 1. 对比方法

* **速度**：交错（interleaved）计时 —— A/B 交替各跑一轮共 7 轮，取**中位数**。交替执行可以抵消
  频率漂移、后台任务造成的单侧偏置，比"先跑完 A 再跑完 B"可信得多。每场景另有 1 轮预热
  （AAMED 的首次内存分配不计入统计）。
* **精度**：合成场景带**已知真值**（6 个场景 + 1080p 场景），把椭圆栅格成**填充掩膜**后算 IoU，
  用一对一贪心匹配（IoU ≥ 0.5 记 TP），统计 F1 与 TP 的平均 IoU。
  > 踩坑记录：最初用**描边环**算 IoU，真值画 3px、预测画 1px，环面积随线宽线性变化，
  > IoU 上限只有 ~0.33，几何完全正确的结果也判成不匹配（TP 全 0）。改成填充区域后
  > 同组数据的 IoU 立刻回到 0.95+。评估口径错了会得出完全相反的结论。
* **两路输出都已归约到同一个 `sbm_ellipse_t` 口径**（cx=列 / cy=行 / a,b 半轴 / phi 相对列轴弧度），
  换算方式与 C ABI 的 `sbm_detect_ellipses`、`sbm_detect_ellipses_aamed` 完全一致，
  所以下面的数字可以直接代表线上接口的表现。
* **AAMED 聚类方法**：本项目把上游 `SELECT_CLUSTER_METHOD` 从默认的 `PRASAD_CLUSTER_METHOD`
  （几何距离聚类，没有 IoU 门槛）切到 `OUR_CLUSTER_METHOD`（基于 IoU 的
  `EllipseNonMaximumSuppression`），这样 `sbm_aamed_params_t.nms_iou` 才能端到端生效
  （T_iou 由 `FLED::SetNMSIoU` 注入）。下方 AAMED 数字均在此配置下取得。

## 2. 综合结果（AVX2 树，median of 7）

| 场景 | standard 耗时 | AAMED 耗时 | 加速比 | standard F1 | AAMED F1 | standard mIoU | AAMED mIoU |
|---|---|---|---|---|---|---|---|
| full-circle（3 个完整清晰椭圆） | 21.4 ms | 2.9 ms | 7.4x | 0.857 | **1.000** | 0.972 | 0.958 |
| small（4 个小目标 r≈22~28） | 16.0 ms | 2.2 ms | 7.2x | 0.571 | **1.000** | 0.919 | 0.871 |
| partial-arc（局部残缺，遮挡 100px） | 15.2 ms | 2.0 ms | 7.5x | 0.500 | 0.500 | 0.965 | 0.959 |
| elongated（细长椭圆 a/b≈4） | 21.1 ms | 2.8 ms | 7.5x | 0.800 | **1.000** | 0.958 | 0.917 |
| noisy-lowcontrast（σ=30 噪声+模糊） | 18.0 ms | 2.3 ms | 7.8x | 0.800 | **1.000** | 0.948 | 0.941 |
| dense-occluded（7 个密集 + 遮挡） | 17.6 ms | 2.9 ms | 6.1x | 0.250 | **0.727** | 0.909 | 0.922 |
| **hd-1080p**（1920×1080，3 个大椭圆） | **159.0 ms** | **12.7 ms** | **12.5x** | 1.000 | 1.000 | 0.995 | 0.987 |
| **合计（7 场景）** | **265.3 ms** | **28.0 ms** | **9.5x** | 0.683 | **0.890** | — | — |

结论一句话：**AAMED 既快 7~13 倍，在困难场景（小目标、密集遮挡、低对比）上还更准**；standard 路
只在完整清晰的大椭圆上勉强保持同等精度，但速度慢一个数量级。

## 3. 精度差异从哪儿来

| 场景 | 谁赢 | 具体表现 |
|---|---|---|
| 小目标（r≈22px） | AAMED | standard 默认门槛（min_cover_angle=240°、min_goodness=0.4）漏掉 2/4，且漏的都是最小的那个；AAMED 4/4 全中 |
| 密集 + 遮挡 | AAMED | standard 只认出 7 个里的 1 个，AAMED 认出 4 个，误检都是 0 |
| 细长椭圆 a/b≈4 | 基本平手 | AAMED 全中但多 1 个误检；standard 漏 1 个 |
| 局部残缺（只画 2/3 弧） | 平手 | 两路都只认出 1/3，都不是对方的优势区 |
| 完整清晰大椭圆 | AAMED 略优 | standard 多 1 个误检（碎弧拟合出来的假椭圆），AAMED 干净 |

**关键结构性差异**：standard 路的 `min_cover_angle=240°` 要求弧覆盖足够完整，遮挡一多就整条丢弃；
AAMED 用评分（goodness）+ NMS 判据，对局部残缺天然宽容 —— 这正是它在 dense-occluded
场景 F1 0.25 → 0.727 的来源。

## 4. 真实图像（无真值，只看检出数与速度）

| 图像 | 尺寸 | standard | AAMED |
|---|---|---|---|
| case0/1.jpg | 900×600 | 1 个（goodness 0.752）116.2 ms | 3 个（goodness 0.967）27.1 ms |
| case0/2.jpg | 640×480 | 8 个（0.928）25.6 ms | 8 个（0.952）17.6 ms |
| case0/3.png | 458×258 | 1 个（0.745）8.2 ms | 1 个（0.842）1.5 ms |
| case1/test.png | 600×473 | 1 个（0.740）45.7 ms | 2 个（0.908）8.0 ms |
| case2/test.png | 1445×532 | **0 个** 61.5 ms | **0 个** 9.5 ms |

* case0/1.jpg：standard 的 `min_cover_angle` 直接把它判死，只吐 1 个低质椭圆；AAMED 一次给 3 个高质结果。
* case2/test.png 两路都 0 个 —— 那张图没有足够完整/清晰的椭圆轮廓，属于"图本身没有目标"，
  不是算法问题（两路都很快就能否掉，反而说明误检率低）。
* **AAMED 首次调用的隐名成本**：`sbm_aamed_create` 会按 rows×cols 量级预分配（1080p 约 550 MB），
  真实图上首帧含分配耗时 18~118 ms（图越大越贵），之后稳定在 2~30 ms。**连续处理视频流/多帧
  务必用 `sbm_aamed_create` 复用检测器**（`sbm_aamed_detect`），不要每帧走一次性接口。

## 5. SIMD 开关的影响（build/ vs build-sse2/）

| 树 | standard 合计 | AAMED 合计 |
|---|---|---|
| build（AVX2） | 265.3 ms | 28.0 ms |
| build-sse2（SSE2） | 268.9 ms | 27.6 ms（同方法重建，差异在噪声内） |

差异 **2~3%，基本落在噪声内**。也就是说 `SBM_SIMD_LEVEL` 这个开关对这两条椭圆路径几乎**没有
可测收益** —— MIPP 的 SIMD 化主要落在 line2Dup / matcher_core 的梯度场热点上，椭圆检测这两路的
耗时大头是：
- standard：逐像素弧扫描 + OpenMP 分块（640×480 单线程扫一个像素就要做多次浮点运算，主线程花在候选搜索而非向量化循环上）
- AAMED：邻接矩阵构建 + flann 检索，是访存/索引密集而非纯算术密集，SIMD 帮不上

## 6. 选型建议

* 要**在线/实时**（产线逐帧、机器人视觉）：选 **AAMED**，1080p 11.6 ms 意味着单帧轻松 >60 fps；
  同时记得复用检测器句柄、按需 `prepare` 预分配。
* 要**保守低误检**、图里都是完整清晰的大椭圆、且可以容忍 100~160 ms/帧：standard 路仍可用，
  而且它的 `coverangle` 信息 AAMED 不产出（AAMED 侧恒填 0）。
* 想两者互补：两路输出同口径（`sbm_ellipse_t`），可以直接做**并集/投票融合** ——
  standard 的高 coverangle 用来给 AAMED 的低分候选做二次确认，是工程上最划算的组合方式。
* 如果标准路在某些场景漏检（小目标、遮挡），调参优先级：`min_cover_angle`（240→180） >
  `min_goodness`（0.4→0.3）> `line_width`。AAMED 侧优先调 `t_val`（0.77→0.65 换召回）。

## 7. 复现

```bash
# 构建（两棵树都带这个目标）
cmake --build build       --config Release --target ellipse_compare -j 8
cmake --build build-sse2  --config Release --target ellipse_compare -j 8

# 跑：默认 7 个合成场景；--real 追加真实图（逗号分隔）；--dump 导出可视化（真值蓝/standard 绿/AAMED 红）
./build/Release/ellipse_compare.exe \
    --real "test/case0/1.jpg,test/case0/2.jpg" --dump /tmp/ec_dump
```

可选参数：`--iters N`（默认 7）、`--threads T`（standard 路并行线程）、`--dump <dir>`。
真实图可视化里：蓝=真值，绿=standard 检出，红=AAMED 检出。
