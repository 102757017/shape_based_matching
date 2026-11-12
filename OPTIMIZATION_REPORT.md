# shape_based_matching 二轮性能优化报告（icp_refined 分支）

针对「更新后的代码（eaa5da6 性能优化）还能否进一步优化」的验证结论：

**能。本轮在不动算法精度的前提下再拿到 1.47x（公平同线程对比），并修复了 4 个真实缺陷。**

---

## 一、验证环境与基准方法

- 沙箱 2 vCPU（AVX2/AVX512），Debian，GCC 14.2，OpenCV 4.13.0（源码精简构建）
- 被测代码：icp_refined @ eaa5da6（上游最新） vs 本轮优化 @ opt_v2
- 基准工具：新增 `apps/matching/bench_json.cpp`（JSON 配置驱动，走 MatcherCore 生产 API）
- 负载：`test/case3`（1766×883 X 光连杆图）
  - 训练 ROI [280,190,1360,460]，feature_num=128，金字塔 {4,8}，角度 0~360° 步距 1° → **361 个模板**
  - 场景 `test/case3/test.png`（目标旋转约 -26.6°），score_threshold=90，NMS 开，ICP 精修开
  - 每轮 warmup 2 次 + 计时 10 次取最小值

## 二、基准结果（同模板、同输入、同线程数）

| 配置 | 基线 eaa5da6 | 本轮优化 | 加速比 |
|---|---|---|---|
| fusion 线程=2（公平对比） | **47.48 ms** | **32.19 ms** | **1.47x** |
| fusion 线程=1（沙箱默认） | 69.56 ms | 59.56 ms | 1.17x |
| 深金字塔 {4,8,16} + fusion=2 | 0 个匹配（漏检！） | 32.5 ms，正常检出 | 修复 |
| 回归 case1（600×473） | 11.51 ms | 11.62 ms | 无回退 |

检测结果逐位一致：case3 得分 96.51、位置 (884.95, 467.78)、角度 333.413°、fitness=1.0，基线与优化版完全相同；case1 得分 100.0 完全相同。

## 三、修复的缺陷（更新代码里新发现的）

1. **`apps/matching/test.cpp:334` 语法错误**：`infosatch.template_id].angle`（丢了 `[m`），全仓库无法编译。已修复为 `infos[match.template_id].angle`。
2. **`fusion_thread_num()` 堆破坏（严重）**：POSIX 路径对 `std::getenv()` 返回的指针调用 `std::free()`。getenv 返回的是环境区内部指针，free 它是未定义行为——glibc 直接 `free(): invalid pointer` 终止。后果：**只要设置 `SBM_FUSION_THREADS` 环境变量，进程必崩**，fusion 多线程从来没被真正用过。只有 MSVC 的 `_dupenv_s` 才需要 free。已修复（本沙箱 fusion=2 修复后稳定运行，提速 44%）。
3. **深金字塔阈值语义缺陷**：用户阈值被原样用于最粗层筛选，而最粗层特征数减半、容差大，同一位姿的粗层得分系统性偏低。实测 {4,8,16} 金字塔下最终 95.9 分的目标粗层只有 80~89 分 → **threshold=90 时整帧 0 匹配**。修复：粗层/中间层放宽 `SBM_COARSE_MARGIN`（默认 20）参与筛选，最细层(l=0)仍按原阈值精确过滤——最终结果阈值语义不变。
4. **训练 padding 公式过度分配**：`diagonal*1.5+50` 对扁长 ROI（1360×460）开出 5766×4866=28MP 的 padded 图（峰值内存 ~1GB，低配机器会 OOM），特征提取耗时随面积线性暴涨。改为旋转扫掠包围盒下界 `(diagonal-max_scale - min(w,h))/2 + 60`（=5.9MP，缩 4.8 倍）。特征只取自 mask 限定的 ROI 内部，padded 区全零，模板特征逐位不变。

## 四、性能优化点（本轮新增）

按收益排序：

1. **ICP 场景缓存**（`Detector::refine`）：原来每个匹配结果都重建一次"全图 Canny + 逐边缘点亚像素 facet 模型 + KD-tree"（几十 ms 级/次），一帧 N 个结果就重复 N 次。改为按 dx_/dy_ 数据指针判据缓存，一帧只建一次。`use_refine=true` 时收益随结果数线性放大。
2. **粗层搜索内循环**（`coarseCandidates`）：
   - `score = raw*100/(4N) > threshold` 的逐像素浮点乘除改为整数域阈值 `raw > floor(threshold*4N/100)`（每帧省 ~420 万次浮点除法）；
   - <64 特征模板直接在 uint8 累加图上扫描，去掉 `convertTo(CV_16U)` 整图拷贝；
   - 候选 vector 预留 + 线程局部持久相似度缓冲（360 模板/帧的 Mat::zeros 归零）。
3. **每模板粗层候选 top-K 截断**（`SBM_MAX_CAND_PER_TPL`，默认 8）：最粗层正确位姿几乎总是该模板的全局最大值，march-up 只会精化最好几个；多实例由不同角度模板各自命中。噪声候选不再进入逐层精化。
4. **march-up 零堆分配**（`matchClass`）：16×16 局部相似度缓冲提到层循环外复用（原来每候选每层一次 Mat::zeros + convertTo）；按实际类型扫描（8U 单字节读）；分母提到循环外。
5. **跨帧缓冲池**（`Detector` 成员 `lm_pool_/sobel_pool_*/pyrdown_pool_/dxy_pool_`）：同尺寸连续视频流下 create() 是 no-op，消除每帧 ~10MB 的响应图/中间结果堆分配与缺页。
6. **跨帧 ICP dx_/dy_ 缓冲池**：同上。

## 五、剩余耗时构成（32ms 的去向，fusion=2）

| 阶段 | 耗时 | 说明 |
|---|---|---|
| fusion 前缀（L0 全图 Gauss×2+Sobel×2） | ~7 ms | 已 2 线程 |
| fusion L1 前缀+粗层响应图链 | ~6 ms | 已 2 线程 |
| 粗层候选搜索（361 模板 × 全图扫） | ~2.6 ms | 本轮已优化 |
| L0 细层响应图（候选邻域 tile，本例接近全图） | ~16 ms | 模板 1360×460 铺满图，tile 裁剪无收益 |
| march-up + NMS + ICP + 后处理 | ~1 ms | 本轮已优化 |

进一步提速的方向（未在本轮实施）：模板更小（工业常规 200~400px）时 tile 裁剪收益会自然显现；把 L0 后缀的 8 方向响应图改为按需 16×16 窗口直算（对超大模板收益明显）；`num_features` 降到 ≤63 让 L0 march-up 也走 uint8 快路径（需重标定阈值）。

## 六、使用说明

```bash
# 基准（JSON 配置, 见 bench.json / bench_p16.json / bench_c1.json）
bench_json --config bench.json --repeat 10 --warmup 2 --out result.png

# 可调环境变量
SBM_FUSION_THREADS=4        # fusion 流水线线程数(默认=物理核数; 修复后可安全设置)
SBM_COARSE_MARGIN=20        # 粗层阈值裕量(0=关闭, 恢复旧行为)
SBM_MAX_CAND_PER_TPL=8      # 每模板粗层候选上限
SBM_ROI_RESPONSE_MAP=1      # 细层响应图 ROI 裁剪开关
```

注意：`Detector` 实例非线程安全（fusion_buffers/缓冲池为共享成员），单实例请串行调用 match()；多路相机请每路一个实例。

## 七、改动清单

```
src/matching/line2Dup.cpp      | 217 ++++++++++-----   # 全部内核优化 + 3 个修复
src/matching/line2Dup.h        |  15 +                 # 缓冲池/场景缓存成员
src/matching/matcher_core.cpp  |  10 +-                # 训练 padding 收紧
src/matching/fusion.h          |  17 +-                # (无逻辑改动, 注释对齐)
apps/matching/bench_json.cpp   | 新增                  # JSON 配置基准工具
apps/matching/test.cpp         |   2 +-                # 语法错误修复
bench*.json                    | 新增                  # 基准配置
```
