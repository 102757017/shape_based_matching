# shape_based_matching  

### For this branch, see [refine icp issue](https://github.com/meiqua/shape_based_matching/issues/100) for detailed introduction

update:   
**[fusion implementation to run faster!](https://github.com/meiqua/shape_based_matching/issues/77)**  
[Transforms in shape-based matching](./Transforms%20in%20shape-based%20matching.pdf)  
[pose refine with icp branch](https://github.com/meiqua/shape_based_matching/tree/icp2D), 0.1-0.5 degree accuracy   
[icp + subpixel branch](https://github.com/meiqua/shape_based_matching/tree/subpixel), < 0.1 degree accuracy  
[icp + subpixel + sim3(previous is so3) branch](https://github.com/meiqua/shape_based_matching/tree/sim3), deal with scale error  

try to implement halcon shape based matching, refer to machine vision algorithms and applications, page 317 3.11.5, written by halcon engineers  
We find that shape based matching is the same as linemod. [linemod pdf](Gradient%20Response%20Maps%20for%20Real-TimeDetection%20of%20Textureless%20Objects.pdf)  

halcon match solution guide for how to select matching methods([halcon documentation](https://www.mvtec.com/products/halcon/documentation/#reference_manual)):  
![match](./match.png)  

## steps

1. change apps/matching/test.cpp 里写死的测试数据前缀为你本地顶层文件夹

2. in cmakeList line 23, change /opt/ros/kinetic to somewhere opencv3 can be found(if opencv3 is installed in default env then don't need to)

3. cmake make & run. To learn usage, see different tests in test.cpp. Particularly, scale_test are fully commented.

NOTE: On windows, it's confirmed that visual studio 17 works fine, but there are some problems with MIPP in vs13. You may want old codes without [MIPP](https://github.com/aff3ct/MIPP): [old commit](https://github.com/meiqua/shape_based_matching/tree/fc3560a1a3bc7c6371eacecdb6822244baac17ba)  

## thoughts about the method

The key of shape based matching, or linemod, is using gradient orientation only. Though both edge and orientation are resistant to disturbance,
edge have only 1bit info(there is an edge or not), so it's hard to dig wanted shapes out if there are too many edges, but we have to have as many edges as possible if we want to find all the target shapes. It's quite a dilemma.  

However, gradient orientation has much more info than edge, so we can easily match shape orientation in the overwhelming img orientation by template matching across the img.  

Speed is also important. Thanks to the speeding up magic in linemod, we can handle 1000 templates in 20ms or so.  

[Chinese blog about the thoughts](https://www.zhihu.com/question/39513724/answer/441677905)  

## improvment

Comparing to opencv linemod src, we improve from 6 aspects:  

1. delete depth modality so we don't need virtual func, this may speed up  

2. opencv linemod can't use more than 63 features. Now wo can have up to 8191  

3. simple codes for rotating and scaling img for training. see test.cpp for examples  

4. nms for accurate edge selection  

5. one channel orientation extraction to save time, slightly faster for gray img

6. use [MIPP](https://github.com/aff3ct/MIPP) for multiple platforms SIMD, for example, x86 SSE AVX, arm neon.
   To have better performance, we have extended MIPP to uint8_t for some instructions.(Otherwise we can only use
   half feature points to avoid int8_t overflow)  

7. rotate features directly to speed up template extractions; selectScatteredFeatures more 
evenly; exautive select all features if not enough rather than abort templates(but features <= 4 will abort)

## some test

### Example for circle shape  

#### You can imagine how many circles we will find if use edges  
![circle1](test/case0/1.jpg)
![circle1](test/case0/result/1.png)  

#### Not that circular  
![circle2](test/case0/2.jpg)
![circle2](test/case0/result/2.png)  

#### Blur  
![circle3](test/case0/3.png)
![circle3](test/case0/result/3.png)  

### circle template before and after nms  

#### before nms

![before](test/case0/features/no_nms_templ.png)

#### after nms

![after](test/case0/features/nms_templ.png)  

### Simple example for arbitary shape

Well, the example is too simple to show the robustness  
running time: 1024x1024, 60ms to construct response map, 7ms for 360 templates  

test img & templ features  
![test](./test/case1/result.png)  
![templ](test/case1/templ.png)  


### noise test  

![test2](test/case2/result/together.png)  

## ellipse detection (integrated)

Integrated [standard-ellipse-detection](https://github.com/memory-overflow/standard-ellipse-detection) (MIT) as a vendored library at `src/ellipse/ellipse_detection/`.

Adaptations for this project (OpenCV 4.13 + MSVC, no LAPACK):
1. Replaced the vestigial LAPACK `dggev_` call in `fitEllipse` with `Eigen::GeneralizedEigenSolver` (same semantics: eigenvalue = alpha/beta). Eigen path: `EIGEN3_ROOT` (same as cuda_icp).
2. Rewrote `cvcannyapi.cpp` from OpenCV C API (`CvMat`/`cvGetMat`/`cvSobel`) to OpenCV 4 C++ API; algorithm unchanged.
3. Guards for Eigen port: 0-inlier fit returns `nullptr` (Eigen RealQZ hangs on all-zero matrices where LAPACK returned gracefully), NaN conic coefficients are rejected, and `EllipseIter` bails out on non-finite coefficients.

Build (with the existing build dir):
```
cmake --build build --target ellipse_detect_demo --config Release
```

Usage:
```
build/Release/ellipse_detect_demo.exe <image> [output.png] [options]
options:
  --polarity <0|-1|1>          ellipse polarity, 0=all (default 0)
  --line-width <px>            ellipse line width in pixels (default 2.0)
  --min-cover-angle <deg>      coverage threshold in degrees (default 240)
  --min-goodness <0~1>         final quality threshold (default 0.4)
  --candidate-goodness <0~1>   candidate pre-filter threshold (default 0.3)
```
Lower `--min-cover-angle`/`--min-goodness` to detect more heavily occluded ellipses, at the cost of possible false/less accurate fits.

API:
```cpp
#include "detect.h"
// 1) 原签名(默认参数, 行为不变)
zgh::detectEllipse(gray.data, gray.rows, gray.cols, ells, /*polarity=*/0, /*line_width=*/2.0);
// 2) 参数结构体版本(阈值可调)
zgh::DetectParams params;               // 默认值 == 原硬编码行为
params.min_cover_angle = 150;           // 放宽完整度
params.min_goodness = 0.25;             // 放宽质量
zgh::detectEllipse(gray.data, gray.rows, gray.cols, ells, params);
```

NOTE coordinate convention: `Ellipse::o.x` is the **row** and `o.y` is the **col**; `phi` is measured in the (row, col) plane. When drawing with `cv::ellipse`, use center `(o.y, o.x)`, axes `(a, b)` and angle `90° - phi`. `ellipse_debug.exe` dumps per-stage intermediates (gradient, arcs, candidates) for troubleshooting.

### Performance

`ellipse_bench.exe <image> [--iters N] [--runs M] [--micro K]` measures throughput and prints a result signature so you can verify that optimizations did not change the output.

The dominant cost used to be the per-point ellipse scan (`EllipseIter`), which was ~70% of total time. Two fixes in `types.hpp` (both behavior-preserving — every detected ellipse is bit-identical):

1. **`EllipseIter::operator++()` returned `EllipseIter` by value.** Every `++iter` therefore deep-copied the internal `std::stack<Pixel>`, which holds an entire column of candidate pixels. The prefix increment now returns `EllipseIter&` (same for `RectIter`). Iterator step went 401 ns -> 57 ns (7x).
2. **`Ellipse::distopoint()` recomputed `cos/sin(phi)`, `a*a`, `b*b` on every call.** These are now cached in `Ellipse::FastGeom` and refreshed only when the ellipse changes (`Ellipse::distFast(x, y)`).

Measured on `build/Release/ellipse_bench.exe` (MSVC Release, profile macros OFF):

| image | before | after | speedup |
|---|---|---|---|
| test5.jpg 473x291 | 90.3 ms | 33.5 ms | 2.7x |
| test1.jpg 309x236 | 33.4 ms | 12.9 ms | 2.6x |
| synth_ellipse.bmp 640x480 | 35.2 ms | 24.2 ms | 1.5x |

### Multithreading

`DetectParams::num_threads` (0 = auto / all cores, 1 = off) controls it; exposed as `EllipseParams.num_threads` in Python, `sbm_ellipse_params_t.num_threads` in the C ABI and `EllipseParams.NumThreads` in C#.

Parallelized (all read-only on the shared gradient field, so results are bit-identical to serial):

- `gaussianSampler` — the two convolution passes, per column / per row
- `calculateGradient3` — the per-row `atan2` angle loop
- `getValidInitialEllipseSet` — the arc-pair fitting loop (outer `i`)
- the candidate scoring loop in `detectEllipse`

`subdetect` (the refine stage) **must stay serial**: an accepted ellipse writes `angles[addr] = ANGLE_NOT_DEF`, i.e. it consumes its own support pixels, so later candidates see a different gradient field. LSD region-growing is likewise serial. Those two are the remaining ~30%.

Threading uses a ~100-line `ThreadPool` in `include/parallel.hpp`, **not OpenMP**. On MSVC the default OpenMP wait policy makes worker threads spin for milliseconds after each parallel region; with only two regions per frame that ate the entire gain (4 threads came out *slower* than 1 thread) unless `OMP_WAIT_POLICY=passive` was set — which a third-party caller cannot be expected to do. The pool's workers sleep on a condition variable instead, and it adds no `vcomp140.dll` dependency to the shipped binaries.

Measured on `ellipse_bench.exe` (8 logical cores), same images as above, medians:

| image | 1 thread | auto (8) | speedup from threading |
|---|---|---|---|
| test5.jpg 473x291 | 34.0 ms | 20.6 ms | 1.65x |
| test1.jpg 309x236 | 13.4 ms | 10.0 ms | 1.35x |
| synth_ellipse.bmp 640x480 | 25.1 ms | 17.9 ms | 1.40x |

Combined with the scan fixes above, total is **90 ms -> ~20 ms (4.4x)** for test5. Output is deterministic: the result signature is identical for every thread count (verified).

Caveat if you call `detectEllipse` from your own multiple threads: parallel regions are serialized by an internal dispatch lock, so they queue rather than oversubscribe. Do not call it from inside a `parallelFor` body.

Remaining hotspots: LSD region-growing (~15%) and `subdetect` (~8%), both inherently sequential, and the 6x6 generalized eigen-solve in `fitEllipse` (~100 calls/image). Halíř–Flusser would turn the last one into a 3x3 standard eigenproblem, but it changes the numerics and therefore the detected ellipses, so it was left alone.

### Python binding

```python
import shape_based_matching_py as sbm   # build/Release/shape_based_matching_py.cp311-win_amd64.pyd
ells = sbm.detect_ellipses(gray_or_bgr_u8)          # 默认参数
p = sbm.EllipseParams()                             # 阈值可调
p.min_cover_angle, p.min_goodness = 150, 0.25
ells = sbm.detect_ellipses(img, p)
# 每项: center_row/center_col/a/b/phi(度, cv::ellipse 口径)/goodness/coverangle/polarity
```
Smoke test: `python py_tests/test_detect_ellipses.py` (在 build/Release 下运行)。

### C ABI (任何语言)

```c
#include "c_api/matcher_c.h"
sbm_ellipse_params_t p; sbm_ellipse_params_init(&p);
p.min_cover_angle = 150;
sbm_ellipse_t out[64];
int n = sbm_detect_ellipses(&image, &p, out, 64);   // n = 检出数, 按 goodness 降序
```
坐标口径: `cx=列(x), cy=行(y)`, `phi` 为相对 x 轴弧度 (cv::ellipse 兼容)。错误详情 `sbm_last_error(NULL)`。

### C# (P/Invoke)

`csharp/MatcherNative.cs` 已含 `EllipseParams` / `Ellipse` / `EllipseDetector`:
```csharp
var p = EllipseParams.Default();
p.MinCoverAngle = 150;
Ellipse[] ells = EllipseDetector.Detect(grayBytes, width, height, stride: 0, p);
// 或用 RawImage 包住 OpenCvSharp Mat.Data (行优先连续内存)
```
可运行示例: `csharp/EllipseDetectSample/` (`dotnet run` 直接体验, 内含零依赖的 24 位 BMP 解码)。
原生依赖: `shape_based_matching_c.dll` + `opencv_world4130.dll` (均在 build/Release)。

## some issues you may want to know  
Well, issues are not clearly classified and many questions are discussed in one issue sometimes. For better reference, some typical discussions are pasted here.  

[object too small?](https://github.com/meiqua/shape_based_matching/issues/13#issuecomment-474780205)  
[failure case?](https://github.com/meiqua/shape_based_matching/issues/19#issuecomment-481153907)  
[how to run even faster?](https://github.com/meiqua/shape_based_matching/issues/21#issuecomment-489664586)  

