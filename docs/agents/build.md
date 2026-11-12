# 构建环境（跨机器可移植）

本仓库依赖 Visual Studio（MSVC）、OpenCV、Eigen、Python（pybind11 绑定）。
各机器路径不同，因此**仓库不写死任何绝对路径**：VS/cmake/ninja 用 `vswhere`
自动定位，依赖位置由各机器通过**环境变量**或**被 git 忽略的本地文件**注入。

## 一、快速开始

```bat
cd shape_based_matching
REM 1) 准备本机依赖路径（每台机器做一次）
copy scripts\configure.local.bat.example scripts\configure.local.bat
REM   然后编辑 configure.local.bat，填入本机的 OpenCV_DIR / EIGEN3_ROOT / PYTHON_EXECUTABLE

REM 2) 配置 + 构建（VS/cmake/ninja 由脚本自动定位，无需手动加 PATH）
scripts\configure.bat
scripts\build.bat
```

产物在 `build/` 下（exe / dll / pyd 等）。

## 二、依赖注入的两种方式（任选其一，脚本优先读本地文件）

| 变量 | 含义 | 示例 |
|------|------|------|
| `OpenCV_DIR` | OpenCV 的 `build` 目录，须含 `OpenCVConfig.cmake` | `E:/ToolChain/opencv4/build` |
| `EIGEN3_ROOT` | Eigen 根目录，须含 `Eigen/Core`（header-only） | `E:/ToolChain/eigen-3.4.0` |
| `PYTHON_EXECUTABLE` | Python 解释器，用于 pybind11 绑定 | `E:/python311/python.exe` |

- **方式 A（推荐，最省事）**：复制 `scripts/configure.local.bat.example` 为
  `scripts/configure.local.bat`（已被 `.gitignore` 忽略），在里面对应 `set`
  三个变量。`configure.bat` 会自动 `call` 它。
- **方式 B（CI / 临时）**：在调用前直接设环境变量，例如
  `set OpenCV_DIR=...` 后再跑 `configure.bat`。

> 注意：CMake 自身也有回退逻辑（`-DOpenCV_DIR` → 环境变量 `OpenCV_DIR` →
> 本机默认路径），但**本仓库要求显式提供**，缺失时脚本会给出明确报错而非静默失败。

## 三、关键坑（跨机器必读）

1. **OpenCV 运行时必须是 `vc16`**：官方 Windows 预编译包都是 `vc16`，脚本已固定
   传 `-DOpenCV_ARCH=x64 -DOpenCV_RUNTIME=vc16`。原因：OpenCV 4.x 的 cmake
   只识别到 VS2022(MSVC193x)，VS2026(MSVC1950) 会让 `OpenCV_RUNTIME` 为空而报
   “no binaries compatible”。换机器只要 OpenCV 是官方 Windows 包即可，无需改。
2. **`USE_CUDA` 默认 `OFF`**：无 CUDA 工具链的机器也能编（`.cu` 会被排除，走 CPU
   分支）；有 GPU 的机器在 configure 时追加 `-DUSE_CUDA=ON` 即可启用。
3. **`SBM_SIMD_LEVEL` 默认 `AVX2`**（针对本机 i5-8250U）：若目标机器 CPU 不支持
   AVX2，运行期可能 `SIGILL`。这种机器 configure 时追加 `-DSBM_SIMD_LEVEL=SSE2`
   （或 `NONE` 退回 SSE2 基线）。
4. **编出的 DLL 与消费端 OpenCV 版本必须一致**（最容易在"换机器重编后部署"时踩）：
   链接期就把 `opencv_world4XXX.dll` 版本写死进 DLL；C# 工程部署目录里必须放**同一
   版本**的运行时，否则加载即报"找不到指定的模块" / `DllNotFoundException`。**唯一
   的版本真相源是本机 `scripts/configure.local.bat` 的 `OpenCV_DIR`**——它指向哪个
   OpenCV `build`，编出的 DLL 就绑定哪个 world 版本。换版本 = 改 `OpenCV_DIR` 后清
   `build/` 重配重编。C 侧 ↔ C# 侧的完整 ABI 契约（导出名 / 结构体 / 调用约定 / 位数）
   及消费端路径的"每台机器不同"配置，见 `docs/agents/consumer.md`；换完版本跑
   `scripts/check_abi.bat`（已挂在 `smoke.bat` 末尾）即可自动校验。

## 四、本地偏离上游的改动（换机器 / 拉取上游时需留意）

- `src/cuda_icp/CMakeLists.txt`：本机无 CUDA 工具链，已将 `USE_CUDA=OFF` 时
  的库改为**只编 `.cpp`、排除 `.cu`**（原上游无条件把 `.cu` 编入库，会在无 nvcc
  的机器上 configure 失败）。CUDA 入口都由 `#ifdef CUDA_ON` 守卫，OFF 走 CPU 分支。
  装了 CUDA Toolkit 的机器可恢复为原行为并开 `USE_CUDA=ON`。

## 五、本机（开发机 A）示例路径（仅作格式参考，非通用值）

```
OpenCV_DIR        = E:/ToolChain/opencv4_13/build   (OpenCV 4.13.0；版本随本机 configure.local.bat 而定，勿据此判断版本)
EIGEN3_ROOT       = E:/ToolChain/eigen-3.4.0        (Eigen 3.4.0, header-only)
PYTHON_EXECUTABLE = E:/python311/python.exe         (Python 3.11 + numpy + pybind11)
```

> OpenCV 版本**不要**写死判断：它取决于你本机 `configure.local.bat` 里的 `OpenCV_DIR`。
> 上面 4.13.0 是当前开发机的取值；换机器/换版本以你自己的 `configure.local.bat` 为准。
> 编出的 DLL 链接哪个 `opencv_world4XXX.dll` 由它决定，必须与消费端运行时一致
> （见 `docs/agents/consumer.md`）。

> 早期会话里临时放在 `E:/ToolChain/configure_sbm.bat`、`build_sbm.bat` 的脚本已被
> 本仓库的 `scripts/configure.bat`、`scripts/build.bat` 取代，请改用后者。

## 六、冒烟检查（smoke test，仓库的轻量 guardrail）

`scripts/smoke.bat` 把「配置 → 构建 → 运行 demo → 断言通过」串成**一个可执行检查**，
是仓库目前唯一的自动化兜底（无 CI、无 pre-commit）：

```bat
scripts\smoke.bat
```

- 它依次调用 `configure.bat`、`build.bat`（均为增量，已配置/已构建时很快），
  再以 `--no-view` 非交互方式运行 `shape_based_matching_test --case case1`。
- 断言两条：程序退出码为 0，且输出含完成标记 `test end`。
- 任一项不满足即打印 `[FAIL]` 并退出非 0，同时把程序输出 dump 出来便于排查。
- 未来要上 CI，直接加一个 job 调这一个文件即可，无需另写逻辑。

> 注：它覆盖的是「demo 能编译并跑通自洽测试」这一基本回归；不覆盖 ICP 开关、
> 训练参数等其它路径（见上面「本地偏离」与 `apps/matching/test.cpp` 的命令行参数）。

## 七、性能剖析与调优开关

匹配耗时几乎全部在 `Detector::match` 内部的"响应图构建"上（fusion 流水线，逐像素
算梯度方向直方图 + 量化 + 线性化），模板相似度计算只占很小一部分。定位耗时时按下面
的顺序开开关：

```bat
REM 1) 分层看：每层金字塔的响应图构建 vs 模板匹配的耗时
cmake -S . -B build -DSBM_DEBUG_MATCH_TIME=ON
REM 2) 再往下钻：fusion 每个算子、每个线程的耗时（输出噪声较大）
cmake -S . -B build -DSBM_FUSION_PROFILE=ON
```

两个开关默认都是 `OFF`。关掉时既不打印，也不再为每次节点 update 取时间戳。

测量本身要注意几点：

- **用 `--repeat N` 取最小值。** 笔记本 CPU 有频率/温度波动，单次计时能差出 50%：
  ```bat
  shape_based_matching_test.exe --case case3 --data test/ --no-view --repeat 15
  ```
  输出里的 `match time: X ms (min of 15)` 才是可比的数。
- **`SBM_FUSION_THREADS` 控制 fusion 的并行线程数**，默认取"逻辑核数 / 2"（即物理核数）。
  想要更多并行（例如确认 SMT 是否有收益）可以临时覆盖：
  ```bat
  set SBM_FUSION_THREADS=8
  ```
  这个值原来写死是 4，在 6 核以上的机器上会浪费算力。

## 八、本仓库已做过的性能优化（避免重复劳动 / 回退）

在 `case3`（1766x883 场景图 + 69 个角度模板）上，下列改动把 `match()` 从约 33ms
降到约 21ms（同一台机、`--repeat 15` 取最小值）：

1. **`Hist3x3Node` 改为 SWAR 实现**：原来的 `votes_of_ori[8]` 动态下标只能落栈，
   每像素 9 次读改写 + 扫 8 项；改成 8 个 4bit 计数器塞进一个 32 位字 + 查表累加，
   单线程耗时 6.4ms → 1.4ms。该算子已用随机数据与原实现做过逐位等价性验证。
2. **`copyToWith0Bound` 改为逐行 memcpy/memset**：原来逐像素判越界，而 level 0 的
   输入常是 3 通道 BGR，这一步占了 Level-0 近 1/4 的时间。
3. **fusion 线程数 4 → 物理核数**（见上）。流水线是纯 tile 并行，基本线性换算力。
4. **fusion 的耗时打印改为编译期开关**：原来每次 `process()` 都无条件构造
   stringstream + 打印，且每次节点 update 都取时间戳。

5. **细层响应图按候选区域裁剪**：level 0（全分辨率）的响应图只服务于"在 level 1 候选点
   附近做 16x16 局部精化"，原先无论有没有候选都算整幅图。现在把 fusion 链拆成
   "全图前缀（灰度→两次高斯→两个 Sobel，顺带产出下一层输入图和 ICP 用的 dx_/dy_）"
   与"按需后缀（量化→投票→spread→响应图→linearize）"，先用最粗层粗搜出全部候选，
   再按候选覆盖的 tile 补算细层后缀。数值与原来整条链连续跑一致。

   注意实现的是 **tile 级选择**而不是单一外接矩形：多个目标分散在图上时，并集外接矩形
   会退化成整幅图，而按 tile 标记只算真正命中的块。最粗层仍全图粗搜，且保持单条完整链，
   避免为它多启动一次流水线。

   调优 / 观测开关：

   - `SBM_ROI_RESPONSE_MAP=0`：关掉裁剪，退回每层全图算，用于 A/B 对比。
   - `SBM_ROI_VERBOSE=1`：打印每层实际参与计算的 tile 数 / 总数与候选矩形数。
   - `Detector::enable_roi_response_map`：C++ 侧的等效开关。

   候选覆盖 >=85% 的 tile 时会自动退回全图，避免白付裁剪的调度开销。实测（`--repeat 15`
   取最小值）case3 的 `match()` 从 27.7ms 降到 23.6ms（约 -15%），case1 从 8.5ms 降到
   8.0ms；开关前后的匹配结果逐项一致（ICP 角度仅有 1e-8 量级差异）。

试过但**没有收益**、别再重复的方向（都实测过）：

- 把 `response1x1` 与 `linearizeTxT` 合并成一个算子：那 8 张中间响应图本来就在
  tile 缓冲里、是缓存命中的，合并反而多了一次查表，实测略慢。
- 改 `tileRows/tileCols`（32x256 → 64x256 / 64x512 / 96x384 / 128x256）：
  差异在噪声范围内，结论是 halo 重复计算不是瓶颈。
- 重排 `linearizeTxT` 的循环（按行相位一次读完，输入流量降到 1/T）：读写总量不变，
  瓶颈在 1 字节散写造成的 cache line RFO，实测无改善。

尚未尝试、且**会动精度**的路子：干脆跳过 level 0（定位精度从 ±4px 退到 ±8px，
靠 ICP 追回亚像素）。
