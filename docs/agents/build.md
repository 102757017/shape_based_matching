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

## 四、本地偏离上游的改动（换机器 / 拉取上游时需留意）

- `src/cuda_icp/CMakeLists.txt`：本机无 CUDA 工具链，已将 `USE_CUDA=OFF` 时
  的库改为**只编 `.cpp`、排除 `.cu`**（原上游无条件把 `.cu` 编入库，会在无 nvcc
  的机器上 configure 失败）。CUDA 入口都由 `#ifdef CUDA_ON` 守卫，OFF 走 CPU 分支。
  装了 CUDA Toolkit 的机器可恢复为原行为并开 `USE_CUDA=ON`。

## 五、本机（开发机 A）示例路径（仅作格式参考，非通用值）

```
OpenCV_DIR        = E:/ToolChain/opencv4/build      (OpenCV 4.10)
EIGEN3_ROOT       = E:/ToolChain/eigen-3.4.0        (Eigen 3.4.0, header-only)
PYTHON_EXECUTABLE = E:/python311/python.exe         (Python 3.11 + numpy + pybind11)
```

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
