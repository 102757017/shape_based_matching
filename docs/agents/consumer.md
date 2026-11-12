# C# 消费端（ABI 契约）

本仓库产出的 `shape_based_matching_c.dll`（C API）被一个**独立的 C# 工程**通过
P/Invoke 调用，做形状匹配的上层业务。这份文档定义两侧必须遵守的契约，以及如何在本机
配置消费端路径、如何做契约校验。

## 一、契约内容（C 侧 ↔ C# 侧必须同步）

| 维度 | C 侧真相源 | C# 侧 | 不同步的后果 |
|------|-----------|-------|-------------|
| 导出函数名 | `bindings/c_api/matcher_c.h` 里的 `SBM_API sbm_xxx` | `MatcherNative.cs` 的 `[DllImport("shape_based_matching_c")]` | `EntryPointNotFoundException`（运行时） |
| 跨边界结构体字段 | 同上头文件里的 `SbXxx` 结构体 | C# 里的 `[StructLayout]` 对应 struct | 内存错位、读取到垃圾值或崩溃 |
| 调用约定 | 头文件里的 `__cdecl` / `__stdcall` | DllImport 的 `CallingConvention` | 栈不平衡、随机崩溃 |
| 位数 | 编译产物 x64 | C# 工程 `AnyCPU` + `Prefer32Bit=false`（64 位运行） | x86 DLL 会 `BadImageFormatException` |

> 经验：本次会话最初被怀疑"优化改了接口"，但 git 比对 + 导出表 dump 证明 C API
> 未被改动；真正的事故是 **OpenCV 运行时版本不匹配**（见下）。所以一旦 C# 侧报
> "找不到模块 / 找不到入口点"，先排查下面两件事，而不是假设接口被改。

## 二、OpenCV 运行时版本必须一致（最容易踩的坑）

- `shape_based_matching_c.dll` 在链接期就**写死**了它依赖的 `opencv_world4XXX.dll`
  版本（例如 4.13.0 → `opencv_world4130.dll`）。
- C# 工程部署目录下必须放**同一个版本**的 `opencv_world4XXX.dll`，否则加载即报
  "找不到指定的模块" / `DllNotFoundException`。
- **唯一的版本真相源**是本机 `scripts/configure.local.bat` 里的 `OpenCV_DIR`：它
  指向哪个 OpenCV `build` 目录，编出来的 DLL 就绑定哪个 world 版本。换版本 = 改
  `OpenCV_DIR` 后清 `build/` 重配重编。

## 三、消费端路径是"每台机器不同"的，不要写死

C# 工程是另一个仓库（peer repo），在不同电脑上的盘符 / 目录都不一样。**本仓库的任何
文件都不得硬编码消费端绝对路径。** 各机器把路径写进自己的 `scripts/configure.local.bat`
（git 忽略，不会污染仓库）：

```bat
REM 消费端部署目录：放有 shape_based_matching_c.dll 与 opencv_world4XXX.dll
set "SBM_CONSUMER_LIBS=E:/C_Sharp/robot_vision/libs"
REM 消费端的 P/Invoke 封装源文件（用于导出名 vs DllImport 差集校验）
set "SBM_CONSUMER_CS=E:/C_Sharp/robot_vision/Robot_Vision/VisModel/Models/ShapeBasedMatching/MatcherNative.cs"
```

上面只是**本机示例**，换机器改成你自己的路径即可。两个变量都是可选的：没配时
`check_abi.bat` 只做"位数 + 报出链接的 OpenCV 版本"，跳过与消费端的逐项比对。

## 四、契约校验（确定性护栏）

`scripts/check_abi.bat` 会：

1. 断言 DLL 是 **x64**；
2. 解析 DLL 的 PE 导入表，报出它链接的 `opencv_world4XXX.dll`，并检查该文件是否
   存在于 `SBM_CONSUMER_LIBS`（不一致直接 FAIL）；
3. （配了 `SBM_CONSUMER_CS` 时）把 DLL 导出名与 C# 的 `DllImport` 方法名做一次差集，
   C# 声明了但 DLL 没导出的，列为 `EntryPointNotFoundException` 风险。

运行方式（它会先 `call configure.local.bat` 自动拿到本机消费端路径）：

```bat
scripts\check_abi.bat
```

也可把它接进 CI：它已经挂到 `scripts/smoke.bat` 末尾（消费端未配置时自动跳过），
所以直接跑 `scripts\smoke.bat` 即可同时覆盖"demo 自洽测试 + ABI 契约"。

## 五、排查清单（C# 报 DLL 相关错误时按此顺序）

1. **位数不匹配** → 看 C# 工程平台目标与 DLL 是否同为 x64。
2. **找不到模块 / DllNotFoundException** → 几乎都是 OpenCV world 版本不一致，
   用 `check_abi.bat` 或 `dump_imports` 比对 DLL 链接版本 vs `libs/` 里实际文件。
3. **找不到入口点 / EntryPointNotFoundException** → 用 `check_abi.bat` 的
   导出 vs DllImport 差集，确认 C 头是否改名 / 删函数。
4. **加载成功但运行期报错码（code=-N）** → 与接口/依赖无关，是 DLL 内部逻辑问题，
   看 C 侧返回的错误信息定位具体函数。
