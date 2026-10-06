# ============================================================================
# shape_based_matching —— 根 CMake（编排器）
#
# 本文件只负责：全局工具链（SIMD / OpenCV / OpenMP / MIPP）、跨模块 option、
# 运行时 DLL 拷贝辅助函数，以及按依赖顺序把各模块挂进来。
# 每个算法/接口/可执行入口的具体编译规则都在各自的 CMakeLists.txt 里。
#
# 目录职责：
#   src/cuda_icp       ICP 精化库
#   src/matching       模板匹配内核（line2Dup + matcher_core + auto_threshold + fusion + mask_painter）
#   src/ellipse        两路椭圆检测（standard + AAMED）集成 + 适配层
#   bindings/c_api     C ABI（供 C# P/Invoke）
#   bindings/python    pybind11 绑定
#   apps/matching      模板匹配 demo / 测试
#   apps/ellipse       椭圆检测 demo / 基准 / 横向对比
#   vendored 上游库直接置于 src 各模块内：MIPP→src/matching/MIPP，
#   ellipse_detection→src/ellipse/ellipse_detection，aamed→src/ellipse/aamed
# ============================================================================
cmake_minimum_required(VERSION 3.15)
project(shape_based_matching)

OPTION(USE_PYBIND "Enable pybind" ON)

# SIMD 指令集级别: NONE(默认 SSE2 基线) / SSE2 / SSE4.2 / AVX / AVX2 / AVX512F
# 本机 i5-8250U 支持 AVX2, 默认选 AVX2 以获得 MIPP + 编译器向量化的最大收益。
set(SBM_SIMD_LEVEL "AVX2" CACHE STRING "SIMD ISA level (NONE/SSE2/SSE4.2/AVX/AVX2/AVX512F)")
set_property(CACHE SBM_SIMD_LEVEL PROPERTY STRINGS "NONE" "SSE2" "SSE4.2" "AVX" "AVX2" "AVX512F")

# pybind11 3.x 需要 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 根据编译器设置选项
if(MSVC)
    # MSVC 编译器
    message(STATUS "Using MSVC compiler")

    # 基本优化选项
    set(OPTIMIZATION_FLAGS_RELEASE "/O2 /Ob2 /GL")
    set(PLATFORM_COMPILE_FLAGS "")

    # MSVC 的 OpenMP 选项
    set(OPENMP_FLAGS "/openmp")

    # SIMD: MIPP 依据编译器预定义宏(__SSE2__/__AVX__/__AVX2__)选择向量化实现,
    # 这些宏由 /arch 开关引入, 所以这里给 /arch 就等于给 MIPP 选代。
    # NONE 退回 SSE2 基线; 目标 CPU 不支持选中的级别会在运行时触发 SIGILL,
    # 换机器时请用 -DSBM_SIMD_LEVEL=... 重配。
    set(SIMD_FLAGS "")
    string(TOUPPER "${SBM_SIMD_LEVEL}" _simd)
    if(_simd STREQUAL "SSE2")
        set(SIMD_FLAGS "/arch:SSE2")
    elseif(_simd STREQUAL "SSE4.2")
        set(SIMD_FLAGS "/arch:SSE4.2")
    elseif(_simd STREQUAL "AVX")
        set(SIMD_FLAGS "/arch:AVX")
    elseif(_simd STREQUAL "AVX2")
        set(SIMD_FLAGS "/arch:AVX2")
    elseif(_simd STREQUAL "AVX512F")
        set(SIMD_FLAGS "/arch:AVX512")
    else()
        message(WARNING "Unknown SBM_SIMD_LEVEL=[${SBM_SIMD_LEVEL}], fallback to SSE2 baseline")
        set(SIMD_FLAGS "/arch:SSE2")
    endif()
    message(STATUS "SIMD level: ${SBM_SIMD_LEVEL} -> ${SIMD_FLAGS}")

    # 警告和调试选项
    # 注意: /W3 不再放在全局 CMAKE_CXX_FLAGS 里, 而是由各模块 CMakeLists 用
    #   add_compile_options(/W3) 仅作用于自研目标 —— 这样 vendored 库(ellipse_detection/aamed)
    #   自己设的 /W2 不会和全局 /W3 冲突, 也就不再产生 D9025 "正在重写 /W3 (用 /W2)"。
    # /wd4464: 相对包含路径含 ".."
    # /wd4309: 第三方 MIPP(mipp_impl_SSE.hxx) 里 SIMD 常量的截断, 改第三方库不划算
    # /wd4849: MSVC 只支持 OpenMP 2.0, edge_scene.cpp 的 collapse(2) 会被忽略(不影响正确性)
    # /wd4244 /wd4267 /wd4305: double→int/float 截断、size_t→int 等精度转换, 主要出自 vendored 头
    #   (defines.h/types.hpp/FLED.h) 在被自研 TU 包含时透出; 这些与 vendored 库自身 /wd 策略一致。
    set(WARNING_FLAGS "/wd4464 /wd4309 /wd4849 /wd4244 /wd4267 /wd4305")
    set(DEBUG_FLAGS "/Zi /Od /RTC1")

    # 字符集
    set(CHAR_SET_FLAGS "/utf-8 /EHsc")

    # 设置通用编译标志
    set(COMMON_COMPILE_FLAGS "${CHAR_SET_FLAGS} ${WARNING_FLAGS} ${SIMD_FLAGS}")

    # 设置 CMake 变量
    set(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS}")

    # 设置调试和发布选项
    # 注意：不用 /MDd 和 /RTC1 —— 本机 CPython 无调试库，pybind 扩展必须以非 _DEBUG 编译
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od")
    set(CMAKE_CXX_FLAGS_RELEASE "${OPTIMIZATION_FLAGS_RELEASE} ${OPENMP_FLAGS}")

    # 链接器选项: /GL(编译期全程序优化, 见 OPTIMIZATION_FLAGS_RELEASE) 必须由 /LTCG 落地,
    # 且必须作用于静态库/共享库/pybind 模块而不只是 EXE, 否则优化会丢失。
    # /INCREMENTAL:NO 消除 "LNK4075 忽略 /INCREMENTAL(由于 /LTCG)" —— Release 本就不该增量链接
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /INCREMENTAL:NO")
    # 静态库用 lib.exe 链接，不认 /INCREMENTAL，只保留 /LTCG（配合 /GL 做全程序优化）
    set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "/LTCG")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "/LTCG /INCREMENTAL:NO")
    set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "/LTCG /INCREMENTAL:NO")

else()
    # GCC/Clang 编译器
    message(STATUS "Using GCC/Clang compiler")

    # OpenMP 选项
    set(OPENMP_FLAGS "-fopenmp")

    # 优化选项
    set(OPTIMIZATION_FLAGS_RELEASE "-O3")

    # SIMD 选项: 与 MSVC 分支共用 SBM_SIMD_LEVEL, 保持两边行为一致
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm|aarch64")
        set(PLATFORM_COMPILE_FLAGS "-mfpu=neon")
    else()
        string(TOUPPER "${SBM_SIMD_LEVEL}" _simd)
        if(_simd STREQUAL "SSE2")
            set(PLATFORM_COMPILE_FLAGS "-msse2")
        elseif(_simd STREQUAL "SSE4.2")
            set(PLATFORM_COMPILE_FLAGS "-msse4.2")
        elseif(_simd STREQUAL "AVX")
            set(PLATFORM_COMPILE_FLAGS "-mavx")
        elseif(_simd STREQUAL "AVX2")
            set(PLATFORM_COMPILE_FLAGS "-mavx2")
        elseif(_simd STREQUAL "AVX512F")
            set(PLATFORM_COMPILE_FLAGS "-mavx512f")
        else()
            set(PLATFORM_COMPILE_FLAGS "-march=native")
        endif()
    endif()

    # 警告选项
    set(WARNING_FLAGS "-Wall -Wno-sign-compare")

    # 调试信息
    set(DEBUG_FLAGS "-g2 -ggdb")

    # 通用编译标志
    set(COMMON_COMPILE_FLAGS "${OPENMP_FLAGS} ${WARNING_FLAGS}")

    # 设置 CMake 变量
    set(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 ${DEBUG_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${OPTIMIZATION_FLAGS_RELEASE}")
endif()

# 默认构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# 如果希望显示匹配时间，取消注释下面的行
# add_definitions(-DDEBUG_MATCH_TIME)

# 添加 OpenMP 包查找
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
else()
    message(WARNING "OpenMP not found")
endif()

# ============ OpenCV 4 ============
# 指向 OpenCV 4.13.0 预编译包（如需更换版本/路径，修改下面一行即可）
set(OpenCV_DIR "E:/programing/opencv-4.13.0/opencv/build" CACHE PATH "OpenCV directory")
find_package(OpenCV 4 REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})

# Debug 配置下也使用 release 版 OpenCV 库（配合 /MD，避免 _ITERATOR_DEBUG_LEVEL 冲突）
if(MSVC)
    foreach(_cvlib ${OpenCV_LIBS})
        if(TARGET ${_cvlib})
            set_target_properties(${_cvlib} PROPERTIES
                MAP_IMPORTED_CONFIG_DEBUG RELEASE
                MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
        endif()
    endforeach()
endif()

# MIPP（vendored SIMD 头文件库）的 include 路径由 src/matching/CMakeLists.txt
# 通过 _core 目标的 PUBLIC 暴露，这里不再全局注入。

# icp for refine
option(USE_CUDA "use cuda or not" OFF)

if(USE_CUDA)
    if(NOT WIN32)
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-10.0")
    endif()
    add_definitions(-DCUDA_ON)
endif()

# 所有配置统一使用 release CRT(/MD)：
#  - 本机 CPython 无调试库，pybind 扩展必须非 _DEBUG 编译
#  - 链接的是 release 版 OpenCV 预编译库，用 /MD 避免 _ITERATOR_DEBUG_LEVEL / RuntimeLibrary 冲突
# 用 CMAKE_MSVC_RUNTIME_LIBRARY 让 CMake 自己在每个配置都输出 /MD（而不是靠 add_compile_options
# 去覆盖 Debug 下默认的 /MDd），这样不会再出现 D9025“正在重写 /MDd (用 /MD)”的警告。
# 必须放在 add_subdirectory 之前，子目录目标才能继承。
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

# ============ 跨模块 option ============
option(USE_ELLIPSE_DETECTION "Build ellipse_detection library and demo" ON)
option(USE_AAMED "Build AAMED ellipse detection library and demo" ON)
option(USE_C_ABI "Build shape_based_matching_c.dll for C# / other languages" ON)
option(ELLIPSE_COMPARE "Build ellipse_compare (standard vs AAMED)" ON)
option(ELLIPSE_BENCH_PROFILE "Enable per-phase timing in ellipse_bench" ON)
option(USE_AAMED_CHECK "Build AAMED output-convention self-check (off by default)" OFF)
option(SBM_COPY_OPENCV_DEBUG_DLL "同时拷贝 OpenCV 调试版运行时(opencv_world4130d.dll)" OFF)
set(SBM_OPENCV_RUNTIME_EXCLUDE "ffmpeg" CACHE STRING
    "拷贝 OpenCV 运行时时按文件名排除的关键字, 分号分隔")

# 随产物拷贝的 OpenCV 运行时。默认只拷 release 变体并排除视频编解码附加库。
set(OpenCV_RUNTIME_BIN "${OpenCV_DIR}/x64/vc16/bin")

# 唯一的 OpenCV 运行时缓存：所有目标通过“硬链接”共享同一份 DLL。
#  - 仍保证每个输出目录里都有 opencv_world4130.dll（C# P/Invoke / python import
#    按相邻目录加载，无需改 PATH）；
#  - 但磁盘只算一份：硬链接指向同一 inode，不额外占空间，且不需要管理员权限
#    （比符号链接更适合 Windows 构建机）。
set(SBM_OPENCV_RUNTIME_CACHE "${CMAKE_BINARY_DIR}/opencv_runtime")

function(sbm_copy_opencv_runtime target_name)
    if(NOT EXISTS "${OpenCV_RUNTIME_BIN}")
        return()
    endif()
    file(GLOB _all_dlls "${OpenCV_RUNTIME_BIN}/*.dll")
    set(_picked "")
    foreach(_dll ${_all_dlls})
        get_filename_component(_name "${_dll}" NAME_WE)
        string(TOLOWER "${_name}" _low)
        set(_skip FALSE)
        foreach(_kw ${SBM_OPENCV_RUNTIME_EXCLUDE})
            string(TOLOWER "${_kw}" _kw_low)
            if(_low MATCHES "${_kw_low}")
                set(_skip TRUE)
            endif()
        endforeach()
        if(NOT _skip AND NOT SBM_COPY_OPENCV_DEBUG_DLL AND _low MATCHES "d$")
            set(_skip TRUE)      # opencv_world4130d.dll 之类的调试变体
        endif()
        if(NOT _skip)
            list(APPEND _picked "${_dll}")
        endif()
    endforeach()
    if(NOT _picked)
        return()
    endif()

    file(MAKE_DIRECTORY "${SBM_OPENCV_RUNTIME_CACHE}")

    foreach(_dll ${_picked})
        get_filename_component(_fname "${_dll}" NAME)
        set(_master "${SBM_OPENCV_RUNTIME_CACHE}/${_fname}")
        set(_link   "$<TARGET_FILE_DIR:${target_name}>/${_fname}")
        # 1) 先把 DLL 拷到唯一缓存目录（copy_if_different，多目标并发也安全；
        #    OpenCV 升级后内容变化会自动更新，所有硬链接随之生效）
        # 2) 在目标输出目录里建“硬链接”指向同一份文件。用“唯一临时名 + 删除 + 改名”
        #    的写法保证并发安全、可重入：同一目录被多个目标共用时也不会因
        #    create_hardlink 目标已存在而失败（rename 会覆盖已存在的 _link）。
        set(_tmplink "$<TARGET_FILE_DIR:${target_name}>/${_fname}.${target_name}.tmp")
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" "${_master}"
            COMMAND ${CMAKE_COMMAND} -E create_hardlink "${_master}" "${_tmplink}"
            COMMAND ${CMAKE_COMMAND} -E remove -f "${_link}"
            COMMAND ${CMAKE_COMMAND} -E rename "${_tmplink}" "${_link}"
        )
    endforeach()
endfunction()

# ============ 模块编排（顺序即依赖顺序） ============
add_subdirectory(src/cuda_icp)        # 产出 cuda_icp
add_subdirectory(src/matching)        # 产出 shape_based_matching_core（依赖 cuda_icp）
add_subdirectory(src/ellipse)         # 接入两路椭圆检测 + aamed 适配层
add_subdirectory(bindings/c_api)      # C ABI（依赖 core + 椭圆检测）
add_subdirectory(bindings/python)     # pybind（依赖 core + 椭圆检测 + cuda_icp）
add_subdirectory(apps/matching)       # 模板匹配 demo / 测试
add_subdirectory(apps/ellipse)        # 椭圆检测 demo / 基准 / 横向对比
