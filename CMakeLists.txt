cmake_minimum_required(VERSION 3.15)
project(shape_based_matching)

OPTION(USE_PYBIND "Enable pybind" ON)

# pybind11 3.x 需要 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 根据编译器设置选项
if(MSVC)
    # MSVC 编译器
    message(STATUS "Using MSVC compiler")

    # 基本优化选项
    set(OPTIMIZATION_FLAGS_RELEASE "/O2")
    set(PLATFORM_COMPILE_FLAGS "")

    # MSVC 的 OpenMP 选项
    set(OPENMP_FLAGS "/openmp")

    # 警告和调试选项
    # /wd4464: 相对包含路径含 ".."
    # /wd4309: 第三方 MIPP(mipp_impl_SSE.hxx) 里 SIMD 常量的截断, 改第三方库不划算
    # /wd4849: MSVC 只支持 OpenMP 2.0, edge_scene.cpp 的 collapse(2) 会被忽略(不影响正确性)
    set(WARNING_FLAGS "/W3 /wd4464 /wd4309 /wd4849")
    set(DEBUG_FLAGS "/Zi /Od /RTC1")

    # 字符集
    set(CHAR_SET_FLAGS "/utf-8 /EHsc")

    # 设置通用编译标志
    set(COMMON_COMPILE_FLAGS "${CHAR_SET_FLAGS} ${WARNING_FLAGS}")

    # 设置 CMake 变量
    set(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS}")

    # 设置调试和发布选项
    # 注意：不用 /MDd 和 /RTC1 —— 本机 CPython 无调试库，pybind 扩展必须以非 _DEBUG 编译
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od")
    set(CMAKE_CXX_FLAGS_RELEASE "${OPTIMIZATION_FLAGS_RELEASE} ${OPENMP_FLAGS}")

    # 链接器选项
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG")

else()
    # GCC/Clang 编译器
    message(STATUS "Using GCC/Clang compiler")

    # OpenMP 选项
    set(OPENMP_FLAGS "-fopenmp")

    # 优化选项
    set(OPTIMIZATION_FLAGS_RELEASE "-O3")

    # SIMD 选项
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm|aarch64")
        set(PLATFORM_COMPILE_FLAGS "-mfpu=neon")
    else()
        set(PLATFORM_COMPILE_FLAGS "-march=native")
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

# include MIPP headers
include_directories(${INCLUDE_DIRECTORIES} "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/")

# icp for refine
option(USE_CUDA "use cuda or not" OFF)

if(USE_CUDA)
    if(NOT WIN32)
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-10.0")
    endif()
    add_definitions(-DCUDA_ON)
endif()

# Debug 配置统一使用 release CRT(/MD)：本机 CPython 无调试库，pybind 扩展必须非 _DEBUG 编译
# 必须放在 add_subdirectory(cuda_icp) 之前，子目录才能继承
if(MSVC)
    add_compile_options("$<$<CONFIG:Debug>:/MD>")
endif()

add_subdirectory(cuda_icp)

# ============ 纯 C++ 内核 ============
# 训练/加载/匹配的完整实现。不依赖 Python / NumPy, Python(通过 pybind 薄壳) 与
# C#(通过 c_api 的 C ABI) 共用这一份代码。
add_library(${PROJECT_NAME}_core STATIC
    line2Dup.cpp
    core/matcher_core.cpp
)
target_include_directories(${PROJECT_NAME}_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/core")
target_link_libraries(${PROJECT_NAME}_core PUBLIC ${OpenCV_LIBS} cuda_icp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME}_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# OpenCV 运行时 DLL 所在目录 (供 pybind 模块与 C ABI 模块拷贝依赖)
set(OpenCV_RUNTIME_BIN "${OpenCV_DIR}/x64/vc16/bin")

# 把 OpenCV 运行时拷到目标旁边, 保证加载时能找到
function(sbm_copy_opencv_runtime target_name)
    if(EXISTS "${OpenCV_RUNTIME_BIN}")
        file(GLOB _opencv_dlls "${OpenCV_RUNTIME_BIN}/*.dll")
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_opencv_dlls} $<TARGET_FILE_DIR:${target_name}>
            COMMAND_EXPAND_LISTS
        )
    endif()
endfunction()

# ============ C ABI (供 C# P/Invoke, 也可被其它语言直接链接) ============
option(USE_C_ABI "Build shape_based_matching_c.dll for C# / other languages" ON)
if (USE_C_ABI)
    add_library(${PROJECT_NAME}_c SHARED c_api/matcher_c.cpp)
    target_include_directories(${PROJECT_NAME}_c PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/core"
        "${CMAKE_CURRENT_SOURCE_DIR}/c_api"
    )
    target_link_libraries(${PROJECT_NAME}_c PRIVATE ${PROJECT_NAME}_core)
    if(MSVC)
        # 与 pybind 模块一致: Debug 也用 release CRT(/MD)
        target_compile_options(${PROJECT_NAME}_c PRIVATE "$<$<CONFIG:Debug>:/MD>")
    endif()
    sbm_copy_opencv_runtime(${PROJECT_NAME}_c)
endif()

# 涂抹制模 demo(用 mask_painter.h 交互式生成 object_mask)
add_executable(paint_mask_demo paint_mask_demo.cpp line2Dup.cpp)
target_link_libraries(paint_mask_demo ${OpenCV_LIBS} cuda_icp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(paint_mask_demo OpenMP::OpenMP_CXX)
endif()

# test exe
add_executable(${PROJECT_NAME}_test line2Dup.cpp test.cpp)

# 把测试数据目录的绝对路径写进程序，运行时不再依赖工作目录
target_compile_definitions(${PROJECT_NAME}_test PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/")

# 链接库
target_link_libraries(${PROJECT_NAME}_test ${OpenCV_LIBS} cuda_icp)

# 添加 OpenMP 链接
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME}_test OpenMP::OpenMP_CXX)
endif()

# ============ pybind ============
if (USE_PYBIND)
    # ---------- 1. 定位 Python 解释器 ----------
    # 优先使用 CONDA_PREFIX；否则默认使用本机的 Python 3.11
    if(DEFINED ENV{CONDA_PREFIX})
        message(STATUS "Detected active Conda environment: $ENV{CONDA_PREFIX}")
        set(PYTHON_EXECUTABLE "$ENV{CONDA_PREFIX}/python.exe" CACHE FILEPATH "Python executable")
    else()
        set(PYTHON_EXECUTABLE "D:/portable program/python311/python.exe" CACHE FILEPATH "Python executable")
    endif()

    # ---------- 2. 查找 Python（Development.Module 足够编译扩展） ----------
    set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE FILEPATH "Python3 executable" FORCE)
    # 本机安装的是 release 版 CPython，没有 pythonXXX_d.lib。
    # 强制 pybind11 在 Debug 配置下也链接 release Python 库（否则 LNK1104: python311_d.lib）。
    set(PYTHON_IS_DEBUG FALSE CACHE BOOL "" FORCE)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    message(STATUS "Using Python: ${Python3_EXECUTABLE} (version ${Python3_VERSION})")

    # ---------- 3. 通过 pip 安装的 pybind11 定位 CMake 配置 ----------
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_PROBE_RESULT
    )
    if(NOT PYBIND11_PROBE_RESULT EQUAL 0)
        message(FATAL_ERROR "pybind11 not installed for ${Python3_EXECUTABLE}. Run: \"${Python3_EXECUTABLE}\" -m pip install pybind11")
    endif()
    find_package(pybind11 CONFIG REQUIRED)
    message(STATUS "pybind11 found: ${pybind11_DIR}")

    # Debug 编译时 CPython pyconfig.h 会通过 #pragma 自动链接 pythonXXX_d.lib，
    # 但普通 CPython 不带调试库。这里复制 release 库改名作为替代（标准 workaround）。
    if(WIN32)
        set(PY_DEBUG_LIB_DIR "${CMAKE_BINARY_DIR}/py_debug_lib")
        set(PY_DEBUG_LIB_NAME "python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}_d.lib")
        # Development.Module 组件下 Python3_LIBRARY 可能为空，直接从解释器目录推导
        get_filename_component(_py_root "${Python3_EXECUTABLE}" DIRECTORY)
        set(PY_RELEASE_LIB "${Python3_LIBRARY}")
        if(NOT PY_RELEASE_LIB OR NOT EXISTS "${PY_RELEASE_LIB}")
            set(PY_RELEASE_LIB "${_py_root}/libs/python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.lib")
        endif()
        file(MAKE_DIRECTORY "${PY_DEBUG_LIB_DIR}")
        if(EXISTS "${PY_RELEASE_LIB}" AND NOT EXISTS "${PY_DEBUG_LIB_DIR}/${PY_DEBUG_LIB_NAME}")
            file(COPY_FILE "${PY_RELEASE_LIB}" "${PY_DEBUG_LIB_DIR}/${PY_DEBUG_LIB_NAME}")
            message(STATUS "Created fake debug python lib: ${PY_DEBUG_LIB_DIR}/${PY_DEBUG_LIB_NAME}")
        elseif(NOT EXISTS "${PY_RELEASE_LIB}")
            message(WARNING "Python release lib not found at ${PY_RELEASE_LIB}; Debug link may fail")
        endif()
    endif()

    # ---------- 4. 查找 NumPy include 目录 ----------
    set(PYTHON_NUMPY_INCLUDE_DIR "")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_PATH_TEMP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE NUMPY_RESULT
    )
    if(NUMPY_RESULT EQUAL 0 AND NUMPY_PATH_TEMP)
        string(STRIP "${NUMPY_PATH_TEMP}" NUMPY_PATH)
        if(EXISTS "${NUMPY_PATH}/numpy/arrayobject.h")
            set(PYTHON_NUMPY_INCLUDE_DIR "${NUMPY_PATH}")
            message(STATUS "Found NumPy include dir: ${NUMPY_PATH}")
        else()
            message(STATUS "NumPy include path invalid: [${NUMPY_PATH}/numpy/arrayobject.h]")
        endif()
    else()
        message(STATUS "NumPy probe failed: result=${NUMPY_RESULT} output=[${NUMPY_PATH_TEMP}]")
    endif()
    if(NOT PYTHON_NUMPY_INCLUDE_DIR)
        message(FATAL_ERROR "NumPy not found for ${Python3_EXECUTABLE}. Run: \"${Python3_EXECUTABLE}\" -m pip install numpy")
    endif()

    # ---------- 5. 编译 pybind 模块 (薄壳, 逻辑在 _core 里) ----------
    set(SOURCE_PYBIND
        pybind11/pybind11.cpp
        pybind11/py_matcher.cpp
        pybind11/np2mat/ndarray_converter.cpp
    )
    pybind11_add_module(${PROJECT_NAME}_py ${SOURCE_PYBIND})
    if(MSVC)
        # 追加在 pybind11_add_module 自动的 /MDd 之后，保证 /MD 生效
        target_compile_options(${PROJECT_NAME}_py PRIVATE "$<$<CONFIG:Debug>:/MD>")
    endif()
    if(WIN32)
        target_link_directories(${PROJECT_NAME}_py PRIVATE "${PY_DEBUG_LIB_DIR}")
    endif()
    target_link_libraries(${PROJECT_NAME}_py PRIVATE
        ${PROJECT_NAME}_core        # 纯 C++ 内核 (静态链接, 不含 Python 依赖)
        ${OpenCV_LIBS}
        cuda_icp
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${PROJECT_NAME}_py PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_include_directories(${PROJECT_NAME}_py PRIVATE
        ${PYTHON_NUMPY_INCLUDE_DIR}
        "${CMAKE_CURRENT_SOURCE_DIR}/core"
    )

    # 构建后把 OpenCV DLL 复制到模块旁边，保证 import 时能加载
    sbm_copy_opencv_runtime(${PROJECT_NAME}_py)
endif()
