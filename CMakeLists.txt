# 1. CMake version
cmake_minimum_required(VERSION 3.15)

# 2. Project definition
project(shape_based_matching_py CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------- Compile Options -----------------
# 强制使用Release模式
set(CMAKE_BUILD_TYPE Release)

if(MSVC)
    # MSVC编译器优化选项
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W3 /wd4996 /O2 /fp:fast")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /Oi /Ot /Oy /GL /arch:AVX2")
    
    # 开启OpenMP
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    endif()
    
    # AVX/AVX2指令集
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    
else()
    # GCC/Clang编译器优化选项
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3 -ffast-math -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    
    # 开启OpenMP
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif()
    
    # SIMD指令集
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2 -mfma -msse4.2")
endif()

# ----------------- Dependencies -----------------
# OpenCV
if(DEFINED ENV{OpenCV_DIR})
    set(OpenCV_DIR "$ENV{OpenCV_DIR}")
    message(STATUS "Using OpenCV_DIR: ${OpenCV_DIR}")
else()
    set(OpenCV_DIR "C:/opencv/opencv/build")
    message(STATUS "Using default OpenCV_DIR: ${OpenCV_DIR}")
endif()

find_package(OpenCV 4 REQUIRED)
message(STATUS "OpenCV found: ${OpenCV_VERSION}")

# Python and pybind11
find_package(pybind11 REQUIRED)

# ----------------- Python Module Target -----------------
# 确保模块名称正确
pybind11_add_module(shape_based_matching_py
    line2Dup.cpp
    pybind11/pybind11.cpp
    pybind11/np2mat/ndarray_converter.cpp
)

# Link libraries
target_link_libraries(shape_based_matching_py PRIVATE 
    ${OpenCV_LIBS}
)

# 链接OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(shape_based_matching_py PRIVATE OpenMP::OpenMP_CXX)
endif()

# Include directories for the target
target_include_directories(shape_based_matching_py PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/"
)

# 添加编译定义
target_compile_definitions(shape_based_matching_py PRIVATE 
    VERSION_INFO=${PROJECT_VERSION}
)

# 在 Windows 上设置正确的输出名称
if(WIN32)
    set_target_properties(shape_based_matching_py PROPERTIES
        SUFFIX ".pyd"
        PREFIX ""
    )
endif()

# 添加额外的优化选项
if(MSVC)
    target_compile_options(shape_based_matching_py PRIVATE
        $<$<CONFIG:RELEASE>:
        /O2      # 最大优化（速度）
        /Ob2     # 内联扩展
        /Oi      # 生成内部函数
        /Ot      # 偏好快速代码
        /Oy      # 省略帧指针
        /GL      # 全程序优化
        /arch:AVX2  # AVX2指令集
        >
    )
    
    target_link_options(shape_based_matching_py PRIVATE
        $<$<CONFIG:RELEASE>:
        /LTCG    # 链接时代码生成
        /OPT:REF # 消除未使用的函数和数据
        /OPT:ICF # 相同的COMDAT折叠
        >
    )
else()
    target_compile_options(shape_based_matching_py PRIVATE
        $<$<CONFIG:RELEASE>:
        -O3
        -mavx
        -mavx2
        -mfma
        -msse4.2
        -ffast-math
        -ftree-vectorize
        -fopt-info-vec
        >
    )
endif()
