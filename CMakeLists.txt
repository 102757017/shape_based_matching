# 1. CMake version
cmake_minimum_required(VERSION 3.15)

# 2. Project definition
project(shape_based_matching_py CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------- Compile Options -----------------
if(MSVC)
    # Release 模式下的优化选项
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /Oi /Ot /Oy /GL /MP")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /openmp /arch:AVX2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /fp:fast /DNDEBUG")
    
    # 通用警告设置
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W3 /wd4996")
    
    # 链接时优化
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /LTCG")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -fopenmp -ffast-math -DNDEBUG")
endif()

# 设置默认为 Release 构建
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

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

# 查找 OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(WARNING "OpenMP not found - parallelization will be disabled")
endif()

# ----------------- Python Module Target -----------------
pybind11_add_module(shape_based_matching_py
    line2Dup.cpp
    pybind11/pybind11.cpp
    pybind11/np2mat/ndarray_converter.cpp
)

# Link libraries
target_link_libraries(shape_based_matching_py PRIVATE 
    ${OpenCV_LIBS}
)

# 链接 OpenMP（如果找到）
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

# 为 Release 模式添加额外的优化定义
target_compile_definitions(shape_based_matching_py PRIVATE 
    $<$<CONFIG:Release>:NDEBUG>
)

# 在 Windows 上设置正确的输出名称
if(WIN32)
    set_target_properties(shape_based_matching_py PROPERTIES
        SUFFIX ".pyd"
        PREFIX ""
    )
endif()