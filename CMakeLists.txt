# 1. CMake version
cmake_minimum_required(VERSION 3.15)

# 2. Project definition
project(shape_based_matching_py CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 强制使用Release模式
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ----------------- Compile Options -----------------
if(MSVC)
    # MSVC编译器优化选项
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W3")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /GL /arch:AVX2")
    
    # 针对Release模式的额外优化
    add_compile_options(
        "$<$<CONFIG:RELEASE>:/O2>"      # 最大优化（速度）
        "$<$<CONFIG:RELEASE>:/GL>"      # 全程序优化
        "$<$<CONFIG:RELEASE>:/arch:AVX2>"  # AVX2指令集
    )
    
    # 链接时优化
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /LTCG")
    
else()
    # GCC/Clang编译器优化选项
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    
    # SIMD指令集
    add_compile_options(
        "$<$<CONFIG:RELEASE>:-O3>"
        "$<$<CONFIG:RELEASE>:-mavx>"
        "$<$<CONFIG:RELEASE>:-mavx2>"
        "$<$<CONFIG:RELEASE>:-mfma>"
    )
endif()

# ----------------- OpenMP Support -----------------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    
    if(MSVC)
        # MSVC specific OpenMP flags
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif()
else()
    message(WARNING "OpenMP not found - performance may be degraded")
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
