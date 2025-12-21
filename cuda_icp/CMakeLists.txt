# opencv
find_package(OpenCV 3 REQUIRED)
list(APPEND icp_inc ${OpenCV_INCLUDE_DIRS})
list(APPEND icp_lib ${OpenCV_LIBS})

if(USE_CUDA)
    # cuda
    find_package(CUDA REQUIRED)
    
    # MSVC 下使用简单的 CUDA 选项
    if(MSVC)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O3")
    else()
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O3 --default-stream per-thread")
    endif()
    
    list(APPEND icp_inc ${CUDA_INCLUDE_DIRS})
    list(APPEND icp_lib ${CUDA_LIBRARIES})
endif()

# eigen
set(EIGEN3_ROOT "C:/Users/sdf63/Downloads/eigen-3.4.0")
if(EXISTS "${EIGEN3_ROOT}/Eigen/Core")
    message(STATUS "Using Eigen from: ${EIGEN3_ROOT}")
    list(APPEND icp_inc ${EIGEN3_ROOT})
else()
    message(FATAL_ERROR "Eigen not found at: ${EIGEN3_ROOT}")
endif()

# src
SET(icp_cuda_srcs  icp.cu scene/common.cu scene/edge_scene/edge_scene.cu scene/kdtree_scene/kdtree_scene.cu)
SET(icp_srcs  icp.cpp scene/common.cpp scene/edge_scene/edge_scene.cpp scene/kdtree_scene/kdtree_scene.cpp)

# 启用 OpenMP - 使用现代 CMake 方式
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found for cuda_icp")
endif()

# 设置源代码特定属性
if(MSVC)
    # 为 MSVC 禁用某些警告
    add_compile_options(/wd4464 /wd4068 /wd4127 /wd4100 /wd4189)
endif()

# 创建库
add_library(cuda_icp
    ${icp_srcs}
    ${icp_cuda_srcs}
)

# 包含目录
target_include_directories(cuda_icp PUBLIC ${icp_inc})

# 链接库
target_link_libraries(cuda_icp PUBLIC ${icp_lib})

# 添加 OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(cuda_icp PUBLIC OpenMP::OpenMP_CXX)
endif()

# MSVC 特定设置
if(MSVC)
    # 设置目标属性
    set_target_properties(cuda_icp PROPERTIES
        COMPILE_FLAGS "/EHsc /utf-8"
    )
    
    # 发布模式下添加优化标志
    target_compile_options(cuda_icp 
        PRIVATE 
        "$<$<CONFIG:Release>:/O2 /openmp>"
        "$<$<CONFIG:Debug>:/Od>"
    )
endif()

# 设置输出目录
set_target_properties(cuda_icp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)