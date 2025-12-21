
# opencv
find_package(OpenCV 3 REQUIRED)
list(APPEND icp_inc ${OpenCV_INCLUDE_DIRS})
list(APPEND icp_lib ${OpenCV_LIBS})


if(USE_CUDA)
# cuda
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O3 --default-stream per-thread -Xcompiler -fopenmp")
list(APPEND icp_inc ${CUDA_INCLUDE_DIRS})
list(APPEND icp_lib ${CUDA_LIBRARIES})
endif()


# eigen
# eigen - 直接设置路径，不需要 find_package
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


if(USE_CUDA)
CUDA_COMPILE(icp_cuda_objs ${icp_cuda_srcs})
endif()

# lib & test exe
add_library(cuda_icp
               ${icp_srcs}
               ${icp_cuda_srcs}
               ${icp_cuda_objs} 
)
target_include_directories(cuda_icp PUBLIC ${icp_inc})
target_link_libraries(cuda_icp PUBLIC ${icp_lib})
