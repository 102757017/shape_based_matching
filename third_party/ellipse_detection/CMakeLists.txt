# standard-ellipse-detection (第三方库, MIT License)
# 来源: https://github.com/memory-overflow/standard-ellipse-detection
#
# 集成说明(相对上游的改动):
# 1. LAPACK dggev_ 广义特征值分解改用 Eigen 头文件库实现(GeneralizedEigenSolver,
#    语义一致) -> Windows/MSVC 无需编译 LAPACK/gfortran
# 2. cvcannyapi.cpp 从 OpenCV 老 C API(CvMat/cvGetMat/cvSobel) 重写为 OpenCV 4 C++ API
#    算法逻辑保持逐行一致
# 3. CMake 现代化: 使用本工程的 OpenCV target(本文件被主 CMakeLists add_subdirectory)

set(ELLIPSE_DETECTION_SOURCES
    src/compute.cpp
    src/cvcannyapi.cpp
    src/detect.cpp
    src/lsd.cpp
    src/unitily.cpp
)

add_library(ellipse_detection STATIC ${ELLIPSE_DETECTION_SOURCES})

target_include_directories(ellipse_detection PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# OpenCV 4 (由主工程 find_package 提供)
target_link_libraries(ellipse_detection PUBLIC ${OpenCV_LIBS})

# Eigen (替代 LAPACK 的广义特征值分解), 与 cuda_icp 共用同一份
set(EIGEN3_ROOT "E:/programing/eigen-3.4.0" CACHE PATH "Eigen root directory")
if(EXISTS "${EIGEN3_ROOT}/Eigen/Core")
    target_include_directories(ellipse_detection PUBLIC "${EIGEN3_ROOT}")
else()
    message(FATAL_ERROR "Eigen not found at: ${EIGEN3_ROOT} (needed by ellipse_detection)")
endif()

target_compile_features(ellipse_detection PUBLIC cxx_std_17)

if(MSVC)
    # 第三方代码警告较多, 降噪到 W2, 不影响主工程
    target_compile_options(ellipse_detection PRIVATE /W2 /wd4244 /wd4267 /wd4018 /wd4305)
    # 原代码为 C++11 风格, 避免 MSVC 对隐式转换的过度告警干扰主构建日志
endif()
