# AAMED —— Arc Adjacency Matrix based Fast Ellipse Detection (第三方库, BSD-2-Clause)
# 来源: https://github.com/Probably-Zero/AAMED
#
# 集成说明(相对上游的改动):
# 1. 原工程是 C++11 + OpenCV 3 的一堆零散 .cpp; 这里按本工程惯例收进 include/ + src/,
#    并去掉上游 main.cpp (demo 改由工程根的 aamed_demo.cpp 承担)。
# 2. 源码原本是 GBK 编码的中文注释, 已整体转 UTF-8, 与本工程 /utf-8 一致 (纯注释, 无语义改动)。
# 3. OpenCV 4 兼容性: 实测 cv::flann::Index / LinearIndexParams / radiusSearch 在 4.x 仍在,
#    CV_PI / CV_8UC1 / uchar 等也都在, 无需改动算法代码。
# 4. CMake 现代化: 使用本工程的 OpenCV target (本文件被主 CMakeLists add_subdirectory)。

set(AAMED_SOURCES
    src/Contours.cpp
    src/EllipseNonMaximumSuppression.cpp
    src/FLED.cpp
    src/FLED_Initialization.cpp
    src/FLED_PrivateFunctions.cpp
    src/FLED_drawAndWriteFunctions.cpp
    src/Group.cpp
    src/LinkMatrix.cpp
    src/Node_FC.cpp
    src/Segmentation.cpp
    src/Validation.cpp
    src/adaptApproxPolyDP.cpp
    src/adaptApproximateContours.cpp
)

add_library(aamed STATIC ${AAMED_SOURCES})

target_include_directories(aamed PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# OpenCV 4 (由主工程 find_package 提供)
target_link_libraries(aamed PUBLIC ${OpenCV_LIBS})

target_compile_features(aamed PUBLIC cxx_std_17)

# 第三方代码警告较多, 降噪到 W2, 不影响主工程
if(MSVC)
    target_compile_options(aamed PRIVATE /W2 /wd4244 /wd4267 /wd4018 /wd4305)
else()
    target_compile_options(aamed PRIVATE -Wno-sign-compare -Wno-deprecated-declarations)
endif()
