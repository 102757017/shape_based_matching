# ============================================================================
# 椭圆检测模块：接入两路第三方算法（standard + AAMED）并暴露统一适配层。
# 各 demo / 对比工具由 apps/ellipse 构建（本文件只负责算法集成）。
# ============================================================================

# standard-ellipse-detection（MIT）：弧段扫描 + 精化拟合
if(USE_ELLIPSE_DETECTION)
    add_subdirectory(ellipse_detection)
endif()

# AAMED / FLED（BSD-2-Clause）：弧邻接矩阵 + 弧段分组投票
if(USE_AAMED)
    add_subdirectory(aamed)

    # AAMED 适配层（header-only）：把上游 FLED 输出归一化到 sbm_ellipse_t 口径
    # （cx=列 / cy=行 / a,b 半轴 / phi 相对列轴弧度），与 standard 路完全一致，
    # 这样两种检测器产出的结果可以混用、共用同一套后处理。
    # 链接 aamed 以传递其 PUBLIC include（FLED.h 所在目录）。
    add_library(${PROJECT_NAME}_aamed_adapter INTERFACE)
    target_include_directories(${PROJECT_NAME}_aamed_adapter INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${PROJECT_NAME}_aamed_adapter INTERFACE aamed)
endif()
