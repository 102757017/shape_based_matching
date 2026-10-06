# ============================================================================
# 椭圆检测相关的可执行入口（demo / 调试 / 基准 / 横向对比），
# 依赖 src/ellipse 接入的两路检测算法与适配层。
# ============================================================================
if(MSVC)
    # 自研代码用 /W3（vendored 库设 /W2，避免全局 /W3 冲突产生 D9025）
    add_compile_options(/W3)
endif()

# ---------- standard-ellipse-detection 三件套 ----------
if(USE_ELLIPSE_DETECTION)
    add_executable(ellipse_detect_demo ellipse_detect_demo.cpp)
    target_link_libraries(ellipse_detect_demo PRIVATE ellipse_detection ${OpenCV_LIBS})
    sbm_copy_opencv_runtime(ellipse_detect_demo)

    # 逐级输出中间结果（梯度 / 弧 / 候选），用于排查
    add_executable(ellipse_debug ellipse_debug.cpp)
    target_link_libraries(ellipse_debug PRIVATE ellipse_detection ${OpenCV_LIBS})
    target_compile_definitions(ellipse_debug PRIVATE ELLIPSE_DEBUG_PROBE)
    sbm_copy_opencv_runtime(ellipse_debug)

    # 性能基准（可选分阶段计时）
    add_executable(ellipse_bench ellipse_bench.cpp)
    target_link_libraries(ellipse_bench PRIVATE ellipse_detection ${OpenCV_LIBS})
    sbm_copy_opencv_runtime(ellipse_bench)
    if(ELLIPSE_BENCH_PROFILE)
        target_compile_definitions(ellipse_bench PRIVATE SBM_ELLIPSE_PROFILE)
    endif()
endif()

# ---------- AAMED demo ----------
if(USE_AAMED)
    add_executable(aamed_demo aamed_demo.cpp)
    target_link_libraries(aamed_demo PRIVATE ${PROJECT_NAME}_aamed_adapter ${OpenCV_LIBS})
    sbm_copy_opencv_runtime(aamed_demo)
endif()

# ---------- 两路横向对比基准（需要两路都编译） ----------
if(ELLIPSE_COMPARE AND USE_ELLIPSE_DETECTION AND USE_AAMED)
    add_executable(ellipse_compare ${CMAKE_SOURCE_DIR}/test/ellipse_compare.cpp)
    target_link_libraries(ellipse_compare PRIVATE ellipse_detection ${PROJECT_NAME}_aamed_adapter aamed ${OpenCV_LIBS})
    target_compile_features(ellipse_compare PRIVATE cxx_std_17)
    sbm_copy_opencv_runtime(ellipse_compare)
endif()

# ---------- AAMED 输出口径自检（默认关闭） ----------
if(USE_AAMED_CHECK)
    add_executable(aamed_fitcheck ${CMAKE_SOURCE_DIR}/test/aamed_fitcheck.cpp)
    target_link_libraries(aamed_fitcheck PRIVATE ${PROJECT_NAME}_aamed_adapter aamed ${OpenCV_LIBS})
    sbm_copy_opencv_runtime(aamed_fitcheck)
endif()
