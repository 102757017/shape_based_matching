# ============================================================================
# C ABI：把 shape_based_matching_core + 两路椭圆检测包成 extern "C" 平铺接口，
# 供 C# P/Invoke 或任意能链接 .dll 的语言使用。
# ============================================================================
if(MSVC)
    # 自研代码用 /W3（vendored 库设 /W2，避免全局 /W3 冲突产生 D9025）
    add_compile_options(/W3)
endif()

if(USE_C_ABI)
    add_library(${PROJECT_NAME}_c SHARED matcher_c.cpp)
    target_include_directories(${PROJECT_NAME}_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    target_link_libraries(${PROJECT_NAME}_c PRIVATE
        ${PROJECT_NAME}_core
        ellipse_detection
    )
    if(USE_AAMED)
        target_link_libraries(${PROJECT_NAME}_c PRIVATE ${PROJECT_NAME}_aamed_adapter)
    endif()

    sbm_copy_opencv_runtime(${PROJECT_NAME}_c)
endif()
