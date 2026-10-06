# ============================================================================
# 模板匹配相关的可执行入口（demo / 测试），依赖 src/matching 内核。
# ============================================================================
if(MSVC)
    # 自研代码用 /W3（vendored 库设 /W2，避免全局 /W3 冲突产生 D9025）
    add_compile_options(/W3)
endif()

# 涂抹制模 demo（交互式生成 object_mask）
add_executable(paint_mask_demo paint_mask_demo.cpp)
target_link_libraries(paint_mask_demo PRIVATE ${PROJECT_NAME}_core)
if(OpenMP_CXX_FOUND)
    target_link_libraries(paint_mask_demo PRIVATE OpenMP::OpenMP_CXX)
endif()
sbm_copy_opencv_runtime(paint_mask_demo)

# 模板匹配测试 exe
add_executable(${PROJECT_NAME}_test test.cpp)
# 把测试数据目录的绝对路径写进程序，运行时不再依赖工作目录
target_compile_definitions(${PROJECT_NAME}_test PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/test/")
target_link_libraries(${PROJECT_NAME}_test PRIVATE ${PROJECT_NAME}_core)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME}_test PRIVATE OpenMP::OpenMP_CXX)
endif()
sbm_copy_opencv_runtime(${PROJECT_NAME}_test)

# fusion 流水线（SIMD 融合）对比测试 exe
add_executable(${PROJECT_NAME}_test_fusion test_fusion.cpp)
target_compile_definitions(${PROJECT_NAME}_test_fusion PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/test/")
target_link_libraries(${PROJECT_NAME}_test_fusion PRIVATE ${PROJECT_NAME}_core)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME}_test_fusion PRIVATE OpenMP::OpenMP_CXX)
endif()
sbm_copy_opencv_runtime(${PROJECT_NAME}_test_fusion)
