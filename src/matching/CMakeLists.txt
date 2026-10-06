# ============================================================================
# 模板匹配内核（纯 C++，无 Python / NumPy 依赖）
# 业务逻辑只写在这一份代码里，Python(pybind 薄壳) 与 C#(c_api 的 C ABI)
# 共用本库。对外接口只用 POD 结构体 + cv::Mat。
# ============================================================================
if(MSVC)
    # 自研代码用 /W3（vendored 库在各自目录设 /W2，避免全局 /W3 与其冲突产生 D9025）
    add_compile_options(/W3)
endif()

add_library(${PROJECT_NAME}_core STATIC
    line2Dup.cpp
    matcher_core.cpp
    auto_threshold.cpp
)

# 暴露本目录，使 line2Dup.h / matcher_core.h / fusion.h / mask_painter.h
# 可被 test / bindings 等消费方按纯文件名引用。
# 同时暴露 MIPP（vendored SIMD 头文件库），matching 内核与 test 均依赖它。
target_include_directories(${PROJECT_NAME}_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/MIPP)

target_link_libraries(${PROJECT_NAME}_core PUBLIC cuda_icp ${OpenCV_LIBS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME}_core PUBLIC OpenMP::OpenMP_CXX)
endif()
