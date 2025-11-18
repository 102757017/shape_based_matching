# 步骤 1: 提高 CMake 最低版本要求
cmake_minimum_required(VERSION 3.15)

# 步骤 2: 定义项目名称和 C++ 标准
project(shape_based_matching_py CXX) # 将项目名称与 setup.py 中的模块名统一
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------- 编译选项 (保持您的优化) -----------------
# Option for pybind is now implicitly ON since we are building a Python module
# SET(CMAKE_BUILD_TYPE "Release") # 这行应该由 setup.py 控制，建议移除

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    set(PLATFORM_COMPILE_FLAGS "-mfpu=neon")
else()
    set(PLATFORM_COMPILE_FLAGS "-march=native")
endif()

set(COMMON_COMPILE_FLAGS "-fopenmp -Wall -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS} $ENV{CXXFLAGS}")
# setup.py 会通过 CMAKE_BUILD_TYPE 来设置 Debug/Release 模式，无需在这里硬编码
# ----------------------------------------------------------------

# ----------------- 依赖项 -----------------
# 步骤 3: 找到 OpenCV
# 移除旧的 CMAKE_PREFIX_PATH 设置，这些路径应该由环境或 CMake 参数提供
find_package(OpenCV 4 REQUIRED)

# 步骤 4: 找到 Python 和 pybind11 (这是核心)
# find_package(PythonLibs 3 REQUIRED) # 通常 pybind11 会自动找到，可以先注释掉
find_package(pybind11 REQUIRED)

# MIPP 头文件
# 使用 target_include_directories 替代 include_directories
# 这是一种更现代、更精确的做法
# include_directories("${CMAKE_CURRENT_SOURCE_DIR}/MIPP/")
# -----------------------------------------------------------------

# ----------------- 定义 Python 模块 -----------------
# 步骤 5: 移除旧的、手动的源文件列表
# set(SOURCE_PYBIND ...) # <--- 移除

# 假设您的绑定代码在一个名为 pybind.cpp 的文件中，而核心逻辑在 line2Dup.cpp
# 如果您的所有代码都在一个文件里，就只写那个文件名

# 步骤 6: 使用现代的 pybind11_add_module
# 它会自动处理所有 Python 相关的包含目录和链接
pybind11_add_module(${PROJECT_NAME}  # 模块名
    # 源文件列表
    line2Dup.cpp 
    pybind11/pybind11.cpp # 您绑定代码的源文件
    pybind11/np2mat/ndarray_converter.cpp # 您的 ndarray_converter 源文件
)

# 步骤 7: 使用现代的方式链接和包含
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/" # 将 MIPP 头文件目录关联到目标
)
# -----------------------------------------------------------------

# ----------------- 测试用的可执行文件 (如果需要) -----------------
# 将它放在 if(BUILD_TESTING) 块中是一个好习惯
# add_executable(shape_based_matching_test test.cpp)
# target_link_libraries(shape_based_matching_test PRIVATE ${OpenCV_LIBS})
# target_include_directories(shape_based_matching_test PRIVATE 
#     ${OpenCV_INCLUDE_DIRS}
#     "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/"
# )
