# 1. CMake version
cmake_minimum_required(VERSION 3.15)

# 2. Project definition
project(shape_based_matching_py CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------- Compile Options (No change) -----------------
# ... (your compile flags here) ...

# ----------------- Dependencies -----------------
# OpenCV
find_package(OpenCV 4 REQUIRED)

# Python and pybind11
# find_package(pybind11) 会自动生效，因为 setup.py 传入了 -Dpybind11_DIR
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# 对于 NumPy，我们不再使用 find_package，而是直接使用传入的路径
# 这更加直接和可靠
# find_package(NumPy REQUIRED) <-- 移除或注释掉这一行

# -------------------------------------------------

# ----------------- Python Module Target -----------------
pybind11_add_module(${PROJECT_NAME}
    line2Dup.cpp
    pybind11/pybind11.cpp
    pybind11/np2mat/ndarray_converter.cpp
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${OpenCV_LIBS}
    Python::Python
)

# Include directories for the target
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}        # <-- 使用从 setup.py 传入的变量
    "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/"
)
# ---------------------------------------------------------
