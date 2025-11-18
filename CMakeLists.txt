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

# Python, pybind11, and NumPy
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(NumPy REQUIRED)  # <-- 让 CMake 自动寻找 NumPy

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
    ${NumPy_INCLUDE_DIRS}        # <-- 将 NumPy 的头文件路径添加进来
    "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/"
)
# ---------------------------------------------------------
