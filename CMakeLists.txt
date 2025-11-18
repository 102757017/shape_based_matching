# 1. CMake version
cmake_minimum_required(VERSION 3.15)

# 2. Project definition
project(shape_based_matching_py CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------- Compile Options -----------------
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W3 /wd4996")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()

# ----------------- Dependencies -----------------
# OpenCV - 使用环境变量中的路径
if(DEFINED ENV{OpenCV_DIR})
    set(OpenCV_DIR "$ENV{OpenCV_DIR}")
    message(STATUS "Using OpenCV_DIR: ${OpenCV_DIR}")
else()
    set(OpenCV_DIR "C:/opencv/opencv/build")
    message(STATUS "Using default OpenCV_DIR: ${OpenCV_DIR}")
endif()

find_package(OpenCV 4 REQUIRED)
message(STATUS "OpenCV found:")
message(STATUS "  Version: ${OpenCV_VERSION}")
message(STATUS "  Libraries: ${OpenCV_LIBS}")
message(STATUS "  Include dirs: ${OpenCV_INCLUDE_DIRS}")

# Python and pybind11 - 自动检测当前 Python 版本
# 不指定版本，让 CMake 自动查找当前 Python 解释器
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

message(STATUS "Python found:")
message(STATUS "  Interpreter: ${Python_EXECUTABLE}")
message(STATUS "  Version: ${Python_VERSION}")
message(STATUS "  Include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "  Library: ${Python_LIBRARIES}")

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
    ${NUMPY_INCLUDE_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/"
)

# 添加编译定义
target_compile_definitions(${PROJECT_NAME} PRIVATE 
    VERSION_INFO=${PROJECT_VERSION}
)
