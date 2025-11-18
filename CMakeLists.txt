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
# OpenCV - 增强查找逻辑
message(STATUS "Looking for OpenCV...")

# 首先检查环境变量
if(DEFINED ENV{OpenCV_DIR})
    set(OpenCV_DIR "$ENV{OpenCV_DIR}")
    message(STATUS "Using OpenCV_DIR from environment: ${OpenCV_DIR}")
endif()

# 尝试查找 OpenCV
find_package(OpenCV 4 QUIET)

if(NOT OpenCV_FOUND)
    # 如果标准查找失败，尝试常见路径
    message(STATUS "Standard OpenCV find failed, trying alternative paths...")
    
    set(possible_paths
        "C:/opencv/build"
        "C:/tools/opencv/build" 
        "C:/opencv"
        "C:/tools/opencv"
    )
    
    foreach(path IN LISTS possible_paths)
        if(EXISTS "${path}")
            message(STATUS "Trying path: ${path}")
            set(OpenCV_DIR "${path}")
            find_package(OpenCV 4 QUIET)
            if(OpenCV_FOUND)
                message(STATUS "Found OpenCV at: ${path}")
                break()
            endif()
        endif()
    endforeach()
endif()

if(OpenCV_FOUND)
    message(STATUS "OpenCV found:")
    message(STATUS "  Version: ${OpenCV_VERSION}")
    message(STATUS "  Libraries: ${OpenCV_LIBS}")
    message(STATUS "  Include dirs: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV 4 not found. Please install OpenCV and ensure OpenCVConfig.cmake is available.")
endif()

# Python and pybind11
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

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
