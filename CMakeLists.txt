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
# OpenCV
if(DEFINED ENV{OpenCV_DIR})
    set(OpenCV_DIR "$ENV{OpenCV_DIR}")
    message(STATUS "Using OpenCV_DIR: ${OpenCV_DIR}")
else()
    set(OpenCV_DIR "C:/opencv/opencv/build")
    message(STATUS "Using default OpenCV_DIR: ${OpenCV_DIR}")
endif()

find_package(OpenCV 4 REQUIRED)
message(STATUS "OpenCV found: ${OpenCV_VERSION}")

# Python and pybind11
find_package(pybind11 REQUIRED)

# ----------------- Python Module Target -----------------
# 确保模块名称正确
pybind11_add_module(shape_based_matching_py
    line2Dup.cpp
    pybind11/pybind11.cpp
    pybind11/np2mat/ndarray_converter.cpp
)

# Link libraries
target_link_libraries(shape_based_matching_py PRIVATE 
    ${OpenCV_LIBS}
)

# Include directories for the target
target_include_directories(shape_based_matching_py PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/"
)

# 添加编译定义
target_compile_definitions(shape_based_matching_py PRIVATE 
    VERSION_INFO=${PROJECT_VERSION}
)

# 在 Windows 上设置正确的输出名称
if(WIN32)
    set_target_properties(shape_based_matching_py PROPERTIES
        SUFFIX ".pyd"
        PREFIX ""
    )
endif()
