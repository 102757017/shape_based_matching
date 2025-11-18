# 1. CMake version
cmake_minimum_required(VERSION 3.15)

# 2. Project definition
project(shape_based_matching_py CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 3. 包含 vcpkg 工具链（如果存在）
if(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg/scripts/buildsystems/vcpkg.cmake")
endif()

# ----------------- Compile Options -----------------
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W3 /wd4996")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()

# ----------------- Dependencies -----------------
# OpenCV - 改进查找逻辑
find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
    # 如果标准查找失败，尝试手动设置路径
    if(DEFINED ENV{OpenCV_DIR})
        set(OpenCV_DIR $ENV{OpenCV_DIR})
        find_package(OpenCV 4 REQUIRED)
    else()
        # 尝试常见安装路径
        set(possible_paths
            "C:/tools/opencv/build"
            "C:/opencv/build"
            "C:/vcpkg/installed/x64-windows"
        )
        foreach(path IN LISTS possible_paths)
            if(EXISTS "${path}/OpenCVConfig.cmake")
                set(OpenCV_DIR "${path}")
                find_package(OpenCV 4 REQUIRED)
                break()
            endif()
        endforeach()
    endif()
endif()

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV 4 not found. Please install OpenCV and set OpenCV_DIR environment variable.")
endif()

message(STATUS "Found OpenCV: ${OpenCV_DIR}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

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

# 添加编译定义（如果需要）
target_compile_definitions(${PROJECT_NAME} PRIVATE 
    VERSION_INFO=${PROJECT_VERSION}
)
