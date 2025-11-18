# 1. Higher CMake version and modern Python finding
cmake_minimum_required(VERSION 3.15)

# 2. Project definition
project(shape_based_matching_py CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------- Compile Options -----------------
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    set(PLATFORM_COMPILE_FLAGS "-mfpu=neon")
else()
    set(PLATFORM_COMPILE_FLAGS "-march=native")
endif()

set(COMMON_COMPILE_FLAGS "-fopenmp -Wall -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS} $ENV{CXXFLAGS}")
# ----------------------------------------------------

# ----------------- Dependencies -----------------
# OpenCV
find_package(OpenCV 4 REQUIRED)

# Python and pybind11
# Modern way to find Python, which silences the warning from pybind11
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
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
    Python::Python  # Link to the modern Python target
)

# Include directories for the target
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # The fix for "line2Dup.h not found"
    ${OpenCV_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/"
)
# ---------------------------------------------------------
