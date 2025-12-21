cmake_minimum_required(VERSION 3.15)
project(shape_based_matching)

OPTION(USE_PYBIND "Enable pybind" ON)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 根据编译器设置选项
if(MSVC)
    # MSVC 编译器
    message(STATUS "Using MSVC compiler")
    
    # 基本优化选项
    set(OPTIMIZATION_FLAGS_RELEASE "/O2")
    set(PLATFORM_COMPILE_FLAGS "")
    
    # MSVC 的 OpenMP 选项 - 简化版本，避免冲突
    set(OPENMP_FLAGS "/openmp")
    
    # 警告和调试选项
    set(WARNING_FLAGS "/W3 /wd4464")
    set(DEBUG_FLAGS "/Zi /Od /RTC1")
    
    # 字符集
    set(CHAR_SET_FLAGS "/utf-8 /EHsc")
    
    # 设置通用编译标志
    set(COMMON_COMPILE_FLAGS "${CHAR_SET_FLAGS} ${WARNING_FLAGS}")
    
    # 设置 CMake 变量
    set(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS}")
    
    # 设置调试和发布选项
    set(CMAKE_CXX_FLAGS_DEBUG "${DEBUG_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${OPTIMIZATION_FLAGS_RELEASE} ${OPENMP_FLAGS}")
    
    # 链接器选项
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG")
    
else()
    # GCC/Clang 编译器
    message(STATUS "Using GCC/Clang compiler")
    
    # OpenMP 选项
    set(OPENMP_FLAGS "-fopenmp")
    
    # 优化选项
    set(OPTIMIZATION_FLAGS_RELEASE "-O3")
    
    # SIMD 选项
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm|aarch64")
        set(PLATFORM_COMPILE_FLAGS "-mfpu=neon")
    else()
        set(PLATFORM_COMPILE_FLAGS "-march=native")
    endif()
    
    # 警告选项
    set(WARNING_FLAGS "-Wall -Wno-sign-compare")
    
    # 调试信息
    set(DEBUG_FLAGS "-g2 -ggdb")
    
    # 通用编译标志
    set(COMMON_COMPILE_FLAGS "${OPENMP_FLAGS} ${WARNING_FLAGS}")
    
    # 设置 CMake 变量
    set(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 ${DEBUG_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${OPTIMIZATION_FLAGS_RELEASE}")
endif()

# 默认构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# 如果希望显示匹配时间，取消注释下面的行
# add_definitions(-DDEBUG_MATCH_TIME)

# 添加 OpenMP 包查找
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
else()
    message(WARNING "OpenMP not found")
endif()

# opencv
set(OpenCV_DIR "C:/Users/sdf63/Downloads/opencv3/build/x64/vc15/lib" CACHE PATH "OpenCV directory")
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# include MIPP headers
include_directories(${INCLUDE_DIRECTORIES} "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/")

# icp for refine
option(USE_CUDA "use cuda or not" OFF)

if(USE_CUDA)
    if(NOT WIN32)
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-10.0")
    endif()
    add_definitions(-DCUDA_ON)
endif()

add_subdirectory(cuda_icp)

# test exe
add_executable(${PROJECT_NAME}_test line2Dup.cpp test.cpp)

# 链接库
target_link_libraries(${PROJECT_NAME}_test ${OpenCV_LIBS} cuda_icp)

# 添加 OpenMP 链接
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME}_test OpenMP::OpenMP_CXX)
endif()


function(find_numpy)
    if(NOT PYTHON_EXECUTABLE)
        return()
    endif()
    
    #     ?   Python   ? NumPy ·  
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_PATH_TEMP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE NUMPY_RESULT
    )
    
    #     ·  
    if(NUMPY_RESULT EQUAL 0 AND NUMPY_PATH_TEMP)
        # ?     п?  ?  ?  з 
        string(STRIP "${NUMPY_PATH_TEMP}" NUMPY_PATH)
        string(REGEX REPLACE "[ \t\r\n]+" "" NUMPY_PATH "${NUMPY_PATH}")
        
        #   ?·  
        if(EXISTS "${NUMPY_PATH}" AND EXISTS "${NUMPY_PATH}/numpy/arrayobject.h")
            set(PYTHON_NUMPY_INCLUDE_DIR "${NUMPY_PATH}" PARENT_SCOPE)
            message(STATUS "Found NumPy include dir: ${NUMPY_PATH}")
        else()
            set(PYTHON_NUMPY_INCLUDE_DIR "" PARENT_SCOPE)
            message(WARNING "NumPy path found but invalid: ${NUMPY_PATH}")
        endif()
    else()
        #    ?   ·  
        set(NUMPY_PATHS
            "C:/Python312/Lib/site-packages/numpy/core/include"
            "D:/ProgramData/Anaconda3/Lib/site-packages/numpy/core/include"
            "$ENV{USERPROFILE}/AppData/Local/Programs/Python/Python312/Lib/site-packages/numpy/core/include"
        )
        
        foreach(PATH IN LISTS NUMPY_PATHS)
            if(EXISTS "${PATH}/numpy/arrayobject.h")
                set(PYTHON_NUMPY_INCLUDE_DIR "${PATH}" PARENT_SCOPE)
                message(STATUS "Found NumPy include dir (fallback): ${PATH}")
                return()
            endif()
        endforeach()
        
        set(PYTHON_NUMPY_INCLUDE_DIR "" PARENT_SCOPE)
        message(WARNING "NumPy not found")
    endif()
endfunction()

# pybind
if (USE_PYBIND)
    find_package(PythonLibs 3 REQUIRED)
    find_package(pybind11 REQUIRED)
    find_numpy()
    set(SOURCE_PYBIND
        pybind11/pybind11.cpp
        pybind11/np2mat/ndarray_converter.cpp)
    include_directories(${INCLUDE_DIRECTORIES}
            ${PYTHON_INCLUDE_DIRS}
            ${PYTHON_NUMPY_INCLUDE_DIR})
    pybind11_add_module(${PROJECT_NAME}_py line2Dup.cpp ${SOURCE_PYBIND})
    target_link_libraries(${PROJECT_NAME}_py PRIVATE ${OpenCV_LIBS})
endif()