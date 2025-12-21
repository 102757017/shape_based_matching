cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 14)
project(shape_based_matching)


# debug or release
SET(CMAKE_BUILD_TYPE "Release")

# add this if want to show match time
#add_definitions(-DDEBUG_MATCH_TIME)

# arm or x86
IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    SET(PLATFORM_COMPILE_FLAGS "-mfpu=neon")
ELSE()
    SET(PLATFORM_COMPILE_FLAGS "-march=native")

    # some places of the algorithm are designed for 128 SIMD
    # so 128 SSE may slightly faster than 256 AVX, you may want this
#    SET(PLATFORM_COMPILE_FLAGS "-msse -msse2 -msse3 -msse4 -mssse3")  # SSE only
ENDIF()

# SET(PLATFORM_COMPILE_FLAGS "-DMIPP_NO_INTRINSICS")  # close SIMD
#SET(COMMON_COMPILE_FLAGS "-fopenmp -Wall -Wno-sign-compare")
SET(COMMON_COMPILE_FLAGS "-fopenmp -Wall")
SET(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS} $ENV{CXXFLAGS}")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")

function(find_numpy)
    if(NOT PYTHON_EXECUTABLE)
        return()
    endif()
    
    #     ?   Python   ? NumPy     
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_PATH_TEMP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE NUMPY_RESULT
    )
    
    #         
    if(NUMPY_RESULT EQUAL 0 AND NUMPY_PATH_TEMP)
        # ?       ?  ?  ?     
        string(STRIP "${NUMPY_PATH_TEMP}" NUMPY_PATH)
        string(REGEX REPLACE "[ \t\r\n]+" "" NUMPY_PATH "${NUMPY_PATH}")
        
        #   ?    
        if(EXISTS "${NUMPY_PATH}" AND EXISTS "${NUMPY_PATH}/numpy/arrayobject.h")
            set(PYTHON_NUMPY_INCLUDE_DIR "${NUMPY_PATH}" PARENT_SCOPE)
            message(STATUS "Found NumPy include dir: ${NUMPY_PATH}")
        else()
            set(PYTHON_NUMPY_INCLUDE_DIR "" PARENT_SCOPE)
            message(WARNING "NumPy path found but invalid: ${NUMPY_PATH}")
        endif()
    else()
        #    ?       
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


# opencv
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:/Users/sdf63/Downloads/opencv3/build/x64/vc15/lib")
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# include MIPP headers
include_directories (${INCLUDE_DIRECTORIES} "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/")


# icp for refine
option(USE_CUDA "use cuda or not" OFF)

if(USE_CUDA)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.0)
add_definitions(-DCUDA_ON)
endif()

add_subdirectory(cuda_icp)


# test exe
add_executable(${PROJECT_NAME}_test line2Dup.cpp test.cpp)
target_link_libraries(${PROJECT_NAME}_test ${OpenCV_LIBS} cuda_icp)

