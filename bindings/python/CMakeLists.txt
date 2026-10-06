# ============================================================================
# Python binding（pybind11 薄壳）：逻辑全在 _core 里，这里只做类型转换与导出。
# ============================================================================
if(MSVC)
    # 自研代码用 /W3（vendored 库设 /W2，避免全局 /W3 冲突产生 D9025）
    add_compile_options(/W3)
endif()

if(USE_PYBIND)
    # ---------- 1. 定位 Python 解释器 ----------
    if(DEFINED ENV{CONDA_PREFIX})
        set(PYTHON_EXECUTABLE "$ENV{CONDA_PREFIX}/python.exe" CACHE FILEPATH "Python executable")
    elseif(EXISTS "D:/portable program/python311/python.exe")
        set(PYTHON_EXECUTABLE "D:/portable program/python311/python.exe" CACHE FILEPATH "Python executable")
    elseif(DEFINED ENV{PYTHON_EXECUTABLE})
        set(PYTHON_EXECUTABLE "$ENV{PYTHON_EXECUTABLE}" CACHE FILEPATH "Python executable")
    endif()

    set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE FILEPATH "Python3 executable" FORCE)
    set(PYTHON_IS_DEBUG FALSE CACHE BOOL "" FORCE)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    message(STATUS "Using Python: ${Python3_EXECUTABLE} (version ${Python3_VERSION})")

    # ---------- 2. 查找 pybind11 ----------
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_PROBE_RESULT
    )
    if(NOT PYBIND11_PROBE_RESULT EQUAL 0)
        message(FATAL_ERROR "pybind11 not installed for ${Python3_EXECUTABLE}. Run: \"${Python3_EXECUTABLE}\" -m pip install pybind11")
    endif()
    find_package(pybind11 CONFIG REQUIRED)
    message(STATUS "pybind11 found: ${pybind11_DIR}")

    # ---------- 3. Debug 下伪造 pythonXX_d.lib（标准 workaround） ----------
    if(WIN32)
        set(PY_DEBUG_LIB_DIR "${CMAKE_BINARY_DIR}/py_debug_lib")
        set(PY_DEBUG_LIB_NAME "python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}_d.lib")
        get_filename_component(_py_root "${Python3_EXECUTABLE}" DIRECTORY)
        set(PY_RELEASE_LIB "${Python3_LIBRARY}")
        if(NOT PY_RELEASE_LIB OR NOT EXISTS "${PY_RELEASE_LIB}")
            set(PY_RELEASE_LIB "${_py_root}/libs/python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.lib")
        endif()
        file(MAKE_DIRECTORY "${PY_DEBUG_LIB_DIR}")
        if(EXISTS "${PY_RELEASE_LIB}" AND NOT EXISTS "${PY_DEBUG_LIB_DIR}/${PY_DEBUG_LIB_NAME}")
            file(COPY_FILE "${PY_RELEASE_LIB}" "${PY_DEBUG_LIB_DIR}/${PY_DEBUG_LIB_NAME}")
            message(STATUS "Created fake debug python lib: ${PY_DEBUG_LIB_DIR}/${PY_DEBUG_LIB_NAME}")
        elseif(NOT EXISTS "${PY_RELEASE_LIB}")
            message(WARNING "Python release lib not found at ${PY_RELEASE_LIB}; Debug link may fail")
        endif()
    endif()

    # ---------- 4. 查找 NumPy include ----------
    set(PYTHON_NUMPY_INCLUDE_DIR "")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_PATH_TEMP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE NUMPY_RESULT
    )
    if(NUMPY_RESULT EQUAL 0 AND NUMPY_PATH_TEMP)
        string(STRIP "${NUMPY_PATH_TEMP}" NUMPY_PATH)
        if(EXISTS "${NUMPY_PATH}/numpy/arrayobject.h")
            set(PYTHON_NUMPY_INCLUDE_DIR "${NUMPY_PATH}")
            message(STATUS "Found NumPy include dir: ${NUMPY_PATH}")
        else()
            message(STATUS "NumPy include path invalid: [${NUMPY_PATH}/numpy/arrayobject.h]")
        endif()
    else()
        message(STATUS "NumPy probe failed: result=${NUMPY_RESULT} output=[${NUMPY_PATH_TEMP}]")
    endif()
    if(NOT PYTHON_NUMPY_INCLUDE_DIR)
        message(FATAL_ERROR "NumPy not found for ${Python3_EXECUTABLE}. Run: \"${Python3_EXECUTABLE}\" -m pip install numpy")
    endif()

    # ---------- 5. 编译 pybind 模块 ----------
    pybind11_add_module(${PROJECT_NAME}_py
        pybind11.cpp
        py_matcher.cpp
        np2mat/ndarray_converter.cpp
    )
    if(WIN32)
        target_link_directories(${PROJECT_NAME}_py PRIVATE "${PY_DEBUG_LIB_DIR}")
    endif()
    target_include_directories(${PROJECT_NAME}_py PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PYTHON_NUMPY_INCLUDE_DIR}
    )
    target_link_libraries(${PROJECT_NAME}_py PRIVATE
        ${PROJECT_NAME}_core
        ellipse_detection
        cuda_icp
        ${OpenCV_LIBS}
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${PROJECT_NAME}_py PRIVATE OpenMP::OpenMP_CXX)
    endif()

    sbm_copy_opencv_runtime(${PROJECT_NAME}_py)
endif()
