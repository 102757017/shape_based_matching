@echo off
REM ============================================================================
REM shape_based_matching - portable configure script. No machine-specific paths are
REM hardcoded. Dependency locations are injected via ENV vars or a local override
REM file, both of which stay out of the repo.
REM   1) ENV vars: OpenCV_DIR, EIGEN3_ROOT, PYTHON_EXECUTABLE
REM   2) local file: configure.local.bat in this dir, git-ignored. Copy it from
REM      configure.local.bat.example and fill in this machine's real paths.
REM VS, cmake and ninja are auto-located via vswhere, so this script never hardcodes
REM a VS version and does not depend on PATH being pre-set.
REM IMPORTANT: keep this file ASCII-only. cmd parses it under the system codepage,
REM so UTF-8 text and parentheses inside REM comments both break the parser. The
REM Chinese documentation lives in docs/agents/build.md, not in this script.
REM ============================================================================
setlocal

REM 0) optional machine-local override, git-ignored, absence is fine
set "SCRIPT_DIR=%~dp0"
if exist "%SCRIPT_DIR%configure.local.bat" call "%SCRIPT_DIR%configure.local.bat"

REM 1) locate Visual Studio, prefer an install that has the C++ tools, else latest
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [ERROR] vswhere.exe not found. Install Visual Studio with the Desktop development with C++ workload.
    exit /b 1
)
set "VSINSTALL="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL (
    for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -property installationPath`) do set "VSINSTALL=%%i"
)
if not defined VSINSTALL (
    echo [ERROR] No Visual Studio install found.
    exit /b 1
)

REM 2) load the VS environment for x64 and use the VS-bundled cmake and ninja
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "CMAKE=%VSINSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not exist "%CMAKE%" (
    echo [ERROR] VS-bundled cmake not found: %CMAKE%
    exit /b 1
)
set "PATH=%VSINSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

REM 3) validate the dependencies and fail loudly instead of silently
if not defined OpenCV_DIR (
    echo [ERROR] OpenCV_DIR is not set. Set the ENV var or define it in configure.local.bat.
    echo         set "OpenCV_DIR=path to the OpenCV build dir that contains OpenCVConfig.cmake"
    exit /b 1
)
if not exist "%OpenCV_DIR%\OpenCVConfig.cmake" (
    echo [ERROR] OpenCV_DIR is invalid, OpenCVConfig.cmake not found: %OpenCV_DIR%
    exit /b 1
)
if not defined EIGEN3_ROOT (
    echo [ERROR] EIGEN3_ROOT is not set. Point it at the dir that contains Eigen/Core.
    exit /b 1
)
if not exist "%EIGEN3_ROOT%\Eigen\Core" (
    echo [ERROR] EIGEN3_ROOT is invalid, Eigen/Core not found: %EIGEN3_ROOT%
    exit /b 1
)
if not defined PYTHON_EXECUTABLE (
    echo [ERROR] PYTHON_EXECUTABLE is not set. The Python interpreter is needed for the pybind11 binding.
    exit /b 1
)
if not exist "%PYTHON_EXECUTABLE%" (
    echo [ERROR] PYTHON_EXECUTABLE does not exist: %PYTHON_EXECUTABLE%
    exit /b 1
)

REM 4) run cmake. Notes for cross-machine builds:
REM   - OpenCV 4.x cmake does not recognise VS2026, whose MSVC is 1950, so the vc16
REM     runtime must be forced. The official Windows prebuilt packages are all vc16,
REM     which keeps this consistent across machines.
REM   - USE_CUDA defaults to OFF, so machines without a CUDA toolkit still build, the
REM     .cu files are excluded. Machines that have a GPU can add -DUSE_CUDA=ON.
REM   - SBM_SIMD_LEVEL defaults to AVX2, this machine's baseline. If the target CPU
REM     lacks AVX2, append -DSBM_SIMD_LEVEL=SSE2 or NONE to avoid a runtime SIGILL.
set "BUILD_DIR=%SCRIPT_DIR%..\build"
"%CMAKE%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=%OpenCV_DIR% -DOpenCV_ARCH=x64 -DOpenCV_RUNTIME=vc16 -DEIGEN3_ROOT=%EIGEN3_ROOT% -DPYTHON_EXECUTABLE=%PYTHON_EXECUTABLE% -B "%BUILD_DIR%" -S "%SCRIPT_DIR%.."
if errorlevel 1 exit /b 1
echo [DONE] Config written to %BUILD_DIR%. Next: run scripts\build.bat
