@echo off
REM ============================================================================
REM shape_based_matching - portable build script.
REM Dependencies are already cached in build/CMakeCache.txt from configure, so only
REM the VS environment is needed here.
REM
REM ASCII-only on purpose (see configure.bat header for why).
REM ============================================================================
setlocal

set "SCRIPT_DIR=%~dp0"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSINSTALL="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL (
    for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -property installationPath`) do set "VSINSTALL=%%i"
)
if not defined VSINSTALL (
    echo [ERROR] No Visual Studio install found.
    exit /b 1
)
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CMAKE=%VSINSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "PATH=%VSINSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

"%CMAKE%" --build "%SCRIPT_DIR%..\build" --config Release
if errorlevel 1 exit /b 1
echo [DONE] Build artifacts are in %SCRIPT_DIR%..\build\
