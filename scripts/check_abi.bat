@echo off
REM ============================================================================
REM shape_based_matching - ABI contract check. A deterministic guardrail proving
REM the freshly built DLL keeps its contract with the external C# consumer:
REM   - the binary is x64
REM   - the OpenCV world DLL it links matches the runtime in the consumer libs
REM   - optional: exported symbols cover the C# P/Invoke names
REM The consumer location is machine-specific, so per-machine values live in
REM configure.local.bat as SBM_CONSUMER_LIBS and SBM_CONSUMER_CS (git-ignored).
REM ASCII-only on purpose, no parentheses in REM comments.
REM ============================================================================
setlocal
set "SCRIPT_DIR=%~dp0"
set "REPO=%SCRIPT_DIR%.."

REM load per-machine settings, including the consumer paths when present
if exist "%SCRIPT_DIR%configure.local.bat" call "%SCRIPT_DIR%configure.local.bat"

REM locate python: prefer the configured interpreter, then py, then python
set "PY="
if defined PYTHON_EXECUTABLE set "PY=%PYTHON_EXECUTABLE%"
if not defined PY (
    where py >nul 2>nul && set "PY=py -3"
)
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo [ERROR] check_abi: no Python interpreter found. Set PYTHON_EXECUTABLE or put python on PATH.
    exit /b 1
)

REM the freshly built DLL
set "DLL=%REPO%\build\bindings\c_api\shape_based_matching_c.dll"
if not exist "%DLL%" (
    echo [ERROR] check_abi: built DLL not found: %DLL%
    echo         Run scripts\build.bat first.
    exit /b 1
)

%PY% "%SCRIPT_DIR%check_abi.py" --dll "%DLL%" --libs "%SBM_CONSUMER_LIBS%" --cs "%SBM_CONSUMER_CS%"
exit /b %errorlevel%
