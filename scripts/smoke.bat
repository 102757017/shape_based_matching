@echo off
REM ============================================================================
REM shape_based_matching - smoke test. This is the repo's lightweight guardrail.
REM It configures, builds, runs the matching demo in non-interactive mode, then
REM asserts the run finished cleanly. A CI job can call this single file.
REM
REM configure and build are incremental: a no-op when already up to date, so
REM re-running is cheap. ASCII-only on purpose, no parentheses in REM comments.
REM ============================================================================
setlocal

set "SCRIPT_DIR=%~dp0"

REM 1) configure and build via the existing portable scripts
call "%SCRIPT_DIR%configure.bat"
if errorlevel 1 exit /b 1
call "%SCRIPT_DIR%build.bat"
if errorlevel 1 exit /b 1

REM 2) locate the built test executable
set "EXE=%SCRIPT_DIR%..\build\apps\matching\shape_based_matching_test.exe"
if not exist "%EXE%" (
    echo [ERROR] smoke: test executable not found: %EXE%
    exit /b 1
)

REM 3) run the demo non-interactively and capture output
echo [SMOKE] running shape_based_matching_test --case case1 --no-view
"%EXE%" --case case1 --no-view > "%TEMP%\sbm_smoke.log" 2>&1
set "RUN_RC=%errorlevel%"

REM 4) assert the run printed the completion marker and exited zero
findstr /C:"test end" "%TEMP%\sbm_smoke.log" >nul
if errorlevel 1 (
    echo [FAIL] smoke: "test end" marker not found in output
    echo -------- output --------
    type "%TEMP%\sbm_smoke.log"
    exit /b 1
)
if not "%RUN_RC%"=="0" (
    echo [FAIL] smoke: test executable exited with code %RUN_RC%
    type "%TEMP%\sbm_smoke.log"
    exit /b 1
)

REM 5) optional ABI contract check - no-op when consumer paths are unconfigured
if exist "%SCRIPT_DIR%check_abi.bat" call "%SCRIPT_DIR%check_abi.bat"
if errorlevel 1 exit /b 1

echo [PASS] smoke: shape_based_matching_test completed cleanly
exit /b 0
