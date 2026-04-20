@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
chcp 65001 > nul
color 0a
title "ASMODEUS SWARM | Liquid Topological Intelligence"

:: Set Paths
set "APP_DIR=%~dp0"
set "EXPERT_DIR=%APP_DIR%expert_checkpoints"
set "PYTHON_EXE="

:: Find Python
if exist "%APP_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%APP_DIR%.venv\Scripts\python.exe"
) else if exist "%APP_DIR%..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%APP_DIR%..\.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python.exe"
)

echo [SYSTEM] Using Python: "%PYTHON_EXE%"

:: Clear Port 8000
echo [SYSTEM] Clearing port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo [SYSTEM] Killing process %%a...
    taskkill /f /pid %%a >nul 2>&1
)

:: Expert Checkpoint Verification
if not exist "%EXPERT_DIR%" mkdir "%EXPERT_DIR%"
set "ASMODEUS_EXPERT_DIR=%EXPERT_DIR%"
"%PYTHON_EXE%" -c "import os; from tests.expert_fixtures import build_expert_store; build_expert_store(os.environ['ASMODEUS_EXPERT_DIR'], num_specialists=100); print('[SYSTEM] Manifold Checkpoints Ready.')"

:menu
echo.
echo ========================================================
echo   WE ARE ASMODEUS ^| LIQUID TOPOLOGICAL SWARM
echo ========================================================
echo   [1] Launch Web Manifold (Recommended)
echo   [2] Launch Swarm CLI
echo   [3] Exit
echo ========================================================
set /p choice="Select Interface [1-3]: "

if "%choice%"=="1" goto launch_web
if "%choice%"=="2" goto launch_cli
if "%choice%"=="3" exit /b 0
echo Invalid choice.
goto menu

:launch_web
echo [SYSTEM] Launching Web Interface at http://localhost:8000...
"%PYTHON_EXE%" asmodeus/web_ui.py
pause
exit /b 0

:launch_cli
echo [SYSTEM] Booting Collective Swarm CLI...
"%PYTHON_EXE%" cli.py --response-mode llm --persona ultron --queen-id "Qwen/Qwen2.5-7B-Instruct" --worker-id "Qwen/Qwen2.5-3B-Instruct" --sentry-id "Qwen/Qwen2.5-1.5B-Instruct" --expert-dir "%EXPERT_DIR%" --require-gpu --hybrid-memory --hybrid-gpu-fraction 0.80 --idle-autonomy
pause
exit /b 0
