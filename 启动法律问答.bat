@echo off
setlocal enabledelayedexpansion
title Legal QA System

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo.
echo ======================================================
echo     Legal QA System - Launcher
echo ======================================================
echo.

:: Step 1: Find Python
echo [1/3] Checking Python...
where python >nul 2>nul
if not %errorlevel%==0 (
    echo [ERROR] Python not found in PATH!
    echo Please install Python 3.9+ from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)
python --version 2>nul
echo [OK] Python found
echo.

:: Step 2: Setup environment (venv + deps + config)
echo [2/3] Setting up environment...
echo.
echo This may take a few minutes on first run...
echo.

python setup_launcher.py
if not %errorlevel%==0 (
    echo.
    echo [FAIL] Setup failed. See above for details.
    pause
    exit /b 1
)

echo.
echo [OK] Environment ready
echo.

:: Use venv Python if available, otherwise system Python
set "PYTHON_EXE=%PROJECT_DIR%.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [!] venv not found, using system Python
    set "PYTHON_EXE=python"
)

:: Step 3: Launch Streamlit
echo [3/3] Launching...
echo.
echo ======================================================
echo     Starting Legal QA System...
echo ======================================================
echo.
echo   Opening browser at http://localhost:8501
echo   Press Ctrl+C to stop
echo.

start "" http://localhost:8501 2>nul
"%PYTHON_EXE%" -m streamlit run src/ui/streamlit_app.py --server.port 8501

echo.
echo System stopped.
pause
