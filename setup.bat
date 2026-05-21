@echo off
REM GrantMatch Setup & Run Script for Windows

echo ==================================================
echo GrantMatch ML Pipeline Setup
echo ==================================================

REM Step 1: Check Python
echo.
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
python --version
echo [OK] Python found

REM Step 2: Create virtual environment
echo.
echo [2/4] Setting up virtual environment...
if not exist "venv" (
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM Activate venv
call venv\Scripts\activate.bat

REM Step 3: Install dependencies
echo.
echo [3/4] Installing dependencies...
echo (This may take 5-10 minutes...)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo [OK] Dependencies installed

REM Step 4: Verify data
echo.
echo [4/4] Verifying data files...
python verify_data.py

REM Step 5: Ready to run
echo.
echo ==================================================
echo Setup Complete!
echo ==================================================
echo.
echo To run the training pipeline:
echo   python main.py
echo.
echo Expected runtime: 5-10 minutes
echo.
pause
