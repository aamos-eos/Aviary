@echo off
echo ============================================================
echo Aircraft Performance Repository Installation (Windows)
echo ============================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo Error: pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo Installing standard requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install standard requirements
    pause
    exit /b 1
)


echo.
echo Installing atlas package...
pip install -e .
if errorlevel 1 (
    echo Warning: Failed to install atlas package
)

::echo.
::echo Installing PyTorch (CUDA 11.8)...
::pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
::if errorlevel 1 (
::    echo Warning: Failed to install PyTorch
::)

echo.
echo ============================================================
echo Installation completed!
echo ============================================================
echo.
pause 