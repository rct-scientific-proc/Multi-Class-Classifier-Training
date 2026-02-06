@echo off
REM Python virtual environment setup script (Batch version)

setlocal enabledelayedexpansion

REM Python base path - modify this if needed
set "PYTHON_BASE_PATH=C:\Users\ryant\SDK\Python\3.10.11"
if not "%~1"=="" set "PYTHON_BASE_PATH=%~1"

REM Python executable path
set "PYTHON_EXE=%PYTHON_BASE_PATH%\python.exe"

REM Check if Python executable exists
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python executable not found at: %PYTHON_EXE%
    exit /b 1
) else (
    echo [OK] Python executable found at: %PYTHON_EXE%
)

REM Check Python version >= 3.10
for /f "tokens=2 delims= " %%v in ('"%PYTHON_EXE%" --version 2^>^&1') do set "PYTHON_VERSION=%%v"
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set "MAJOR=%%a"
    set "MINOR=%%b"
)
if %MAJOR% LSS 3 (
    echo [ERROR] Python version must be ^>= 3.10. Found: %PYTHON_VERSION%
    exit /b 1
)
if %MAJOR% EQU 3 if %MINOR% LSS 10 (
    echo [ERROR] Python version must be ^>= 3.10. Found: %PYTHON_VERSION%
    exit /b 1
)
echo [OK] Python version: %PYTHON_VERSION%

REM Create virtual environment in the current directory .venv
set "VENV_PATH=%CD%\.venv"

if exist "%VENV_PATH%" (
    echo [INFO] Removing existing virtual environment at: %VENV_PATH%
    rmdir /s /q "%VENV_PATH%"
)

echo [INFO] Creating virtual environment at: %VENV_PATH%
"%PYTHON_EXE%" -m venv "%VENV_PATH%"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
)

REM Get path to the virtual environment's python
set "VENV_PYTHON=%VENV_PATH%\Scripts\python.exe"

REM Upgrade pip
echo [INFO] Upgrading pip in the virtual environment...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    exit /b 1
)

REM Upgrade again just to be safe
echo [INFO] Upgrading pip again to ensure latest version...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip on second attempt.
    exit /b 1
)

REM Upgrade setuptools and wheel
echo [INFO] Upgrading setuptools and wheel...
"%VENV_PYTHON%" -m pip install --upgrade setuptools wheel
if errorlevel 1 (
    echo [ERROR] Failed to upgrade setuptools and wheel.
    exit /b 1
)

REM Ask user for CUDA version or CPU only
echo.
echo Select CUDA version for PyTorch installation:
echo [0] cpu
echo [1] cu126
echo [2] cu118
echo [3] cu130
echo.
set /p "SELECTION=Enter the number corresponding to your choice (default 0 for cpu): "

if "%SELECTION%"=="" set "SELECTION=0"

REM Set the chosen CUDA version
if "%SELECTION%"=="0" (
    set "CHOSEN_CUDA=cpu"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
) else if "%SELECTION%"=="1" (
    set "CHOSEN_CUDA=cu126"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
) else if "%SELECTION%"=="2" (
    set "CHOSEN_CUDA=cu118"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu118"
) else if "%SELECTION%"=="3" (
    set "CHOSEN_CUDA=cu130"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu130"
) else (
    echo [WARN] Invalid selection. Defaulting to 'cpu'.
    set "CHOSEN_CUDA=cpu"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
)

echo [INFO] You selected: %CHOSEN_CUDA%

REM Install torch and torchvision based on chosen CUDA version
echo [INFO] Installing torch and torchvision for %CHOSEN_CUDA%...
"%VENV_PYTHON%" -m pip install torch torchvision --index-url %TORCH_INDEX%
if errorlevel 1 (
    echo [ERROR] Failed to install torch and torchvision.
    exit /b 1
)

REM Install final dependencies
echo [INFO] Installing additional dependencies: onnx, onnxscript, onnxruntime, tqdm, scikit-learn, matplotlib, pyzmq...
"%VENV_PYTHON%" -m pip install onnx onnxscript onnxruntime tqdm scikit-learn matplotlib pyzmq
if errorlevel 1 (
    echo [ERROR] Failed to install additional dependencies.
    exit /b 1
)

echo.
echo [SUCCESS] Setup completed successfully!
echo To activate the virtual environment, run:
echo     .venv\Scripts\activate.bat
echo.

endlocal
