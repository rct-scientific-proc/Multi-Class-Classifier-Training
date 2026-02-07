@echo off
REM Python virtual environment setup script (Batch version)
REM
REM Usage:
REM   setup.bat                                    # Interactive mode
REM   setup.bat "C:\Python311"                     # Specify Python path
REM   setup.bat "C:\Python311" cu126               # Specify Python path and CUDA version
REM   setup.bat cu126                              # Use default Python, specify CUDA
REM
REM CudaVersion options: cpu, cu118, cu126, cu128, cu130

setlocal enabledelayedexpansion

REM Python base path - modify this if needed
set "PYTHON_BASE_PATH=%USERPROFILE%\AppData\Local\Programs\Python\Python313"
set "CUDA_ARG="

REM Parse arguments - check if first arg is a CUDA version
set "ARG1=%~1"
set "ARG2=%~2"

REM Check if ARG1 is a known CUDA version (meaning Python path was skipped)
if "%ARG1%"=="cpu" (
    set "CUDA_ARG=cpu"
) else if "%ARG1%"=="cu128" (
    set "CUDA_ARG=cu128"
) else if "%ARG1%"=="cu126" (
    set "CUDA_ARG=cu126"
) else if "%ARG1%"=="cu128" (
    set "CUDA_ARG=cu128"
) else if "%ARG1%"=="cu130" (
    set "CUDA_ARG=cu130"
) else if not "%ARG1%"=="" (
    REM First arg is not a CUDA version, treat it as Python path
    set "PYTHON_BASE_PATH=%ARG1%"
    set "CUDA_ARG=%ARG2%"
) else (
    REM First arg is empty, check second arg
    set "CUDA_ARG=%ARG2%"
)

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

REM Check if CUDA version was provided via command line
if not "%CUDA_ARG%"=="" (
    REM Validate provided CUDA version
    if "%CUDA_ARG%"=="cpu" (
        set "CHOSEN_CUDA=cpu"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
        echo [INFO] Using CUDA version from argument: cpu
    ) else if "%CUDA_ARG%"=="cu128" (
        set "CHOSEN_CUDA=cu128"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
        echo [INFO] Using CUDA version from argument: cu128
    ) else if "%CUDA_ARG%"=="cu126" (
        set "CHOSEN_CUDA=cu126"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
        echo [INFO] Using CUDA version from argument: cu126
    ) else if "%CUDA_ARG%"=="cu130" (
        set "CHOSEN_CUDA=cu130"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu130"
        echo [INFO] Using CUDA version from argument: cu130
    ) else (
        echo [ERROR] Invalid CUDA version: %CUDA_ARG%. Valid options: cpu, cu128, cu126, cu130
        exit /b 1
    )
) else (
    REM Ask user for CUDA version or CPU only
    echo.
    echo Select CUDA version for PyTorch installation:
    echo [0] cpu
    echo [1] cu126
    echo [2] cu128
    echo [3] cu130
    echo.
    set /p "SELECTION=Enter the number corresponding to your choice (default 0 for cpu): "

    if "!SELECTION!"=="" set "SELECTION=0"

    REM Set the chosen CUDA version
    if "!SELECTION!"=="0" (
        set "CHOSEN_CUDA=cpu"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
    ) else if "!SELECTION!"=="1" (
        set "CHOSEN_CUDA=cu126"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
    ) else if "!SELECTION!"=="2" (
        set "CHOSEN_CUDA=cu128"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
    ) else if "!SELECTION!"=="3" (
        set "CHOSEN_CUDA=cu130"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu130"
    ) else (
        echo [WARN] Invalid selection. Defaulting to 'cpu'.
        set "CHOSEN_CUDA=cpu"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
    )
    echo [INFO] You selected: !CHOSEN_CUDA!
)

REM Install torch and torchvision based on chosen CUDA version
echo [INFO] Installing torch and torchvision for %CHOSEN_CUDA%...
"%VENV_PYTHON%" -m pip install torch torchvision --index-url %TORCH_INDEX%
if errorlevel 1 (
    echo [ERROR] Failed to install torch and torchvision.
    exit /b 1
)

REM Install final dependencies
echo [INFO] Installing additional dependencies: onnx, onnxscript, onnxruntime, tqdm, scikit-learn, matplotlib, seaborn...
"%VENV_PYTHON%" -m pip install onnx onnxscript onnxruntime tqdm scikit-learn matplotlib seaborn
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
