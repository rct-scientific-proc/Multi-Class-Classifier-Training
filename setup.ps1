# Python virtual environment setup script

# Python base path parameter
param (
    [string]$PythonBasePath = "C:\Users\ryant\SDK\Python3.14"
)

# Python executable path
$pythonExe = Join-Path $PythonBasePath "\python.exe"


# Check if Python executable exists
if (-not (Test-Path $pythonExe)) {
    Write-Host "Python executable not found at: $pythonExe" -ForegroundColor Red
    exit 1
} else {
    Write-Host "Python executable found at: $pythonExe" -ForegroundColor Green
}

# Check Python version >= 3.10
$pythonVersionOutput = & $pythonExe --version 2>&1
if ($pythonVersionOutput -match "Python (\d+)\.(\d+)") {
    $majorVersion = [int]$Matches[1]
    $minorVersion = [int]$Matches[2]
    if ($majorVersion -lt 3 -or ($majorVersion -eq 3 -and $minorVersion -lt 10)) {
        Write-Host "[ERROR] Python version must be >= 3.10. Found: $pythonVersionOutput" -ForegroundColor Red
        exit 1
    } else {
        Write-Host "[OK] Python version: $pythonVersionOutput" -ForegroundColor Green
    }
} else {
    Write-Host "[ERROR] Could not determine Python version." -ForegroundColor Red
    exit 1
}

# Create virtual environment in the current directory .venv
$venvPath = Join-Path (Get-Location) ".venv"

if (Test-Path $venvPath) {
    Write-Host "Removing existing virtual environment at: $venvPath" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venvPath
}

Write-Host "Creating virtual environment at: $venvPath" -ForegroundColor Green
& $pythonExe -m venv $venvPath
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to create virtual environment." -ForegroundColor Red
    exit 1
}

# Get path to the virtual environment's python
$venvPython = Join-Path $venvPath "Scripts\python.exe"

# Upgrade pip
Write-Host "Upgrading pip in the virtual environment..." -ForegroundColor Green
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to upgrade pip." -ForegroundColor Red
    exit 1
}

# Upgrade again just to be safe
Write-Host "Upgrading pip again to ensure latest version..." -ForegroundColor Green
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to upgrade pip on second attempt." -ForegroundColor Red
    exit 1
}

# Upgrade setuptools and wheel
Write-Host "Upgrading setuptools and wheel..." -ForegroundColor Green
& $venvPython -m pip install --upgrade setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to upgrade setuptools and wheel." -ForegroundColor Red
    exit 1
}

# Install torch and torchvision, allowed cpu, cu126, cu118, cu130
# Ask user for CUDA version or CPU only
$cudaOptions = @("cpu", "cu126", "cu118", "cu130")
Write-Host "Select CUDA version for PyTorch installation:" -ForegroundColor Cyan
for ($i = 0; $i -lt $cudaOptions.Count; $i++) {
    Write-Host "[$i] $($cudaOptions[$i])"
}
$selection = Read-Host "Enter the number corresponding to your choice (default 0 for cpu)"
if ([string]::IsNullOrWhiteSpace($selection)) {
    $selection = 0
}

if ($selection -ge 0 -and $selection -lt $cudaOptions.Count) {
    $chosenCuda = $cudaOptions[$selection]
    Write-Host "You selected: $chosenCuda" -ForegroundColor Green
} else {
    Write-Host "Invalid selection. Defaulting to 'cpu'." -ForegroundColor Yellow
    $chosenCuda = "cpu"
}

# Install torch and torchvision based on chosen CUDA version
Write-Host "Installing torch and torchvision for $chosenCuda..." -ForegroundColor Green
switch ($chosenCuda) {
    "cpu" {
        & $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    }
    "cu126" {
        & $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    }
    "cu118" {
        & $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    }
    "cu130" {
        & $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    }
    default {
        Write-Host "Unknown CUDA option selected. Exiting." -ForegroundColor Red
        exit 1
    }
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install torch and torchvision." -ForegroundColor Red
    exit 1
}

# Install final dependencies, onnx onnxscript onnxruntime tqdm scikit-learn matplotlib pyzmq
Write-Host "Installing additional dependencies: onnx, onnxscript, onnxruntime, tqdm, scikit-learn, matplotlib, pyzmq..." -ForegroundColor Green
& $venvPython -m pip install onnx onnxscript onnxruntime tqdm scikit-learn matplotlib pyyaml seaborn
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install additional dependencies." -ForegroundColor Red
    exit 1
}

Write-Host "Setup completed successfully. To activate the virtual environment, run:" -ForegroundColor Green
Write-Host "`t`".\$venvPath\Scripts\Activate.ps1`"" -ForegroundColor Yellow



