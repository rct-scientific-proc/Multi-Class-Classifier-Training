#!/bin/bash
# Python virtual environment setup script (Bash version)
#
# Usage:
#   ./setup.sh                                    # Interactive mode
#   ./setup.sh /usr/bin/python3                   # Specify Python path
#   ./setup.sh /usr/bin/python3 cu126             # Specify Python path and CUDA version
#   ./setup.sh "" cu126                           # Use default Python, specify CUDA
#
# CudaVersion options: cpu, cu118, cu126, cu130

# Python base path - modify this or pass as argument
PYTHON_BASE_PATH="${1:-/usr/bin}"

# CUDA version from command line (optional)
CUDA_ARG="${2:-}"

# Python executable path
PYTHON_EXE="$PYTHON_BASE_PATH/python"

# Check if Python executable exists
if [ ! -f "$PYTHON_EXE" ]; then
    echo "[ERROR] Python executable not found at: $PYTHON_EXE"
    exit 1
else
    echo "[OK] Python executable found at: $PYTHON_EXE"
fi

# Check Python version >= 3.10
PYTHON_VERSION=$($PYTHON_EXE --version 2>&1 | awk '{print $2}')
MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]; }; then
    echo "[ERROR] Python version must be >= 3.10. Found: $PYTHON_VERSION"
    exit 1
else
    echo "[OK] Python version: $PYTHON_VERSION"
fi

# Create virtual environment in the current directory .venv
VENV_PATH="$(pwd)/.venv"

if [ -d "$VENV_PATH" ]; then
    echo "[INFO] Removing existing virtual environment at: $VENV_PATH"
    rm -rf "$VENV_PATH"
fi

echo "[INFO] Creating virtual environment at: $VENV_PATH"
"$PYTHON_EXE" -m venv "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to create virtual environment."
    exit 1
fi

# Get path to the virtual environment's python
VENV_PYTHON="$VENV_PATH/bin/python"

# Upgrade pip
echo "[INFO] Upgrading pip in the virtual environment..."
"$VENV_PYTHON" -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to upgrade pip."
    exit 1
fi

# Upgrade again just to be safe
echo "[INFO] Upgrading pip again to ensure latest version..."
"$VENV_PYTHON" -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to upgrade pip on second attempt."
    exit 1
fi

# Upgrade setuptools and wheel
echo "[INFO] Upgrading setuptools and wheel..."
"$VENV_PYTHON" -m pip install --upgrade setuptools wheel
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to upgrade setuptools and wheel."
    exit 1
fi

# Check if CUDA version was provided via command line
if [ -n "$CUDA_ARG" ]; then
    # Validate provided CUDA version
    case "$CUDA_ARG" in
        cpu)
            CHOSEN_CUDA="cpu"
            TORCH_INDEX="https://download.pytorch.org/whl/cpu"
            echo "[INFO] Using CUDA version from argument: cpu"
            ;;
        cu128)
            CHOSEN_CUDA="cu128"
            TORCH_INDEX="https://download.pytorch.org/whl/cu128"
            echo "[INFO] Using CUDA version from argument: cu128"
            ;;
        cu126)
            CHOSEN_CUDA="cu126"
            TORCH_INDEX="https://download.pytorch.org/whl/cu126"
            echo "[INFO] Using CUDA version from argument: cu126"
            ;;
        cu130)
            CHOSEN_CUDA="cu130"
            TORCH_INDEX="https://download.pytorch.org/whl/cu130"
            echo "[INFO] Using CUDA version from argument: cu130"
            ;;
        *)
            echo "[ERROR] Invalid CUDA version: $CUDA_ARG. Valid options: cpu, cu118, cu126, cu130"
            exit 1
            ;;
    esac
else
    # Ask user for CUDA version or CPU only
    echo ""
    echo "Select CUDA version for PyTorch installation:"
    echo "[0] cpu"
    echo "[1] cu126"
    echo "[2] cu128"
    echo "[3] cu130"
    echo ""
    read -p "Enter the number corresponding to your choice (default 0 for cpu): " SELECTION

    # Default to cpu if empty
    SELECTION="${SELECTION:-0}"

    # Set the chosen CUDA version
    case "$SELECTION" in
        0)
            CHOSEN_CUDA="cpu"
            TORCH_INDEX="https://download.pytorch.org/whl/cpu"
            ;;
        1)
            CHOSEN_CUDA="cu126"
            TORCH_INDEX="https://download.pytorch.org/whl/cu126"
            ;;
        2)
            CHOSEN_CUDA="cu128"
            TORCH_INDEX="https://download.pytorch.org/whl/cu128"
            ;;
        3)
            CHOSEN_CUDA="cu130"
            TORCH_INDEX="https://download.pytorch.org/whl/cu130"
            ;;
        *)
            echo "[WARN] Invalid selection. Defaulting to 'cpu'."
            CHOSEN_CUDA="cpu"
            TORCH_INDEX="https://download.pytorch.org/whl/cpu"
            ;;
    esac
    echo "[INFO] You selected: $CHOSEN_CUDA"
fi

# Install torch and torchvision based on chosen CUDA version
echo "[INFO] Installing torch and torchvision for $CHOSEN_CUDA..."
"$VENV_PYTHON" -m pip install torch torchvision --index-url "$TORCH_INDEX"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install torch and torchvision."
    exit 1
fi

# Install final dependencies
echo "[INFO] Installing additional dependencies: onnx, onnxscript, onnxruntime, tqdm, scikit-learn, matplotlib, seaborn..."
"$VENV_PYTHON" -m pip install onnx onnxscript onnxruntime tqdm scikit-learn matplotlib seaborn
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install additional dependencies."
    exit 1
fi

echo ""
echo "[SUCCESS] Setup completed successfully!"
echo "To activate the virtual environment, run:"
echo "    source .venv_linux/bin/activate"
echo ""
