#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Detect the operating system
detect_os() {
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "linux"
    fi
}

OS=$(detect_os)
echo "Detected OS: $OS"

# Function to find Python executable
find_python() {
    # Try common Python commands in order
    for cmd in python3 python py; do
        if command -v "$cmd" &> /dev/null; then
i            # Verify the command actually works (not just a Microsoft Store redirect)
            if "$cmd" --version &> /dev/null; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    echo "python"  # fallback
}
# figure out which Python command to use
# This will be used to create the venv and install packages
# and register the Jupyter kernel
PYTHON_CMD=$(find_python)
echo "Using Python command: $PYTHON_CMD"

# 1. Create the virtual environment
$PYTHON_CMD -m venv .sttpipelinevenv

echo "Activating new venv '.sttpipelinevenv'"
# 2. Activate the virtual environment (cross-platform with fallback)
activate_venv() {
    local activated=false
    
    # Try Linux/Unix style first
    if [[ -f ".sttpipelinevenv/bin/activate" ]]; then
        echo "Using Linux/Unix activation path"
        source .sttpipelinevenv/bin/activate
        activated=true
    # Fallback to Windows style
    elif [[ -f ".sttpipelinevenv/Scripts/activate" ]]; then
        echo "Using Windows activation path"
        source .sttpipelinevenv/Scripts/activate
        activated=true
    else
        echo "❌ Activation script not found in either bin/ or Scripts/"
        exit 1
    fi
    
    if [[ "$activated" == true ]]; then
        echo "✅ Virtual environment activated successfully"
    fi
}

activate_venv

# Function to find pip executable (cross-platform)
find_pip() {
    # Try common pip commands in order
    for cmd in pip3 pip; do
        if command -v "$cmd" &> /dev/null; then
            echo "$cmd"
            return 0
        fi
    done
    echo "pip"  # fallback
}

PIP_CMD=$(find_pip)
echo "Using pip command: $PIP_CMD"

# 3. Upgrade pip
$PIP_CMD install --upgrade pip

# 4. Install dependencies from requirements.txt
$PIP_CMD install -r requirements.txt

# 5. Ensure that an Install of ipykernel is in the venv
$PIP_CMD install ipykernel

# 6. Register the kernel with a name matching the venv
$PYTHON_CMD -m ipykernel install --user --name=sttpipelinevenv --display-name="Python (sttpipelinevenv)"

echo "✅ Virtual environment 'sttpipelinevenv' created, dependencies installed, and Jupyter kernel registered."