#!/bin/bash

.sttpipelinevenv/bin/activate || .sttpipelinevenv/Scripts/activateecho "Activating new venv '.sttpipelinevenv'"
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
echo "Virtual environment activated. You can now run your Python scripts with the activated environment."