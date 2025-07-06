#!/bin/bash
# LLM Agent Environment Activation Script

PROJECT_DIR="$HOME/ScratchSpace/API_TESTING/LM_STUDIO"
VENV_DIR="$PROJECT_DIR/venvs/llmstudioapi7525"
echo "activating venv @ $VENV_DIR"

if [ -d "$VENV_DIR" ]; then
    echo "🚀 Activating LLM Agent environment..."
    cd "$PROJECT_DIR"
    source "$VENV_DIR/bin/activate"
    echo "✅ Environment activated!"
    echo "Current directory: $(pwd)"
    echo "Python: $(which python)"
    echo ""
    echo "Available scripts:"
    if [ -f "remote_llm_test.py" ]; then
        echo "  - python remote_llm_test.py <ngrok_url>"
    fi
    if [ -f "test_llm_agent.py" ]; then
        echo "  - python test_llm_agent.py"
    fi
    echo ""
    echo "To deactivate: deactivate"
else
    echo "❌ Virtual environment not found at $VENV_DIR"
    echo "Please run setup_linux_venv.sh first"
fi