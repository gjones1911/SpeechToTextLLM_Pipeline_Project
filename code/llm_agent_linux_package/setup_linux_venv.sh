#!/bin/bash

# Linux Virtual Environment Setup Script for LLM Agent
# This script sets up a Python virtual environment on Linux for LLM Agent testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐧 Setting up LLM Agent on Linux${NC}"
echo "=================================="

# Check Python version
echo -e "${YELLOW}📋 Checking Python installation...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    echo "Please install Python 3.8 or higher:"
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-venv python3-pip"
    echo "  RHEL/CentOS: sudo yum install python3 python3-venv python3-pip"
    echo "  Fedora: sudo dnf install python3 python3-venv python3-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo -e "${GREEN}✅ Python found: $PYTHON_VERSION${NC}"

# Check if venv module is available
if ! python3 -m venv --help &> /dev/null; then
    echo -e "${RED}❌ Python venv module not available${NC}"
    echo "Please install python3-venv:"
    echo "  Ubuntu/Debian: sudo apt install python3-venv"
    echo "  RHEL/CentOS: sudo yum install python3-venv"
    exit 1
fi

echo -e "${GREEN}✅ Python venv module available${NC}"

# Check pip
if ! python3 -m pip --version &> /dev/null; then
    echo -e "${RED}❌ pip not available${NC}"
    echo "Please install pip:"
    echo "  Ubuntu/Debian: sudo apt install python3-pip"
    echo "  RHEL/CentOS: sudo yum install python3-pip"
    exit 1
fi

echo -e "${GREEN}✅ pip available${NC}"

# Create project directory
PROJECT_DIR="$HOME/llm_agent_remote"
echo -e "${YELLOW}📁 Creating project directory: $PROJECT_DIR${NC}"

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create virtual environment
VENV_DIR="venv"
echo -e "${YELLOW}🔧 Creating virtual environment: $VENV_DIR${NC}"

if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists. Removing...${NC}"
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
echo -e "${GREEN}✅ Virtual environment created${NC}"

# Activate virtual environment
echo -e "${YELLOW}🚀 Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo -e "${YELLOW}⬆️  Upgrading pip...${NC}"
pip install --upgrade pip

# Check which requirements file to use
if [ -f "requirements.txt" ]; then
    REQUIREMENTS_FILE="requirements.txt"
    echo -e "${BLUE}📦 Using full requirements.txt${NC}"
elif [ -f "requirements-linux-minimal.txt" ]; then
    REQUIREMENTS_FILE="requirements-linux-minimal.txt"
    echo -e "${BLUE}📦 Using minimal requirements${NC}"
else
    echo -e "${YELLOW}📦 No requirements file found, installing minimal packages...${NC}"
    pip install requests python-dotenv
    echo -e "${GREEN}✅ Minimal packages installed${NC}"
    REQUIREMENTS_FILE=""
fi

# Install requirements if file exists
if [ ! -z "$REQUIREMENTS_FILE" ]; then
    echo -e "${YELLOW}📦 Installing packages from $REQUIREMENTS_FILE...${NC}"
    pip install -r "$REQUIREMENTS_FILE"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ All packages installed successfully${NC}"
    else
        echo -e "${RED}❌ Some packages failed to install${NC}"
        echo -e "${YELLOW}⚠️  Trying minimal installation...${NC}"
        pip install requests python-dotenv
    fi
fi

# Show installed packages
echo -e "${BLUE}📋 Installed packages:${NC}"
pip list

# Create activation script
echo -e "${YELLOW}📝 Creating activation script...${NC}"
cat > activate_llm_agent.sh << 'EOF'
#!/bin/bash
# LLM Agent Environment Activation Script

PROJECT_DIR="$HOME/llm_agent_remote"
VENV_DIR="$PROJECT_DIR/venv"

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
EOF

chmod +x activate_llm_agent.sh

# Deactivate for now
deactivate

echo ""
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo ""
echo -e "${BLUE}📁 Project directory: $PROJECT_DIR${NC}"
echo -e "${BLUE}🐍 Virtual environment: $PROJECT_DIR/venv${NC}"
echo ""
echo -e "${CYAN}🚀 To use the environment:${NC}"
echo "  cd $PROJECT_DIR"
echo "  source venv/bin/activate"
echo ""
echo -e "${CYAN}📄 Or use the activation script:${NC}"
echo "  cd $PROJECT_DIR"
echo "  ./activate_llm_agent.sh"
echo ""
echo -e "${CYAN}📂 Next steps:${NC}"
echo "  1. Copy LLM Agent files to: $PROJECT_DIR"
echo "  2. Activate environment: source $PROJECT_DIR/venv/bin/activate"
echo "  3. Test connection: python remote_llm_test.py <ngrok_url>"
echo ""
echo -e "${YELLOW}💡 Files to copy from Windows:${NC}"
echo "  - remote_llm_test.py (minimal testing)"
echo "  - llm_agent.py (full functionality)"
echo "  - test_llm_agent_enhanced.py (comprehensive testing)"
echo ""
echo -e "${GREEN}Happy testing! 🌐${NC}"
