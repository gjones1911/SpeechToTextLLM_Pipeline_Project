# 🐧 LLM Agent Linux Testing Package

This package contains everything needed to test the LLM Agent on Linux systems connecting to a Windows machine running LM Studio.

## 📋 Current Configuration

**ngrok URL:** `https://0987-2601-840-8702-17f0-3d62-8389-f210-1869.ngrok-free.app`

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Make setup script executable
chmod +x setup_linux_venv.sh

# Run setup (creates virtual environment and installs dependencies)
./setup_linux_venv.sh
```

### 2. Activate Environment
```bash
# Navigate to project directory
cd ~/llm_agent_remote

# Activate virtual environment
source venv/bin/activate
```

### 3. Test Connection
```bash
# Simple test
python remote_llm_test.py https://0987-2601-840-8702-17f0-3d62-8389-f210-1869.ngrok-free.app

# Enhanced test (if available)
python test_llm_agent_enhanced.py
```

## 📁 Package Contents

- **`remote_llm_test.py`** - Simple test script for basic connection testing
- **`llm_agent.py`** - Full LLM Agent class with all features
- **`test_llm_agent_enhanced.py`** - Comprehensive test suite
- **`requirements.txt`** - Full requirements for all features
- **`requirements-linux-minimal.txt`** - Minimal requirements for basic testing
- **`setup_linux_venv.sh`** - Automated setup script
- **`README.md`** - This file

## 🔧 Manual Setup (Alternative)

If the automated setup doesn't work:

```bash
# Install Python 3 and pip (if not installed)
sudo apt update
sudo apt install python3 python3-venv python3-pip

# Create virtual environment
python3 -m venv ~/llm_agent_remote/venv

# Activate environment
source ~/llm_agent_remote/venv/bin/activate

# Install minimal requirements
pip install requests python-dotenv

# Or install full requirements
pip install -r requirements.txt
```

## 🧪 Testing Examples

### Basic Connection Test
```bash
python remote_llm_test.py https://0987-2601-840-8702-17f0-3d62-8389-f210-1869.ngrok-free.app
```

### Using Full LLM Agent
```python
from llm_agent import create_lmstudio_agent

# Connect to remote LM Studio
agent = create_lmstudio_agent('https://0987-2601-840-8702-17f0-3d62-8389-f210-1869.ngrok-free.app')

# Test connection
health = agent.get_health_status()
print(f"Status: {health['agent_status']}")

# Send message
response = agent.send_message("Hello from Linux!")
print(response)

# Batch processing
questions = ["What is AI?", "Explain Python", "What is Linux?"]
responses = agent.batch_messages(questions)

# Save conversation
agent.save_conversation("chat_session.json")
```

### Interactive Chat
```bash
# Run enhanced test for interactive features
python test_llm_agent_enhanced.py
```

## ⚠️ SSL Certificate Issues

If you encounter SSL certificate errors:

```bash
# For curl commands, use -k flag
curl -k -H "ngrok-skip-browser-warning: true" "https://0987-2601-840-8702-17f0-3d62-8389-f210-1869.ngrok-free.app/v1/models"

# For Python, the scripts handle SSL verification automatically
```

## 🐛 Troubleshooting

### Connection Issues
1. Verify the Windows machine is running LM Studio with server enabled
2. Check that ngrok tunnel is active
3. Test the ngrok URL in a browser first
4. Ensure firewall isn't blocking the connection

### Python Issues
1. Make sure Python 3.8+ is installed
2. Verify virtual environment is activated
3. Check that all required packages are installed
4. Try the minimal requirements if full installation fails

### Package Issues
1. Ensure all files were copied correctly
2. Make scripts executable: `chmod +x *.sh`
3. Check file permissions and ownership

## 📞 Support

If you encounter issues:
1. Check the ngrok URL is still active
2. Verify LM Studio is running on the Windows machine
3. Test local connectivity first
4. Check the troubleshooting section above

## 🎉 Success Criteria

A successful test should show:
- ✅ Connection to ngrok URL
- ✅ Model list retrieval
- ✅ Chat message exchange
- ✅ Response time under 5 seconds
- ✅ Stable connection for multiple messages

---

**Happy testing! 🌐🚀**
