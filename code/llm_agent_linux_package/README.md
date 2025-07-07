# LLM Agent Package

Flexible LLM API client with robust error handling, conversation management, and multi-provider support.

## 🎯 Overview

The LLM Agent package provides a unified interface for interacting with various Large Language Model APIs, including Gradio, LM Studio, OpenAI, and custom endpoints. It features conversation history management, automatic retry logic, and comprehensive error handling.

## � Files

### **`llm_agent.py`** ⭐ Main LLM Client

**Purpose**: Universal LLM API client with conversation management

**Key Features**:
- **🌐 Multi-Provider Support**: Gradio, LM Studio, OpenAI-compatible APIs
- **💭 Conversation History**: Optional context management with interactive controls
- **🔄 Auto-Retry Logic**: Exponential backoff for failed requests
- **🛠️ Configurable Parameters**: Temperature, max tokens, model selection
- **📊 Health Monitoring**: Connection testing and performance metrics
- **🔧 Runtime Controls**: Enable/disable history, clear conversations, get stats

**Basic Usage**:
```python
from llm_agent import LLMAgent

# Create agent for LM Studio
agent = LLMAgent("https://24be-174-161-24-253.ngrok-free.app", api_type="lmstudio")

# Send message with history
response = agent.send_message("Hello, how are you?")
print(response)

# Control conversation history
agent.set_maintain_history(False)  # Disable history
agent.clear_history()              # Clear conversation
stats = agent.get_conversation_stats()  # Get statistics
```

**Conversation History Management**:
```python
# Enable/disable history
agent = LLMAgent(url, maintain_history=True)   # Default: enabled
agent = LLMAgent(url, maintain_history=False)  # Disable at creation

# Runtime control
agent.set_maintain_history(False)  # Disable during session
agent.set_maintain_history(True)   # Re-enable during session

# History operations
agent.clear_history()                    # Clear conversation
stats = agent.get_conversation_stats()   # Get message counts, timing
history = agent.get_history()           # Export conversation
agent.save_conversation("chat.json")    # Save to file
```

**Multi-Provider Examples**:
```python
# LM Studio (local or ngrok)
lm_agent = LLMAgent("http://localhost:1234", api_type="lmstudio")

# Gradio application
gradio_agent = LLMAgent("http://localhost:7860", api_type="gradio")

# OpenAI-compatible
openai_agent = LLMAgent("https://api.openai.com", api_type="openai", api_key="your-key")

# Custom endpoint
custom_agent = LLMAgent("https://your-api.com", api_type="custom")
```

### **`remote_llm_test.py`** 🧪 Remote Testing Tool

**Purpose**: Simplified client for testing remote LM Studio via ngrok

**Features**:
- Lightweight testing without dependencies
- Interactive chat sessions
- Connection validation
- ngrok-optimized headers

**Usage**:
```bash
# Test remote LM Studio connection
python remote_llm_test.py https://24be-174-161-24-253.ngrok-free.app

# Interactive testing session
python remote_llm_test.py https://24be-174-161-24-253.ngrok-free.app
# Type messages, 'quit' to exit, 'config' to show settings
```

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
python remote_llm_test.py https://24be-174-161-24-253.ngrok-free.app
```

### Using Full LLM Agent
```python
from llm_agent import create_lmstudio_agent

# Connect to remote LM Studio
agent = create_lmstudio_agent('https://24be-174-161-24-253.ngrok-free.app')

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
curl -k -H "ngrok-skip-browser-warning: true" "https://24be-174-161-24-253.ngrok-free.app/v1/models"

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
## 🔧 Integration with TranscriberAgent

The `LLMAgent` is primarily used by the `TranscriberAgent` for voice-to-text-to-LLM workflows:

```python
from transcriber_test_script import TranscriberAgent

# TranscriberAgent automatically uses LLMAgent internally
agent = TranscriberAgent(
    llm_url="https://24be-174-161-24-253.ngrok-free.app",
    api_type="lmstudio",
    maintain_history=True  # Passed through to LLMAgent
)

# Interactive conversation with voice input and history management
agent.interactive_mode()
```

## � New Features

### Conversation History Management
- **Programmatic Control**: Set via constructor or runtime methods
- **Interactive Commands**: Toggle during active sessions  
- **Statistics Tracking**: Message counts, timing, conversation length
- **Persistence**: Save/load conversation history
- **Universal Support**: Works with all API types

### Enhanced Error Handling
- **Automatic Model Fallback**: Uses "any" when no model specified (fixes LM Studio 500 errors)
- **Retry Logic**: Exponential backoff for transient failures
- **Connection Testing**: Validate API availability before use
- **Health Monitoring**: Track response times and success rates

For complete voice-to-LLM workflows, use the `TranscriberAgent` which integrates this package with speech recognition capabilities.
