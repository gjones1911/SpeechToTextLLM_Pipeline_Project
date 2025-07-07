# Core Application Code

This directory contains the main application code for the Speech-to-Text LLM Pipeline Project.

## 📁 Structure

```text
code/
├── __init__.py                     # Package initialization
├── transcriber_test_script.py     ⭐ TranscriberAgent - Primary interface
├── llm_agent_linux_package/       # LLM API client package
│   ├── llm_agent.py               ⭐ Flexible LLM API client
│   ├── remote_llm_test.py         # Remote LM Studio testing
│   └── ...
└── voice_processing/              # Voice processing modules
    ├── __init__.py                # Package initialization
    ├── api_chatbot.py             # Legacy voice-to-API application
    ├── multi_engine_stt.py        ⭐ Robust STT engine
    ├── test_api_server.py         ⭐ Mock API server for testing
    └── test_complete_pipeline.py  ⭐ Comprehensive test suite
```

## 🚀 Quick Start

### Primary Interface (Recommended)
```bash
# Activate environment
bash activate_venv.sh

# Basic voice chat with LM Studio
python test_transcriber_agent.py https://your-ngrok-url.ngrok-free.app --api-type lmstudio

# Push-to-talk mode with conversation history
python test_transcriber_agent.py https://your-ngrok-url.ngrok-free.app --api-type lmstudio --soundmode button

# Continuous listening without history
python test_transcriber_agent.py https://your-ngrok-url.ngrok-free.app --api-type lmstudio --soundmode continuous --no-history
```

### Legacy Interface
```bash
# Start test server (Terminal 1)
python code/voice_processing/test_api_server.py

# Start voice chat (Terminal 2)
python code/voice_processing/api_chatbot.py --gradio-url "http://localhost:7860"
```

## 📚 Documentation

- **Main README**: `../README.md` - Complete project overview
- **Technical Docs**: `../STT_TECHNICAL_DOCUMENTATION.md` - Method documentation
- **API Guide**: `../API_INTEGRATION_GUIDE.md` - API integration instructions
- **Voice Processing**: `voice_processing/README.md` - Core module documentation

## 🎯 Main Entry Points

| File | Purpose | Usage |
|------|---------|-------|
| `transcriber_test_script.py` | **Primary Interface** | `python test_transcriber_agent.py <url> --api-type lmstudio` |
| `llm_agent_linux_package/llm_agent.py` | **LLM API Client** | Import: `from code.llm_agent_linux_package.llm_agent import LLMAgent` |
| `voice_processing/multi_engine_stt.py` | **STT Engine** | Import: `from code.voice_processing.multi_engine_stt import MultiEngineSTT` |
| `voice_processing/api_chatbot.py` | Legacy application | `python code/voice_processing/api_chatbot.py --gradio-url URL` |
| `voice_processing/test_api_server.py` | Development server | `python code/voice_processing/test_api_server.py` |

## 🆕 New Features

- **🤖 TranscriberAgent**: Unified voice-to-LLM interface with conversation history
- **💭 Conversation History**: Optional context management across all API types
- **🎛️ Dual Listening Modes**: Push-to-talk and continuous voice detection
- **🌐 Enhanced API Support**: Improved LM Studio, Gradio, and OpenAI compatibility
- **🔧 Interactive Controls**: Runtime history management and statistics

For detailed usage instructions, see the main project [README.md](../README.md).