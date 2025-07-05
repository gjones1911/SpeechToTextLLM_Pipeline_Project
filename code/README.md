# Core Application Code

This directory contains the main application code for the Speech-to-Text LLM Pipeline Project.

## 📁 Structure

```text
code/
├── __init__.py                   # Package initialization
└── voice_processing/            # Main application package
    ├── __init__.py              # Package initialization
    ├── api_chatbot.py           ⭐ Main voice-to-API application
    ├── multi_engine_stt.py      ⭐ Robust STT engine
    ├── test_api_server.py       ⭐ Mock API server for testing
    └── test_complete_pipeline.py ⭐ Comprehensive test suite
```

## 🚀 Quick Start

From the project root directory:

```bash
# Activate environment
bash activate_venv.sh

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
| `voice_processing/api_chatbot.py` | Main application | `python code/voice_processing/api_chatbot.py --gradio-url URL` |
| `voice_processing/test_api_server.py` | Development server | `python code/voice_processing/test_api_server.py` |
| `voice_processing/test_complete_pipeline.py` | System validation | `python code/voice_processing/test_complete_pipeline.py` |

For detailed usage instructions, see the main project [README.md](../README.md).