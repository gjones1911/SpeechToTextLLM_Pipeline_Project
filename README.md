# Speech-to-Text LLM Pipeline Project

## 🎯 Overview

A robust, cross-platform Speech-to-Text (STT) pipeline designed for seamless integration with chatbot APIs. This project provides a production-ready voice-to-text solution with memory safety, multi-engine fallback, and API connectivity for modern chatbot frameworks.

## ✨ Key Features

- **🎙️ Robust STT Engine**: OpenAI Whisper + Google Speech Recognition fallback
- **🔗 API Integration**: Ready-to-use connections for Gradio and LM Studio APIs
- **🛡️ Memory Safety**: Comprehensive safeguards against memory allocation issues
- **🌐 Cross-Platform**: Windows and Linux support
- **🔄 Auto-Fallback**: Seamless switching between STT engines and API endpoints
- **⚡ Production-Ready**: Error handling, logging, and comprehensive testing

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd SpeechToTextLLM_Pipeline_Project

# Create and activate virtual environment
bash generate_venv.sh
bash activate_venv.sh

# Install dependencies
pip install -r requirements.txt
```

### 2. Test the System

```bash
# Start the test API server (in terminal 1)
python code/voice_processing/test_api_server.py

# Start voice chat (in terminal 2)
python code/voice_processing/api_chatbot.py --gradio-url "http://localhost:7860"
```

### 3. Use with Real APIs

```bash
# Gradio integration
python code/voice_processing/api_chatbot.py --gradio-url "http://your-gradio-server:7860"

# LM Studio integration
python code/voice_processing/api_chatbot.py --lm-studio-url "http://localhost:1234"

# Both APIs with fallback
python code/voice_processing/api_chatbot.py \
  --gradio-url "http://your-gradio-server:7860" \
  --lm-studio-url "http://localhost:1234"
```

## 📁 Project Structure

```text
SpeechToTextLLM_Pipeline_Project/
├── 📖 Documentation
│   ├── README.md                        # This file - project overview
│   ├── STT_TECHNICAL_DOCUMENTATION.md   # Detailed method documentation
│   ├── API_INTEGRATION_GUIDE.md         # API integration guide
│   ├── COMPLETION_SUMMARY.md            # Project completion summary
│   ├── CLEANUP_ANALYSIS.md              # File cleanup documentation
│   └── PROJECT_STATUS.md                # Current project status
├── ⚙️ Configuration
│   ├── requirements.txt                 # Python dependencies
│   ├── generate_venv.sh                # Virtual environment setup
│   └── activate_venv.sh                # Environment activation
└── 🐍 Core Application
    └── code/
        ├── README.md                   # Code directory overview
        └── voice_processing/
            ├── README.md               # Module documentation
            ├── api_chatbot.py          ⭐ Main voice-to-API application
            ├── multi_engine_stt.py     ⭐ Robust STT engine
            ├── test_api_server.py      ⭐ Mock API server for testing
            └── test_complete_pipeline.py ⭐ Comprehensive test suite
```

## 🔧 Core Components

### 1. Voice-to-API Chatbot (`api_chatbot.py`)
The main application that combines voice input with API-based chatbots:
- Records speech using sounddevice
- Transcribes using multi-engine STT
- Sends to Gradio/LM Studio APIs
- Returns responses to user

### 2. Multi-Engine STT (`multi_engine_stt.py`)
Robust speech-to-text engine with fallback:
- **Primary**: OpenAI Whisper (local, high quality)
- **Fallback**: Google Speech Recognition (online, reliable)
- Memory-safe audio processing
- Automatic engine switching

### 3. Test API Server (`test_api_server.py`)
Mock Gradio-compatible server for development:
- Gradio-style API endpoints
- Conversation history tracking
- Realistic response simulation
- Health monitoring

### 4. Test Suite (`test_complete_pipeline.py`)
Comprehensive validation system:
- Import and dependency tests
- STT engine validation
- Memory usage monitoring
- API connectivity checks

## 🎮 Usage Examples

### Interactive Voice Chat
```bash
python code/voice_processing/api_chatbot.py --mode interactive
# Press Enter to record voice, type to send text, 'quit' to exit
```

### Single Voice Query
```bash
python code/voice_processing/api_chatbot.py --mode single --duration 5
# Records 5 seconds of speech and returns response
```

### Custom Configuration
```bash
python code/voice_processing/api_chatbot.py \
  --gradio-url "http://your-server:7860" \
  --lm-studio-url "http://localhost:1234" \
  --duration 4.0 \
  --mode interactive
```

## 🔗 API Integration

### Gradio Integration
Connect to any Gradio-based chatbot:
```python
from code.voice_processing.api_chatbot import VoiceAPIChat

voice_chat = VoiceAPIChat(gradio_url="http://localhost:7860")
result = voice_chat.single_voice_query(duration=4.0)
print(f"Response: {result['response']}")
```

### LM Studio Integration
Connect to local LLMs via LM Studio:
```python
voice_chat = VoiceAPIChat(lm_studio_url="http://localhost:1234")
success, transcript = voice_chat.transcribe_speech(duration=3.0)
response = voice_chat.send_to_chatbot(transcript)
```

## 📊 Technical Specifications

### Audio Processing
- **Sample Rate**: 16 kHz (Whisper optimized)
- **Format**: 32-bit float (processing), 16-bit int (storage)
- **Channels**: Mono (1 channel)
- **Duration**: 1-30 seconds (configurable)
- **Memory Safety**: 5-100 MB limits per recording

### STT Engines
- **Whisper**: `tiny` model, CPU-only, FP32 precision
- **Google**: Online API with automatic retry
- **Fallback**: Automatic switching on failure
- **Memory Usage**: <200 MB total

### API Support
- **Gradio**: Native Gradio API format support
- **LM Studio**: OpenAI-compatible endpoints
- **Custom APIs**: Extensible architecture for new APIs
- **Fallback**: Automatic API switching

### System Requirements
- **OS**: Windows 10+ or Linux (Ubuntu 18.04+)
- **Python**: 3.8+ (tested with 3.11)
- **Memory**: 4+ GB RAM recommended
- **Internet**: Optional (for Google STT and online APIs)

## 🛠️ Development

### Running Tests
```bash
# Full test suite
python code/voice_processing/test_complete_pipeline.py

# Start development server
python code/voice_processing/test_api_server.py
```

### Adding Custom APIs
Extend the `VoiceAPIChat` class to support new APIs:
```python
class YourCustomAPI:
    def test_connection(self) -> bool:
        # Test API connectivity
        pass
    
    def send_message(self, message: str) -> Optional[str]:
        # Send message and return response
        pass
```

### Debugging
- Enable verbose logging: `logging.basicConfig(level=logging.DEBUG)`
- Check individual components with test files
- Monitor memory usage with built-in tracking
- Use health check endpoints for API status

## 🚨 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "No STT engines available" | Install missing dependencies: `pip install openai-whisper sounddevice` |
| "No chatbot APIs available" | Start API servers (Gradio/LM Studio) or use test server |
| "Memory allocation error" | Reduce recording duration, restart application |
| "Microphone not found" | Check audio permissions and device connections |

### Performance Tips
- Use shorter recording durations (3-5 seconds) for faster response
- Keep API servers on local network for low latency
- Restart application if memory usage grows over time
- Use test server for development to avoid API rate limits

## 📚 Documentation

- **[Technical Documentation](STT_TECHNICAL_DOCUMENTATION.md)**: Detailed method explanations and learning guide
- **[API Integration Guide](API_INTEGRATION_GUIDE.md)**: Complete API setup and usage instructions
- **[Completion Summary](COMPLETION_SUMMARY.md)**: Project development history and achievements

## 🎯 Use Cases

- **Voice Assistants**: Add voice input to existing chatbots
- **Accessibility**: Voice control for applications
- **Prototyping**: Quick voice interface development
- **Education**: Learning STT and API integration
- **Research**: Voice interaction experiments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI Whisper team for the excellent STT model
- Google Speech Recognition for reliable fallback
- Gradio team for the API framework
- SoundDevice library for cross-platform audio

---

**Ready to add voice interaction to your applications!** 🎙️🤖 
