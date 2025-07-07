# Speech-to-Text LLM Pipeline Project

## 🎯 Overview

A robust, cross-platform Speech-to-Text (STT) pipeline designed for seamless integration with chatbot APIs. This project provides a production-ready voice-to-text solution with memory safety, multi-engine fallback, and API connectivity for modern chatbot frameworks.

## ✨ Key Features

- **🎙️ Robust STT Engine**: OpenAI Whisper + Google Speech Recognition fallback
- **🔗 API Integration**: Ready-to-use connections for Gradio, LM Studio, and OpenAI APIs
- **💭 Conversation History**: Optional conversation context maintenance across interactions
- **🎛️ Dual Listening Modes**: Push-to-talk (SPACE key) and continuous voice detection
- **🛡️ Memory Safety**: Comprehensive safeguards against memory allocation issues
- **🌐 Cross-Platform**: Windows and Linux support
- **🔄 Auto-Fallback**: Seamless switching between STT engines and API endpoints
- **⚡ Production-Ready**: Error handling, logging, and comprehensive testing

## 🚀 Quick Start

### 0. Update ngrok URL (If using ngrok)

```bash
# Automatically update all ngrok URLs in the project
./update_ngrok_url.sh https://your-new-ngrok-url.ngrok-free.app

# Or on Windows
update_ngrok_url.bat https://your-new-ngrok-url.ngrok-free.app
```

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

### 2. Basic Voice-to-LLM Chat

```bash
# Quick test with LM Studio (ngrok or local)
python test_transcriber_agent.py https://your-ngrok-url.ngrok-free.app --api-type lmstudio --test-only

# Interactive session with conversation history
python test_transcriber_agent.py https://your-ngrok-url.ngrok-free.app --api-type lmstudio

# Continuous listening mode (hands-free)
python test_transcriber_agent.py https://your-ngrok-url.ngrok-free.app --api-type lmstudio --soundmode continuous
```

### 3. Legacy Testing

```bash
# Start the test API server (in terminal 1)
python code/voice_processing/test_api_server.py

# Start voice chat (in terminal 2)
python code/voice_processing/api_chatbot.py --gradio-url "http://localhost:7860"
```

### 4. Advanced Usage

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

## 🏗️ Core Classes

### TranscriberAgent (`code/transcriber_test_script.py`)
**Primary interface for voice-to-LLM interaction** - integrates STT and LLM with flexible conversation management:

- **🎙️ Dual Listening Modes**: Push-to-talk (SPACE) and continuous voice detection
- **🤖 Multi-API Support**: Gradio, LM Studio, OpenAI-compatible endpoints  
- **💭 Conversation History**: Optional context maintenance with interactive controls
- **🔧 Configurable STT**: Auto-selection, Whisper, or Google Speech Recognition
- **🧪 Test Integration**: Single-shot testing and interactive session modes

### LLMAgent (`code/llm_agent_linux_package/llm_agent.py`)
**Flexible LLM API client** with robust error handling and conversation management:

- **🌐 Universal API Support**: Works with Gradio, LM Studio, OpenAI, and custom endpoints
- **💭 History Management**: Optional conversation context with programmatic controls
- **🔄 Auto-Retry**: Built-in retry logic with exponential backoff
- **🛠️ Configurable**: Generation parameters, timeouts, and model selection
- **📊 Health Monitoring**: Connection testing and performance metrics

### MultiEngineSTT (`code/voice_processing/multi_engine_stt.py`)
**Robust speech-to-text** with automatic fallback and memory safety:

- **🎯 Auto-Fallback**: Whisper → Google Speech Recognition failover
- **🛡️ Memory Safety**: Prevents memory allocation issues
- **⚙️ Configurable**: Engine selection, audio parameters, and quality settings

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

### TranscriberAgent (Recommended)
The primary interface for voice-to-LLM interaction with both push-to-talk and continuous listening modes:

```bash
# Basic usage with push-to-talk mode
python test_transcriber_agent.py http://localhost:7860

# Continuous listening mode
python test_transcriber_agent.py http://localhost:1234 --soundmode continuous

# LM Studio with conversation history disabled
python test_transcriber_agent.py http://localhost:1234 --api-type lmstudio --no-history

# Single test interaction
python test_transcriber_agent.py http://localhost:7860 --test-only
```

**TranscriberAgent Features:**
- **🎙️ Dual Listening Modes**: Push-to-talk (SPACE key) or continuous (voice detection)
- **🤖 Multi-API Support**: Gradio, LM Studio, OpenAI-compatible endpoints
- **💭 Conversation History**: Maintain context across interactions (optional)
- **🔧 Configurable**: STT engine selection, recording duration, silence detection
- **🧪 Test Mode**: Single interaction testing

### Conversation History Management
Control conversation context and memory across interactions:

```bash
# Enable conversation history (default)
python test_transcriber_agent.py <url> --api-type lmstudio

# Disable conversation history for fresh interactions
python test_transcriber_agent.py <url> --api-type lmstudio --no-history

# Interactive history controls during session:
>>> history          # Check current status  
>>> history on       # Enable conversation memory
>>> history off      # Disable conversation memory
>>> clear            # Clear conversation history
>>> stats            # Show conversation statistics
```

**History Features:**
- **🔧 Programmatic Control**: Set via command-line or code
- **🎛️ Interactive Toggle**: Change during active sessions
- **📊 Conversation Stats**: Message counts, roles, timing
- **🗑️ Selective Clearing**: Reset without ending session
- **🌐 Universal Support**: Works with all API types (Gradio, LM Studio, OpenAI)

### Text-to-Speech (TTS) Features
Enhanced voice output with multi-engine support and interactive controls:

```bash
# Enable/disable TTS during session
>>> tts on/off         # Toggle text-to-speech
>>> tts test           # Test current TTS settings

# Voice configuration
>>> tts rate 200       # Set speech rate (50-400 WPM)
>>> tts volume 0.8     # Set volume (0.0-1.0)
>>> tts voice en-US-1  # Change voice (use 'tts voices' to see options)
>>> tts engine sapi    # Switch TTS engine

# Voice information
>>> tts voices         # List available voices for all engines
>>> tts config         # Show current TTS configuration
>>> tts info           # Show detailed engine information
```

**TTS Features:**
- **🎭 Multi-Engine Support**: Windows SAPI, Google TTS, eSpeak, Azure, Mozilla TTS
- **🎤 Voice Selection**: Choose from system and online voices
- **🔧 Real-time Configuration**: Adjust rate, volume, pitch during conversation
- **🔄 Auto-Fallback**: Seamless switching between TTS engines
- **🌐 Cross-Platform**: Works on Windows, Linux, and macOS
- **💾 File Output**: Save speech to audio files (optional)

### Legacy Voice Chat
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
