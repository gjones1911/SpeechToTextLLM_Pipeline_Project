# Voice Processing Module

The core speech-to-text and API integration module for the Speech-to-Text LLM Pipeline Project.

## 🎯 Module Overview

This module provides robust voice processing components used by the TranscriberAgent and legacy applications. It includes advanced STT engines, API testing tools, and comprehensive validation systems.

## 📁 Files

### 🎙️ Core Components

#### **`multi_engine_stt.py`** ⭐ Primary STT Engine

**Purpose**: Robust speech-to-text engine with automatic fallback - **Used by TranscriberAgent**

**Features**:

- OpenAI Whisper (primary, local, high quality)
- Google Speech Recognition (fallback, online, reliable)
- Memory-safe audio processing (fixed 30GB memory bug)
- Cross-platform audio recording
- Automatic engine switching on failure
- Integration with TranscriberAgent conversation system

**Usage**:

```python
from multi_engine_stt import MultiEngineSTT

stt = MultiEngineSTT()
text = stt.voice_to_text(duration=4.0)
print(f"You said: {text}")
```

**Primary Integration**: Used by `TranscriberAgent` in `../transcriber_test_script.py`

#### **`api_chatbot.py`** 📜 Legacy Application

**Purpose**: Original voice-to-API chatbot system (superseded by TranscriberAgent)

**Features**:

- Voice input recording and transcription
- Gradio and LM Studio API integration
- Automatic fallback between APIs and STT engines
- Interactive chat sessions and single queries
- Comprehensive error handling and recovery

**Note**: For new projects, use `TranscriberAgent` instead:

```bash
# Modern approach (recommended)
python test_transcriber_agent.py <url> --api-type lmstudio

# Legacy approach (still functional)
python code/voice_processing/api_chatbot.py --lm-studio-url <url>
```

### 🧪 Development Tools

#### **`test_api_server.py`** ⭐ Development Server

**Purpose**: Mock Gradio-compatible API server for testing

**Features**:

- Gradio-style `/api/predict` endpoint
- Conversation history tracking
- Health monitoring endpoints
- Realistic chatbot response simulation

**Usage**:

```bash
# Start development server
python test_api_server.py
# Server runs on http://localhost:7860

# Test with curl
curl -X POST http://localhost:7860/api/predict \
  -H "Content-Type: application/json" \
  -d '{"data": ["Hello!"], "session_hash": "test"}'
```

#### **`test_complete_pipeline.py`** ⭐ Test Suite

**Purpose**: Comprehensive validation of the entire system

**Features**:

- Import and dependency validation
- STT engine functionality testing
- Memory usage monitoring
- API connectivity verification
- Full pipeline integration tests

**Usage**:

```bash
# Run complete test suite
python test_complete_pipeline.py

# Output shows all test results and system status
```

## 🚀 Quick Start

### 1. Test the Complete System

```bash
# Terminal 1: Start mock API server
python test_api_server.py

# Terminal 2: Start voice chat
python api_chatbot.py --gradio-url "http://localhost:7860"

# Terminal 3: Run validation tests
python test_complete_pipeline.py
```

### 2. Production Usage

```bash
# Connect to real Gradio API
python api_chatbot.py --gradio-url "https://your-space.hf.space"

# Connect to LM Studio
python api_chatbot.py --lm-studio-url "http://localhost:1234"
```

### 3. Programmatic Integration

```python
from api_chatbot import VoiceAPIChat

# Initialize voice chat system
voice_chat = VoiceAPIChat(
    gradio_url="http://localhost:7860",
    lm_studio_url="http://localhost:1234"
)

# Single voice query
result = voice_chat.single_voice_query(duration=4.0)
print(f"User: {result['transcript']}")
print(f"Bot: {result['response']}")

# Interactive session
voice_chat.voice_chat_session()
```

## 🛡️ Key Safety Features

### Memory Safety

- **Fixed**: 30GB memory allocation bug in Whisper
- **Monitoring**: Real-time memory usage tracking
- **Limits**: Configurable memory thresholds
- **Cleanup**: Automatic garbage collection

### Error Recovery

- **STT Fallback**: Whisper → Google Speech Recognition
- **API Fallback**: Gradio → LM Studio → Offline mode
- **Network Resilience**: Automatic retry with backoff
- **Graceful Degradation**: System continues with reduced functionality

### Cross-Platform Support

- **Windows**: Native support with Git Bash
- **Linux**: Native support with bash
- **Audio**: Cross-platform microphone access
- **Dependencies**: Platform-specific package handling

## 📊 Performance Characteristics

### Audio Processing

- **Sample Rate**: 16 kHz (Whisper optimized)
- **Format**: 32-bit float (processing), 16-bit int (storage)
- **Latency**: ~2-5 seconds for 4-second recordings
- **Memory**: <200 MB total system usage

### STT Engines

- **Whisper**: 90-95% accuracy, ~3-4 seconds processing
- **Google**: 85-90% accuracy, ~1-2 seconds processing
- **Fallback Time**: <1 second detection and switching

### API Integration

- **Response Time**: 1-3 seconds (local APIs)
- **Timeout**: 30 seconds (configurable)
- **Fallback**: <2 seconds detection and switching

## 🔧 Configuration

### Environment Variables

```bash
export GRADIO_API_URL="http://localhost:7860"
export LMSTUDIO_API_URL="http://localhost:1234"
export STT_ENGINE="whisper"  # or "google"
export AUDIO_DURATION="4.0"
```

### Command Line Options

```bash
# API endpoints
--gradio-url URL          # Gradio API URL
--lm-studio-url URL       # LM Studio API URL

# Voice settings
--duration SECONDS        # Recording duration (1-30 seconds)
--mode MODE              # "interactive" or "single"

# Engine preferences
--stt-engine ENGINE      # "whisper", "google", or "auto"
--timeout SECONDS        # API timeout (default: 30)
```

## 🚨 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "No microphone found" | Check audio permissions and device connections |
| "STT engines failed" | Verify internet for Google STT, restart for Whisper |
| "API connection failed" | Check server status and URL correctness |
| "Memory error" | Reduce recording duration, restart application |

### Debug Mode

```bash
# Enable verbose logging
python api_chatbot.py --gradio-url "http://localhost:7860" --debug

# Check system status
python test_complete_pipeline.py --verbose
```

## 📚 Related Documentation

- **[Main README](../../README.md)**: Complete project overview and setup
- **[Technical Documentation](../../STT_TECHNICAL_DOCUMENTATION.md)**: Detailed method explanations
- **[API Integration Guide](../../API_INTEGRATION_GUIDE.md)**: Complete API setup instructions
- **[Completion Summary](../../COMPLETION_SUMMARY.md)**: Project history and achievements

---

**Ready for production use!** 🎙️🤖