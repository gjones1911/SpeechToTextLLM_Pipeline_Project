# API Integration Guide - Gradio & LM Studio

This guide explains how to integrate the Speech-to-Text pipeline with external chatbot APIs, specifically Gradio and LM Studio.

## 🎯 Quick Start

### 1. Test with Mock API Server
First, test the integration with our mock server:

```bash
# Terminal 1: Start mock API server
python code/voice_processing/test_api_server.py

# Terminal 2: Test voice-to-API integration
python code/voice_processing/api_chatbot.py --gradio-url "http://localhost:7860"
```

### 2. Connect to Real Gradio API
```bash
# Replace with your actual Gradio URL
python code/voice_processing/api_chatbot.py --gradio-url "https://your-space.hf.space"
```

### 3. Connect to LM Studio
```bash
# Start LM Studio with local server, then:
python code/voice_processing/api_chatbot.py --lmstudio-url "http://localhost:1234"
```

## 🌐 API Types Supported

### Gradio API Integration

**When to use:** When you have a chatbot deployed on Hugging Face Spaces or a custom Gradio interface.

**API Format:**
```python
# Request
POST /api/predict
{
    "data": ["Your message here"],
    "session_hash": "unique-session-id"
}

# Response
{
    "data": ["Bot response here"],
    "duration": 1.2
}
```

**Example Code:**
```python
from voice_processing.api_chatbot import APIChatbot

chatbot = APIChatbot(
    api_type="gradio",
    api_url="http://localhost:7860"
)

# Voice interaction
result = chatbot.voice_to_api_chat(duration=4.0)
print(f"You said: {result['transcript']}")
print(f"Bot replied: {result['response']}")
```

### LM Studio API Integration

**When to use:** When you have LM Studio running locally with a language model.

**API Format:**
```python
# Request
POST /v1/chat/completions
{
    "model": "local-model",
    "messages": [
        {"role": "user", "content": "Your message"}
    ],
    "temperature": 0.7,
    "max_tokens": 150
}

# Response
{
    "choices": [
        {
            "message": {
                "content": "Bot response here"
            }
        }
    ]
}
```

**Example Code:**
```python
from voice_processing.api_chatbot import APIChatbot

chatbot = APIChatbot(
    api_type="lmstudio",
    api_url="http://localhost:1234"
)

# Voice interaction
result = chatbot.voice_to_api_chat(duration=4.0)
print(f"You said: {result['transcript']}")
print(f"Bot replied: {result['response']}")
```

## 🔧 Configuration Options

### Environment Variables
```bash
# Set default API URLs
export GRADIO_API_URL="http://localhost:7860"
export LMSTUDIO_API_URL="http://localhost:1234"

# Set API timeouts
export API_TIMEOUT=30
```

### Command Line Options
```bash
# Gradio
python api_chatbot.py --gradio-url "http://localhost:7860"
python api_chatbot.py --gradio-url "https://your-space.hf.space"

# LM Studio
python api_chatbot.py --lmstudio-url "http://localhost:1234"

# Custom timeout
python api_chatbot.py --gradio-url "http://localhost:7860" --timeout 45
```

### Programmatic Configuration
```python
# Custom configuration
chatbot = APIChatbot(
    api_type="gradio",
    api_url="http://localhost:7860",
    timeout=30,
    session_id="custom-session-id"
)

# Alternative initialization
chatbot = APIChatbot.from_config({
    "api_type": "lmstudio",
    "api_url": "http://localhost:1234",
    "model": "custom-model-name",
    "temperature": 0.5,
    "max_tokens": 200
})
```

### `GradioAPI` Class

**Purpose:** Connects to Gradio-based chatbot interfaces

**Key Methods:**

#### `test_connection() -> bool`
Tests if the Gradio API is accessible
```python
gradio_api = GradioAPI("http://localhost:7860")
if gradio_api.test_connection():
    print("Gradio is ready!")
```

#### `send_message(message: str) -> Optional[str]`
Sends a message to the Gradio chatbot and returns the response
```python
response = gradio_api.send_message("Hello, how are you?")
if response:
    print(f"Bot said: {response}")
```

**Gradio API Format:**
```python
# Request format
payload = {
    "data": [message],
    "fn_index": 0  # Usually 0 for chat function
}

# Response format
{
    "data": ["Bot response here"],
    "duration": 1.23
}
```

### `LMStudioAPI` Class

**Purpose:** Connects to LM Studio's OpenAI-compatible API

**Key Methods:**

#### `test_connection() -> bool`
Tests LM Studio API and lists available models
```python
lm_api = LMStudioAPI("http://localhost:1234")
if lm_api.test_connection():
    print("LM Studio is ready!")
```

#### `send_message(message: str) -> Optional[str]`
Sends message using OpenAI-compatible chat completion format
```python
response = lm_api.send_message("Tell me a joke")
if response:
    print(f"Bot response: {response}")
```

**LM Studio API Format:**
```python
# Request format (OpenAI-compatible)
payload = {
    "model": "local-model",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": message}
    ],
    "max_tokens": 150,
    "temperature": 0.7
}

# Response format
{
    "choices": [
        {
            "message": {
                "content": "Bot response here"
            }
        }
    ]
}
```

### `VoiceAPIChat` Class

**Purpose:** Main coordinator that combines STT with API chatbots

**Key Methods:**

#### `transcribe_speech(duration: float) -> Tuple[bool, str]`
Records and transcribes speech
```python
success, transcript = voice_chat.transcribe_speech(duration=4.0)
if success:
    print(f"You said: {transcript}")
```

#### `send_to_chatbot(message: str) -> Optional[str]`
Sends message to available APIs with automatic fallback
```python
response = voice_chat.send_to_chatbot("Hello there!")
if response:
    print(f"Bot replied: {response}")
```

#### `single_voice_query(duration: float) -> Dict[str, Any]`
Complete voice-to-response pipeline in one call
```python
result = voice_chat.single_voice_query(duration=4.0)
print(f"Success: {result['success']}")
print(f"Transcript: {result['transcript']}")
print(f"Response: {result['response']}")
```

## 🎛️ Configuration Options

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--gradio-url` | `http://localhost:7860` | Gradio API endpoint |
| `--lm-studio-url` | `http://localhost:1234` | LM Studio API endpoint |
| `--mode` | `interactive` | `interactive` or `single` query mode |
| `--duration` | `4.0` | Recording duration in seconds |

### Environment Variables

You can also set these via environment variables:
```bash
export GRADIO_API_URL="http://your-server:7860"
export LM_STUDIO_API_URL="http://localhost:1234"
```

## 🔄 API Fallback Strategy

The system tries APIs in this order:
1. **Gradio API** (primary) - if configured and accessible
2. **LM Studio API** (fallback) - if Gradio fails
3. **Error handling** - graceful failure messages

```python
# Automatic fallback logic
for api_name, api_client in api_attempts:
    try:
        response = api_client.send_message(message)
        if response:
            return response  # Success!
    except Exception:
        continue  # Try next API
```

## 🧪 Testing Your Integration

### 1. Test Individual Components

```bash
# Test STT only
python code/voice_processing/ultra_safe_voice.py

# Test API server only
python code/voice_processing/test_api_server.py

# Test specific API connection
python -c "
from code.voice_processing.api_chatbot import GradioAPI
api = GradioAPI('http://localhost:7860')
print('Connected:', api.test_connection())
"
```

### 2. Test Complete Pipeline

```bash
# Interactive mode
python code/voice_processing/api_chatbot.py --mode interactive

# Single query mode
python code/voice_processing/api_chatbot.py --mode single --duration 3
```

### 3. Test Error Handling

```bash
# Test with invalid URLs to see fallback behavior
python code/voice_processing/api_chatbot.py \
  --gradio-url "http://invalid:9999" \
  --lm-studio-url "http://localhost:1234"
```

## 🔧 Custom API Integration

### Adding Your Own API

To integrate a custom API, create a new class following this pattern:

```python
class YourCustomAPI:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_connection(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def send_message(self, message: str) -> Optional[str]:
        try:
            payload = {"message": message}  # Your API format
            response = self.session.post(
                f"{self.base_url}/chat",
                json=payload
            )
            if response.status_code == 200:
                return response.json()["reply"]  # Extract response
            return None
        except:
            return None
```

Then add it to the `VoiceAPIChat` class initialization.

## 📊 Response Format

The `single_voice_query()` method returns a structured result:

```python
{
    "success": True,                    # Whether the query succeeded
    "transcript": "Hello how are you",  # What the user said
    "response": "I'm doing great!",     # Bot's response
    "api_used": "gradio",              # Which API responded
    "error": None,                     # Error message if failed
    "timestamp": 1672531200.0          # Unix timestamp
}
```

## 🚨 Error Handling

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "No STT engines available" | Missing audio dependencies | Install `sounddevice`, `openai-whisper` |
| "No chatbot APIs available" | API servers not running | Start Gradio/LM Studio servers |
| "Speech transcription failed" | Audio recording issues | Check microphone permissions |
| "API connection failed" | Network/server issues | Verify API URLs and server status |

### Debugging Tips

1. **Enable verbose logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Test each component separately:**
   - Audio recording: `debug_audio.py`
   - STT engines: `test_complete_pipeline.py`
   - API connectivity: Use curl or Postman

3. **Check server logs:**
   - Gradio: Check console output
   - LM Studio: Check application logs

## 🎯 Production Deployment

### Security Considerations

1. **API Authentication:**
   ```python
   # Add API keys to requests
   headers = {"Authorization": f"Bearer {api_key}"}
   response = self.session.post(url, headers=headers, json=payload)
   ```

2. **Input Validation:**
   ```python
   # Sanitize user input
   message = message.strip()[:500]  # Limit length
   if not message or len(message) < 2:
       return "Message too short"
   ```

3. **Rate Limiting:**
   ```python
   import time
   self.last_request = time.time()
   # Add delays between requests
   ```

### Performance Optimization

1. **Connection Pooling:**
   ```python
   # Reuse HTTP connections
   self.session = requests.Session()
   ```

2. **Async Processing:**
   ```python
   import asyncio
   # Use async/await for multiple API calls
   ```

3. **Caching:**
   ```python
   # Cache common responses
   response_cache = {}
   ```

## 🎉 Ready to Use!

Your voice-to-API chatbot system is now ready for:

- ✅ **Gradio Integration** - Connect to Gradio-based chatbots
- ✅ **LM Studio Integration** - Use local LLMs via API
- ✅ **Automatic Fallback** - Robust error handling
- ✅ **Voice Input** - Complete STT pipeline
- ✅ **Flexible Configuration** - Command-line and environment options
- ✅ **Testing Framework** - Comprehensive test utilities

Start with the test server, then connect to your real APIs!
