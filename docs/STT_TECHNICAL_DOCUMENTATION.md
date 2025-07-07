# Speech-to-Text Pipeline - Complete Method Documentation

## 📚 Table of Contents

1. [Core STT Classes](#core-stt-classes)
2. [Audio Processing Methods](#audio-processing-methods)
3. [Transcription Engines](#transcription-engines)
4. [API Integration](#api-integration)
5. [Chatbot Backends](#chatbot-backends)
6. [Memory Management](#memory-management)
7. [Testing & Diagnostics](#testing--diagnostics)
8. [Usage Examples](#usage-examples)

---

## 🎯 Core STT Classes

### `MultiEngineSTT` Class
**File:** `multi_engine_stt.py`

The main speech-to-text coordinator that manages multiple STT engines with automatic fallback.

#### Constructor
```python
stt = MultiEngineSTT()
```

**What it does:**
- Initializes all available STT engines (Whisper, Google Speech Recognition)
- Sets up automatic fallback system
- Configures memory monitoring
- Selects the best available engine as primary

#### Key Methods

##### `voice_to_text(duration: float = 5.0) -> str`
**Purpose:** Complete voice-to-text pipeline in one call

**Parameters:**
- `duration`: Recording duration in seconds (max 30s for safety)

**Returns:** Transcribed text as string

**What it does:**
1. Records audio from microphone
2. Applies memory safety checks
3. Transcribes using best available engine
4. Cleans up memory
5. Returns transcribed text

**Example:**
```python
stt = MultiEngineSTT()
text = stt.voice_to_text(duration=3.0)
print(f"You said: {text}")
```

##### `transcribe(audio_data: np.ndarray, sample_rate: int = 16000) -> str`
**Purpose:** Transcribe pre-recorded audio with engine fallback

**Parameters:**
- `audio_data`: NumPy array containing audio samples
- `sample_rate`: Audio sample rate in Hz

**Returns:** Transcribed text

**What it does:**
1. Tries preferred engine first (usually Whisper)
2. Falls back to secondary engines if primary fails
3. Handles all error cases gracefully
4. Returns best available transcription

---

## 🎤 Audio Processing Methods

### `record_audio(duration: float, sample_rate: int = 16000) -> np.ndarray`
**File:** `multi_engine_stt.py`

**Purpose:** Safely record audio with memory monitoring

**Parameters:**
- `duration`: Recording time (automatically capped at 30s)
- `sample_rate`: Audio sample rate (default 16kHz for Whisper)

**Safety Features:**
- Duration limits to prevent excessive memory usage
- Memory size estimation before recording
- Audio level analysis after recording
- Silence detection

**What it does:**
1. Validates recording parameters
2. Estimates memory requirements
3. Records audio using sounddevice
4. Analyzes audio properties (amplitude, RMS)
5. Returns normalized float32 audio array

**Example:**
```python
audio = stt.record_audio(duration=5.0)
# Audio is automatically validated and memory-safe
```

### `safe_record_audio(duration=3, samplerate=16000)`
**File:** `ultra_safe_voice.py`

**Purpose:** Ultra-conservative audio recording for memory-constrained environments

**Safety Measures:**
- Hard 5-second duration cap
- 16kHz sample rate limit
- Immediate dtype conversion to float32
- Blocking recording to ensure completion

---

## 🧠 Transcription Engines

### Whisper Engine Methods

#### `transcribe_with_whisper(audio_data: np.ndarray) -> str`
**File:** `multi_engine_stt.py`

**Purpose:** High-quality local transcription using OpenAI Whisper

**Memory Safety Features:**
- Pre-transcription memory checks
- Audio size validation (max 100MB)
- Memory usage monitoring during processing
- Conservative model settings to reduce memory

**Whisper Configuration:**
```python
result = model.transcribe(
    audio_data,
    fp16=False,              # Use FP32 for CPU compatibility
    language='en',           # English language
    verbose=False,           # Minimal output
    task='transcribe',       # Transcription task
    temperature=0.0,         # Deterministic output
    best_of=1,              # Single beam
    beam_size=1,            # Minimal beam search
    condition_on_previous_text=False,  # Reduce memory
    without_timestamps=True  # Faster processing
)
```

#### `ultra_safe_transcribe(audio_data)`
**File:** `ultra_safe_voice.py`

**Purpose:** Maximum safety transcription with extensive validation

**Validation Steps:**
1. Empty audio check
2. Memory size validation (5MB limit)
3. Audio length truncation (5 seconds max)
4. Data type normalization
5. Amplitude normalization
6. Silence detection
7. Minimal Whisper settings

### Google Speech Recognition Engine

#### `transcribe_with_google(audio_data: np.ndarray, sample_rate: int) -> str`
**File:** `multi_engine_stt.py`

**Purpose:** Online transcription using Google's Speech Recognition API

**What it does:**
1. Converts float32 audio to int16 for API compatibility
2. Creates proper AudioData object
3. Handles API errors gracefully
4. Provides fallback when Whisper fails

**Error Handling:**
- `UnknownValueError`: No speech detected
- `RequestError`: Network/API issues
- Generic exceptions: Catch-all error handling

---

## 🌐 API Integration

### `APIChatbot` Class
**File:** `api_chatbot.py`

**Purpose:** Connect STT pipeline to external chatbot APIs (Gradio, LM Studio)

#### Constructor Options
```python
# Gradio API
chatbot = APIChatbot(api_type="gradio", api_url="http://localhost:7860")

# LM Studio API
chatbot = APIChatbot(api_type="lmstudio", api_url="http://localhost:1234")
```

#### `call_gradio_api(message: str) -> str`
**Purpose:** Send text to Gradio-hosted chatbot

**API Format:**
```python
payload = {
    "data": [message],
    "session_hash": session_id
}
response = requests.post(f"{api_url}/api/predict", json=payload)
```

**What it does:**
1. Formats message for Gradio API
2. Handles session management
3. Parses response data
4. Manages timeouts and errors

#### `call_lmstudio_api(message: str) -> str`
**Purpose:** Send text to LM Studio local LLM

**API Format:**
```python
payload = {
    "model": "local-model",
    "messages": [{"role": "user", "content": message}],
    "temperature": 0.7,
    "max_tokens": 150
}
response = requests.post(f"{api_url}/v1/chat/completions", json=payload)
```

#### `voice_to_api_chat(duration: float = 4.0) -> dict`
**Purpose:** Complete voice → STT → API → response pipeline

**Process Flow:**
1. **Voice Input:** Record audio for specified duration
2. **STT Processing:** Transcribe using MultiEngineSTT
3. **API Call:** Send text to configured chatbot API
4. **Response Processing:** Parse and format API response
5. **Return Data:** Complete interaction data

**Return Format:**
```python
{
    "transcript": "user's spoken words",
    "response": "chatbot's response",
    "success": True,
    "timing": {
        "recording": 4.0,
        "transcription": 1.2,
        "api_call": 0.8,
        "total": 6.0
    }
}
```

---

## 🤖 Chatbot Backends

### `SimpleChatbot` Class
**File:** `chatbot_final.py`

**Purpose:** Offline rule-based chatbot for testing and fallback

#### `respond(message: str) -> str`
**What it does:**
1. Converts message to lowercase
2. Checks for keyword matches
3. Returns appropriate pre-defined responses
4. Handles unknown inputs gracefully

**Built-in Responses:**
- Greetings: "hello", "hi"
- Status: "how are you"
- Information: "time", "date", "weather"
- Help: "help"
- Goodbyes: "bye", "goodbye"

### `OllamaChatbot` Class
**File:** `chatbot_final.py`

**Purpose:** Local LLM integration via Ollama

#### `initialize() -> bool`
**What it does:**
1. Checks Ollama server availability
2. Retrieves available models
3. Selects default model
4. Validates connection

#### `respond(message: str) -> str`
**API Interaction:**
```python
payload = {
    "model": self.model,
    "prompt": f"User: {message}\nAssistant:",
    "stream": False
}
response = requests.post(f"{base_url}/api/generate", json=payload)
```

### `OpenAIChatbot` Class
**File:** `chatbot_final.py`

**Purpose:** OpenAI ChatGPT integration

#### `initialize() -> bool`
**What it does:**
1. Retrieves API key from environment
2. Initializes OpenAI client
3. Tests connection with minimal request
4. Handles authentication errors

---

## 💾 Memory Management

### Memory Safety Functions

#### `cleanup_memory()`
**File:** `voice_input.py`

**Purpose:** Force garbage collection to free memory

```python
def cleanup_memory():
    gc.collect()
    print("🧹 Memory cleanup performed")
```

#### `_get_memory_usage() -> float`
**File:** `multi_engine_stt.py`

**Purpose:** Monitor current process memory usage

```python
def _get_memory_usage(self):
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB
```

### Memory Safety Checks

#### Audio Recording Limits
- **Duration:** Maximum 30 seconds per recording
- **Sample Rate:** Capped at 16kHz for efficiency
- **Memory Estimation:** Pre-allocation size calculation
- **Size Validation:** Reject recordings over 100MB

#### Transcription Limits
- **Audio Length:** Truncate to 30 seconds max
- **Memory Threshold:** 100MB warning, 5MB ultra-safe limit
- **Model Settings:** Minimal beam search and processing options

---

## 🧪 Testing & Diagnostics

### Comprehensive Test Suite

#### `test_complete_pipeline.py`
**Purpose:** Full system validation

**Test Categories:**
1. **Import Tests:** Verify all dependencies
2. **Memory Tests:** Check memory monitoring
3. **Audio Tests:** Validate microphone access
4. **Whisper Tests:** Model loading and transcription
5. **Google STT Tests:** Online API functionality
6. **Integration Tests:** End-to-end pipeline

#### `test_google_stt.py`
**Purpose:** Isolated Google Speech Recognition testing

**Test Process:**
1. Initialize recognizer
2. Create synthetic audio
3. Test API connectivity
4. Validate error handling

#### `system_diagnostic.py`
**Purpose:** System capability analysis

**Diagnostics:**
- Memory availability
- Audio device enumeration
- CPU capabilities
- Python environment validation

### Debug Utilities

#### `debug_audio.py`
**Purpose:** Audio system debugging

**Features:**
- Device listing and capabilities
- Recording quality analysis
- Level monitoring
- Format validation

#### `ultra_safe_voice.py`
**Purpose:** Minimal memory-safe testing

**Use Cases:**
- Memory-constrained environments
- Debugging memory issues
- Baseline functionality validation

---

## 🚀 Usage Examples

### Basic Voice-to-Text
```python
from voice_processing.multi_engine_stt import MultiEngineSTT

# Initialize STT system
stt = MultiEngineSTT()

# Record and transcribe
text = stt.voice_to_text(duration=5.0)
print(f"Transcription: {text}")
```

### API Chatbot Integration
```python
from voice_processing.api_chatbot import APIChatbot

# Connect to Gradio API
chatbot = APIChatbot(
    api_type="gradio",
    api_url="http://localhost:7860"
)

# Voice interaction
result = chatbot.voice_to_api_chat(duration=4.0)
print(f"You: {result['transcript']}")
print(f"Bot: {result['response']}")
```

### Production Chatbot
```python
from voice_processing.chatbot_final import VoiceChatbot

# Initialize with backend
chatbot = VoiceChatbot(backend_name="simple")
chatbot.initialize()

# Start interactive loop
chatbot.chat_loop()
```

### Memory-Safe Testing
```python
from voice_processing.ultra_safe_voice import test_single_recording

# Ultra-safe voice test
test_single_recording()
```

### Custom Audio Processing
```python
import sounddevice as sd
import numpy as np
from voice_processing.multi_engine_stt import MultiEngineSTT

# Custom recording
audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()

# Process with STT
stt = MultiEngineSTT()
text = stt.transcribe(audio.flatten())
print(f"Result: {text}")
```

---

## 🔧 Configuration Options

### Environment Variables
```bash
# OpenAI integration
export OPENAI_API_KEY="your-api-key"

# Custom API endpoints
export GRADIO_API_URL="http://localhost:7860"
export LMSTUDIO_API_URL="http://localhost:1234"
```

### Audio Settings
```python
# Default configuration
SAMPLE_RATE = 16000      # Whisper-optimized
MAX_DURATION = 30.0      # Safety limit
MAX_MEMORY_MB = 100      # Memory threshold
AUDIO_CHANNELS = 1       # Mono recording
DTYPE = np.float32       # Internal format
```

### API Configuration
```python
# Gradio API settings
GRADIO_TIMEOUT = 30      # Request timeout
GRADIO_SESSION_HASH = "random-uuid"

# LM Studio settings
LMSTUDIO_MODEL = "local-model"
LMSTUDIO_TEMPERATURE = 0.7
LMSTUDIO_MAX_TOKENS = 150
```

---

## 🎯 Key Learning Points

### Architecture Design
- **Multi-engine approach** provides reliability through redundancy
- **Memory safety** prevents system crashes from excessive allocation
- **API abstraction** allows easy integration with various chatbot backends
- **Error handling** ensures graceful degradation when components fail

### Performance Optimization
- **Minimal model settings** reduce memory usage
- **Audio preprocessing** optimizes for each engine's requirements
- **Caching strategies** avoid repeated model loading
- **Resource monitoring** prevents system overload

### Production Readiness
- **Comprehensive testing** validates all components
- **Logging and monitoring** aid in debugging and maintenance
- **Configuration management** allows easy customization
- **Documentation** enables team collaboration and maintenance

This pipeline demonstrates professional software development practices including modularity, error handling, testing, documentation, and performance optimization.

## 🎯 Core Architecture

The STT pipeline follows a modular architecture:

```
Audio Input → Recording → Processing → Transcription → Chatbot API → Response
```

## 📁 File Structure and Purpose

### Core Files

- **`ultra_safe_voice.py`** - Minimal, memory-safe STT implementation
- **`multi_engine_stt.py`** - Production-ready multi-engine STT system
- **`chatbot_final.py`** - Complete chatbot integration
- **`voice_input.py`** - Original STT implementation with detailed memory safety

### Test and Debug Files

- **`test_complete_pipeline.py`** - Comprehensive test suite
- **`debug_audio.py`** - Audio debugging utilities
- **`system_diagnostic.py`** - System resource monitoring

---

## 🔧 Method Documentation

### 1. Audio Recording Methods

#### `safe_record_audio(duration=3, samplerate=16000)`

**Purpose:** Records audio from the microphone with safety constraints

**Parameters:**
- `duration` (int): Recording duration in seconds (max 5s for safety)
- `samplerate` (int): Audio sample rate in Hz (capped at 16kHz)

**Returns:** 
- `numpy.ndarray`: 1D float32 array containing audio samples

**Safety Features:**
- Duration capping to prevent excessive memory usage
- Sample rate limiting for Whisper compatibility
- Automatic conversion to 1D array
- Exception handling for recording failures

**Example Usage:**
```python
audio_data = safe_record_audio(duration=3, samplerate=16000)
# Records 3 seconds of audio at 16kHz
```

**How It Works:**
1. Validates and caps input parameters for safety
2. Uses `sounddevice.rec()` to capture audio from default microphone
3. Records in float32 format to avoid conversion overhead
4. Blocks until recording is complete (`blocking=True`)
5. Flattens multi-dimensional arrays to 1D for processing

---

### 2. Audio Processing Methods

#### `ultra_safe_transcribe(audio_data)`

**Purpose:** Transcribes audio data using Whisper with maximum safety checks

**Parameters:**
- `audio_data` (numpy.ndarray): Audio samples as float32 array

**Returns:**
- `str`: Transcribed text or error message

**Safety Features:**
- Memory size validation (max 5MB)
- Audio length truncation (max 5 seconds)
- Data type normalization
- Silence detection
- Comprehensive error handling

**Processing Steps:**

1. **Validation Phase:**
   ```python
   # Check if audio exists
   if len(audio_data) == 0:
       return "[No audio recorded]"
   
   # Memory safety check
   size_mb = audio_data.nbytes / (1024 * 1024)
   if size_mb > 5:
       return "[Audio too large]"
   ```

2. **Normalization Phase:**
   ```python
   # Ensure float32 format
   if audio_data.dtype != np.float32:
       audio_data = audio_data.astype(np.float32)
   
   # Normalize amplitude to [-1, 1] range
   max_val = np.max(np.abs(audio_data))
   if max_val > 1.0:
       audio_data = audio_data / max_val
   ```

3. **Quality Check:**
   ```python
   # Calculate audio energy to detect silence
   energy = np.mean(np.abs(audio_data))
   if energy < 1e-4:
       return "[Silent audio]"
   ```

4. **Transcription Phase:**
   ```python
   result = model.transcribe(
       audio_data,
       fp16=False,              # Use FP32 for CPU compatibility
       language='en',           # English language
       temperature=0.0,         # Deterministic output
       beam_size=1,            # Single beam for speed
       condition_on_previous_text=False  # Reduce memory usage
   )
   ```

---

### 3. Multi-Engine STT Class

#### `class MultiEngineSTT`

**Purpose:** Provides robust STT with multiple engine fallback

**Engines Supported:**
- **Whisper**: Local, high-quality transcription
- **Google Speech Recognition**: Online, reliable fallback

#### `__init__(self)`

**Purpose:** Initializes all available STT engines

**Process:**
1. Attempts to load Whisper model with memory monitoring
2. Initializes Google Speech Recognition as fallback
3. Sets preferred engine based on availability
4. Reports available engines and preferred choice

#### `_init_whisper(self)`

**Purpose:** Safely initializes Whisper with memory monitoring

**Safety Features:**
- Memory usage tracking during model loading
- Conservative memory limits (warns if >1GB)
- Error handling for initialization failures

**Implementation:**
```python
def _init_whisper(self):
    try:
        initial_memory = self._get_memory_usage()
        model = whisper.load_model("tiny")
        after_memory = self._get_memory_usage()
        
        model_size = after_memory - initial_memory
        if model_size > 1000:  # Warn if >1GB
            print("⚠️ Whisper using excessive memory")
            
        return model
    except Exception as e:
        print(f"Failed to initialize Whisper: {e}")
        return None
```

#### `record_audio(self, duration=5.0, sample_rate=16000)`

**Purpose:** Records audio with comprehensive safety checks

**Features:**
- Duration capping (max 30 seconds)
- Memory estimation before recording
- Audio quality analysis (amplitude, RMS levels)
- User confirmation for large recordings

#### `transcribe_with_whisper(self, audio_data)`

**Purpose:** Transcribes audio using Whisper engine

**Memory Management:**
- Pre-transcription memory check
- Memory usage tracking during transcription
- Post-transcription cleanup

#### `transcribe_with_google(self, audio_data, sample_rate=16000)`

**Purpose:** Transcribes audio using Google Speech Recognition

**Process:**
1. Converts float32 audio to int16 format required by Google API
2. Creates AudioData object for speech_recognition library
3. Calls Google's online transcription service
4. Handles network errors and unknown speech gracefully

**Data Conversion:**
```python
# Convert float32 to int16 for Google API
audio_int16 = (audio_data * 32767).astype(np.int16)

# Create AudioData object
audio_data_sr = sr.AudioData(
    audio_int16.tobytes(),
    sample_rate,
    2  # 2 bytes per sample (int16)
)
```

#### `transcribe(self, audio_data, sample_rate=16000)`

**Purpose:** Main transcription method with automatic fallback

**Fallback Logic:**
1. Attempts transcription with preferred engine
2. If preferred engine fails, tries remaining engines
3. Returns first successful transcription
4. Reports which engine succeeded

**Implementation:**
```python
engines_to_try = [self.preferred_engine] + [
    engine for engine in self.engines.keys() 
    if engine != self.preferred_engine
]

for engine_name in engines_to_try:
    try:
        if engine_name == 'whisper':
            result = self.transcribe_with_whisper(audio_data)
        elif engine_name == 'google':
            result = self.transcribe_with_google(audio_data, sample_rate)
        
        if result and result.strip():
            return result
    except Exception as e:
        continue  # Try next engine
```

---

### 4. Memory Management Methods

#### `cleanup_memory()`

**Purpose:** Forces garbage collection to free memory

**Usage:** Called before and after major operations

```python
def cleanup_memory():
    gc.collect()
    print("🧹 Memory cleanup performed")
```

#### `_get_memory_usage(self)`

**Purpose:** Returns current process memory usage in MB

```python
def _get_memory_usage(self):
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024
```

---

### 5. Error Handling Patterns

#### Exception Hierarchy

1. **Recording Errors**: Microphone access, hardware issues
2. **Memory Errors**: Insufficient RAM, allocation failures  
3. **Transcription Errors**: Model failures, network issues
4. **Format Errors**: Invalid audio data, conversion issues

#### Error Recovery Strategies

```python
try:
    # Primary operation
    result = primary_method()
except SpecificError as e:
    # Specific handling
    result = fallback_method()
except Exception as e:
    # Generic handling
    result = safe_default_value()
finally:
    # Cleanup
    cleanup_resources()
```

---

### 6. Configuration Parameters

#### Audio Settings

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `duration` | 3-5s | 1-30s | Recording length |
| `sample_rate` | 16000 Hz | 8000-48000 Hz | Audio quality |
| `channels` | 1 (mono) | 1-2 | Audio channels |
| `dtype` | float32 | int16/float32 | Data format |

#### Memory Limits

| Resource | Limit | Purpose |
|----------|-------|---------|
| Audio size | 5-100 MB | Prevent memory overflow |
| Duration | 30s max | Reasonable processing time |
| Model memory | 1GB warning | Monitor Whisper usage |

#### Whisper Settings

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `fp16` | False | CPU compatibility |
| `beam_size` | 1 | Speed optimization |
| `temperature` | 0.0 | Deterministic output |
| `language` | 'en' | English optimization |

---

### 7. Testing and Debugging

#### Test Methods

- **`test_imports()`**: Verifies all dependencies are available
- **`test_whisper_model()`**: Tests Whisper initialization and basic transcription
- **`test_google_stt()`**: Tests Google Speech Recognition
- **`test_audio_devices()`**: Verifies microphone access
- **`test_memory_usage()`**: Monitors system resources

#### Debug Utilities

- **Audio level analysis**: RMS, peak amplitude measurement
- **Memory tracking**: Real-time usage monitoring  
- **Device enumeration**: Lists available audio hardware
- **Format validation**: Checks audio data integrity

---

## 🎓 Learning Outcomes

After studying this code, you'll understand:

1. **Audio Processing**: How to capture and process audio data safely
2. **Memory Management**: Techniques for preventing memory leaks and overflow
3. **Error Handling**: Robust fallback strategies for production systems
4. **API Integration**: How to connect STT with various services
5. **Testing**: Comprehensive validation of complex systems
6. **Cross-platform Development**: Windows/Linux compatibility

---

## 🚀 Next Steps

1. Study the method implementations in detail
2. Run the test suite to see methods in action
3. Experiment with different parameters
4. Integrate with your preferred chatbot APIs
5. Extend with additional STT engines or features

This documentation provides the foundation for understanding and extending the STT pipeline for your specific needs.
