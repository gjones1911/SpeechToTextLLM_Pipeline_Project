#!/usr/bin/env python3
"""
Robust Text-to-Speech Pipeline with Multiple Engine Support

This module provides a reliable TTS pipeline that can fallback between:
1. Windows SAPI (Windows native, system voices)
2. Festival (cross-platform, open source)
3. eSpeak (cross-platform, lightweight)
4. Azure Speech Services (online, enterprise-grade)
5. Google Text-to-Speech (online, high quality)
6. OpenTTS/Mozilla TTS (local, neural voices)  ***
7. Elevenlabs (online, "high quality", not free!!!!!)

Features:
- Cross-platform engine detection and initialization
- Automatic engine fallback
- Voice selection and configuration
- Speed, pitch, and volume controls
- Audio output to speakers or file
- Memory usage monitoring
- Robust error handling
"""

import os
import sys
import platform
import tempfile
import time
import gc
import psutil
import subprocess
import threading
from typing import Optional, Dict, Any, List, Tuple
import warnings
from pathlib import Path
from TTS.api import TTS
import torch
from dotenv import dotenv_values
import os
import tempfile
from pydub import AudioSegment
import httpx

# Check if a CUDA-compatible GPU is available
gpu_available = torch.cuda.is_available()

# PULL my environment variables (API keys so I can use them)
config = dotenv_values("../env/env_config")

# Set them as environment variables
for key, value in config.items():
    os.environ[key] = value

# # Now you can access them via os.environ
# print("ELEVEN_LABS_KEY: ", os.environ.get("ELEVEN_LABS_API_KEY"))


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class ModelOptions:
    mozilla_models = [
        # 🗣️ Single-speaker English
        "tts_models/en/ljspeech/tacotron2-DDC",
        "tts_models/en/ljspeech/tacotron2-DDC_ph",
        "tts_models/en/ljspeech/glow-tts",
        "tts_models/en/ljspeech/vits",
        "tts_models/en/ljspeech/fast_pitch",
        "tts_models/en/jenny/jenny",  # lightweight, good for faster demos
    
        # 🧑‍🤝‍🧑 Multi-speaker English
        "tts_models/en/vctk/vits",
        "tts_models/en/vctk/tacotron2-DDC",
        "tts_models/en/vctk/fast_pitch",
        
        # 🌍 Multilingual / Multi-speaker
        "tts_models/multilingual/multi-dataset/your_tts",
        "tts_models/multilingual/multi-dataset/xtts_v2",  # best for zero-shot speaker cloning
        "tts_models/multilingual/multi-dataset/bark",     # bark TTS port (experimental)
    
        # 🌐 Other languages (examples)
        "tts_models/de/thorsten/tacotron2-DCA",   # German
        "tts_models/fr/siwis/glow-tts",           # French
        "tts_models/es/mai/tacotron2-DDC",        # Spanish
        "tts_models/ja/kokoro/tacotron2-DDC",     # Japanese
    
        # 🧪 Experimental / Niche
        "tts_models/en/ljspeech/tacotron2-DDC-GST",  # with Global Style Tokens
        "tts_models/en/ljspeech/transformer-TTS",
    ]


class MultiEngineTTS:
    """Text-to-Speech engine with multiple backend support"""
    
    def __init__(self, preferred_engine: Optional[str] = None, model_selection=5, **kwargs):
        """
        Initialize TTS with automatic engine detection
        
        Args:
            preferred_engine: Preferred engine name ('sapi', 'festival', 'espeak', 'azure', 'google', 'mozilla')
        """
        self.engines = {}
        self.preferred_engine = preferred_engine
        self.system_platform = platform.system().lower()
        self.voices = {}
        self.default_config = {
            'rate': 200,      # Words per minute
            'volume': 0.8,    # Volume level (0.0 to 1.0)
            'pitch': 0,       # Pitch adjustment (-50 to +50)
            'voice': None     # Voice ID (None = default)
        }
        self.current_config = self.default_config.copy()
        self.model_options = ModelOptions()
        self.selected_model = model_selection
        self.output_file = ""
        print(f"🎭 Initializing MultiEngineTTS on {self.system_platform}")
        self._init_engines()
        
    def _init_engines(self):
        """Initialize available TTS engines based on platform"""
        
        # 1. Platform-specific engines first
        if self.system_platform == 'windows':
            self._init_sapi()
        elif self.system_platform in ['linux', 'darwin']:
            self._init_festival()
            
        # 2. Cross-platform engines
        self._init_espeak()
        self._init_gTTS()
        
        # 3. Optional cloud engines
        self._init_azure_tts()
        self._init_eleven_labs()
        
        # 4. Neural/Local engines
        self._init_mozilla_tts()
        
        
        if not self.engines:
            raise RuntimeError("No TTS engines could be initialized!")
            
        # Set preferred engine if not specified
        if not self.preferred_engine or self.preferred_engine not in self.engines:
            if self.system_platform == 'windows' and 'sapi' in self.engines:
                self.preferred_engine = 'sapi'
            elif 'espeak' in self.engines:
                self.preferred_engine = 'espeak'
            elif 'gtts' in self.engines:
                self.preferred_engine = 'gtts'
            elif 'elevenlabs' in self.engines:
                self.prefered_engine = 'elevenlabs'
            else:
                self.preferred_engine = list(self.engines.keys())[0]
                
        print(f"🎯 Preferred engine: {self.preferred_engine}")
        print(f"📋 Available engines: {list(self.engines.keys())}")
        
        # Load available voices
        self._load_voices()

    def get_elevenlabs_voices(self, ):
        return ["Rachel", "Antoni", "Bella"]
    
    def _init_eleven_labs(self, ):
        """Intialization for elven labs (API Key Required)"""
        try:
            from elevenlabs.client import ElevenLabs
            import simpleaudio as sa
            

            self.engines["elevenlabs"] = {
                # create client as generation engine
                'engine': ElevenLabs(api_key=os.environ.get("ELEVEN_LABS_API_KEY"), timeout=httpx.Timeout(60.0)),
                'simpleaudio': sa, # used to store audio file
                'type': 'online',
                'model': "eleven_monolingual_v1", 
                'voice': self.get_elevenlabs_voices()[0]
            }
            print("✅ Eleven Labs engine initialized")
        except Exception as e:
            print(f"⚠️ Eleven Labs initialization failed: {e}")
        return
    
    def _init_sapi(self):
        """Initialize Windows SAPI engine"""
        if self.system_platform != 'windows':
            return
            
        try:
            import win32com.client
            
            # Test SAPI availability
            sapi = win32com.client.Dispatch("SAPI.SpVoice")
            voices = sapi.GetVoices()
            
            if voices.Count > 0:
                self.engines['sapi'] = {
                    'engine': sapi,
                    'type': 'native',
                    'voices': voices
                }
                print("✅ Windows SAPI engine initialized")
            else:
                print("⚠️ Windows SAPI available but no voices found")
                
        except Exception as e:
            print(f"⚠️ Windows SAPI initialization failed: {e}")
    
    def _init_festival(self):
        """Initialize Festival engine (Linux/macOS)"""
        try:
            # Check if festival is installed
            result = subprocess.run(['festival', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                self.engines['festival'] = {
                    'type': 'command',
                    'command': 'festival',
                    'test_passed': True
                }
                print("✅ Festival engine initialized")
            else:
                print("⚠️ Festival not found in PATH")
                
        except Exception as e:
            print(f"⚠️ Festival initialization failed: {e}")
    
    def _init_espeak(self):
        """Initialize eSpeak engine (cross-platform)"""
        try:
            # Check if espeak is installed
            result = subprocess.run(['espeak', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                self.engines['espeak'] = {
                    'type': 'command',
                    'command': 'espeak',
                    'test_passed': True
                }
                print("✅ eSpeak engine initialized")
            else:
                print("⚠️ eSpeak not found in PATH")
                
        except Exception as e:
            print(f"⚠️ eSpeak initialization failed: {e}")
    
    def _init_gTTS(self):
        """Initialize Google Text-to-Speech"""
        try:
            from gtts import gTTS
            import pygame
            
            # Test with a simple phrase
            test_tts = gTTS(text="test", lang='en', slow=False)
            
            self.engines['gtts'] = {
                'type': 'online',
                'module': gTTS,
                'pygame': pygame,
                'test_passed': True
            }
            print("✅ Google TTS (gTTS) engine initialized")
            
        except ImportError:
            print("⚠️ gTTS not installed (pip install gtts pygame)")
        except Exception as e:
            print(f"⚠️ gTTS initialization failed: {e}")
    
    def _init_azure_tts(self):
        """Initialize Azure Speech Services"""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # Check for Azure key in environment
            speech_key = os.getenv('AZURE_SPEECH_KEY')
            speech_region = os.getenv('AZURE_SPEECH_REGION', 'eastus')
            
            if speech_key:
                speech_config = speechsdk.SpeechConfig(
                    subscription=speech_key, 
                    region=speech_region
                )
                
                self.engines['azure'] = {
                    'type': 'online',
                    'config': speech_config,
                    'module': speechsdk,
                    'test_passed': True
                }
                print("✅ Azure Speech Services initialized")
            else:
                print("⚠️ Azure TTS available but AZURE_SPEECH_KEY not set")
                
        except ImportError:
            print("⚠️ Azure Speech SDK not installed")
        except Exception as e:
            print(f"⚠️ Azure TTS initialization failed: {e}")
    
    def _init_mozilla_tts(self):
        """Initialize Mozilla TTS (local neural voices)"""
        try:
            import TTS
            from TTS.api import TTS as TTSEngine
            mozzilla_models = []
            # Try to load a lightweight model
            selected_model = self.model_options.mozilla_models[self.selected_model]
            print(f"using model: {selected_model}")
            print(f"GPU?: {gpu_available}")
            tts_engine = TTSEngine(
                                   # model_name="tts_models/en/ljspeech/tacotron2-DDC", 
                                   model_name=selected_model,
                                   gpu=gpu_available,
                                   progress_bar=True)
            
            self.engines['mozilla'] = {
                'type': 'neural',
                'engine': tts_engine,
                'test_passed': True
            }
            print("✅ Mozilla TTS engine initialized")
            
        except ImportError:
            print("⚠️ Mozilla TTS not installed (pip install TTS)")
        except Exception as e:
            print(f"⚠️ Mozilla TTS initialization failed: {e}")
    
    def _load_voices(self):
        """Load available voices for each engine"""
        
        for engine_name, engine_info in self.engines.items():
            try:
                if engine_name == 'sapi' and 'voices' in engine_info:
                    voices = []
                    for i in range(engine_info['voices'].Count):
                        voice = engine_info['voices'].Item(i)
                        voices.append({
                            'id': i,
                            'name': voice.GetDescription(),
                            'gender': 'unknown'  # SAPI doesn't easily expose this
                        })
                    self.voices[engine_name] = voices
                    
                elif engine_name == 'espeak':
                    # eSpeak has built-in voices
                    try:
                        result = subprocess.run(['espeak', '--voices'], 
                                              capture_output=True, text=True, timeout=5)
                        voices = []
                        for line in result.stdout.split('\n')[1:]:  # Skip header
                            if line.strip():
                                parts = line.split()
                                if len(parts) >= 4:
                                    voices.append({
                                        'id': parts[1],
                                        'name': parts[3],
                                        'gender': 'unknown'
                                    })
                        self.voices[engine_name] = voices
                    except Exception:
                        self.voices[engine_name] = [{'id': 'en', 'name': 'English', 'gender': 'unknown'}]
                        
                elif engine_name == 'gtts':
                    # gTTS supports multiple languages
                    self.voices[engine_name] = [
                        {'id': 'en', 'name': 'English (US)', 'gender': 'female'},
                        {'id': 'en-uk', 'name': 'English (UK)', 'gender': 'female'},
                        {'id': 'es', 'name': 'Spanish', 'gender': 'female'},
                        {'id': 'fr', 'name': 'French', 'gender': 'female'},
                        {'id': 'de', 'name': 'German', 'gender': 'female'},
                        {'id': 'it', 'name': 'Italian', 'gender': 'female'},
                        {'id': 'pt', 'name': 'Portuguese', 'gender': 'female'},
                        {'id': 'ru', 'name': 'Russian', 'gender': 'female'},
                        {'id': 'ja', 'name': 'Japanese', 'gender': 'female'},
                        {'id': 'ko', 'name': 'Korean', 'gender': 'female'},
                        {'id': 'zh', 'name': 'Chinese', 'gender': 'female'},
                    ]
                    
                else:
                    # Default voice for other engines
                    self.voices[engine_name] = [{'id': 'default', 'name': 'Default', 'gender': 'unknown'}]
                    
            except Exception as e:
                print(f"⚠️ Failed to load voices for {engine_name}: {e}")
                self.voices[engine_name] = [{'id': 'default', 'name': 'Default', 'gender': 'unknown'}]
    
    def get_available_voices(self, engine_name: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Get available voices for engines
        
        Args:
            engine_name: Specific engine name, or None for all engines
            
        Returns:
            Dictionary of engine names to voice lists
        """
        if engine_name:
            return {engine_name: self.voices.get(engine_name, [])}
        return self.voices.copy()
    
    def set_voice_config(self, **kwargs):
        """
        Configure voice parameters
        
        Args:
            rate: Speech rate (words per minute, 50-400)
            volume: Volume level (0.0 to 1.0)
            pitch: Pitch adjustment (-50 to +50)
            voice: Voice ID or name
        """
        
        for key, value in kwargs.items():
            if key in self.current_config:
                if key == 'rate':
                    self.current_config[key] = max(50, min(400, int(value)))
                elif key == 'volume':
                    self.current_config[key] = max(0.0, min(1.0, float(value)))
                elif key == 'pitch':
                    self.current_config[key] = max(-50, min(50, int(value)))
                else:
                    self.current_config[key] = value
                    
        print(f"🔧 Voice config updated: {self.current_config}")
    
    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def speak_with_sapi(self, text: str) -> bool:
        """Speak text using Windows SAPI"""
        if 'sapi' not in self.engines:
            raise RuntimeError("SAPI engine not available")
            
        try:
            sapi = self.engines['sapi']['engine']
            
            # Configure voice settings
            if self.current_config['voice'] is not None:
                voices = self.engines['sapi']['voices']
                voice_id = self.current_config['voice']
                
                # Handle both integer index and string voice selection
                if isinstance(voice_id, str):
                    # Try to parse as integer first
                    try:
                        voice_id = int(voice_id)
                    except ValueError:
                        # If not integer, search by name
                        for i in range(voices.Count):
                            voice = voices.Item(i)
                            voice_name = voice.GetDescription().lower()
                            if str(voice_id).lower() in voice_name:
                                voice_id = i
                                break
                        else:
                            print(f"⚠️ Voice '{self.current_config['voice']}' not found, using default")
                            voice_id = None
                
                # Set the voice by index
                if isinstance(voice_id, int) and 0 <= voice_id < voices.Count:
                    selected_voice = voices.Item(voice_id)
                    sapi.Voice = selected_voice
                    print(f"🎤 Using SAPI voice {voice_id}: {selected_voice.GetDescription()}")
                        
            # Set rate (SAPI uses -10 to +10, map from 50-400 WPM)
            rate_mapped = max(-10, min(10, int((self.current_config['rate'] - 200) / 20)))
            sapi.Rate = rate_mapped
            
            # Set volume (SAPI uses 0 to 100)
            sapi.Volume = int(self.current_config['volume'] * 100)
            
            # Speak the text
            sapi.Speak(text)
            return True
            
        except Exception as e:
            print(f"❌ SAPI speech failed: {e}")
            return False
    
    def speak_with_espeak(self, text: str) -> bool:
        """Speak text using eSpeak"""
        if 'espeak' not in self.engines:
            raise RuntimeError("eSpeak engine not available")
            
        try:
            cmd = ['espeak']
            
            # Configure voice settings
            if self.current_config['voice']:
                cmd.extend(['-v', str(self.current_config['voice'])])
            
            # Set speed (words per minute)
            cmd.extend(['-s', str(self.current_config['rate'])])
            
            # Set volume (0-200, default 100)
            volume = int(self.current_config['volume'] * 200)
            cmd.extend(['-a', str(volume)])
            
            # Set pitch (0-99, default 50)
            pitch = int(50 + self.current_config['pitch'])
            cmd.extend(['-p', str(pitch)])
            
            # Add text
            cmd.append(text)
            
            # Execute
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ eSpeak speech failed: {e}")
            return False
    
    def speak_with_festival(self, text: str) -> bool:
        """Speak text using Festival"""
        if 'festival' not in self.engines:
            raise RuntimeError("Festival engine not available")
            
        try:
            # Create temporary file with text
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(text)
                temp_file = f.name
            
            try:
                # Use festival to speak
                cmd = ['festival', '--tts', temp_file]
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                return result.returncode == 0
                
            finally:
                # Clean up temp file
                os.unlink(temp_file)
                
        except Exception as e:
            print(f"❌ Festival speech failed: {e}")
            return False
    
    def speak_with_gtts(self, text: str, output_file: Optional[str] = None) -> bool:
        """Speak text using Google TTS"""
        if 'gtts' not in self.engines:
            raise RuntimeError("gTTS engine not available")
            
        try:
            gTTS = self.engines['gtts']['module']
            pygame = self.engines['gtts']['pygame']
            
            # Determine language
            lang = self.current_config.get('voice', 'en')
            
            # Validate language code
            valid_langs = ['en', 'es', 'fr', 'de', 'it', 'ja', 'ko', 'zh', 'pt', 'ru']
            if lang not in valid_langs:
                # Try to find a match
                lang_lower = str(lang).lower()
                for valid_lang in valid_langs:
                    if lang_lower.startswith(valid_lang) or valid_lang in lang_lower:
                        lang = valid_lang
                        break
                else:
                    lang = 'en'  # fallback
            
            print(f"🌐 Using gTTS language: {lang}")
            
            # Create TTS object
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # Save to file
            if output_file:
                tts.save(output_file)
                print(f"💾 Audio saved to: {output_file}")
                return True
            else:
                # Play immediately
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                    temp_audio = f.name
                
                try:
                    tts.save(temp_audio)
                    
                    # Initialize pygame mixer
                    pygame.mixer.init()
                    pygame.mixer.music.load(temp_audio)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to complete
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                        
                    return True
                    
                finally:
                    pygame.mixer.quit()
                    if os.path.exists(temp_audio):
                        os.unlink(temp_audio)
                        
        except Exception as e:
            print(f"❌ gTTS speech failed: {e}")
            return False

    def speak_with_elevenlabs(self, text: str, output_file: Optional[str] = None) -> bool:
        if 'elevenlabs' not in self.engines:
            raise RuntimeError("Eleven labs engine not available")
        try:
            client = self.engines['elevenlabs']['engine']
            sa = self.engines['elevenlabs']['simpleaudio']
            audio = client.generate(
                text = text,
                voice=self.engines['elevenlabs']['voice'],
                model=self.engines['elevenlabs']['model']
            )
            try:
                wave_obj = sa.WaveObject.from_wave_file("output.wav")
                wave_obj.play()
            except Exception as ex:
                print(f"❌ Could use player for Eleven Labs TTS speech: {ex}")
                print("Falling back on file method")
                self.output_file = output_file
                with open(self.output_file) as f:
                    f.write(audio)
                
            return True
        except Exception as ex:
            print(f"❌ Eleven Labs TTS speech failed: {ex}")
            return False
            
    
    def speak_with_azure(self, text: str, output_file: Optional[str] = None) -> bool:
        """Speak text using Azure Speech Services"""
        if 'azure' not in self.engines:
            raise RuntimeError("Azure TTS engine not available")
            
        try:
            speechsdk = self.engines['azure']['module']
            speech_config = self.engines['azure']['config']
            self.output_file = output_file
            # Configure synthesis
            if output_file:
                audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
            else:
                audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
            
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config, 
                audio_config=audio_config
            )
            
            # Perform synthesis
            result = synthesizer.speak_text_async(text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return True
            else:
                print(f"❌ Azure synthesis failed: {result.reason}")
                return False
                
        except Exception as e:
            print(f"❌ Azure TTS speech failed: {e}")
            return False
    
    def speak_with_mozilla(self, text: str, output_file: Optional[str] = None) -> bool:
        """Speak text using Mozilla TTS"""
        if 'mozilla' not in self.engines:
            raise RuntimeError("Mozilla TTS engine not available")
            
        try:
            tts_engine = self.engines['mozilla']['engine']
            
            if output_file:
                # Generate to file
                tts_engine.tts_to_file(text=text, file_path=output_file)
                print(f"💾 Audio saved to: {output_file}")
                return True
            else:
                # Generate to temporary file and play
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_audio = f.name
                
                try:
                    tts_engine.tts_to_file(text=text, file_path=temp_audio)
                    
                    # Play using system command
                    if self.system_platform == 'windows':
                        os.system(f'start /min wmplayer "{temp_audio}"')
                    elif self.system_platform == 'darwin':
                        os.system(f'afplay "{temp_audio}"')
                    else:
                        os.system(f'aplay "{temp_audio}" 2>/dev/null || paplay "{temp_audio}" 2>/dev/null')
                    
                    return True
                    
                finally:
                    # Clean up after a delay
                    def cleanup():
                        time.sleep(5)  # Wait for playback
                        if os.path.exists(temp_audio):
                            os.unlink(temp_audio)
                    
                    threading.Thread(target=cleanup, daemon=True).start()
                    
        except Exception as e:
            print(f"❌ Mozilla TTS speech failed: {e}")
            return False
    
    def _chunk_text(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """
        Split text into speakable chunks at natural boundaries
        
        Args:
            text: Text to chunk
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences first
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed limit
            if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence is too long, split by words
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 > max_chunk_size:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                                temp_chunk = word
                            else:
                                # Single word is too long, just truncate
                                chunks.append(word[:max_chunk_size])
                        else:
                            temp_chunk += " " + word if temp_chunk else word
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


    def process_text_to_speech_elevenlabs(self, text: str, output_file: str = 'tts_output.mp3', max_chunk_size: int = 400) -> str:
        chunks = self._chunk_text(text, max_chunk_size=max_chunk_size)
        audio_segments = []
    
        for i, chunk in enumerate(chunks):
            audio_gen = self.engines['elevenlabs']['engine'].text_to_speech.convert(
                text=chunk,
                voice_id="21m00Tcm4TlvDq8ikWAM",  # Customize as needed
                model_id=self.engines['elevenlabs']['model'],
                output_format="mp3_44100_128",
            )
            audio_bytes = b"".join(audio_gen)
            audio_segments.append(audio_bytes)
    
        final_audio = b"".join(audio_segments)
        temp_path = os.path.join(tempfile.gettempdir(), output_file)
    
        with open(temp_path, "wb") as f:
            f.write(final_audio)
    
        return temp_path


    def process_text_to_speech_mozilla(self, text: str, output_file: str = 'tts_output.wav', max_chunk_size: int = 400) -> str:
        chunks = self._chunk_text(text, max_chunk_size=max_chunk_size)
        combined_path = os.path.join(tempfile.gettempdir(), output_file)
        print(f"Chunks: {len(chunks)}")
        combined_audio = AudioSegment.empty()
    
        for i, chunk in enumerate(chunks):
            temp_chunk_path = os.path.join(tempfile.gettempdir(), f"mozilla_chunk_{i}.wav")
            self.engines['mozilla']['engine'].tts_to_file(text=chunk, file_path=temp_chunk_path)
    
            segment = AudioSegment.from_wav(temp_chunk_path)
            combined_audio += segment
    
        combined_audio.export(combined_path, format="wav")
    
        return combined_path

  
    def process_and_speak(self, text: str, engine: str=None, output_file: str='tts_output.wav', chunk_size=400):
        if engine is None:
            engine = self.prefered_engine
        if engine in self.engines:
            if engine == "mozilla":
                print("using mozilla")
                return self.process_text_to_speech_mozilla(text, output_file=output_file, max_chunk_size=chunk_size)
            elif engine == "elevenlabs":
                print("using elevenlabs")
                return self.process_text_to_speech_elevenlabs(text, output_file=output_file, max_chunk_size=chunk_size)
        else:
            print(f"No viable engine chosen: {engine}")
            return False
 
    def speak(self, text: str, engine: Optional[str] = None, output_file: Optional[str] = None, chunk_size: int = 1000) -> bool:
        """
        Speak text using the best available engine with fallback and chunking
        
        Args:
            text: Text to speak
            engine: Specific engine to use (None for auto-selection)
            output_file: Save audio to file instead of playing
            chunk_size: Maximum characters per chunk (default 1000)
            
        Returns:
            True if speech was successful
        """
        
        if not text or not text.strip():
            print("⚠️ No text to speak")
            return False
        
        # Split text into chunks if it's long
        text_chunks = self._chunk_text(text, chunk_size)
        
        if len(text_chunks) > 1:
            print(f"📝 Speaking text in {len(text_chunks)} chunks...")
        
        overall_success = True
        
        for i, chunk in enumerate(text_chunks):
            if len(text_chunks) > 1:
                print(f"🔊 Speaking chunk {i+1}/{len(text_chunks)} ({len(chunk)} chars)")
            
            chunk_success = self._speak_single_chunk(chunk, engine, output_file)
            
            if not chunk_success:
                print(f"❌ Failed to speak chunk {i+1}")
                overall_success = False
                # Continue with next chunk instead of failing completely
            
            # Small pause between chunks for natural flow
            if i < len(text_chunks) - 1:
                time.sleep(0.3)
        
        return overall_success
    
    def _speak_single_chunk(self, text: str, engine: Optional[str] = None, output_file: Optional[str] = None) -> bool:
        """
        Speak a single text chunk using the best available engine with fallback
        
        Args:
            text: Text chunk to speak
            engine: Specific engine to use (None for auto-selection)
            output_file: Save audio to file instead of playing
            
        Returns:
            True if speech was successful
        """
        
        # Determine engines to try
        if engine and engine in self.engines:
            engines_to_try = [engine]
        else:
            engines_to_try = [self.preferred_engine] + [
                eng for eng in self.engines.keys() 
                if eng != self.preferred_engine
            ]
        
        initial_memory = self._get_memory_usage()
        
        for engine_name in engines_to_try:
            try:
                success = False
                if engine_name == 'sapi':
                    success = self.speak_with_sapi(text)
                elif engine_name == 'espeak':
                    success = self.speak_with_espeak(text)
                elif engine_name == 'festival':
                    success = self.speak_with_festival(text)
                elif engine_name == 'gtts':
                    success = self.speak_with_gtts(text, output_file)
                elif engine_name == 'azure':
                    success = self.speak_with_azure(text, output_file)
                elif engine_name == 'mozilla':
                    print("Speaking with mozilla")
                    success = self.speak_with_mozilla(text, output_file)
                elif engine_name == 'elevenlabs':
                    print("Speaking with elevenlabs")
                    success = self.speak_with_elevenlabs(text, output_file)
                
                if success:
                    after_memory = self._get_memory_usage()
                    memory_used = after_memory - initial_memory
                    
                    # Clean up memory
                    gc.collect()
                    return True
                else:
                    print(f"⚠️ {engine_name} returned failure")
                    
            except Exception as e:
                print(f"❌ {engine_name} failed: {e}")
                continue
        
        print("❌ All TTS engines failed for this chunk")
        return False
    
    def test_engines(self, output_file: Optional[str]="outfile.mp3") -> Dict[str, bool]:
        """Test all available engines with a simple phrase"""
        
        test_text = "Hello, this is a test of the text to speech engine."
        results = {}
        
        print("🧪 Testing all TTS engines...")
        print("=" * 50)
        
        for engine_name in self.engines.keys():
            print(f"\n🎯 Testing {engine_name}...")
            try:
                success = self.speak(test_text, engine=engine_name, output_file=output_file)
                results[engine_name] = success
                
                if success:
                    print(f"✅ {engine_name} test passed")
                else:
                    print(f"❌ {engine_name} test failed")
                    
                # Small delay between tests
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ {engine_name} test error: {e}")
                results[engine_name] = False
        
        print(f"\n📊 Test Results: {results}")
        return results
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get detailed information about all engines"""
        
        info = {
            'platform': self.system_platform,
            'preferred_engine': self.preferred_engine,
            'current_config': self.current_config.copy(),
            'engines': {},
            'voices': self.voices.copy()
        }
        
        for engine_name, engine_data in self.engines.items():
            info['engines'][engine_name] = {
                'type': engine_data.get('type', 'unknown'),
                'available': True,
                'voice_count': len(self.voices.get(engine_name, []))
            }
        
        return info


def main():
    """Test the multi-engine TTS pipeline"""
    
    print("🎭 Multi-Engine Text-to-Speech Pipeline Test")
    print("=" * 50)
    
    try:
        # Initialize TTS system
        tts = MultiEngineTTS()
        
        print(f"\n📋 Available engines: {list(tts.engines.keys())}")
        print(f"🎯 Preferred engine: {tts.preferred_engine}")
        
        # Show available voices
        print("\n🎤 Available voices:")
        voices = tts.get_available_voices()
        for engine, voice_list in voices.items():
            print(f"  {engine}: {len(voice_list)} voices")
            for voice in voice_list[:3]:  # Show first 3 voices
                print(f"    - {voice['name']} ({voice['id']})")
        
        print("\n🧪 Running engine tests...")
        test_results = tts.test_engines()
        
        print("\n🎯 Interactive mode - Enter text to speak (or commands):")
        print("Commands:")
        print("  'quit' or 'q' - Exit")
        print("  'test' - Test all engines")
        print("  'voices' - Show available voices") 
        print("  'config' - Show current configuration")
        print("  'set rate <number>' - Set speech rate")
        print("  'set volume <0.0-1.0>' - Set volume")
        print("  'set voice <id>' - Set voice")
        print("  'engine <name>' - Switch preferred engine")
        
        while True:
            try:
                user_input = input("\n💬 > ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'q', 'exit']:
                    break
                    
                elif user_input.lower() == 'test':
                    tts.test_engines()
                    
                elif user_input.lower() == 'voices':
                    voices = tts.get_available_voices()
                    for engine, voice_list in voices.items():
                        print(f"🎤 {engine}: {len(voice_list)} voices")
                        for voice in voice_list:
                            print(f"    - {voice['name']} ({voice['id']})")
                            
                elif user_input.lower() == 'config':
                    config = tts.current_config
                    print(f"🔧 Current configuration:")
                    for key, value in config.items():
                        print(f"    {key}: {value}")
                        
                elif user_input.lower().startswith('set '):
                    parts = user_input.split()
                    if len(parts) >= 3:
                        param = parts[1].lower()
                        value = ' '.join(parts[2:])
                        
                        try:
                            if param in ['rate', 'volume', 'pitch']:
                                value = float(value) if param == 'volume' else int(value)
                            tts.set_voice_config(**{param: value})
                        except ValueError:
                            print(f"❌ Invalid value for {param}: {value}")
                    else:
                        print("❌ Usage: set <parameter> <value>")
                        
                elif user_input.lower().startswith('engine '):
                    engine_name = user_input[7:].strip()
                    if engine_name in tts.engines:
                        tts.preferred_engine = engine_name
                        print(f"🎯 Switched to {engine_name} engine")
                    else:
                        print(f"❌ Engine '{engine_name}' not available")
                        print(f"Available: {list(tts.engines.keys())}")
                        
                else:
                    # Speak the text
                    print(f"🗣️ Speaking: '{user_input}'")
                    success = tts.speak(user_input)
                    if not success:
                        print("❌ Speech failed")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
        print("\n👋 Goodbye!")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
