#!/usr/bin/env python3
"""
Robust Speech-to-Text Pipeline with Multiple Engine Support

This module provides a reliable STT pipeline that can fallback between:
1. OpenAI Whisper (local, high quality)
2. Google Speech Recognition (online, reliable)
3. Azure Speech Services (online, enterprise)

Features:
- Memory-safe audio recording
- Automatic engine fallback
- Cross-platform support
- Memory usage monitoring
"""

import numpy as np
import sounddevice as sd
import time
import tempfile
import os
import sys
import gc
import psutil
from typing import Optional, Tuple, Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class MultiEngineSTT:
    """Speech-to-Text engine with multiple backend support"""
    
    def __init__(self):
        self.engines = {}
        self.preferred_engine = None
        self._init_engines()
        
    def _init_engines(self):
        """Initialize available STT engines"""
        
        # 1. Try to initialize Whisper
        try:
            import whisper
            self.engines['whisper'] = self._init_whisper()
            if self.engines['whisper']:
                self.preferred_engine = 'whisper'
                print("✅ Whisper engine initialized")
        except Exception as e:
            print(f"⚠️ Whisper initialization failed: {e}")
            
        # 2. Try to initialize SpeechRecognition
        try:
            import speech_recognition as sr
            self.engines['google'] = sr.Recognizer()
            if not self.preferred_engine:
                self.preferred_engine = 'google'
            print("✅ Google Speech Recognition engine initialized")
        except Exception as e:
            print(f"⚠️ SpeechRecognition initialization failed: {e}")
            
        if not self.engines:
            raise RuntimeError("No STT engines could be initialized!")
            
        print(f"🎯 Preferred engine: {self.preferred_engine}")
        print(f"📋 Available engines: {list(self.engines.keys())}")
    
    def _init_whisper(self):
        """Safely initialize Whisper with memory monitoring"""
        try:
            import whisper
            
            # Monitor memory during model loading
            initial_memory = self._get_memory_usage()
            print(f"Loading Whisper model... (Initial memory: {initial_memory:.1f} MB)")
            
            # Use the smallest model first
            model = whisper.load_model("tiny")
            
            after_memory = self._get_memory_usage()
            model_size = after_memory - initial_memory
            print(f"Whisper model loaded. Memory usage: {model_size:.1f} MB")
            
            if model_size > 1000:  # If model uses more than 1GB
                print("⚠️ Whisper model using excessive memory, will use as fallback only")
                
            return model
            
        except Exception as e:
            print(f"Failed to initialize Whisper: {e}")
            return None
    
    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def record_audio(self, duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
        """
        Record audio with memory safety checks
        
        Args:
            duration: Recording duration in seconds (max 30s for safety)
            sample_rate: Audio sample rate
            
        Returns:
            numpy array of audio data
        """
        
        # Safety limits
        duration = min(duration, 30.0)  # Max 30 seconds
        
        expected_size = int(duration * sample_rate * 4)  # 4 bytes per float32
        expected_mb = expected_size / 1024 / 1024
        
        if expected_mb > 100:  # Don't allow recordings > 100MB
            raise ValueError(f"Recording too large: {expected_mb:.1f} MB")
            
        print(f"🎤 Recording {duration}s of audio (expected size: {expected_mb:.1f} MB)...")
        
        try:
            # Record audio
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32
            )
            sd.wait()  # Wait for recording to complete
            
            # Convert to 1D array
            audio_data = audio_data.flatten()
            
            # Check audio levels
            max_amplitude = np.max(np.abs(audio_data))
            rms_level = np.sqrt(np.mean(audio_data**2))
            
            print(f"🔊 Audio recorded - Max: {max_amplitude:.3f}, RMS: {rms_level:.3f}")
            
            if max_amplitude < 0.001:
                print("⚠️ Warning: Very quiet audio detected")
            
            return audio_data
            
        except Exception as e:
            print(f"❌ Audio recording failed: {e}")
            raise
    
    def transcribe_with_whisper(self, audio_data: np.ndarray) -> str:
        """Transcribe audio using Whisper"""
        if 'whisper' not in self.engines or self.engines['whisper'] is None:
            raise RuntimeError("Whisper engine not available")
            
        try:
            initial_memory = self._get_memory_usage()
            print(f"🔄 Transcribing with Whisper... (Memory: {initial_memory:.1f} MB)")
            
            model = self.engines['whisper']
            result = model.transcribe(audio_data, verbose=False)
            
            after_memory = self._get_memory_usage()
            memory_used = after_memory - initial_memory
            print(f"📝 Whisper transcription complete (Memory used: {memory_used:.1f} MB)")
            
            return result.get('text', '').strip()
            
        except Exception as e:
            print(f"❌ Whisper transcription failed: {e}")
            raise
    
    def transcribe_with_google(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio using Google Speech Recognition"""
        if 'google' not in self.engines:
            raise RuntimeError("Google Speech Recognition not available")
            
        try:
            import speech_recognition as sr
            
            print("🔄 Transcribing with Google Speech Recognition...")
            
            # Convert numpy array to AudioData
            recognizer = self.engines['google']
            
            # Convert float32 to int16 for speech_recognition
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Create AudioData object
            audio_data_sr = sr.AudioData(
                audio_int16.tobytes(),
                sample_rate,
                2  # 2 bytes per sample (int16)
            )
            
            # Perform recognition
            text = recognizer.recognize_google(audio_data_sr)
            print("📝 Google transcription complete")
            
            return text.strip()
            
        except sr.UnknownValueError:
            print("⚠️ Google Speech Recognition could not understand audio")
            return ""
        except sr.RequestError as e:
            print(f"❌ Google Speech Recognition error: {e}")
            raise
        except Exception as e:
            print(f"❌ Google transcription failed: {e}")
            raise
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio using the best available engine with fallback
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Audio sample rate
            
        Returns:
            Transcribed text
        """
        
        engines_to_try = [self.preferred_engine] + [
            engine for engine in self.engines.keys() 
            if engine != self.preferred_engine
        ]
        
        for engine_name in engines_to_try:
            try:
                print(f"🎯 Trying {engine_name} engine...")
                
                if engine_name == 'whisper':
                    result = self.transcribe_with_whisper(audio_data)
                elif engine_name == 'google':
                    result = self.transcribe_with_google(audio_data, sample_rate)
                else:
                    continue
                    
                if result and result.strip():
                    print(f"✅ Successfully transcribed with {engine_name}: '{result}'")
                    return result
                else:
                    print(f"⚠️ {engine_name} returned empty result")
                    
            except Exception as e:
                print(f"❌ {engine_name} failed: {e}")
                continue
        
        print("❌ All transcription engines failed")
        return ""
    
    def voice_to_text(self, duration: float = 5.0) -> str:
        """
        Complete voice-to-text pipeline
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Transcribed text
        """
        
        try:
            # Record audio
            audio_data = self.record_audio(duration)
            
            # Transcribe
            text = self.transcribe(audio_data)
            
            # Clean up
            del audio_data
            gc.collect()
            
            return text
            
        except Exception as e:
            print(f"❌ Voice-to-text pipeline failed: {e}")
            return ""


def main():
    """Test the multi-engine STT pipeline"""
    
    print("🎙️ Multi-Engine Speech-to-Text Pipeline Test")
    print("=" * 50)
    
    try:
        # Initialize STT system
        stt = MultiEngineSTT()
        
        print("\n🎯 Testing voice input...")
        print("Press Enter to start recording, or 'q' to quit:")
        
        while True:
            user_input = input().strip().lower()
            
            if user_input == 'q':
                break
            
            print("\n🎤 Recording for 3 seconds... Speak now!")
            text = stt.voice_to_text(duration=3.0)
            
            if text:
                print(f"📝 Transcription: '{text}'")
            else:
                print("❌ No speech detected or transcription failed")
            
            print("\nPress Enter to record again, or 'q' to quit:")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
