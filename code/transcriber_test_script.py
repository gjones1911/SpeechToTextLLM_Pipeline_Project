#!/usr/bin/env python3
"""
TranscriberAgent - Voice-to-LLM Interactive System

Combines MultiEngineSTT for speech recognition with LLMAgent for intelligent responses.
Supports both noise detection and push-to-talk activation modes.

Usage:
    agent = TranscriberAgent("http://localhost:7860", listen_mode="push_to_talk")
    agent.interactive_mode()
"""

from voice_processing.multi_engine_stt import MultiEngineSTT
from llm_agent_linux_package.llm_agent import LLMAgent

import sys
import time
import os
import threading
import queue
import numpy as np
import sounddevice as sd
import logging
from typing import Optional, Dict, Any, Tuple

try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False
    print("⚠️  Warning: 'keyboard' module not installed. Push-to-talk mode will be limited.")

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranscriberAgent:
    """
    Interactive voice-to-LLM agent that combines speech recognition with LLM responses.
    
    Features:
    - Two listening modes: noise detection and push-to-talk
    - Automatic STT engine fallback (Whisper -> Google)
    - LLM API integration with configurable endpoints
    - Real-time audio processing and transcription
    """
    
    def __init__(self, 
                 llm_url: str, 
                 api_type: str = "gradio",
                 listen_mode: str = "push_to_talk",
                 stt_engine: str = "auto",
                 duration: float = 4.0,
                 silence_threshold: float = 0.01,
                 silence_duration: float = 1.5,
                 maintain_history: bool = True,
                 **llm_kwargs):
        """
        Initialize the TranscriberAgent
        
        Args:
            llm_url: URL for the LLM API endpoint
            api_type: Type of LLM API ("gradio", "lmstudio", "openai")
            listen_mode: "push_to_talk" or "noise_detection"
            stt_engine: "auto", "whisper", or "google"
            duration: Recording duration for push-to-talk mode
            silence_threshold: Audio amplitude threshold for noise detection
            silence_duration: Silence duration to stop recording (noise detection)
            maintain_history: Whether to maintain conversation history
            **llm_kwargs: Additional arguments for LLMAgent
        """
        self.llm_url = llm_url
        self.api_type = api_type
        self.listen_mode = listen_mode
        self.stt_engine = stt_engine
        self.duration = duration
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.maintain_history = maintain_history
        
        # Initialize STT engine
        print("🎙️ Initializing Speech-to-Text engine...")
        self.audio_transcriber = MultiEngineSTT()
        
        # Initialize LLM agent
        print(f"🤖 Initializing LLM Agent ({api_type}) at {llm_url}...")
        self.llm_agent = LLMAgent(
            base_url=llm_url,
            api_type=api_type,
            maintain_history=maintain_history,
            **llm_kwargs
        )
        
        # Test LLM connection
        connection_test = self.llm_agent.test_connection()
        if connection_test["status"] != "connected":
            print(f"⚠️  Warning: LLM connection test failed: {connection_test}")
        else:
            print(f"✅ LLM connection successful!")
        
        # Audio recording state
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        
        print(f"🎯 TranscriberAgent initialized:")
        print(f"   - Listen Mode: {listen_mode}")
        print(f"   - STT Engine: {stt_engine}")
        print(f"   - LLM API: {api_type} at {llm_url}")
        print(f"   - Recording Duration: {duration}s")
        if listen_mode == "noise_detection":
            print(f"   - Silence Threshold: {silence_threshold}")
            print(f"   - Silence Duration: {silence_duration}s")
    
    def listen_and_transcribe(self, duration: Optional[float] = None) -> Tuple[bool, str]:
        """
        Listen for audio input and transcribe it using the STT engine.
        
        Args:
            duration: Override default recording duration
            
        Returns:
            Tuple of (success, transcript)
        """
        record_duration = duration or self.duration
        
        try:
            print(f"🎧 Listening for {record_duration}s...")
            
            # Use MultiEngineSTT's voice_to_text method
            transcript = self.audio_transcriber.voice_to_text(duration=record_duration)
            
            if transcript and transcript.strip():
                print(f"📝 Transcribed: '{transcript}'")
                return True, transcript
            else:
                print("❌ No speech detected or transcription failed")
                return False, ""
                
        except Exception as e:
            error_msg = f"❌ Error during transcription: {e}"
            print(error_msg)
            logger.error(error_msg)
            return False, ""
    
    def listen_with_noise_detection(self) -> Tuple[bool, str]:
        """
        Listen with noise detection - start recording on sound, stop on silence.
        
        Returns:
            Tuple of (success, transcript)
        """
        print("🎧 Listening for speech (noise detection mode)...")
        print("   Speak when ready - recording will auto-start and stop")
        
        try:
            # Monitor audio levels to detect speech
            chunk_size = 1024
            silence_chunks = 0
            max_silence_chunks = int(self.silence_duration * self.sample_rate / chunk_size)
            recording_started = False
            audio_data = []
            
            def audio_callback(indata, frames, time, status):
                nonlocal silence_chunks, recording_started, audio_data
                
                if status:
                    logger.warning(f"Audio callback status: {status}")
                
                # Calculate RMS amplitude
                rms = np.sqrt(np.mean(indata**2))
                
                if not recording_started:
                    # Check if speech started
                    if rms > self.silence_threshold:
                        recording_started = True
                        print("🔴 Recording started...")
                        audio_data = [indata.copy()]
                        silence_chunks = 0
                else:
                    # Recording in progress
                    audio_data.append(indata.copy())
                    
                    if rms < self.silence_threshold:
                        silence_chunks += 1
                        if silence_chunks >= max_silence_chunks:
                            print("⏹️  Silence detected, stopping recording")
                            raise sd.CallbackStop()
                    else:
                        silence_chunks = 0
            
            # Start monitoring audio
            with sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=chunk_size,
                dtype=np.float32
            ):
                try:
                    # Wait for recording to complete
                    while True:
                        time.sleep(0.1)
                except sd.CallbackStop:
                    pass
            
            if not recording_started or not audio_data:
                print("❌ No speech detected")
                return False, ""
            
            # Concatenate audio data
            full_audio = np.concatenate(audio_data, axis=0).flatten()
            
            # Transcribe using MultiEngineSTT
            print("🔄 Transcribing...")
            transcript = self.audio_transcriber.transcribe(full_audio, self.sample_rate)
            
            if transcript and transcript.strip():
                print(f"📝 Transcribed: '{transcript}'")
                return True, transcript
            else:
                print("❌ Transcription failed")
                return False, ""
                
        except Exception as e:
            error_msg = f"❌ Error in noise detection: {e}"
            print(error_msg)
            logger.error(error_msg)
            return False, ""
    
    def listen_push_to_talk(self) -> Tuple[bool, str]:
        """
        Listen with push-to-talk - record while space bar is held.
        
        Returns:
            Tuple of (success, transcript)
        """
        if not HAS_KEYBOARD:
            print("❌ Keyboard module not available, falling back to timed recording")
            return self.listen_and_transcribe()
        
        print("🎧 Push-to-talk mode: Hold SPACE to record, release to stop")
        
        try:
            audio_data = []
            is_recording = False
            
            def audio_callback(indata, frames, time, status):
                nonlocal audio_data, is_recording
                if is_recording:
                    audio_data.append(indata.copy())
            
            # Start audio stream
            stream = sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                dtype=np.float32
            )
            
            with stream:
                print("   Press and hold SPACE to start recording...")
                
                # Wait for space key press
                keyboard.wait('space')
                is_recording = True
                print("� Recording... (release SPACE to stop)")
                
                # Record while space is held
                while keyboard.is_pressed('space'):
                    time.sleep(0.01)
                
                is_recording = False
                print("⏹️  Recording stopped")
            
            if not audio_data:
                print("❌ No audio recorded")
                return False, ""
            
            # Concatenate audio data
            full_audio = np.concatenate(audio_data, axis=0).flatten()
            
            if len(full_audio) < self.sample_rate * 0.5:  # Less than 0.5 seconds
                print("❌ Recording too short")
                return False, ""
            
            # Transcribe using MultiEngineSTT
            print("🔄 Transcribing...")
            transcript = self.audio_transcriber.transcribe(full_audio, self.sample_rate)
            
            if transcript and transcript.strip():
                print(f"📝 Transcribed: '{transcript}'")
                return True, transcript
            else:
                print("❌ Transcription failed")
                return False, ""
                
        except Exception as e:
            error_msg = f"❌ Error in push-to-talk: {e}"
            print(error_msg)
            logger.error(error_msg)
            return False, ""
    
    def send_to_llm(self, message: str) -> str:
        """
        Send transcribed text to LLM and get response.
        
        Args:
            message: The transcribed text to send
            
        Returns:
            LLM response text
        """
        try:
            print(f"🤖 Sending to LLM: '{message[:50]}{'...' if len(message) > 50 else ''}'")
            
            response = self.llm_agent.send_message(message)
            
            if response and not response.startswith("Error"):
                print(f"💬 LLM Response: {response}")
                return response
            else:
                error_msg = f"❌ LLM error: {response}"
                print(error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"❌ Error sending to LLM: {e}"
            print(error_msg)
            logger.error(error_msg)
            return error_msg
    
    def process_voice_input(self) -> Optional[str]:
        """
        Complete voice-to-LLM pipeline: listen, transcribe, and get LLM response.
        
        Returns:
            LLM response or None if failed
        """
        # Step 1: Listen and transcribe based on mode
        if self.listen_mode == "noise_detection":
            success, transcript = self.listen_with_noise_detection()
        elif self.listen_mode == "push_to_talk":
            success, transcript = self.listen_push_to_talk()
        else:
            # Fallback to timed recording
            success, transcript = self.listen_and_transcribe()
        
        if not success or not transcript:
            return None
        
        # Step 2: Send to LLM
        response = self.send_to_llm(transcript)
        return response if not response.startswith("❌") else None
    
    def interactive_mode(self):
        """
        Start interactive voice chat session.
        Continuously listens for voice input and provides LLM responses.
        """
        print("\n" + "="*60)
        print("🎙️🤖 INTERACTIVE VOICE CHAT SESSION")
        print("="*60)
        print(f"Mode: {self.listen_mode}")
        print(f"STT Engine: {self.stt_engine}")
        print(f"LLM API: {self.api_type} at {self.llm_url}")
        
        if self.listen_mode == "push_to_talk" and not HAS_KEYBOARD:
            print("⚠️  Warning: Keyboard module not available, using timed recording")
        
        print("\nCommands:")
        print("  - Type 'quit' or 'exit' to end session")
        print("  - Type 'clear' to clear conversation history")
        print("  - Type 'stats' to show conversation statistics")
        print("  - Type 'history' to check history status")
        print("  - Type 'history on/off' to toggle conversation history")
        
        if self.listen_mode == "push_to_talk":
            print("  - Hold SPACE to record voice input")
        else:
            print("  - Speak naturally, recording will auto-start/stop")
        
        print("\nStarting session... (Ctrl+C to force quit)")
        print("-" * 60)
        
        session_start = time.time()
        interaction_count = 0
        
        try:
            while True:
                try:
                    # Check for text commands
                    print(f"\n[{interaction_count + 1}] Ready for input...")
                    
                    # For push-to-talk, also check for keyboard input
                    if self.listen_mode == "push_to_talk":
                        print("Type a command or hold SPACE for voice input:")
                    
                    # Non-blocking input check (simplified approach)
                    user_input = input(">>> ").strip()
                    
                    if user_input.lower() in ['quit', 'exit']:
                        print("👋 Goodbye!")
                        break
                    elif user_input.lower() == 'clear':
                        self.clear_conversation()
                        continue
                    elif user_input.lower() == 'stats':
                        stats = self.get_conversation_stats()
                        print(f"� Conversation Stats: {stats}")
                        continue
                    elif user_input.lower() == 'history':
                        current_setting = self.get_maintain_history()
                        print(f"💭 Conversation history: {'ON' if current_setting else 'OFF'}")
                        continue
                    elif user_input.lower() in ['history on', 'history_on']:
                        self.set_maintain_history(True)
                        continue
                    elif user_input.lower() in ['history off', 'history_off']:
                        self.set_maintain_history(False)
                        continue
                    elif user_input:
                        # Text input - send directly to LLM
                        response = self.send_to_llm(user_input)
                        if response:
                            interaction_count += 1
                        continue
                    
                    # Voice input processing
                    print("🎧 Processing voice input...")
                    response = self.process_voice_input()
                    
                    if response:
                        interaction_count += 1
                        print(f"✅ Interaction {interaction_count} completed")
                    else:
                        print("❌ Interaction failed, try again")
                    
                except KeyboardInterrupt:
                    print("\n⚠️  Interrupted by user")
                    break
                except Exception as e:
                    print(f"❌ Error in interaction: {e}")
                    logger.error(f"Interactive mode error: {e}")
                    continue
        
        except KeyboardInterrupt:
            print("\n👋 Session ended by user")
        
        # Session summary
        session_duration = time.time() - session_start
        print(f"\n📊 Session Summary:")
        print(f"   Duration: {session_duration:.1f} seconds")
        print(f"   Interactions: {interaction_count}")
        print(f"   Conversation: {len(self.llm_agent.messages)} messages")
        
        # Cleanup
        self.llm_agent.close()
        print("🧹 Resources cleaned up")

    def get_maintain_history(self) -> bool:
        """Get current conversation history maintenance setting"""
        return self.maintain_history
    
    def set_maintain_history(self, maintain: bool) -> None:
        """Set whether to maintain conversation history"""
        old_setting = self.maintain_history
        self.maintain_history = maintain
        self.llm_agent.set_maintain_history(maintain)
        print(f"💭 Conversation history maintenance changed from {old_setting} to {maintain}")
    
    def clear_conversation(self) -> None:
        """Clear the conversation history"""
        if hasattr(self.llm_agent, 'clear_history'):
            self.llm_agent.clear_history()
            print("🧹 Conversation history cleared")
        else:
            print("⚠️  No conversation history to clear")
    
    def get_conversation_stats(self) -> dict:
        """Get statistics about the current conversation"""
        if hasattr(self.llm_agent, 'get_conversation_stats'):
            return self.llm_agent.get_conversation_stats()
        else:
            return {"message_count": 0, "error": "Stats not available"}


def main():
    """Example usage and testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TranscriberAgent - Voice-to-LLM Interactive System")
    parser.add_argument("--llm-url", default="http://localhost:7860", help="LLM API URL")
    parser.add_argument("--api-type", default="gradio", choices=["gradio", "lmstudio", "openai"], help="LLM API type")
    parser.add_argument("--listen-mode", default="push_to_talk", choices=["push_to_talk", "noise_detection"], help="Listening mode")
    parser.add_argument("--stt-engine", default="auto", choices=["auto", "whisper", "google"], help="STT engine preference")
    parser.add_argument("--duration", type=float, default=4.0, help="Recording duration for push-to-talk")
    parser.add_argument("--test", action="store_true", help="Run quick test instead of interactive mode")
    
    args = parser.parse_args()
    
    try:
        # Create TranscriberAgent
        agent = TranscriberAgent(
            llm_url=args.llm_url,
            api_type=args.api_type,
            listen_mode=args.listen_mode,
            stt_engine=args.stt_engine,
            duration=args.duration
        )
        
        if args.test:
            # Quick test
            print("\n🧪 Running quick test...")
            response = agent.process_voice_input()
            if response:
                print(f"✅ Test successful: {response}")
            else:
                print("❌ Test failed")
        else:
            # Interactive mode
            agent.interactive_mode()
            
    except Exception as e:
        print(f"❌ Failed to start TranscriberAgent: {e}")
        logger.error(f"Main execution error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())