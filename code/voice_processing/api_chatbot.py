#!/usr/bin/env python3
"""
API-Based Speech-to-Text Chatbot
Integrates STT pipeline with Gradio and LM Studio APIs

This module provides a complete voice-to-API chatbot system that:
- Records and transcribes speech using the robust STT pipeline
- Sends transcribed text to Gradio API (primary)
- Falls back to LM Studio API if Gradio fails
- Returns spoken responses back to the user

Usage:
    python api_chatbot.py --gradio-url "http://localhost:7860" --lm-studio-url "http://localhost:1234"
"""

import argparse
import sys
import os
import json
import time
import requests
import asyncio
from typing import Optional, Dict, Any, Tuple
import logging

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_processing.multi_engine_stt import MultiEngineSTT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GradioAPI:
    """Gradio API client for chatbot interactions"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = timeout

    def test_connection(self) -> bool:
        """Test if Gradio API is accessible"""
        try:
            response = self.session.get(f"{self.base_url}/info", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Gradio API connection successful")
                return True
            else:
                logger.warning(f"⚠️ Gradio API returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Gradio API connection failed: {e}")
            return False
    
    def send_message(self, message: str) -> Optional[str]:
        """
        Send message to Gradio chatbot API
        
        Args:
            message: User message to send
            
        Returns:
            Bot response or None if failed
        """
        try:
            # Gradio API format for chat completion
            payload = {
                "data": [message],
                "fn_index": 0  # Usually the first function for chat
            }
            
            response = self.session.post(
                f"{self.base_url}/api/predict",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                # Extract response from Gradio's response format
                if "data" in result and len(result["data"]) > 0:
                    bot_response = result["data"][0]
                    logger.info(f"✅ Gradio response received: {len(bot_response)} chars")
                    return bot_response
                else:
                    logger.warning("⚠️ Gradio returned empty response")
                    return None
            else:
                logger.error(f"❌ Gradio API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Gradio API request failed: {e}")
            return None

class LMStudioAPI:
    """LM Studio API client for local LLM interactions"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 30
        
    def test_connection(self) -> bool:
        """Test if LM Studio API is accessible"""
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                model_count = len(models.get('data', []))
                logger.info(f"✅ LM Studio API connection successful ({model_count} models)")
                return True
            else:
                logger.warning(f"⚠️ LM Studio API returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ LM Studio API connection failed: {e}")
            return False
    
    def send_message(self, message: str) -> Optional[str]:
        """
        Send message to LM Studio API (OpenAI-compatible)
        
        Args:
            message: User message to send
            
        Returns:
            Bot response or None if failed
        """
        try:
            # OpenAI-compatible chat completion format
            payload = {
                "model": "local-model",  # LM Studio uses this for loaded model
                "messages": [
                    {"role": "system", "content": "You are a helpful voice assistant. Keep responses concise and conversational."},
                    {"role": "user", "content": message}
                ],
                "max_tokens": 150,
                "temperature": 0.7,
                "stream": False
            }
            
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    bot_response = result["choices"][0]["message"]["content"].strip()
                    logger.info(f"✅ LM Studio response received: {len(bot_response)} chars")
                    return bot_response
                else:
                    logger.warning("⚠️ LM Studio returned empty response")
                    return None
            else:
                logger.error(f"❌ LM Studio API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ LM Studio API request failed: {e}")
            return None

class VoiceAPIChat:
    """Main voice-to-API chatbot coordinator"""
    
    def __init__(self, gradio_url: str = None, lm_studio_url: str = None):
        self.stt = None
        self.gradio_api = None
        self.lm_studio_api = None
        self.available_apis = []
        
        # Initialize STT
        self._init_stt()
        
        # Initialize APIs
        if gradio_url:
            self._init_gradio(gradio_url)
        if lm_studio_url:
            self._init_lm_studio(lm_studio_url)
            
        if not self.available_apis:
            raise RuntimeError("No chatbot APIs are available!")
    
    def _init_stt(self):
        """Initialize the speech-to-text system"""
        try:
            logger.info("🎙️ Initializing Speech-to-Text...")
            self.stt = MultiEngineSTT()
            logger.info(f"✅ STT initialized with engines: {list(self.stt.engines.keys())}")
        except Exception as e:
            raise RuntimeError(f"STT initialization failed: {e}")
    
    def _init_gradio(self, gradio_url: str):
        """Initialize Gradio API connection"""
        try:
            logger.info(f"🌐 Connecting to Gradio API: {gradio_url}")
            self.gradio_api = GradioAPI(gradio_url)
            if self.gradio_api.test_connection():
                self.available_apis.append("gradio")
                logger.info("✅ Gradio API ready")
            else:
                logger.warning("⚠️ Gradio API not available")
        except Exception as e:
            logger.error(f"❌ Gradio API initialization failed: {e}")
    
    def _init_lm_studio(self, lm_studio_url: str):
        """Initialize LM Studio API connection"""
        try:
            logger.info(f"🤖 Connecting to LM Studio API: {lm_studio_url}")
            self.lm_studio_api = LMStudioAPI(lm_studio_url)
            if self.lm_studio_api.test_connection():
                self.available_apis.append("lm_studio")
                logger.info("✅ LM Studio API ready")
            else:
                logger.warning("⚠️ LM Studio API not available")
        except Exception as e:
            logger.error(f"❌ LM Studio API initialization failed: {e}")
    
    def transcribe_speech(self, duration: float = 4.0) -> Tuple[bool, str]:
        """
        Record and transcribe speech
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            (success, transcription)
        """
        try:
            logger.info(f"🎤 Recording speech for {duration} seconds...")
            transcription = self.stt.voice_to_text(duration)
            
            if transcription and transcription.strip():
                # Filter out common error messages
                error_indicators = ["[", "silent", "error", "no speech"]
                if any(indicator in transcription.lower() for indicator in error_indicators):
                    logger.warning(f"⚠️ STT returned error/empty: {transcription}")
                    return False, transcription
                
                logger.info(f"✅ Speech transcribed: '{transcription}'")
                return True, transcription
            else:
                logger.warning("⚠️ No speech detected")
                return False, "No speech detected"
                
        except Exception as e:
            logger.error(f"❌ Speech transcription failed: {e}")
            return False, f"Transcription error: {e}"
    
    def send_to_chatbot(self, message: str) -> Optional[str]:
        """
        Send message to chatbot APIs with fallback
        
        Args:
            message: Transcribed message to send
            
        Returns:
            Chatbot response or None if all APIs fail
        """
        
        # Try APIs in order of preference
        api_attempts = []
        
        if "gradio" in self.available_apis:
            api_attempts.append(("Gradio", self.gradio_api))
        if "lm_studio" in self.available_apis:
            api_attempts.append(("LM Studio", self.lm_studio_api))
        
        for api_name, api_client in api_attempts:
            try:
                logger.info(f"🔄 Sending to {api_name} API...")
                response = api_client.send_message(message)
                
                if response and response.strip():
                    logger.info(f"✅ {api_name} responded successfully")
                    return response
                else:
                    logger.warning(f"⚠️ {api_name} returned empty response")
                    
            except Exception as e:
                logger.error(f"❌ {api_name} API error: {e}")
                continue
        
        logger.error("❌ All chatbot APIs failed")
        return None
    
    def voice_chat_session(self):
        """Interactive voice chat session"""
        
        logger.info("🎯 Voice API Chat Ready!")
        print("\n" + "=" * 50)
        print("🎙️ Voice-to-API Chatbot")
        print("=" * 50)
        print(f"Available APIs: {', '.join(self.available_apis)}")
        print(f"STT Engines: {', '.join(self.stt.engines.keys())}")
        print("\nControls:")
        print("  - Press Enter: Start voice recording")
        print("  - Type message: Send text directly")
        print("  - Type 'quit': Exit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n💬 Voice or text (Enter for voice): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                # Text input mode
                if user_input:
                    print(f"📝 Sending text: '{user_input}'")
                    response = self.send_to_chatbot(user_input)
                    
                    if response:
                        print(f"🤖 Bot: {response}")
                    else:
                        print("❌ No response from chatbot APIs")
                    continue
                
                # Voice input mode
                print("\n🎤 Listening... Speak now!")
                
                success, transcript = self.transcribe_speech(duration=4.0)
                
                if not success:
                    print(f"❌ {transcript}")
                    continue
                
                print(f"📝 You said: '{transcript}'")
                
                # Send to chatbot
                response = self.send_to_chatbot(transcript)
                
                if response:
                    print(f"🤖 Bot: {response}")
                else:
                    print("❌ Chatbot APIs are not responding")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Session error: {e}")
                continue
    
    def single_voice_query(self, duration: float = 4.0) -> Dict[str, Any]:
        """
        Process a single voice query and return structured result
        
        Args:
            duration: Recording duration
            
        Returns:
            Dictionary with query results
        """
        
        result = {
            "success": False,
            "transcript": "",
            "response": "",
            "api_used": None,
            "error": None,
            "timestamp": time.time()
        }
        
        try:
            # Record and transcribe
            success, transcript = self.transcribe_speech(duration)
            result["transcript"] = transcript
            
            if not success:
                result["error"] = "Speech transcription failed"
                return result
            
            # Send to chatbot
            response = self.send_to_chatbot(transcript)
            
            if response:
                result["success"] = True
                result["response"] = response
                result["api_used"] = self.available_apis[0] if self.available_apis else None
            else:
                result["error"] = "No response from chatbot APIs"
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Single query error: {e}")
        
        return result

def main():
    """Main function with command-line argument parsing"""
    
    parser = argparse.ArgumentParser(description="Voice-to-API Chatbot")
    parser.add_argument(
        '--gradio-url',
        default="http://localhost:7860",
        help='Gradio API URL (default: http://localhost:7860)'
    )
    parser.add_argument(
        '--lm-studio-url',
        default="http://localhost:1234",
        help='LM Studio API URL (default: http://localhost:1234)'
    )
    parser.add_argument(
        '--mode',
        choices=['interactive', 'single'],
        default='interactive',
        help='Chat mode: interactive session or single query'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=4.0,
        help='Recording duration in seconds (default: 4.0)'
    )
    
    args = parser.parse_args()
    
    print("🎙️ Voice-to-API Chatbot Starting...")
    print(f"🌐 Gradio URL: {args.gradio_url}")
    print(f"🤖 LM Studio URL: {args.lm_studio_url}")
    
    try:
        # Initialize the voice chat system
        voice_chat = VoiceAPIChat(
            gradio_url=args.gradio_url,
            lm_studio_url=args.lm_studio_url
        )
        
        if args.mode == 'interactive':
            # Start interactive session
            voice_chat.voice_chat_session()
        else:
            # Single query mode
            print(f"\n🎤 Recording for {args.duration} seconds... Speak now!")
            result = voice_chat.single_voice_query(args.duration)
            
            print("\n📋 Results:")
            print(f"Success: {result['success']}")
            print(f"Transcript: '{result['transcript']}'")
            if result['response']:
                print(f"Response: '{result['response']}'")
            if result['error']:
                print(f"Error: {result['error']}")
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
