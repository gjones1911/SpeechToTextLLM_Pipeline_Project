#!/usr/bin/env python3
"""
Test Script for TranscriberAgent

Simple test script to create and run a TranscriberAgent with command-line arguments.
Maps sound modes to listen modes and provides a clean interface for testing.

Usage:
    python test_transcriber_agent.py <url> [--soundmode button|continuous]

Examples:
    python test_transcriber_agent.py http://localhost:7860
    python test_transcriber_agent.py http://localhost:1234 --soundmode continuous
"""

import argparse
import sys
import os
import logging

# Add the code directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from transcriber_test_script import TranscriberAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function to parse arguments and run TranscriberAgent"""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Test Script for TranscriberAgent - Voice-to-LLM Interactive System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s http://localhost:7860
  %(prog)s http://localhost:7860 --soundmode button
  %(prog)s http://localhost:1234 --soundmode continuous
  %(prog)s https://your-gradio-space.hf.space --soundmode button --no-history

Sound Modes:
  button     - Push-to-talk mode (hold SPACE to record)
  continuous - Noise detection mode (auto-start/stop on speech)

History Options:
  --maintain-history - Keep conversation history (default)
  --no-history      - Start fresh each interaction
        """
    )
    
    # Required URL argument
    parser.add_argument(
        "url", 
        help="API URL for the LLM service (e.g., http://localhost:7860)"
    )
    
    # Optional sound mode argument
    parser.add_argument(
        "--soundmode", 
        choices=["button", "continuous"],
        default="button",
        help="Sound recording mode (default: button)"
    )
    
    # Additional optional arguments for advanced configuration
    parser.add_argument(
        "--api-type",
        choices=["gradio", "lmstudio", "openai"],
        default="gradio",
        help="Type of API endpoint (default: gradio)"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Recording duration in seconds for button mode (default: 4.0)"
    )
    
    parser.add_argument(
        "--stt-engine",
        choices=["auto", "whisper", "google"],
        default="auto",
        help="Speech-to-text engine preference (default: auto)"
    )
    
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run a single test interaction instead of interactive mode"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--maintain-history",
        action="store_true",
        default=True,
        help="Maintain conversation history across interactions (default: True)"
    )
    
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable conversation history (shortcut for --maintain-history=False)"
    )
    
    parser.add_argument(
        "--enable-tts",
        action="store_true",
        default=True,
        help="Enable text-to-speech for LLM responses (default: True)"
    )
    
    parser.add_argument(
        "--disable-tts",
        action="store_true",
        help="Disable text-to-speech output"
    )
    
    parser.add_argument(
        "--tts-engine",
        default="auto",
        help="Preferred TTS engine (auto, sapi, espeak, gtts, etc.)"
    )

    # Parse arguments
    args = parser.parse_args()
    
    # Validate URL format
    if not validate_url(args.url):
        print(f"❌ Invalid URL format: {args.url}")
        print("   URL should start with http:// or https://")
        return 1
    
    # Test connection to the API
    print(f"🔍 Testing connection to {args.url}...")
    if not test_connection(args.url, args.api_type):
        print("⚠️  Warning: Connection test failed, but continuing anyway...")
        print("   Make sure your API server is running!")
    print()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Map sound modes to listen modes
    listen_mode_map = {
        "button": "push_to_talk",
        "continuous": "noise_detection"
    }
    
    listen_mode = listen_mode_map[args.soundmode]
    
    # Determine maintain_history setting
    maintain_history = args.maintain_history and not args.no_history
    
    # Determine TTS setting
    enable_tts = not args.disable_tts and args.enable_tts
    
    # Display configuration
    print("🎯 TranscriberAgent Test Configuration:")
    print(f"   URL: {args.url}")
    print(f"   Sound Mode: {args.soundmode} (maps to {listen_mode})")
    print(f"   API Type: {args.api_type}")
    print(f"   STT Engine: {args.stt_engine}")
    print(f"   TTS Enabled: {enable_tts}")
    if enable_tts:
        print(f"   TTS Engine: {args.tts_engine}")
    print(f"   Duration: {args.duration}s")
    print(f"   Maintain History: {maintain_history}")
    print(f"   Test Only: {args.test_only}")
    print()
    
    try:
        # Create TranscriberAgent with mapped parameters
        print("🚀 Initializing TranscriberAgent...")
        agent = TranscriberAgent(
            llm_url=args.url,
            api_type=args.api_type,
            listen_mode=listen_mode,
            stt_engine=args.stt_engine,
            duration=args.duration,
            maintain_history=maintain_history,
            enable_tts=enable_tts,
            tts_engine=args.tts_engine
        )
        
        if args.test_only:
            # Run single test interaction
            print("\n🧪 Running single test interaction...")
            print("=" * 50)
            
            if listen_mode == "push_to_talk":
                print("📝 Instructions: Hold SPACE to record when prompted")
            else:
                print("📝 Instructions: Speak naturally, recording will auto-start/stop")
            
            response = agent.process_voice_input()
            
            if response:
                print(f"\n✅ Test completed successfully!")
                print(f"🤖 Final response: {response}")
            else:
                print(f"\n❌ Test failed - no response received")
                return 1
                
        else:
            # Run interactive mode
            print("\n🎙️ Starting interactive mode...")
            print("=" * 50)
            
            if listen_mode == "push_to_talk":
                print("💡 Tip: Hold SPACE to record voice input")
            else:
                print("💡 Tip: Speak naturally, recording will auto-start/stop")
            
            print("💡 Tip: Type 'quit' to exit, 'clear' to reset conversation")
            print("💡 Tip: Use 'history on/off' to toggle conversation history")
            print("💡 Tip: Use 'tts on/off' to toggle text-to-speech")
            print("💡 Tip: Use 'stats' to see conversation statistics")
            print()
            
            agent.interactive_mode()
        
        print("\n✅ TranscriberAgent test completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        logger.error(f"Test execution error: {e}")
        return 1


def validate_url(url: str) -> bool:
    """
    Basic URL validation
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL appears valid
    """
    if not url:
        return False
    
    # Basic checks
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    
    if len(url) < 10:  # Minimum reasonable URL length
        return False
    
    return True


def test_connection(url: str, api_type: str = "gradio") -> bool:
    """
    Test if the API endpoint is reachable
    
    Args:
        url: API URL to test
        api_type: Type of API
        
    Returns:
        True if connection successful
    """
    try:
        import requests
        
        # SSL bypass configuration for ngrok
        request_kwargs = {
            "verify": False,  # 🔑 Essential for ngrok SSL bypass
            "headers": {"ngrok-skip-browser-warning": "any"},
            "timeout": 5
        }
        
        # Try to connect to the base URL
        response = requests.get(url, **request_kwargs)
        
        if response.status_code == 200:
            print(f"✅ Connection test successful: {url}")
            return True
        else:
            print(f"⚠️  Connection test warning: {url} returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Run main function
    exit_code = main()
    sys.exit(exit_code)
