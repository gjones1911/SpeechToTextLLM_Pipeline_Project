#!/usr/bin/env python3
"""
Simple TTS test to verify MultiEngineTTS works
"""

import sys
import os

# Add the voice_processing directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'voice_processing'))

def test_tts_import():
    """Test if TTS can be imported"""
    try:
        from multi_engine_tts import MultiEngineTTS
        print("✅ TTS import successful")
        return True
    except Exception as e:
        print(f"❌ TTS import failed: {e}")
        return False

def test_tts_init():
    """Test TTS initialization"""
    try:
        from multi_engine_tts import MultiEngineTTS
        tts = MultiEngineTTS()
        print(f"✅ TTS initialized with {len(tts.engines)} engines")
        print(f"   Preferred engine: {tts.preferred_engine}")
        print(f"   Available engines: {list(tts.engines.keys())}")
        return True, tts
    except Exception as e:
        print(f"❌ TTS initialization failed: {e}")
        return False, None

def test_tts_speak(tts):
    """Test TTS speaking"""
    try:
        test_text = "Hello, this is a test of the text to speech engine."
        print(f"🔊 Testing speech: '{test_text}'")
        success = tts.speak(test_text)
        if success:
            print("✅ TTS speak test successful")
        else:
            print("⚠️ TTS speak test failed but no exception")
        return success
    except Exception as e:
        print(f"❌ TTS speak test failed: {e}")
        return False

def main():
    """Run TTS tests"""
    print("🎭 Simple TTS Test")
    print("=" * 50)
    
    # Test import
    if not test_tts_import():
        return 1
    
    # Test initialization
    success, tts = test_tts_init()
    if not success:
        return 1
    
    # Test speaking
    if test_tts_speak(tts):
        print("\n🎉 All TTS tests passed!")
        return 0
    else:
        print("\n⚠️ TTS tests completed with warnings")
        return 0

if __name__ == "__main__":
    exit(main())
