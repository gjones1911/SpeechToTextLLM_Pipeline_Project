#!/usr/bin/env python3
"""
Quick TTS Voice Test Script

Tests different voices to ensure they actually sound different.
"""

import sys
import os

# Add the code directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'code', 'voice_processing'))

from multi_engine_tts import MultiEngineTTS

def main():
    """Test different voices to make sure they actually sound different"""
    
    print("🎭 TTS Voice Difference Test")
    print("=" * 50)
    
    try:
        # Initialize TTS
        tts = MultiEngineTTS()
        
        # Get available voices
        voices = tts.get_available_voices()
        
        print("Available engines and voices:")
        for engine, voice_list in voices.items():
            print(f"\n🔊 {engine.upper()}: {len(voice_list)} voices")
            for i, voice in enumerate(voice_list[:5]):  # Show first 5
                print(f"   {i}: {voice['name']} (ID: {voice['id']})")
        
        # Test different voices if SAPI is available
        if 'sapi' in voices and len(voices['sapi']) > 1:
            print(f"\n🧪 Testing Windows SAPI voices...")
            
            test_text = "Hello, this is a test of voice number"
            
            for i in range(min(3, len(voices['sapi']))):  # Test first 3 voices
                voice_info = voices['sapi'][i]
                print(f"\n🎤 Testing voice {i}: {voice_info['name']}")
                
                # Set the voice
                tts.set_voice_config(voice=i)
                
                # Speak with voice identification
                tts.speak(f"{test_text} {i}. {voice_info['name'].split()[0]}")
                
                input("Press Enter to continue to next voice...")
        
        # Test different engines
        print(f"\n🧪 Testing different engines...")
        
        engines_to_test = []
        if 'sapi' in tts.engines:
            engines_to_test.append(('sapi', 'Windows SAPI'))
        if 'gtts' in tts.engines:
            engines_to_test.append(('gtts', 'Google TTS'))
        if 'espeak' in tts.engines:
            engines_to_test.append(('espeak', 'eSpeak'))
        
        for engine_id, engine_name in engines_to_test:
            print(f"\n🎯 Testing {engine_name}...")
            success = tts.speak(f"This is {engine_name} speaking", engine=engine_id)
            if success:
                print(f"✅ {engine_name} test successful")
            else:
                print(f"❌ {engine_name} test failed")
            
            input("Press Enter to continue to next engine...")
        
        # Test gTTS different languages
        if 'gtts' in tts.engines:
            print(f"\n🌐 Testing Google TTS languages...")
            
            lang_tests = [
                ('en', 'Hello, this is English'),
                ('es', 'Hola, esto es español'),  
                ('fr', 'Bonjour, c\'est français'),
                ('de', 'Hallo, das ist deutsch')
            ]
            
            for lang, text in lang_tests:
                print(f"\n🗣️ Testing {lang}: {text}")
                tts.set_voice_config(voice=lang)
                success = tts.speak(text, engine='gtts')
                if success:
                    print(f"✅ {lang} test successful")
                else:
                    print(f"❌ {lang} test failed")
                
                input("Press Enter for next language...")
        
        print("\n✅ Voice testing completed!")
        print("If all voices sounded the same, there may be an issue with:")
        print("   1. Windows SAPI voice installation")
        print("   2. Audio driver configuration") 
        print("   3. Voice selection logic")
        
    except Exception as e:
        print(f"❌ Error during voice testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
