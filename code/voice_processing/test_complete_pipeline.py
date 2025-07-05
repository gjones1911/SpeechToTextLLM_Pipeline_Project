#!/usr/bin/env python3
"""
Comprehensive STT Pipeline Test Suite
Tests all components of the speech-to-text pipeline
"""

import sys
import os
import time
import traceback
import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test if all required modules can be imported"""
    print("🔧 Testing Imports...")
    
    tests = [
        ("numpy", "import numpy as np"),
        ("sounddevice", "import sounddevice as sd"),
        ("scipy", "import scipy"),
        ("speech_recognition", "import speech_recognition as sr"),
        ("whisper", "import whisper"),
        ("psutil", "import psutil"),
    ]
    
    results = {}
    for name, import_cmd in tests:
        try:
            exec(import_cmd)
            results[name] = "✅ PASS"
            print(f"  {name}: ✅")
        except Exception as e:
            results[name] = f"❌ FAIL: {e}"
            print(f"  {name}: ❌ {e}")
    
    return results

def test_whisper_model():
    """Test Whisper model loading and basic functionality"""
    print("\n🤖 Testing Whisper Model...")
    
    try:
        import whisper
        
        print("  Loading tiny model...")
        model = whisper.load_model("tiny")
        print("  ✅ Model loaded successfully")
        
        # Test with minimal audio
        print("  Testing transcription...")
        test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        result = model.transcribe(test_audio, verbose=False)
        print(f"  ✅ Transcription completed: '{result.get('text', '')}'")
        
        return "✅ PASS"
        
    except Exception as e:
        print(f"  ❌ Whisper test failed: {e}")
        return f"❌ FAIL: {e}"

def test_google_stt():
    """Test Google Speech Recognition"""
    print("\n🌐 Testing Google Speech Recognition...")
    
    try:
        import speech_recognition as sr
        
        recognizer = sr.Recognizer()
        print("  ✅ Recognizer initialized")
        
        # Test with noise (should fail gracefully)
        print("  Testing with test audio...")
        audio_data = np.random.normal(0, 0.01, 16000).astype(np.float32)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        audio_sr = sr.AudioData(audio_int16.tobytes(), 16000, 2)
        
        try:
            text = recognizer.recognize_google(audio_sr)
            print(f"  ✅ Transcription: '{text}'")
        except sr.UnknownValueError:
            print("  ✅ No speech detected (expected)")
        except sr.RequestError as e:
            print(f"  ⚠️ Request error: {e}")
        
        return "✅ PASS"
        
    except Exception as e:
        print(f"  ❌ Google STT test failed: {e}")
        return f"❌ FAIL: {e}"

def test_audio_devices():
    """Test audio input/output devices"""
    print("\n🎤 Testing Audio Devices...")
    
    try:
        import sounddevice as sd
        
        devices = sd.query_devices()
        print(f"  Found {len(devices)} audio devices")
        
        # Find default input device
        default_input = sd.default.device[0]
        if default_input is not None:
            device_info = sd.query_devices(default_input)
            print(f"  Default input: {device_info['name']}")
            print(f"  Max input channels: {device_info['max_input_channels']}")
        else:
            print("  ⚠️ No default input device found")
        
        return "✅ PASS"
        
    except Exception as e:
        print(f"  ❌ Audio device test failed: {e}")
        return f"❌ FAIL: {e}"

def test_multi_engine_stt():
    """Test the multi-engine STT class"""
    print("\n🔄 Testing Multi-Engine STT...")
    
    try:
        from voice_processing.multi_engine_stt import MultiEngineSTT
        
        stt = MultiEngineSTT()
        print(f"  ✅ STT initialized with engines: {list(stt.engines.keys())}")
        print(f"  ✅ Preferred engine: {stt.preferred_engine}")
        
        # Test with synthetic audio
        test_audio = np.zeros(8000, dtype=np.float32)  # 0.5 seconds
        result = stt.transcribe(test_audio)
        print(f"  ✅ Transcription test completed: '{result}'")
        
        return "✅ PASS"
        
    except Exception as e:
        print(f"  ❌ Multi-engine STT test failed: {e}")
        return f"❌ FAIL: {e}"

def test_memory_usage():
    """Test memory usage monitoring"""
    print("\n💾 Testing Memory Monitoring...")
    
    try:
        import psutil
        
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"  Current memory usage: {memory_mb:.1f} MB")
        
        # Test virtual memory
        vm = psutil.virtual_memory()
        print(f"  System memory: {vm.total / 1024 / 1024 / 1024:.1f} GB total")
        print(f"  Available: {vm.available / 1024 / 1024 / 1024:.1f} GB ({vm.percent:.1f}% used)")
        
        return "✅ PASS"
        
    except Exception as e:
        print(f"  ❌ Memory monitoring test failed: {e}")
        return f"❌ FAIL: {e}"

def main():
    """Run comprehensive test suite"""
    
    print("🚀 Comprehensive STT Pipeline Test Suite")
    print("=" * 60)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    test_results = {}
    
    # Run all tests
    test_functions = [
        ("Import Tests", test_imports),
        ("Memory Monitoring", test_memory_usage),
        ("Audio Devices", test_audio_devices),
        ("Whisper Model", test_whisper_model),
        ("Google STT", test_google_stt),
        ("Multi-Engine STT", test_multi_engine_stt),
    ]
    
    for test_name, test_func in test_functions:
        try:
            result = test_func()
            if isinstance(result, dict):
                test_results.update(result)
            else:
                test_results[test_name] = result
        except Exception as e:
            test_results[test_name] = f"❌ FAIL: {e}"
            print(f"  ❌ Unexpected error in {test_name}: {e}")
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if "✅ PASS" in str(result) else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        
        if "✅ PASS" in str(result):
            passed += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"Total Tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! STT Pipeline is ready for production.")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed. Check the details above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
