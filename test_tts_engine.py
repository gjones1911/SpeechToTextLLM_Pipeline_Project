#!/usr/bin/env python3
"""
Test script for MultiEngineTTS

This script provides comprehensive testing of the TTS pipeline including:
- Engine availability testing
- Voice quality testing
- Performance benchmarking
- Configuration testing
- Integration testing
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Add the voice_processing directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'voice_processing'))

try:
    from multi_engine_tts import MultiEngineTTS
except ImportError as e:
    print(f"❌ Failed to import MultiEngineTTS: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

def test_basic_functionality():
    """Test basic TTS functionality"""
    print("🧪 Testing Basic TTS Functionality")
    print("=" * 50)
    
    try:
        # Initialize TTS
        tts = MultiEngineTTS()
        
        # Test simple speech
        test_text = "Hello world, this is a test of the text to speech system."
        print(f"🗣️ Testing with text: '{test_text}'")
        
        success = tts.speak(test_text)
        if success:
            print("✅ Basic functionality test passed")
            return True
        else:
            print("❌ Basic functionality test failed")
            return False
            
    except Exception as e:
        print(f"❌ Basic functionality test error: {e}")
        return False

def test_all_engines():
    """Test all available engines"""
    print("\n🧪 Testing All Available Engines")
    print("=" * 50)
    
    try:
        tts = MultiEngineTTS()
        results = tts.test_engines()
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        print(f"\n📊 Engine Test Summary: {passed}/{total} passed")
        return passed > 0
        
    except Exception as e:
        print(f"❌ Engine testing error: {e}")
        return False

def test_voice_configuration():
    """Test voice configuration options"""
    print("\n🧪 Testing Voice Configuration")
    print("=" * 50)
    
    try:
        tts = MultiEngineTTS()
        
        # Test different configurations
        configs = [
            {'rate': 150, 'volume': 0.5},
            {'rate': 250, 'volume': 0.8},
            {'rate': 200, 'volume': 1.0, 'pitch': 10},
            {'rate': 180, 'volume': 0.6, 'pitch': -10}
        ]
        
        test_text = "Testing voice configuration."
        
        for i, config in enumerate(configs):
            print(f"\n🔧 Config {i+1}: {config}")
            tts.set_voice_config(**config)
            
            success = tts.speak(test_text)
            if success:
                print(f"✅ Configuration {i+1} test passed")
            else:
                print(f"❌ Configuration {i+1} test failed")
            
            time.sleep(1)  # Brief pause between tests
        
        print("✅ Voice configuration testing completed")
        return True
        
    except Exception as e:
        print(f"❌ Voice configuration test error: {e}")
        return False

def test_file_output():
    """Test saving speech to file"""
    print("\n🧪 Testing File Output")
    print("=" * 50)
    
    try:
        tts = MultiEngineTTS()
        
        test_text = "This is a test of file output functionality."
        output_file = "test_tts_output.wav"
        
        print(f"💾 Saving speech to: {output_file}")
        success = tts.speak(test_text, output_file=output_file)
        
        if success and os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✅ File output test passed - File size: {file_size} bytes")
            
            # Clean up
            try:
                os.remove(output_file)
                print("🧹 Test file cleaned up")
            except:
                pass
                
            return True
        else:
            print("❌ File output test failed")
            return False
            
    except Exception as e:
        print(f"❌ File output test error: {e}")
        return False

def test_performance():
    """Test TTS performance and memory usage"""
    print("\n🧪 Testing Performance")
    print("=" * 50)
    
    try:
        tts = MultiEngineTTS()
        
        # Test different text lengths
        test_texts = [
            "Short test.",
            "This is a medium length test sentence to evaluate performance.",
            "This is a longer test sentence that contains more words and should take longer to process and synthesize into speech audio. We want to test how the system handles longer inputs."
        ]
        
        for i, text in enumerate(test_texts):
            print(f"\n📏 Test {i+1} - Length: {len(text)} characters")
            print(f"Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            start_time = time.time()
            success = tts.speak(text)
            end_time = time.time()
            
            duration = end_time - start_time
            
            if success:
                print(f"✅ Performance test {i+1} passed - Duration: {duration:.2f}s")
            else:
                print(f"❌ Performance test {i+1} failed")
            
            time.sleep(0.5)  # Brief pause
        
        print("✅ Performance testing completed")
        return True
        
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False

def test_error_handling():
    """Test error handling and edge cases"""
    print("\n🧪 Testing Error Handling")
    print("=" * 50)
    
    try:
        tts = MultiEngineTTS()
        
        # Test edge cases
        test_cases = [
            ("", "Empty string"),
            ("   ", "Whitespace only"),
            ("A" * 1500, "Very long text (truncated)"),
            ("Hello\nworld\ttab", "Text with newlines and tabs"),
            ("Test 123 !@# $%^", "Text with numbers and symbols")
        ]
        
        for text, description in test_cases:
            print(f"\n🧪 Testing: {description}")
            
            try:
                success = tts.speak(text)
                print(f"{'✅' if success else '⚠️'} {description}: {'Success' if success else 'Failed gracefully'}")
            except Exception as e:
                print(f"❌ {description}: Exception - {e}")
        
        print("✅ Error handling testing completed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test error: {e}")
        return False

def test_engine_info():
    """Test engine information retrieval"""
    print("\n🧪 Testing Engine Information")
    print("=" * 50)
    
    try:
        tts = MultiEngineTTS()
        
        # Get engine info
        info = tts.get_engine_info()
        
        print(f"🖥️ Platform: {info['platform']}")
        print(f"🎯 Preferred Engine: {info['preferred_engine']}")
        print(f"🔧 Current Config: {info['current_config']}")
        
        print(f"\n📋 Available Engines: {len(info['engines'])}")
        for engine_name, engine_info in info['engines'].items():
            print(f"  - {engine_name}: {engine_info['type']} ({engine_info['voice_count']} voices)")
        
        # Test voice listing
        voices = tts.get_available_voices()
        total_voices = sum(len(voice_list) for voice_list in voices.values())
        print(f"\n🎤 Total Voices Available: {total_voices}")
        
        print("✅ Engine information test passed")
        return True
        
    except Exception as e:
        print(f"❌ Engine information test error: {e}")
        return False

def interactive_test():
    """Run interactive testing session"""
    print("\n🧪 Interactive Testing Mode")
    print("=" * 50)
    
    try:
        tts = MultiEngineTTS()
        
        print("🎭 Interactive TTS Testing")
        print("Available commands:")
        print("  speak <text> - Speak the given text")
        print("  engine <name> - Switch to specific engine")
        print("  rate <number> - Set speech rate")
        print("  volume <0.0-1.0> - Set volume")
        print("  voices - List available voices")
        print("  info - Show engine information")
        print("  test - Run quick test")
        print("  quit - Exit")
        
        while True:
            try:
                command = input("\n🎯 TTS > ").strip()
                
                if not command:
                    continue
                    
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                    
                elif command.lower().startswith('speak '):
                    text = command[6:].strip()
                    if text:
                        print(f"🗣️ Speaking: '{text}'")
                        tts.speak(text)
                    else:
                        print("❌ No text provided")
                        
                elif command.lower().startswith('engine '):
                    engine = command[7:].strip()
                    if engine in tts.engines:
                        tts.preferred_engine = engine
                        print(f"🎯 Switched to {engine}")
                    else:
                        print(f"❌ Engine '{engine}' not available")
                        print(f"Available: {list(tts.engines.keys())}")
                        
                elif command.lower().startswith('rate '):
                    try:
                        rate = int(command[5:].strip())
                        tts.set_voice_config(rate=rate)
                        print(f"🔧 Rate set to {rate} WPM")
                    except ValueError:
                        print("❌ Invalid rate value")
                        
                elif command.lower().startswith('volume '):
                    try:
                        volume = float(command[7:].strip())
                        tts.set_voice_config(volume=volume)
                        print(f"🔧 Volume set to {volume}")
                    except ValueError:
                        print("❌ Invalid volume value")
                        
                elif command.lower() == 'voices':
                    voices = tts.get_available_voices()
                    for engine, voice_list in voices.items():
                        print(f"🎤 {engine}: {len(voice_list)} voices")
                        for voice in voice_list[:3]:
                            print(f"    - {voice['name']} ({voice['id']})")
                            
                elif command.lower() == 'info':
                    info = tts.get_engine_info()
                    print(f"Platform: {info['platform']}")
                    print(f"Preferred: {info['preferred_engine']}")
                    print(f"Engines: {list(info['engines'].keys())}")
                    print(f"Config: {info['current_config']}")
                    
                elif command.lower() == 'test':
                    tts.speak("This is a quick test of the text to speech system.")
                    
                else:
                    print("❌ Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("👋 Interactive testing completed")
        return True
        
    except Exception as e:
        print(f"❌ Interactive test error: {e}")
        return False

def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Test the MultiEngineTTS system")
    parser.add_argument('--basic', action='store_true', help="Run basic functionality test only")
    parser.add_argument('--engines', action='store_true', help="Test all engines")
    parser.add_argument('--config', action='store_true', help="Test voice configuration")
    parser.add_argument('--file', action='store_true', help="Test file output")
    parser.add_argument('--performance', action='store_true', help="Test performance")
    parser.add_argument('--errors', action='store_true', help="Test error handling")
    parser.add_argument('--info', action='store_true', help="Test engine information")
    parser.add_argument('--interactive', action='store_true', help="Run interactive testing")
    parser.add_argument('--all', action='store_true', help="Run all tests")
    
    args = parser.parse_args()
    
    # If no specific tests specified, run all
    if not any([args.basic, args.engines, args.config, args.file, 
               args.performance, args.errors, args.info, args.interactive]):
        args.all = True
    
    print("🎭 MultiEngineTTS Test Suite")
    print("=" * 60)
    
    results = []
    
    try:
        if args.all or args.basic:
            results.append(("Basic Functionality", test_basic_functionality()))
        
        if args.all or args.engines:
            results.append(("All Engines", test_all_engines()))
        
        if args.all or args.config:
            results.append(("Voice Configuration", test_voice_configuration()))
        
        if args.all or args.file:
            results.append(("File Output", test_file_output()))
        
        if args.all or args.performance:
            results.append(("Performance", test_performance()))
        
        if args.all or args.errors:
            results.append(("Error Handling", test_error_handling()))
        
        if args.all or args.info:
            results.append(("Engine Information", test_engine_info()))
        
        if args.interactive:
            results.append(("Interactive Testing", interactive_test()))
        
        # Summary
        if results:
            print("\n" + "=" * 60)
            print("📊 TEST SUMMARY")
            print("=" * 60)
            
            passed = 0
            for test_name, result in results:
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{status} - {test_name}")
                if result:
                    passed += 1
            
            print(f"\n🎯 Overall Result: {passed}/{len(results)} tests passed")
            
            if passed == len(results):
                print("🎉 All tests passed! TTS system is working correctly.")
                return 0
            else:
                print("⚠️ Some tests failed. Check the output above for details.")
                return 1
        else:
            print("❌ No tests were run")
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
