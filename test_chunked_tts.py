#!/usr/bin/env python3
"""
Test Chunked TTS Response

Tests the new chunked TTS functionality with long text responses.
"""

import sys
import os

# Add the code directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'code', 'voice_processing'))

from multi_engine_tts import MultiEngineTTS

def main():
    """Test chunked TTS with long text"""
    
    print("🔊 Testing Chunked TTS Response")
    print("=" * 50)
    
    try:
        # Initialize TTS
        tts = MultiEngineTTS()
        
        # Test with a long response (simulating LLM output)
        long_text = """
        This is a test of the new chunked text-to-speech functionality. The system can now handle responses that are longer than 1000 characters by automatically splitting them into smaller, more manageable chunks. Each chunk is spoken sequentially with a brief pause between them for natural flow. This ensures that even very long responses from the language model can be fully vocalized without truncation. The chunking algorithm tries to split at natural sentence boundaries to maintain readability and flow. This is particularly useful for detailed explanations, stories, or comprehensive answers that the LLM might generate. The system maintains all the voice configuration settings across chunks, so the entire response sounds consistent. This represents a significant improvement over the previous system that would simply truncate long responses at 1000 characters.
        """
        
        print(f"📝 Test text length: {len(long_text)} characters")
        print(f"🔤 Text preview: {long_text[:100]}...")
        
        print("\n🎭 Speaking long text with chunking...")
        success = tts.speak(long_text.strip())
        
        if success:
            print("✅ Chunked TTS test completed successfully!")
            print("💡 The entire response should have been spoken in multiple chunks")
        else:
            print("❌ Chunked TTS test failed")
        
        # Test different chunk sizes
        print(f"\n🧪 Testing with smaller chunk size (500 chars)...")
        success = tts.speak(long_text.strip(), chunk_size=500)
        
        if success:
            print("✅ Small chunk test completed!")
        
    except Exception as e:
        print(f"❌ Error during chunked TTS test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
