#!/usr/bin/env python3
"""
Debug script to test LLM connection issues
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

def test_basic_connection():
    """Test basic connection functionality"""
    print("🔍 Testing basic LLM connection...")
    
    try:
        from code.llm_agent_linux_package.llm_agent import LLMAgent
        print("✅ LLMAgent import successful")
        
        # Test with ngrok URL
        ngrok_url = "https://0987-2601-840-8702-17f0-3d62-8389-f210-1869.ngrok-free.app"
        print(f"🌐 Testing connection to: {ngrok_url}")
        
        agent = LLMAgent(ngrok_url, api_type='lmstudio')
        print("✅ LLMAgent creation successful")
        
        # Test connection
        print("🧪 Running connection test...")
        result = agent.test_connection()
        
        print(f"📊 Connection Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
            
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_transcriber_agent():
    """Test TranscriberAgent creation"""
    print("\n🤖 Testing TranscriberAgent...")
    
    try:
        from code.transcriber_test_script import TranscriberAgent
        print("✅ TranscriberAgent import successful")
        
        ngrok_url = "https://0987-2601-840-8702-17f0-3d62-8389-f210-1869.ngrok-free.app"
        
        # Create with minimal settings
        agent = TranscriberAgent(
            llm_url=ngrok_url,
            api_type='lmstudio',
            maintain_history=False
        )
        print("✅ TranscriberAgent creation successful")
        
        # Test LLM connection
        result = agent.llm_agent.test_connection()
        print(f"📊 TranscriberAgent LLM Connection:")
        for key, value in result.items():
            print(f"  {key}: {value}")
            
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 Debug Connection Test")
    print("=" * 50)
    
    # Test 1: Basic LLMAgent
    result1 = test_basic_connection()
    
    # Test 2: TranscriberAgent
    result2 = test_transcriber_agent()
    
    print("\n📋 Summary:")
    print(f"  Basic LLMAgent: {'✅ Success' if result1 and result1.get('status') == 'connected' else '❌ Failed'}")
    print(f"  TranscriberAgent: {'✅ Success' if result2 and result2.get('status') == 'connected' else '❌ Failed'}")
