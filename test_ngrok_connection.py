#!/usr/bin/env python3
"""
Test ngrok connection and maintain_history functionality
"""
import sys
import os

# Add the code directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from code.llm_agent_linux_package.llm_agent import LLMAgent

def test_ngrok_connection():
    """Test connection to ngrok URL with maintain_history functionality"""
    
    ngrok_url = "https://0987-2601-840-8702-17f0-3d62-8389-f210-1869.ngrok-free.app"
    
    print("Testing ngrok connection and maintain_history functionality...")
    print(f"URL: {ngrok_url}")
    print("=" * 60)
    
    try:
        # Test 1: Create agent with history enabled (default)
        print("\n1. Testing with maintain_history=True (default)")
        agent1 = LLMAgent(
            base_url=ngrok_url,
            api_type="lmstudio",
            maintain_history=True
        )
        
        print(f"   maintain_history setting: {agent1.get_maintain_history()}")
        
        # Test connection
        connection_result = agent1.test_connection()
        print(f"   Connection status: {connection_result['status']}")
        
        if connection_result['status'] == 'connected':
            print("   ✅ Connection successful!")
        else:
            print(f"   ⚠️  Connection issues: {connection_result}")
        
        # Test 2: Create agent with history disabled
        print("\n2. Testing with maintain_history=False")
        agent2 = LLMAgent(
            base_url=ngrok_url,
            api_type="lmstudio",
            maintain_history=False
        )
        
        print(f"   maintain_history setting: {agent2.get_maintain_history()}")
        
        # Test 3: Toggle history setting
        print("\n3. Testing history setting toggle")
        print(f"   Before toggle: {agent2.get_maintain_history()}")
        agent2.set_maintain_history(True)
        print(f"   After toggle: {agent2.get_maintain_history()}")
        
        print("\n✅ All tests completed successfully!")
        
        # Clean up
        agent1.close()
        agent2.close()
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ngrok_connection()
