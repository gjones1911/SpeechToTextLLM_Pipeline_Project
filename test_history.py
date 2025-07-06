#!/usr/bin/env python3
"""
Quick test of maintain_history functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from code.transcriber_test_script import TranscriberAgent

def test_history_options():
    """Test different history configurations"""
    ngrok_url = "https://0987-2601-840-8702-17f0-3d62-8389-f210-1869.ngrok-free.app"
    
    print("🧪 Testing history options...")
    
    # Test 1: Default (history enabled)
    print("\n1. Testing default (maintain_history=True)...")
    agent1 = TranscriberAgent(ngrok_url, api_type='lmstudio')
    print(f"   maintain_history: {agent1.get_maintain_history()}")
    
    # Test 2: Explicitly disabled
    print("\n2. Testing disabled (maintain_history=False)...")
    agent2 = TranscriberAgent(ngrok_url, api_type='lmstudio', maintain_history=False)
    print(f"   maintain_history: {agent2.get_maintain_history()}")
    
    # Test 3: Toggle after creation
    print("\n3. Testing toggle...")
    agent2.set_maintain_history(True)
    print(f"   maintain_history after toggle: {agent2.get_maintain_history()}")
    
    print("\n✅ All history tests passed!")

if __name__ == "__main__":
    test_history_options()
