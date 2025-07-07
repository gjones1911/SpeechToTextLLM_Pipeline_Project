#!/usr/bin/env python3
"""
Remote LLM Agent Test Script

This script can be used on a Linux system to test the LLM Agent
connecting to a Windows machine running LM Studio via ngrok.

Usage:
    python3 remote_llm_test.py <ngrok_url>

Example:
    python3 remote_llm_test.py https://abc123.ngrok-free.app
"""

import sys
import json
import time
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Simplified LLM Agent for remote testing (no external dependencies beyond requests)
@dataclass
class GenerationConfig:
    temperature: float = 0.7
    max_tokens: int = 500
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class SimpleLLMAgent:
    """Simplified LLM Agent for remote testing"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.config = GenerationConfig()
        self.session = requests.Session()
        
        # SSL bypass configuration for ngrok
        self.request_kwargs = {
            "verify": False,  # 🔑 Essential for ngrok SSL bypass
            "headers": {"ngrok-skip-browser-warning": "any"},
            "timeout": self.timeout
        }
        
        # Set headers for ngrok
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "LLMAgent-Remote/1.0",
            "ngrok-skip-browser-warning": "true"  # Skip ngrok browser warning
        })
        
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to the remote API"""
        result = {
            "status": "unknown",
            "base_url": self.base_url,
            "models_available": False,
            "chat_available": False,
            "models": []
        }
        
        try:
            # Test models endpoint
            models_url = f"{self.base_url}/v1/models"
            print(f"Testing models endpoint: {models_url}")
            
            response = self.session.get(models_url, **self.request_kwargs)
            response.raise_for_status()
            
            models_data = response.json()
            if "data" in models_data:
                models = [model["id"] for model in models_data["data"]]
                result["models"] = models
                result["models_available"] = len(models) > 0
                print(f"✅ Found {len(models)} models: {models}")
            
        except Exception as e:
            result["models_error"] = str(e)
            print(f"❌ Models endpoint error: {e}")
        
        try:
            # Test chat endpoint
            chat_url = f"{self.base_url}/v1/chat/completions"
            print(f"Testing chat endpoint: {chat_url}")
            
            test_payload = {
                "model": "any",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 50,
                "temperature": 0.7
            }
            
            response = self.session.post(chat_url, json=test_payload, **self.request_kwargs)
            response.raise_for_status()
            
            chat_data = response.json()
            if "choices" in chat_data and chat_data["choices"]:
                test_response = chat_data["choices"][0]["message"]["content"]
                result["chat_available"] = True
                result["test_response"] = test_response
                print(f"✅ Chat test successful: {test_response[:50]}...")
            
        except Exception as e:
            result["chat_error"] = str(e)
            print(f"❌ Chat endpoint error: {e}")
        
        # Overall status
        if result["chat_available"]:
            result["status"] = "connected"
        elif result["models_available"]:
            result["status"] = "partial"
        else:
            result["status"] = "failed"
            
        return result
    
    def send_message(self, message: str, model: Optional[str] = None) -> str:
        """Send a message and get response"""
        try:
            chat_url = f"{self.base_url}/v1/chat/completions"
            
            payload = {
                "model": model or "any",
                "messages": [{"role": "user", "content": message}],
                **self.config.to_dict()
            }
            
            response = self.session.post(chat_url, json=payload, **self.request_kwargs)
            response.raise_for_status()
            
            data = response.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            else:
                return "No response content found"
                
        except Exception as e:
            return f"Error: {e}"
    
    def update_config(self, **kwargs):
        """Update generation configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)


def test_remote_connection(ngrok_url: str):
    """Test connection to remote LM Studio via ngrok"""
    print(f"🔗 Testing remote connection to: {ngrok_url}")
    print("=" * 60)
    
    # Create agent
    agent = SimpleLLMAgent(ngrok_url)
    
    # Test connection
    print("🧪 Running connection test...")
    result = agent.test_connection()
    
    print(f"\n📊 Connection Test Results:")
    print(f"  Status: {result['status']}")
    print(f"  Models Available: {result['models_available']}")
    print(f"  Chat Available: {result['chat_available']}")
    
    if result.get("models"):
        print(f"  Available Models: {result['models']}")
    
    if result.get("test_response"):
        print(f"  Test Response: {result['test_response']}")
    
    if result.get("models_error"):
        print(f"  Models Error: {result['models_error']}")
        
    if result.get("chat_error"):
        print(f"  Chat Error: {result['chat_error']}")
    
    # If successful, run interactive test
    if result["status"] == "connected":
        print("\n✅ Connection successful! Starting interactive test...")
        interactive_test(agent)
    else:
        print(f"\n❌ Connection failed with status: {result['status']}")
        print("Please check:")
        print("  1. LM Studio is running with server enabled")
        print("  2. ngrok tunnel is active")
        print("  3. The ngrok URL is correct")
        print("  4. A model is loaded in LM Studio")


def interactive_test(agent: SimpleLLMAgent):
    """Interactive testing session"""
    print("\n💬 Interactive Test Session")
    print("Type 'quit' to exit, 'config' to show configuration")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'config':
                print(f"Current config: {agent.config.to_dict()}")
                continue
            elif not user_input:
                continue
            
            print("🤔 Sending message...")
            start_time = time.time()
            response = agent.send_message(user_input)
            response_time = time.time() - start_time
            
            print(f"Assistant ({response_time:.1f}s): {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Interactive test completed")


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python3 remote_llm_test.py <ngrok_url>")
        print("Example: python3 remote_llm_test.py https://abc123.ngrok-free.app")
        sys.exit(1)
    
    ngrok_url = sys.argv[1]
    
    # Validate URL
    if not ngrok_url.startswith(('http://', 'https://')):
        print("❌ Error: URL must start with http:// or https://")
        sys.exit(1)
    
    # Test the connection
    test_remote_connection(ngrok_url)


if __name__ == "__main__":
    main()
