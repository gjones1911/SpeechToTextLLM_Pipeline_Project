"""
Enhanced LLM Agent Test Script

Demonstrates advanced features of the LLMAgent class including:
- Health monitoring
- Batch processing
- Conversation management
- System prompts
- Statistics

Usage:
    python test_llm_agent_enhanced.py
"""

import sys
import os
import time
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_agent import LLMAgent, create_gradio_agent, create_lmstudio_agent


def test_health_monitoring():
    """Test health monitoring features"""
    print("🏥 Testing Health Monitoring...")
    
    agent = create_gradio_agent("http://localhost:7860")
    
    # Get health status
    health = agent.get_health_status()
    print(f"Agent Health: {health['agent_status']}")
    print(f"API Reachable: {health['api_reachable']}")
    print(f"Response Time: {health['response_time_ms']}ms")
    print(f"Models Available: {health['models_count']}")
    print(f"Memory Usage: {health['memory_usage']} messages")
    
    return agent


def test_conversation_management():
    """Test conversation save/load and statistics"""
    print("\n💾 Testing Conversation Management...")
    
    agent = create_gradio_agent("http://localhost:7860")
    
    # Set a system prompt
    agent.set_system_prompt("You are a helpful coding assistant. Always provide concise, practical answers.")
    
    # Have a short conversation
    if agent.test_connection()["status"] == "connected":
        responses = [
            agent.send_message("What is Python?"),
            agent.send_message("How do I create a list in Python?"),
            agent.send_message("Thank you!")
        ]
        
        # Get conversation statistics
        stats = agent.get_conversation_stats()
        print(f"Conversation Stats: {json.dumps(stats, indent=2)}")
        
        # Save conversation
        agent.save_conversation("test_conversation.json", "json")
        agent.save_conversation("test_conversation.txt", "txt")
        print("✅ Conversation saved to files")
        
        # Create new agent and load conversation
        new_agent = create_gradio_agent("http://localhost:7860")
        success = new_agent.load_conversation("test_conversation.json")
        if success:
            print(f"✅ Conversation loaded: {len(new_agent.messages)} messages")
            new_stats = new_agent.get_conversation_stats()
            print(f"Loaded Stats: {json.dumps(new_stats, indent=2)}")
    else:
        print("❌ API not available for conversation test")
    
    return agent


def test_batch_processing():
    """Test batch message processing"""
    print("\n📦 Testing Batch Processing...")
    
    agent = create_lmstudio_agent()
    
    if agent.test_connection()["status"] == "connected":
        # Test batch messages
        questions = [
            "What is 2 + 2?",
            "Name a programming language",
            "What color is the sky?"
        ]
        
        print("Processing batch of questions...")
        responses = agent.batch_messages(questions)
        
        for i, (question, response) in enumerate(zip(questions, responses)):
            print(f"Q{i+1}: {question}")
            print(f"A{i+1}: {response[:100]}{'...' if len(response) > 100 else ''}")
            print()
    else:
        print("❌ LM Studio not available for batch test")
    
    return agent


def test_advanced_config():
    """Test advanced configuration features"""
    print("\n⚙️ Testing Advanced Configuration...")
    
    agent = create_gradio_agent("http://localhost:7860")
    
    # Show initial config
    print(f"Initial config: {agent.get_config()}")
    
    # Update multiple parameters
    agent.update_config(
        temperature=0.9,
        max_tokens=200,
        top_p=0.95,
        frequency_penalty=0.1
    )
    
    print(f"Updated config: {agent.get_config()}")
    
    # Test with different settings
    if agent.test_connection()["status"] == "connected":
        print("Testing with high creativity settings...")
        response = agent.send_message("Write a creative story opening in one sentence.")
        print(f"Creative response: {response}")
        
        # Reset to conservative settings
        agent.update_config(temperature=0.3, max_tokens=50)
        print("Testing with conservative settings...")
        response = agent.send_message("What is 5 + 3?")
        print(f"Conservative response: {response}")
    
    return agent


def interactive_enhanced_chat():
    """Enhanced interactive chat with advanced features"""
    print("\n💬 Enhanced Interactive Chat")
    print("Commands:")
    print("  'quit' - Exit chat")
    print("  'clear' - Clear history")
    print("  'config' - Show configuration")
    print("  'stats' - Show conversation statistics")
    print("  'health' - Show health status")
    print("  'save <filename>' - Save conversation")
    print("  'load <filename>' - Load conversation")
    print("  'system <prompt>' - Set system prompt")
    print("  'batch' - Enter batch mode")
    
    # Choose agent type
    agent_type = input("\nChoose agent type (gradio/lmstudio): ").strip().lower()
    if agent_type == "lmstudio":
        agent = create_lmstudio_agent()
    else:
        agent = create_gradio_agent("http://localhost:7860")
    
    print(f"\nStarted {agent.api_type} agent")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
                
            elif user_input.lower() == 'clear':
                agent.clear_history()
                print("🧹 History cleared")
                
            elif user_input.lower() == 'config':
                print(f"Config: {json.dumps(agent.get_config(), indent=2)}")
                
            elif user_input.lower() == 'stats':
                stats = agent.get_conversation_stats()
                print(f"Stats: {json.dumps(stats, indent=2)}")
                
            elif user_input.lower() == 'health':
                health = agent.get_health_status()
                print(f"Health: {json.dumps(health, indent=2)}")
                
            elif user_input.lower().startswith('save '):
                filename = user_input[5:].strip()
                if agent.save_conversation(filename):
                    print(f"✅ Saved to {filename}")
                else:
                    print("❌ Save failed")
                    
            elif user_input.lower().startswith('load '):
                filename = user_input[5:].strip()
                if agent.load_conversation(filename):
                    print(f"✅ Loaded from {filename}")
                else:
                    print("❌ Load failed")
                    
            elif user_input.lower().startswith('system '):
                system_prompt = user_input[7:].strip()
                agent.set_system_prompt(system_prompt)
                print("✅ System prompt set")
                
            elif user_input.lower() == 'batch':
                print("Enter messages (empty line to finish):")
                batch_messages = []
                while True:
                    msg = input(f"Message {len(batch_messages)+1}: ").strip()
                    if not msg:
                        break
                    batch_messages.append(msg)
                
                if batch_messages:
                    print("Processing batch...")
                    responses = agent.batch_messages(batch_messages)
                    for i, (msg, resp) in enumerate(zip(batch_messages, responses)):
                        print(f"Q{i+1}: {msg}")
                        print(f"A{i+1}: {resp[:100]}{'...' if len(resp) > 100 else ''}")
                        print()
                        
            elif not user_input:
                continue
                
            else:
                print("🤔 Thinking...")
                response = agent.send_message(user_input)
                print(f"Assistant: {response}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Cleanup
    agent.close()
    print("\n👋 Chat ended")


def main():
    """Main test function"""
    print("🚀 Enhanced LLM Agent Test Suite")
    print("=" * 50)
    
    # Run all tests
    try:
        test_health_monitoring()
        test_conversation_management() 
        test_batch_processing()
        test_advanced_config()
        
        # Ask if user wants interactive chat
        choice = input("\nStart interactive chat? (y/n): ").strip().lower()
        if choice == 'y':
            interactive_enhanced_chat()
            
    except KeyboardInterrupt:
        print("\n\n👋 Tests interrupted")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    
    print("\n✨ Test suite completed")


if __name__ == "__main__":
    main()
