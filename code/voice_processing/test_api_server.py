#!/usr/bin/env python3
"""
Simple API Server for Testing
Provides a mock Gradio-style API for testing the voice chatbot integration

This is a test server that mimics Gradio API behavior for development purposes.
Run this server to test the API chatbot without needing actual Gradio setup.

Usage:
    python test_api_server.py
    
Then in another terminal:
    python api_chatbot.py --gradio-url "http://localhost:7860"
"""

from flask import Flask, request, jsonify
import time
import random

app = Flask(__name__)

# Simple responses for testing
test_responses = [
    "Hello! I heard what you said. How can I help you today?",
    "That's interesting! Tell me more about that.",
    "I understand. Is there anything specific you'd like to know?",
    "Thanks for sharing that with me. What else is on your mind?",
    "I'm here to help! Feel free to ask me anything.",
    "That's a great question! Let me think about that.",
    "I appreciate you talking with me. What would you like to discuss next?",
]

conversation_history = []

@app.route('/info', methods=['GET'])
def info():
    """Gradio-style info endpoint"""
    return jsonify({
        "version": "1.0.0",
        "api_version": "0.1.0",
        "space_id": "test-chatbot",
        "app_name": "Test API Server"
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Gradio-style prediction endpoint"""
    try:
        data = request.json
        
        if not data or 'data' not in data:
            return jsonify({"error": "Invalid request format"}), 400
        
        # Extract user message
        user_message = data['data'][0] if data['data'] else ""
        
        if not user_message:
            return jsonify({"error": "Empty message"}), 400
        
        # Store in conversation history
        conversation_history.append({
            "user": user_message,
            "timestamp": time.time()
        })
        
        # Generate response
        if "hello" in user_message.lower() or "hi" in user_message.lower():
            response = "Hello! Nice to meet you through voice chat!"
        elif "how are you" in user_message.lower():
            response = "I'm doing great! Thanks for asking. How are you today?"
        elif "weather" in user_message.lower():
            response = "I don't have access to weather data, but I hope it's beautiful where you are!"
        elif "time" in user_message.lower():
            response = f"The current time is {time.strftime('%H:%M:%S')}."
        elif "joke" in user_message.lower():
            jokes = [
                "Why don't scientists trust atoms? Because they make up everything!",
                "Why did the scarecrow win an award? He was outstanding in his field!",
                "Why don't eggs tell jokes? They'd crack each other up!"
            ]
            response = random.choice(jokes)
        else:
            # Use a random response for other messages
            response = random.choice(test_responses)
            # Personalize it slightly
            response += f" You mentioned: '{user_message}'"
        
        # Add conversation context
        if len(conversation_history) > 1:
            response += f" (This is message #{len(conversation_history)} in our conversation)"
        
        # Store bot response
        conversation_history[-1]["bot"] = response
        
        # Return in Gradio format
        return jsonify({
            "data": [response],
            "duration": round(random.uniform(0.5, 2.0), 2),  # Simulate processing time
            "avg_duration": 1.2
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get conversation history (debugging endpoint)"""
    return jsonify({
        "conversation_count": len(conversation_history),
        "history": conversation_history[-10:]  # Last 10 messages
    })

@app.route('/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    conversation_history.clear()
    return jsonify({"message": "History cleared"})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": "running",
        "conversations": len(conversation_history)
    })

if __name__ == '__main__':
    print("🚀 Starting Test API Server...")
    print("📡 API Endpoints:")
    print("  - GET  /info - API information")
    print("  - POST /api/predict - Chat prediction (Gradio format)")
    print("  - GET  /history - Conversation history")
    print("  - POST /clear - Clear conversation history")
    print("  - GET  /health - Health check")
    print("\n🌐 Server running on: http://localhost:7860")
    print("💡 Test with: python api_chatbot.py --gradio-url http://localhost:7860")
    print("🛑 Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=7860, debug=True)
