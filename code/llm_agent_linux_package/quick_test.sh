#!/bin/bash

# Quick Test Script for LLM Agent

echo "🚀 Quick LLM Agent Test"
echo "======================"

NGROK_URL="https://24be-174-161-24-253.ngrok-free.app"

# URL is already set to current ngrok endpoint

echo "🔗 Testing connection to: $NGROK_URL"

# Test with curl first
echo "📡 Testing with curl..."
if command -v curl &> /dev/null; then
    if curl -k -s -H "ngrok-skip-browser-warning: true" "$NGROK_URL/v1/models" | grep -q "data"; then
        echo "✅ curl test successful"
    else
        echo "❌ curl test failed"
        exit 1
    fi
else
    echo "⚠️  curl not available, skipping curl test"
fi

# Test with Python
echo "🐍 Testing with Python..."
if [ -f "remote_llm_test.py" ]; then
    python3 remote_llm_test.py "$NGROK_URL"
else
    echo "❌ remote_llm_test.py not found"
    exit 1
fi
