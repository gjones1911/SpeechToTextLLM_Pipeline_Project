#!/bin/bash

# Quick Test Script for LLM Agent

echo "🚀 Quick LLM Agent Test"
echo "======================"

NGROK_URL="https://0987-2601-840-8702-17f0-3d62-8389-f210-1869.ngrok-free.app"

if [ "$NGROK_URL" = "https://0987-2601-840-8702-17f0-3d62-8389-f210-1869.ngrok-free.app" ]; then
    echo "❌ Please update the ngrok URL in this script"
    echo "   Edit this file and replace https://0987-2601-840-8702-17f0-3d62-8389-f210-1869.ngrok-free.app with your actual ngrok URL"
    exit 1
fi

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
