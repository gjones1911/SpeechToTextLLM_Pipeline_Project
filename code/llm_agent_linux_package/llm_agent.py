"""
LLM Agent Class

A flexible agent class for interacting with various LLM API endpoints.
Supports text-based messaging and parameter adjustment.

Author: Gerald L. Jones Jr.
Date: 2025-07-06
"""

import requests
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from urllib.parse import urljoin
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_API_TYPE = "lmstudio"  # Default API type if not specified (LM Studio)
DEFAULT_BASE_URL = "http://localhost:1234"  # Default base URL for LM Studio
MAX_RETRIES = 3
RETRY_DELAY = 1.0

@dataclass
class GenerationConfig:
    """Configuration for LLM generation parameters"""
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = .95
    top_k: int = 50
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def update(self, **kwargs) -> None:
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")


@dataclass
class Message:
    """Represents a chat message"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }


class LLMAgentError(Exception):
    """Custom exception for LLM Agent errors"""
    pass


class LLMAgent:
    """
    Agent for interacting with LLM API endpoints.
    
    Supports multiple API formats:
    - Gradio apps (like your lmstudio_gradio.py)
    - LM Studio direct API
    - OpenAI-compatible APIs
    - Custom endpoints
    """
    PWD_ENV = "LLM_AGENT_PWD"  # Environment variable for password (if needed)
    API_KEY_ENV = "LLM_AGENT_API_KEY"  # Environment variable for API key (if needed)
    API_TYPES = ["lmstudio", "gradio", "openai", "custom"]  # Supported API types

    def __init__(self, 
                 base_url: str,
                 api_type: str = "lmstudio",
                 api_key: Optional[str] = None,
                 default_model: Optional[str] = None,
                 timeout: int = 30,
                 max_retries: int = MAX_RETRIES,
                 retry_delay: float = RETRY_DELAY,
                 enable_logging: bool = True,
                 maintain_history: bool = True):
        """
        Initialize the LLM Agent
        
        Args:
            base_url: Base URL of the API endpoint
            api_type: Type of API ("lmstudio", "gradio", "openai", "custom") - default is "lmstudio"
            api_key: API key if required (also checks environment variables)
            default_model: Default model to use
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            enable_logging: Whether to enable detailed logging
            maintain_history: Whether to maintain conversation history
        """
        self.base_url = base_url.rstrip('/')
        self.api_type = api_type.lower()
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv(self.API_KEY_ENV)
        
        self.default_model = default_model
        self.maintain_history = maintain_history
        
        # Configure logging
        if enable_logging:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
            
        logger.info(f"Initializing LLM Agent: {api_type} at {base_url}")
        
        # Validate API type
        if self.api_type not in self.API_TYPES:
            raise LLMAgentError(f"Unsupported API type: {api_type}. Supported: {self.API_TYPES}")
        
        # Message history
        self.messages: List[Message] = []
        
        # Generation configuration
        self.config = GenerationConfig()
        
        # Session for connection pooling
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        
        # Set default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "LLMAgent/1.0"
        })
        
        # Request configuration with SSL bypass for ngrok
        self.request_kwargs = {
            "verify": False,  # 🔑 Essential for ngrok SSL bypass
            "headers": {"ngrok-skip-browser-warning": "any"},
            "timeout": self.timeout
        }
        
        # Add ngrok-specific headers for LM Studio mode
        if self.api_type == "lmstudio":
            self.session.headers.update({
                "ngrok-skip-browser-warning": "true"  # Skip ngrok browser warning
            })
        
        # API endpoint mapping
        self._setup_endpoints()
        
        # Connection status
        self._last_connection_test = None
        self._connection_status = None
    
    def _setup_endpoints(self):
        """Setup API endpoints based on API type"""
        if self.api_type == "gradio":
            self.chat_endpoint = f"{self.base_url}/api/predict"
            self.models_endpoint = None  # Gradio apps may not expose models endpoint
        elif self.api_type == "lmstudio":
            self.chat_endpoint = f"{self.base_url}/v1/chat/completions"
            self.models_endpoint = f"{self.base_url}/v1/models"
        elif self.api_type == "openai":
            self.chat_endpoint = f"{self.base_url}/v1/chat/completions"
            self.models_endpoint = f"{self.base_url}/v1/models"
        else:  # custom
            self.chat_endpoint = f"{self.base_url}/chat"
            self.models_endpoint = f"{self.base_url}/models"
            
        logger.info(f"Chat endpoint: {self.chat_endpoint}")
        logger.info(f"Models endpoint: {self.models_endpoint}")
    
    def _make_request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make HTTP request with retry logic"""
        last_exception = None
        print("DEBUG: URL-> ", url)  # Debugging line to check URL
        print("DEBUG: Method-> ", method)  # Debugging line to check HTTP method
        
        # Merge request_kwargs with any additional kwargs
        merged_kwargs = {**self.request_kwargs, **kwargs}
        
        for attempt in range(self.max_retries):
            try:
                if method.upper() not in ["GET", "POST", "PUT", "DELETE"]:
                    raise LLMAgentError(f"Unsupported HTTP method: {method}")
                if method.upper() == "GET":
                    print("DEBUG: GET request detected")
                    # Make the request with SSL bypass
                    response = self.session.get(url, **merged_kwargs)
                else:
                    # For other methods, use the session request
                    print("DEBUG: POST request detected")
                    response = self.session.post(url, **merged_kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    
        # All retries failed
        raise LLMAgentError(f"Request failed after {self.max_retries} attempts: {last_exception}")
    
    def update_config(self, **kwargs) -> None:
        """Update generation configuration"""
        self.config.update(**kwargs)
        logger.info(f"Updated config: {self.config.to_dict()}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current generation configuration"""
        return self.config.to_dict()
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history"""
        message = Message(role=role, content=content)
        self.messages.append(message)
        logger.debug(f"Added message: {role} - {content[:50]}{'...' if len(content) > 50 else ''}")
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        old_count = len(self.messages)
        self.messages.clear()
        logger.info(f"Cleared {old_count} messages from history")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history as list of dictionaries"""
        return [msg.to_dict() for msg in self.messages]
    
    def export_history(self, format_type: str = "json") -> str:
        """Export conversation history in various formats"""
        history = self.get_history()
        
        if format_type.lower() == "json":
            return json.dumps(history, indent=2)
        elif format_type.lower() == "txt":
            lines = []
            for msg in history:
                timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg["timestamp"]))
                lines.append(f"[{timestamp_str}] {msg['role'].upper()}: {msg['content']}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format_type}. Use 'json' or 'txt'")
    
    def load_history(self, history_data: Union[str, List[Dict]]) -> None:
        """Load conversation history from data"""
        if isinstance(history_data, str):
            # Assume JSON string
            history_list = json.loads(history_data)
        else:
            history_list = history_data
            
        self.messages.clear()
        for msg_data in history_list:
            self.messages.append(Message(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=msg_data.get("timestamp")
            ))
        logger.info(f"Loaded {len(self.messages)} messages from history")

    def set_maintain_history(self, maintain: bool) -> None:
        """Set whether to maintain conversation history"""
        old_setting = self.maintain_history
        self.maintain_history = maintain
        logger.info(f"Conversation history maintenance changed from {old_setting} to {maintain}")
    
    def get_maintain_history(self) -> bool:
        """Get current conversation history maintenance setting"""
        return self.maintain_history

    def get_available_models(self) -> List[str]:
        """Fetch available models from the API"""
        if not self.models_endpoint:
            return [self.default_model] if self.default_model else []
        
        try:
            response = self._make_request_with_retry("GET", self.models_endpoint)
            data = response.json()
            
            # Handle different response formats
            if "data" in data:  # OpenAI/LM Studio format
                models = [model["id"] for model in data["data"]]
            elif "models" in data:  # Some custom APIs
                models = data["models"]
            elif isinstance(data, list):  # Direct list
                models = data
            else:
                models = [self.default_model] if self.default_model else []
                
            logger.info(f"Available models: {models}")
            return models
                
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return [self.default_model] if self.default_model else []
    
    def _format_chat_request(self, message: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Format chat request based on API type"""
        model = model or self.default_model or "any"  # Use "any" as fallback for LM Studio
        
        # Build message history for context
        messages = [{"role": msg.role, "content": msg.content} for msg in self.messages]
        messages.append({"role": "user", "content": message})
        
        if self.api_type == "gradio":
            # Gradio format (depends on your specific Gradio app)
            # This assumes the chat function signature: chat(user_input, chathistory)
            return {
                "data": [message, [{"role": msg.role, "content": msg.content} for msg in self.messages]]
            }
        
        elif self.api_type in ["lmstudio", "openai"]:
            # OpenAI-compatible format
            payload = {
                "model": model,
                "messages": messages,
                **self.config.to_dict()
            }
            return payload
        
        else:  # custom
            return {
                "message": message,
                "history": messages,
                "config": self.config.to_dict(),
                "model": model
            }
    
    def send_message(self, 
                    message: str, 
                    model: Optional[str] = None,
                    add_to_history: Optional[bool] = None) -> str:
        """
        Send a message and get response
        
        Args:
            message: The message to send
            model: Model to use (if different from default)
            add_to_history: Whether to add to conversation history (defaults to instance setting)
            
        Returns:
            The assistant's response
        """
        # Use instance setting if not explicitly provided
        if add_to_history is None:
            add_to_history = self.maintain_history
            
        if add_to_history:
            self.add_message("user", message)
        
        try:
            payload = self._format_chat_request(message, model)
            
            response = self._make_request_with_retry("POST", self.chat_endpoint, json=payload)
            response_data = response.json()
            
            # Parse response based on API type
            response_text = self._parse_response(response_data)
            
            if add_to_history:
                self.add_message("assistant", response_text)
            
            return response_text
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {e}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error processing response: {e}"
            logger.error(error_msg)
            return error_msg
    
    def _parse_response(self, response_data: Dict[str, Any]) -> str:
        """Parse response based on API type"""
        try:
            if self.api_type == "gradio":
                # Gradio typically returns: {"data": [result]}
                if "data" in response_data and response_data["data"]:
                    # The response might be in different formats depending on your Gradio app
                    result = response_data["data"][0]
                    if isinstance(result, list) and len(result) > 1:
                        # If it returns [None, chat_history], get the last assistant message
                        chat_history = result[1]
                        if chat_history and isinstance(chat_history, list):
                            last_msg = chat_history[-1]
                            if isinstance(last_msg, dict) and "content" in last_msg:
                                return last_msg["content"]
                    return str(result)
                
            elif self.api_type in ["lmstudio", "openai"]:
                # OpenAI format: {"choices": [{"message": {"content": "..."}}]}
                if "choices" in response_data and response_data["choices"]:
                    return response_data["choices"][0]["message"]["content"]
            
            else:  # custom
                # Try common field names
                for field in ["response", "content", "message", "text"]:
                    if field in response_data:
                        return response_data[field]
            
            # Fallback: return the whole response as string
            return str(response_data)
            
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return f"Error parsing response: {e}"
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to the API"""
        result = {
            "status": "unknown",
            "api_type": self.api_type,
            "base_url": self.base_url,
            "models_available": False,
            "chat_available": False,
            "models": []
        }
        
        # Test models endpoint
        try:
            models = self.get_available_models()
            result["models"] = models
            result["models_available"] = len(models) > 0
        except Exception as e:
            result["models_error"] = str(e)
        
        # Test chat endpoint with a simple message
        try:
            response = self.send_message("Hello", add_to_history=False)
            result["chat_available"] = len(response) > 0 and not response.startswith("Error")
            result["test_response"] = response[:100] + "..." if len(response) > 100 else response
        except Exception as e:
            result["chat_error"] = str(e)
        
        # Overall status
        if result["chat_available"]:
            result["status"] = "connected"
        elif result["models_available"]:
            result["status"] = "partial"
        else:
            result["status"] = "failed"
        
        logger.info(f"Connection test result: {result}")
        self._last_connection_test = result
        return result
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status of the agent and API"""
        health = {
            "agent_status": "healthy",
            "api_reachable": False,
            "response_time_ms": None,
            "models_count": 0,
            "memory_usage": len(self.messages),
            "last_test": self._last_connection_test,
            "config": self.get_config(),
            "endpoints": {
                "chat": self.chat_endpoint,
                "models": self.models_endpoint
            }
        }
        
        # Test response time
        start_time = time.time()
        try:
            test_result = self.test_connection()
            health["response_time_ms"] = int((time.time() - start_time) * 1000)
            health["api_reachable"] = test_result["status"] in ["connected", "partial"]
            health["models_count"] = len(test_result.get("models", []))
            
            if test_result["status"] == "failed":
                health["agent_status"] = "degraded"
                
        except Exception as e:
            health["agent_status"] = "unhealthy"
            health["error"] = str(e)
            
        return health
    
    def stream_message(self, message: str, model: Optional[str] = None) -> str:
        """
        Send a message with streaming support (if API supports it)
        Note: This is a basic implementation. Full streaming would require SSE support.
        """
        # For now, this is the same as send_message but could be extended
        # to support Server-Sent Events (SSE) streaming in the future
        logger.info("Streaming not fully implemented yet, using regular send_message")
        return self.send_message(message, model)
    
    def batch_messages(self, messages: List[str], model: Optional[str] = None) -> List[str]:
        """Send multiple messages and get responses"""
        responses = []
        for i, msg in enumerate(messages):
            logger.info(f"Processing message {i+1}/{len(messages)}")
            try:
                response = self.send_message(msg, model)
                responses.append(response)
            except Exception as e:
                error_response = f"Error processing message {i+1}: {e}"
                logger.error(error_response)
                responses.append(error_response)
                
        return responses
    
    def save_conversation(self, filepath: str, format_type: str = "json") -> bool:
        """Save conversation history to file"""
        try:
            history_data = self.export_history(format_type)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(history_data)
            logger.info(f"Conversation saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False
    
    def load_conversation(self, filepath: str) -> bool:
        """Load conversation history from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                history_data = f.read()
                
            # Try to determine format
            if filepath.endswith('.json'):
                self.load_history(history_data)
            else:
                # Assume JSON format
                self.load_history(history_data)
                
            logger.info(f"Conversation loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            return False
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt"""
        # Remove existing system messages
        self.messages = [msg for msg in self.messages if msg.role != "system"]
        
        # Add new system message at the beginning
        system_msg = Message(role="system", content=system_prompt)
        self.messages.insert(0, system_msg)
        logger.info("System prompt updated")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current conversation"""
        if not self.messages:
            return {"message_count": 0, "roles": {}, "avg_length": 0, "total_chars": 0}
            
        role_counts = {}
        total_chars = 0
        
        for msg in self.messages:
            role_counts[msg.role] = role_counts.get(msg.role, 0) + 1
            total_chars += len(msg.content)
            
        return {
            "message_count": len(self.messages),
            "roles": role_counts,
            "avg_length": total_chars // len(self.messages) if self.messages else 0,
            "total_chars": total_chars,
            "first_message_time": self.messages[0].timestamp if self.messages else None,
            "last_message_time": self.messages[-1].timestamp if self.messages else None
        }

    # Enhanced API Methods for Model Management and Parameter Control
    
    def switch_model(self, model_id: str) -> Dict[str, Any]:
        """
        Switch to a different model
        
        Args:
            model_id: ID of the model to switch to
            
        Returns:
            Dictionary with success status and message
        """
        if self.api_type != "lmstudio":
            return {"success": False, "error": "Model switching only supported for LM Studio"}
        
        try:
            url = f"{self.base_url}/v1/models/manage"
            data = {
                "model_id": model_id,
                "action": "switch"
            }
            
            response = self._make_request_with_retry("POST", url, json=data)
            
            if response.status_code == 200:
                logger.info(f"Successfully switched to model: {model_id}")
                self.default_model = model_id
                return {"success": True, "model": model_id, "message": "Model switched successfully"}
            else:
                error_msg = f"Failed to switch model: {response.status_code}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = f"Error switching model: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def adjust_parameters(self, **params) -> Dict[str, Any]:
        """
        Adjust generation parameters on the server
        
        Args:
            **params: Parameters to adjust (max_tokens, temperature, etc.)
            
        Returns:
            Dictionary with success status and message
        """
        if self.api_type != "lmstudio":
            # For non-LM Studio APIs, just update local config
            self.update_config(**params)
            return {"success": True, "message": "Parameters updated locally", "params": params}
        
        try:
            url = f"{self.base_url}/v1/parameters"
            
            response = self._make_request_with_retry("POST", url, json=params)
            
            if response.status_code == 200:
                # Also update local config
                self.update_config(**params)
                logger.info(f"Successfully adjusted parameters: {params}")
                return {"success": True, "message": "Parameters adjusted on server", "params": params}
            else:
                error_msg = f"Failed to adjust parameters: {response.status_code}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = f"Error adjusting parameters: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Get current generation parameters from the server
        
        Returns:
            Dictionary with current parameters or error
        """
        if self.api_type != "lmstudio":
            return {"success": True, "params": self.config.to_dict(), "source": "local"}
        
        try:
            url = f"{self.base_url}/v1/parameters"
            
            response = self._make_request_with_retry("GET", url)
            
            if response.status_code == 200:
                params = response.json()
                logger.info("Successfully retrieved current parameters")
                return {"success": True, "params": params, "source": "server"}
            else:
                error_msg = f"Failed to get parameters: {response.status_code}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = f"Error getting parameters: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def apply_preset(self, preset_name: str) -> Dict[str, Any]:
        """
        Apply a generation preset (creative, precise, balanced)
        
        Args:
            preset_name: Name of preset ("creative", "precise", "balanced")
            
        Returns:
            Dictionary with success status and message
        """
        if self.api_type != "lmstudio":
            # For non-LM Studio APIs, apply local presets
            presets = {
                "creative": {"temperature": 0.9, "top_p": 0.95, "max_tokens": 1500},
                "precise": {"temperature": 0.3, "top_p": 0.85, "max_tokens": 1000},
                "balanced": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 1200}
            }
            
            if preset_name in presets:
                self.update_config(**presets[preset_name])
                return {"success": True, "preset": preset_name, "message": f"Applied {preset_name} preset locally"}
            else:
                return {"success": False, "error": f"Unknown preset: {preset_name}"}
        
        try:
            url = f"{self.base_url}/v1/presets/{preset_name}/apply"
            
            response = self._make_request_with_retry("POST", url)
            
            if response.status_code == 200:
                logger.info(f"Successfully applied preset: {preset_name}")
                return {"success": True, "preset": preset_name, "message": f"Applied {preset_name} preset on server"}
            else:
                error_msg = f"Failed to apply preset: {response.status_code}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = f"Error applying preset: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def get_available_models(self) -> Dict[str, Any]:
        """
        Get list of available models
        
        Returns:
            Dictionary with models list or error
        """
        try:
            if not self.models_endpoint:
                return {"success": False, "error": "Models endpoint not available for this API type"}
            
            response = self._make_request_with_retry("GET", self.models_endpoint)
            
            if response.status_code == 200:
                models_data = response.json()
                logger.info("Successfully retrieved available models")
                return {"success": True, "models": models_data}
            else:
                error_msg = f"Failed to get models: {response.status_code}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = f"Error getting models: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def close(self) -> None:
        """Clean up resources"""
        if hasattr(self, 'session'):
            self.session.close()
        logger.info("Agent resources cleaned up")
    

# Usage examples and factory functions
def create_gradio_agent(url: str, **kwargs) -> LLMAgent:
    """Create an agent for Gradio apps"""
    return LLMAgent(url, api_type="gradio", **kwargs)

def create_lmstudio_agent(url: str = "http://localhost:1234", **kwargs) -> LLMAgent:
    """Create an agent for LM Studio"""
    return LLMAgent(url, api_type="lmstudio", **kwargs)

def create_openai_agent(url: str, api_key: str, **kwargs) -> LLMAgent:
    """Create an agent for OpenAI-compatible APIs"""
    return LLMAgent(url, api_type="openai", api_key=api_key, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("LLM Agent Class - Example Usage")
    print("=" * 40)
    
    # Test with local Gradio app
    gradio_agent = create_gradio_agent("http://localhost:7860")
    print(f"Created: {gradio_agent}")
    
    # Test connection
    test_result = gradio_agent.test_connection()
    print(f"Connection test: {test_result}")
    
    # Example conversation
    if test_result["status"] == "connected":
        print("\nStarting conversation...")
        gradio_agent.update_config(temperature=0.8, max_tokens=100)
        
        response = gradio_agent.send_message("What's the weather like?")
        print(f"Response: {response}")
        
        print(f"Conversation history: {len(gradio_agent.messages)} messages")
