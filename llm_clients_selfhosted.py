"""
Self-hosted LLM client implementations for Enterprise Deep Research
Supports: Ollama, LocalAI, vLLM, llama.cpp, and other OpenAI-compatible endpoints
"""

import os
import json
import requests
import logging
from typing import Optional, List, Dict, Any, Iterator
from langchain.schema import ChatMessage, HumanMessage, SystemMessage, AIMessage
from langchain.chat_models.base import BaseChatModel
from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for Ollama self-hosted models.
    Ollama provides a simple way to run large language models locally.
    """

    def __init__(
        self,
        model_name: str = "llama3.3:latest",
        base_url: str = None,
        timeout: int = 120,
        max_tokens: int = 4096,
    ):
        """
        Initialize Ollama client.
        
        Args:
            model_name: Name of the Ollama model (e.g., 'llama3.3:latest', 'mistral', 'codellama')
            base_url: Base URL for Ollama API (defaults to environment variable or localhost)
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.model = model_name  # For compatibility
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama service is accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            response.raise_for_status()
            logger.info(f"Connected to Ollama at {self.base_url}")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama at {self.base_url}: {e}")
    
    @traceable
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def invoke(self, messages, config=None, stream=False):
        """
        Invoke Ollama model with the given messages.
        
        Args:
            messages: List of LangChain message objects
            config: Optional configuration
            stream: Whether to stream the response
            
        Returns:
            AI message response or iterator if streaming
        """
        # Convert LangChain messages to Ollama format
        ollama_messages = []
        system_content = ""
        
        for msg in messages:
            if isinstance(msg, SystemMessage) or (hasattr(msg, 'type') and msg.type == "system"):
                system_content = msg.content
            elif isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == "human"):
                ollama_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) or (hasattr(msg, 'type') and msg.type == "ai"):
                ollama_messages.append({"role": "assistant", "content": msg.content})
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "messages": ollama_messages,
            "stream": stream,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }
        
        if system_content:
            payload["system"] = system_content
        
        url = f"{self.base_url}/api/chat"
        
        try:
            logger.info(f"Calling Ollama API with model {self.model_name}")
            
            if stream:
                return self._stream_response(url, payload)
            else:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                content = result.get("message", {}).get("content", "")
                
                return content
        
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise
    
    def _stream_response(self, url: str, payload: Dict[str, Any]) -> Iterator[str]:
        """Stream response from Ollama."""
        with requests.post(url, json=payload, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk:
                            content = chunk["message"].get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "ollama"
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate method required by BaseChatModel."""
        response = self.invoke(messages)
        return ChatMessage(content=response, role="assistant")


class LocalAIClient:
    """
    Client for LocalAI - OpenAI-compatible local inference.
    LocalAI supports multiple model backends including llama.cpp, GPT4All, etc.
    """
    
    def __init__(
        self,
        model_name: str = "llama-3-8b",
        base_url: str = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_tokens: int = 4096,
    ):
        """
        Initialize LocalAI client.
        
        Args:
            model_name: Name of the model in LocalAI
            base_url: Base URL for LocalAI API
            api_key: Optional API key
            timeout: Request timeout
            max_tokens: Maximum tokens to generate
        """
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url or os.getenv("LOCALAI_BASE_URL", "http://localhost:8080")
        self.api_key = api_key or os.getenv("LOCALAI_API_KEY", "")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.model = model_name
    
    @traceable
    def invoke(self, messages, config=None, stream=False):
        """
        Invoke LocalAI model using OpenAI-compatible API.
        """
        # Convert to OpenAI format
        openai_messages = []
        for msg in messages:
            role = "user"
            if isinstance(msg, SystemMessage) or (hasattr(msg, 'type') and msg.type == "system"):
                role = "system"
            elif isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == "human"):
                role = "user"
            elif isinstance(msg, AIMessage) or (hasattr(msg, 'type') and msg.type == "ai"):
                role = "assistant"
            
            openai_messages.append({"role": role, "content": msg.content})
        
        payload = {
            "model": self.model_name,
            "messages": openai_messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
            "stream": stream
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        url = f"{self.base_url}/v1/chat/completions"
        
        try:
            logger.info(f"Calling LocalAI API with model {self.model_name}")
            
            if stream:
                return self._stream_response(url, headers, payload)
            else:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                return content
        
        except Exception as e:
            logger.error(f"Error calling LocalAI API: {e}")
            raise
    
    def _stream_response(self, url: str, headers: Dict, payload: Dict) -> Iterator[str]:
        """Stream response from LocalAI."""
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line and line != b"data: [DONE]":
                    try:
                        if line.startswith(b"data: "):
                            chunk = json.loads(line[6:])
                            if "choices" in chunk:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                    except json.JSONDecodeError:
                        continue
    
    @property
    def _llm_type(self) -> str:
        return "localai"
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.invoke(messages)
        return ChatMessage(content=response, role="assistant")


class VLLMClient:
    """
    Client for vLLM - high-performance inference server.
    vLLM provides OpenAI-compatible API with optimized inference.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        base_url: str = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_tokens: int = 4096,
    ):
        """
        Initialize vLLM client.
        
        Args:
            model_name: Name/path of the model in vLLM
            base_url: Base URL for vLLM API
            api_key: Optional API key
            timeout: Request timeout
            max_tokens: Maximum tokens to generate
        """
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8001")
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.model = model_name
    
    @traceable
    def invoke(self, messages, config=None, stream=False):
        """
        Invoke vLLM model using OpenAI-compatible API.
        """
        # Convert to OpenAI format
        openai_messages = []
        for msg in messages:
            role = "user"
            if isinstance(msg, SystemMessage) or (hasattr(msg, 'type') and msg.type == "system"):
                role = "system"
            elif isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == "human"):
                role = "user"
            elif isinstance(msg, AIMessage) or (hasattr(msg, 'type') and msg.type == "ai"):
                role = "assistant"
            
            openai_messages.append({"role": role, "content": msg.content})
        
        payload = {
            "model": self.model_name,
            "messages": openai_messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": stream
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        url = f"{self.base_url}/v1/chat/completions"
        
        try:
            logger.info(f"Calling vLLM API with model {self.model_name}")
            
            if stream:
                return self._stream_response(url, headers, payload)
            else:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                return content
        
        except Exception as e:
            logger.error(f"Error calling vLLM API: {e}")
            raise
    
    def _stream_response(self, url: str, headers: Dict, payload: Dict) -> Iterator[str]:
        """Stream response from vLLM."""
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line and line != b"data: [DONE]":
                    try:
                        if line.startswith(b"data: "):
                            chunk = json.loads(line[6:])
                            if "choices" in chunk:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                    except json.JSONDecodeError:
                        continue
    
    @property
    def _llm_type(self) -> str:
        return "vllm"
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.invoke(messages)
        return ChatMessage(content=response, role="assistant")


class GenericOpenAICompatibleClient:
    """
    Generic client for any OpenAI-compatible API endpoint.
    Works with: Text Generation WebUI, llama.cpp server, FastChat, etc.
    """
    
    def __init__(
        self,
        model_name: str = "default",
        base_url: str = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_tokens: int = 4096,
        provider_name: str = "openai-compatible",
    ):
        """
        Initialize generic OpenAI-compatible client.
        
        Args:
            model_name: Name of the model
            base_url: Base URL for the API
            api_key: Optional API key
            timeout: Request timeout
            max_tokens: Maximum tokens to generate
            provider_name: Name of the provider for logging
        """
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000")
        self.api_key = api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY", "")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.provider_name = provider_name
        self.model = model_name
    
    @traceable
    def invoke(self, messages, config=None, stream=False):
        """
        Invoke model using OpenAI-compatible API.
        """
        # Convert to OpenAI format
        openai_messages = []
        for msg in messages:
            role = "user"
            if isinstance(msg, SystemMessage) or (hasattr(msg, 'type') and msg.type == "system"):
                role = "system"
            elif isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == "human"):
                role = "user"
            elif isinstance(msg, AIMessage) or (hasattr(msg, 'type') and msg.type == "ai"):
                role = "assistant"
            
            openai_messages.append({"role": role, "content": msg.content})
        
        payload = {
            "model": self.model_name,
            "messages": openai_messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
            "stream": stream
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Try different endpoint patterns
        endpoints = [
            f"{self.base_url}/v1/chat/completions",
            f"{self.base_url}/chat/completions",
            f"{self.base_url}/api/v1/chat/completions",
        ]
        
        last_error = None
        for url in endpoints:
            try:
                logger.info(f"Calling {self.provider_name} API at {url} with model {self.model_name}")
                
                if stream:
                    return self._stream_response(url, headers, payload)
                else:
                    response = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    return content
            
            except Exception as e:
                last_error = e
                logger.debug(f"Failed with endpoint {url}: {e}")
                continue
        
        logger.error(f"All endpoints failed for {self.provider_name}: {last_error}")
        raise last_error
    
    def _stream_response(self, url: str, headers: Dict, payload: Dict) -> Iterator[str]:
        """Stream response from OpenAI-compatible API."""
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line and line != b"data: [DONE]":
                    try:
                        if line.startswith(b"data: "):
                            chunk = json.loads(line[6:])
                            if "choices" in chunk:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                    except json.JSONDecodeError:
                        continue
    
    @property
    def _llm_type(self) -> str:
        return self.provider_name
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.invoke(messages)
        return ChatMessage(content=response, role="assistant")


# Configuration for self-hosted models
SELFHOSTED_MODEL_CONFIGS = {
    "ollama": {
        "available_models": [
            "llama3.3:latest",
            "llama3.2:latest",
            "llama3.1:8b",
            "mistral:latest",
            "codellama:latest",
            "phi3:latest",
            "gemma2:latest",
            "qwen2.5:latest",
        ],
        "default_model": "llama3.3:latest",
        "client_class": OllamaClient,
    },
    "localai": {
        "available_models": [
            "llama-3-8b",
            "mistral-7b",
            "gpt4all-j",
            "vicuna-13b",
        ],
        "default_model": "llama-3-8b",
        "client_class": LocalAIClient,
    },
    "vllm": {
        "available_models": [
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "codellama/CodeLlama-7b-Instruct-hf",
        ],
        "default_model": "meta-llama/Llama-3.1-8B-Instruct",
        "client_class": VLLMClient,
    },
    "openai-compatible": {
        "available_models": ["default"],
        "default_model": "default",
        "client_class": GenericOpenAICompatibleClient,
    },
}


def get_selfhosted_llm_client(provider: str, model_name: Optional[str] = None, **kwargs):
    """
    Factory function to get self-hosted LLM client.
    
    Args:
        provider: Provider name (ollama, localai, vllm, openai-compatible)
        model_name: Optional model name
        **kwargs: Additional arguments for the client
        
    Returns:
        Initialized LLM client
    """
    if provider not in SELFHOSTED_MODEL_CONFIGS:
        raise ValueError(f"Unknown self-hosted provider: {provider}")
    
    config = SELFHOSTED_MODEL_CONFIGS[provider]
    model_name = model_name or config["default_model"]
    client_class = config["client_class"]
    
    logger.info(f"Creating {provider} client with model {model_name}")
    
    return client_class(model_name=model_name, **kwargs)