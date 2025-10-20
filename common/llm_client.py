"""
LLM client wrapper with retries, timeouts, and provider abstraction.
"""
import os
import requests
import time
from typing import Optional, Dict, Any


class LLMClient:
    """Unified client for calling LLM providers (OpenAI, Anthropic, local Ollama, etc.)"""
    
    def __init__(self, provider: str = "openai", model: str = None, timeout: int = 120, max_retries: int = 3):
        self.provider = provider.lower()
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Default models per provider
        if model:
            self.model = model
        elif self.provider == "openai":
            self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        elif self.provider == "anthropic":
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
        elif self.provider == "ollama":
            self.model = os.getenv("OLLAMA_MODEL", "llama2")
        else:
            self.model = "gpt-4"
        
        # API keys
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    def call(self, prompt: str, system_prompt: str = "", max_tokens: int = 4000) -> Dict[str, Any]:
        """
        Call LLM with retry logic.
        Returns: {"text": str, "tokens": int, "cost": float}
        """
        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    return self._call_openai(prompt, system_prompt, max_tokens)
                elif self.provider == "anthropic":
                    return self._call_anthropic(prompt, system_prompt, max_tokens)
                elif self.provider == "ollama":
                    return self._call_ollama(prompt, system_prompt, max_tokens)
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"LLM call failed after {self.max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception("LLM call failed")
    
    def _call_openai(self, prompt: str, system_prompt: str, max_tokens: int) -> Dict[str, Any]:
        """Call OpenAI API."""
        if not self.openai_key:
            raise Exception("OPENAI_API_KEY not set")
        
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        
        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        
        return {"text": text, "tokens": tokens, "cost": 0.0}  # Cost calculation can be added
    
    def _call_anthropic(self, prompt: str, system_prompt: str, max_tokens: int) -> Dict[str, Any]:
        """Call Anthropic API."""
        if not self.anthropic_key:
            raise Exception("ANTHROPIC_API_KEY not set")
        
        headers = {
            "x-api-key": self.anthropic_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        
        text = data["content"][0]["text"]
        tokens = data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0)
        
        return {"text": text, "tokens": tokens, "cost": 0.0}
    
    def _call_ollama(self, prompt: str, system_prompt: str, max_tokens: int) -> Dict[str, Any]:
        """Call local Ollama instance."""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1
            }
        }
        
        resp = requests.post(
            f"{self.ollama_base}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        
        text = data.get("response", "")
        
        return {"text": text, "tokens": 0, "cost": 0.0}

