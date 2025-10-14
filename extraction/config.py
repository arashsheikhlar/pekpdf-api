# SPDX-License-Identifier: AGPL-3.0-only

"""
Configuration service for the extraction system.

This module centralizes all configuration settings for the extraction pipeline,
supporting environment variable overrides and validation.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class ExtractionConfig(BaseSettings):
    """Configuration settings for the extraction system."""
    
    # Cache settings
    cache_ttl_seconds: int = Field(default=86400, description="Cache TTL in seconds (default: 1 day)")
    cache_enabled: bool = Field(default=True, description="Enable caching")
    pipeline_version: str = Field(default="v3-ai-only", description="Pipeline version for cache invalidation")
    
    # Text extraction settings
    max_text_length: int = Field(default=20000, description="Maximum text length for AI processing")
    max_workers: int = Field(default=4, description="Maximum worker threads for parallel processing")
    
    # AI service settings
    ai_timeout: int = Field(default=180, description="AI service timeout in seconds")
    ai_service: str = Field(default="ollama", description="AI service provider (ollama, openai, anthropic)")
    
    # Ollama settings
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    ollama_model: str = Field(default="llama3.1:8b", description="Ollama model name")
    ollama_num_ctx: int = Field(default=16384, description="Ollama context window size")
    ollama_num_predict: int = Field(default=4096, description="Ollama max tokens to predict")
    
    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model name")
    
    # Anthropic settings
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    anthropic_model: str = Field(default="claude-3-haiku-20240307", description="Anthropic model name")
    
    # Synthesis settings
    synthesis_per_file_max: int = Field(default=1200, description="Max characters per file for synthesis")
    synthesis_max_chars: int = Field(default=8000, description="Max total characters for synthesis")
    
    # Upload settings
    upload_folder: str = Field(default="uploads", description="Upload folder path")
    max_file_size: int = Field(default=50 * 1024 * 1024, description="Max file size in bytes (50MB)")
    
    # Plugin settings
    plugins_enabled: bool = Field(default=True, description="Enable plugin system")
    plugin_directory: str = Field(default="extraction/plugins", description="Plugin directory path")
    
    class Config:
        env_prefix = "EXTRACTION_"
        case_sensitive = False
    
    def get_ai_config(self) -> dict:
        """Get AI service configuration."""
        return {
            "service": self.ai_service,
            "timeout": self.ai_timeout,
            "ollama": {
                "base_url": self.ollama_base_url,
                "model": self.ollama_model,
                "num_ctx": self.ollama_num_ctx,
                "num_predict": self.ollama_num_predict,
            },
            "openai": {
                "api_key": self.openai_api_key,
                "model": self.openai_model,
            },
            "anthropic": {
                "api_key": self.anthropic_api_key,
                "model": self.anthropic_model,
            }
        }
    
    def get_cache_config(self) -> dict:
        """Get cache configuration."""
        return {
            "enabled": self.cache_enabled,
            "ttl_seconds": self.cache_ttl_seconds,
            "pipeline_version": self.pipeline_version,
        }
    
    def get_text_config(self) -> dict:
        """Get text extraction configuration."""
        return {
            "max_length": self.max_text_length,
            "max_workers": self.max_workers,
        }
    
    def get_upload_config(self) -> dict:
        """Get upload configuration."""
        return {
            "folder": self.upload_folder,
            "max_file_size": self.max_file_size,
        }
    
    def get_plugin_config(self) -> dict:
        """Get plugin configuration."""
        return {
            "enabled": self.plugins_enabled,
            "directory": self.plugin_directory,
        }
    
    def validate_ai_config(self) -> bool:
        """Validate AI configuration."""
        if self.ai_service == "openai" and not self.openai_api_key:
            return False
        if self.ai_service == "anthropic" and not self.anthropic_api_key:
            return False
        return True
    
    def get_effective_upload_folder(self) -> str:
        """Get the effective upload folder path."""
        if os.path.isabs(self.upload_folder):
            return self.upload_folder
        return os.path.join(os.getcwd(), self.upload_folder)


# Global configuration instance
config = ExtractionConfig()
