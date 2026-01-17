"""
Unified AI client factory that supports both Claude and OpenAI.
"""
import os
from typing import Dict, Any, Optional

from health_data.claude_client import ClaudeClient
from health_data.openai_client import OpenAIClient


def get_ai_client(config=None) -> Any:
    """
    Factory function to get the appropriate AI client based on available API keys.
    
    Args:
        config: Optional Config object for model settings
        
    Returns:
        ClaudeClient or OpenAIClient instance
        
    Raises:
        ValueError: If neither API key is found
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not anthropic_key and not openai_key:
        raise ValueError(
            "No API key found. Please set either ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file"
        )
    
    # Prefer Anthropic if both are available (for backward compatibility)
    if anthropic_key:
        if config:
            return ClaudeClient(
                model=config.claude_model,
                model_alternatives=config.claude_model_alternatives,
                max_tokens=config.claude_max_tokens
            )
        else:
            return ClaudeClient()
    
    # Use OpenAI if only OpenAI key is available
    if openai_key:
        if config:
            try:
                return OpenAIClient(
                    model=config.openai_model,
                    model_alternatives=config.openai_model_alternatives,
                    max_tokens=config.openai_max_tokens
                )
            except AttributeError:
                # Fallback if config doesn't have OpenAI settings (backward compatibility)
                return OpenAIClient()
        else:
            return OpenAIClient()
    
    # Should not reach here, but just in case
    raise ValueError("Unable to initialize AI client")

