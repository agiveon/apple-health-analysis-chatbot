"""
Claude API client for generating code and text responses.
"""
import os
from typing import List, Dict, Any, Optional
from anthropic import Anthropic


class ClaudeClient:
    """Client for interacting with Claude API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-opus-4-5", 
                 model_alternatives: Optional[List[str]] = None, max_tokens: int = 4096):
        """
        Initialize Claude client.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Primary model to use
            model_alternatives: List of alternative models to try if primary fails
            max_tokens: Maximum tokens for responses
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        self.model_alternatives = model_alternatives or [
            "claude-opus-4-5",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet",
            "claude-sonnet-4-20250514",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229"
        ]
        self.max_tokens = max_tokens
    
    def generate_response(self, prompt: str) -> Dict[str, str]:
        """
        Generate a response from Claude (either text or code).
        
        Args:
            prompt: The prompt to send to Claude
            
        Returns:
            Dictionary with 'type' ('text' or 'code') and 'content'
        """
        # Try models in order
        models_to_try = [self.model] + [
            m for m in self.model_alternatives if m != self.model
        ]
        
        last_error = None
        message = None
        
        for model in models_to_try:
            try:
                message = self.client.messages.create(
                    model=model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                break  # Success
            except Exception as e:
                last_error = e
                # If it's not a 404, re-raise immediately
                if "404" not in str(e) and "not_found" not in str(e).lower():
                    raise
                continue
        
        if message is None:
            raise Exception(f"All model attempts failed. Last error: {last_error}")
        
        # Extract response
        response = message.content[0].text.strip()
        
        # Check if it's a text response or code
        if response.startswith("TEXT_RESPONSE:"):
            text_answer = response.replace("TEXT_RESPONSE:", "").strip()
            return {"type": "text", "content": text_answer}
        elif response.startswith("CODE:"):
            code = response.replace("CODE:", "").strip()
            code = self._clean_code(code)
            return {"type": "code", "content": code}
        else:
            # Default: assume it's code if it contains Python-like syntax, otherwise text
            if "fig" in response or "plt." in response or "ax." in response:
                return {"type": "code", "content": self._clean_code(response)}
            else:
                return {"type": "text", "content": response}
    
    def _clean_code(self, code: str) -> str:
        """Remove markdown code blocks from code string"""
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

