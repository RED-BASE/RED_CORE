"""
Base runner class that provides common functionality for all API runners.

This includes rate limiting, retry logic, error handling, and token counting.
"""

import time
import logging
from typing import Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod

from app.core.rate_limiter import check_rate_limit, report_rate_limit_error

logger = logging.getLogger(__name__)


class BaseAPIRunner(ABC):
    """
    Base class for all API runners.
    
    Provides:
    - Centralized rate limiting
    - Consistent retry logic
    - Token counting utilities
    - Error handling patterns
    """
    
    def __init__(self, provider_name: str, api_key: Optional[str] = None):
        """
        Initialize base runner.
        
        Args:
            provider_name: Name matching rate_limits.yaml (e.g., 'openai', 'anthropic')
            api_key: API key for the provider
        """
        self.provider_name = provider_name
        self.api_key = api_key
        self.max_retries = 5
        
    @abstractmethod
    def _make_api_call(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Make the actual API call. Must be implemented by subclasses.
        
        Args:
            prompt: The prompt to send
            **kwargs: Provider-specific parameters
            
        Returns:
            Dict with at least 'model_output' and optionally 'usage'
        """
        pass
    
    @abstractmethod
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens for the given text. Provider-specific implementation.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate response with rate limiting and retry logic.
        
        Args:
            prompt: The prompt to send
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Dict with 'model_output', 'model_name', 'usage', etc.
        """
        # Estimate tokens for rate limiting
        estimated_tokens = self._count_tokens(prompt)
        
        # Add some buffer for response tokens
        estimated_total = estimated_tokens + kwargs.get('max_tokens', 512)
        
        # Rate limiting loop with retries
        for attempt in range(1, self.max_retries + 1):
            try:
                # Check rate limits (pre-emptive for some providers)
                can_proceed, wait_time = check_rate_limit(
                    self.provider_name, 
                    estimated_total,
                    timeout=30.0  # Don't wait too long
                )
                
                if not can_proceed:
                    if attempt < self.max_retries:
                        wait_time = wait_time or 10.0
                        logger.info(f"{self.provider_name}: Rate limit pre-check failed, waiting {wait_time:.1f}s (attempt {attempt}/{self.max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {
                            "model_output": f"[ERROR] Rate limit exceeded after {self.max_retries} attempts",
                            "model_name": f"{self.provider_name}-unknown",
                            "usage": None
                        }
                
                # Make the actual API call
                result = self._make_api_call(prompt, **kwargs)
                
                # Success! Log token usage if available
                if 'usage' in result and result['usage']:
                    logger.debug(f"{self.provider_name}: Used {result['usage'].get('total_tokens', 'unknown')} tokens")
                
                return result
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error
                if self._is_rate_limit_error(error_str):
                    # Extract retry-after if available
                    retry_after = self._extract_retry_after(error_str)
                    report_rate_limit_error(self.provider_name, retry_after)
                    
                    if attempt < self.max_retries:
                        # Use exponential backoff
                        backoff = min(2 ** (attempt - 1), 60)
                        if retry_after:
                            backoff = max(backoff, retry_after)
                        
                        logger.warning(f"{self.provider_name}: Rate limit hit, retrying in {backoff:.1f}s (attempt {attempt}/{self.max_retries})")
                        time.sleep(backoff)
                        continue
                
                # Not a rate limit error or final attempt
                logger.error(f"{self.provider_name}: API error: {error_str}")
                
                # Return error response
                return {
                    "model_output": f"[ERROR] {error_str}",
                    "model_name": f"{self.provider_name}-unknown",
                    "usage": None
                }
        
        # Should not reach here, but just in case
        return {
            "model_output": "[ERROR] Maximum retries exceeded",
            "model_name": f"{self.provider_name}-unknown",
            "usage": None
        }
    
    def _is_rate_limit_error(self, error_str: str) -> bool:
        """Check if error is rate limit related."""
        rate_limit_indicators = [
            "429",
            "rate limit",
            "rate_limit",
            "too many requests",
            "quota exceeded",
            "RateLimitError",
        ]
        error_lower = error_str.lower()
        return any(indicator.lower() in error_lower for indicator in rate_limit_indicators)
    
    def _extract_retry_after(self, error_str: str) -> Optional[float]:
        """
        Try to extract retry-after value from error message.
        
        Some providers include this in the error message.
        """
        # Common patterns:
        # "Retry after 30 seconds"
        # "retry_after: 15"
        # "Retry-After: 60"
        
        import re
        
        patterns = [
            r'retry[_\- ]after[:\s]+(\d+(?:\.\d+)?)',
            r'retry after (\d+(?:\.\d+)?)\s*(?:seconds?)?',
            r'wait (\d+(?:\.\d+)?)\s*(?:seconds?)?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        
        return None
    
    def estimate_tokens(self, text: str) -> int:
        """
        Public method to estimate tokens for a text.
        
        Useful for pre-flight checks before sending requests.
        """
        return self._count_tokens(text)