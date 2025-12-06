"""
Centralized rate limiter for managing API rate limits across all providers.

This module implements a token bucket algorithm with provider-specific
strategies to prevent rate limit errors and optimize throughput.
"""

import os
import time
import threading
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """
    Token bucket implementation for rate limiting.
    
    Tokens are added at a constant rate up to max capacity.
    Requests consume tokens; if not enough tokens, request waits.
    """
    capacity: float
    tokens: float
    refill_rate: float  # tokens per second
    last_refill: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def consume(self, tokens_needed: int, timeout: float = 60.0) -> bool:
        """
        Attempt to consume tokens. Blocks until tokens available or timeout.
        
        Args:
            tokens_needed: Number of tokens to consume
            timeout: Max seconds to wait for tokens
            
        Returns:
            True if tokens consumed, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                # Refill bucket based on time passed
                now = time.time()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
                self.last_refill = now
                
                # Check if we have enough tokens
                if self.tokens >= tokens_needed:
                    self.tokens -= tokens_needed
                    return True
            
            # Not enough tokens, wait a bit
            wait_time = min(0.1, (tokens_needed - self.tokens) / self.refill_rate)
            time.sleep(wait_time)
        
        return False
    
    def available_tokens(self) -> float:
        """Get current available tokens after refill."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            return min(self.capacity, self.tokens + elapsed * self.refill_rate)


@dataclass
class RateLimitConfig:
    """Configuration for a specific provider's rate limits."""
    tokens_per_minute: Optional[int] = None
    input_tokens_per_minute: Optional[int] = None  # For Anthropic
    output_tokens_per_minute: Optional[int] = None  # For Anthropic
    requests_per_minute: int = 60
    strategy: str = "reactive"  # "pre_emptive" or "reactive"
    buffer_percent: float = 10.0
    initial_backoff: float = 1.0
    max_backoff: float = 60.0
    backoff_multiplier: float = 2.0
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RateLimitConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def get_effective_token_limit(self) -> int:
        """Get effective token limit considering buffer."""
        limit = self.tokens_per_minute or (self.input_tokens_per_minute or 1000000)
        return int(limit * (1 - self.buffer_percent / 100))
    
    def get_effective_request_limit(self) -> int:
        """Get effective request limit considering buffer."""
        return int(self.requests_per_minute * (1 - self.buffer_percent / 100))


class CentralizedRateLimiter:
    """
    Centralized rate limiter that manages rate limits for all API providers.
    
    Features:
    - Token bucket algorithm for smooth rate limiting
    - Provider-specific strategies (pre-emptive vs reactive)
    - Environment variable overrides
    - Thread-safe for parallel execution
    - Detailed logging of rate limit events
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize rate limiter with configuration.
        
        Args:
            config_path: Path to rate_limits.yaml (defaults to app/config/rate_limits.yaml)
        """
        self.config_path = config_path or Path(__file__).parent.parent / "config" / "rate_limits.yaml"
        self.configs: Dict[str, RateLimitConfig] = {}
        self.token_buckets: Dict[str, TokenBucket] = {}
        self.request_buckets: Dict[str, TokenBucket] = {}
        self.backoff_until: Dict[str, float] = defaultdict(float)  # provider -> timestamp
        self._lock = threading.Lock()
        
        # Load configuration
        self._load_config()
        
        # Initialize buckets
        self._initialize_buckets()
        
        logger.info("Centralized rate limiter initialized")
    
    def _load_config(self):
        """Load rate limit configuration from YAML and environment."""
        # Load base config from YAML
        if self.config_path.exists():
            with open(self.config_path) as f:
                yaml_config = yaml.safe_load(f)
                
            for provider, config in yaml_config.get('providers', {}).items():
                self.configs[provider] = RateLimitConfig.from_dict(config)
        
        # Override with environment variables
        self._apply_env_overrides()
        
        # Log loaded configurations
        for provider, config in self.configs.items():
            logger.debug(f"{provider} rate limits: {config}")
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            'OPENAI_RATE_LIMIT_TPM': ('openai', 'tokens_per_minute'),
            'OPENAI_RATE_LIMIT_RPM': ('openai', 'requests_per_minute'),
            'ANTHROPIC_RATE_LIMIT_ITPM': ('anthropic', 'input_tokens_per_minute'),
            'ANTHROPIC_RATE_LIMIT_OTPM': ('anthropic', 'output_tokens_per_minute'),
            'ANTHROPIC_RATE_LIMIT_RPM': ('anthropic', 'requests_per_minute'),
            'GOOGLE_RATE_LIMIT_TPM': ('google', 'tokens_per_minute'),
            'GOOGLE_RATE_LIMIT_RPM': ('google', 'requests_per_minute'),
            'MISTRAL_RATE_LIMIT_TPM': ('mistral', 'tokens_per_minute'),
            'MISTRAL_RATE_LIMIT_RPM': ('mistral', 'requests_per_minute'),
        }
        
        for env_var, (provider, field) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                try:
                    if provider not in self.configs:
                        self.configs[provider] = RateLimitConfig()
                    setattr(self.configs[provider], field, int(value))
                    logger.info(f"Override {provider}.{field} = {value} from {env_var}")
                except ValueError:
                    logger.error(f"Invalid value for {env_var}: {value}")
    
    def _initialize_buckets(self):
        """Initialize token buckets for each provider."""
        for provider, config in self.configs.items():
            # Token bucket
            token_limit = config.get_effective_token_limit()
            tokens_per_second = token_limit / 60.0
            self.token_buckets[provider] = TokenBucket(
                capacity=token_limit,
                tokens=token_limit,  # Start full
                refill_rate=tokens_per_second
            )
            
            # Request bucket
            request_limit = config.get_effective_request_limit()
            requests_per_second = request_limit / 60.0
            self.request_buckets[provider] = TokenBucket(
                capacity=request_limit,
                tokens=request_limit,  # Start full
                refill_rate=requests_per_second
            )
    
    def check_and_consume(
        self, 
        provider: str, 
        tokens: int,
        is_input: bool = True,
        timeout: float = 60.0
    ) -> Tuple[bool, Optional[float]]:
        """
        Check rate limits and consume tokens if available.
        
        Args:
            provider: API provider name (e.g., 'openai', 'anthropic')
            tokens: Number of tokens to consume
            is_input: True for input tokens, False for output (only matters for Anthropic)
            timeout: Max seconds to wait for tokens
            
        Returns:
            Tuple of (success, wait_time_if_failed)
        """
        if provider not in self.configs:
            # No rate limit config for this provider
            return True, None
        
        config = self.configs[provider]
        
        # Check if we're in backoff period
        if time.time() < self.backoff_until[provider]:
            wait_time = self.backoff_until[provider] - time.time()
            logger.warning(f"{provider}: In backoff period, wait {wait_time:.1f}s")
            return False, wait_time
        
        # Pre-emptive strategy: check before making request
        if config.strategy == "pre_emptive":
            # Check token bucket
            token_bucket = self.token_buckets[provider]
            if not token_bucket.consume(tokens, timeout=0):  # Non-blocking check
                available = token_bucket.available_tokens()
                wait_time = (tokens - available) / (token_bucket.refill_rate)
                logger.info(f"{provider}: Pre-emptive wait {wait_time:.1f}s for {tokens} tokens")
                
                # Actually wait
                if not token_bucket.consume(tokens, timeout=timeout):
                    return False, timeout
            
            # Check request bucket
            request_bucket = self.request_buckets[provider]
            if not request_bucket.consume(1, timeout=timeout):
                return False, timeout
        
        # For reactive strategy, we don't block here
        # Just track usage for reporting
        else:
            with self._lock:
                # Update buckets without blocking
                self.token_buckets[provider].consume(tokens, timeout=0)
                self.request_buckets[provider].consume(1, timeout=0)
        
        return True, None
    
    def report_rate_limit_error(self, provider: str, retry_after: Optional[float] = None):
        """
        Report that a rate limit error occurred.
        
        Args:
            provider: API provider that returned 429
            retry_after: Retry-After header value if provided
        """
        config = self.configs.get(provider, RateLimitConfig())
        
        # Calculate backoff time
        current_backoff = config.initial_backoff
        if provider in self.backoff_until:
            # Already in backoff, increase it
            last_backoff = self.backoff_until[provider] - time.time()
            if last_backoff > 0:
                current_backoff = min(
                    last_backoff * config.backoff_multiplier,
                    config.max_backoff
                )
        
        # Use retry_after if provided, otherwise use calculated backoff
        backoff_time = retry_after or current_backoff
        self.backoff_until[provider] = time.time() + backoff_time
        
        logger.warning(f"{provider}: Rate limit hit, backing off for {backoff_time:.1f}s")
    
    def get_provider_status(self, provider: str) -> Dict:
        """
        Get current status for a provider.
        
        Returns dict with available tokens, requests, and backoff status.
        """
        if provider not in self.configs:
            return {"configured": False}
        
        token_bucket = self.token_buckets[provider]
        request_bucket = self.request_buckets[provider]
        
        return {
            "configured": True,
            "available_tokens": int(token_bucket.available_tokens()),
            "available_requests": int(request_bucket.available_tokens()),
            "token_capacity": int(token_bucket.capacity),
            "request_capacity": int(request_bucket.capacity),
            "in_backoff": time.time() < self.backoff_until[provider],
            "backoff_until": self.backoff_until[provider] if time.time() < self.backoff_until[provider] else None,
            "strategy": self.configs[provider].strategy,
        }
    
    def get_all_status(self) -> Dict[str, Dict]:
        """Get status for all configured providers."""
        return {provider: self.get_provider_status(provider) for provider in self.configs}


# Global rate limiter instance
_rate_limiter: Optional[CentralizedRateLimiter] = None


def get_rate_limiter() -> CentralizedRateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = CentralizedRateLimiter()
    return _rate_limiter


def check_rate_limit(provider: str, tokens: int, timeout: float = 60.0) -> Tuple[bool, Optional[float]]:
    """
    Convenience function to check rate limits.
    
    Args:
        provider: API provider name
        tokens: Number of tokens to consume
        timeout: Max seconds to wait
        
    Returns:
        Tuple of (success, wait_time_if_failed)
    """
    limiter = get_rate_limiter()
    return limiter.check_and_consume(provider, tokens, timeout=timeout)


def report_rate_limit_error(provider: str, retry_after: Optional[float] = None):
    """
    Convenience function to report rate limit errors.
    
    Args:
        provider: API provider that returned 429
        retry_after: Retry-After header value if provided
    """
    limiter = get_rate_limiter()
    limiter.report_rate_limit_error(provider, retry_after)