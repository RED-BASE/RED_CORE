# API Runners Context

**Last Updated**: 2025-06-07 by Claude Code

## 🎯 Purpose

Unified interface layer for AI model APIs, providing consistent interaction patterns across different vendors while handling vendor-specific requirements.

## 🏗️ Architecture Pattern

All runners implement the same interface for seamless model swapping:

```python
class BaseRunner:
    def generate(self, prompt: str, **kwargs) -> dict
    def set_system_prompt(self, prompt: str) -> None
    def set_persona(self, persona: dict) -> None  # Optional
```

## 📁 Model Implementations

### **`anthropic_runner.py`** - Claude Models
```python
# Supported Models:
- claude-opus-4-20250514
- claude-sonnet-4-20250514  
- claude-3-7-sonnet-20250219
- claude-3-5-sonnet-20241022
- claude-3-5-haiku-20241022
- claude-3-opus-20240229
```

**Key Features:**
- System prompt handling via `set_system_prompt()`
- Message format conversion for Claude API
- Retry logic with exponential backoff
- Rate limiting compliance

**Recent Fix:** Proper model initialization for LLM evaluator

### **`openai_runner.py`** - GPT Models
```python
# Supported Models:
- gpt-4.1 (1M context)
- gpt-4.1-mini
- gpt-4.1-nano
- gpt-4o
- gpt-4o-mini
```

**Key Features:**
- Conversation history management
- Usage tracking (token counts)
- Temperature control
- Function calling support (future)

### **`google_runner.py`** - Gemini Models
```python
# Supported Models:
- gemini-2.5-pro-preview-06-05
- gemini-2.5-flash-preview-05-20
- gemini-2.0-flash
- gemini-2.0-flash-lite (evaluation default)
- gemini-1.5-pro (deprecated)
- gemini-1.5-flash (deprecated)
```

**Key Features:**
- Thinking capability support
- Cost-optimized evaluation model
- Safety settings configuration
- Content filtering integration

### **`llama_cpp_runner.py`** - Local Models
```python
# Purpose: Local model execution
# Models: Any GGUF format model
- mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

**Key Features:**
- Local inference (no API costs)
- Custom model loading
- Hardware acceleration support

### **`xai_runner.py`** - XAI Grok Models
```python
# Supported Models:
- grok-beta
```

**Key Features:**
- Uncensored approach for edge case testing
- Real-time data access capabilities
- OpenAI-compatible API format
- Rate limiting and error handling

### **`deepseek_runner.py`** - DeepSeek Models
```python
# Supported Models:
- deepseek-chat
- deepseek-coder
```

**Key Features:**
- Strong Chinese AI perspective
- Excellent coding capabilities
- Extended timeout for code generation
- OpenAI-compatible API format

### **`mistral_runner.py`** - Mistral AI Models
```python
# Supported Models:
- mistral-large-latest
- mistral-small-latest
```

**Key Features:**
- European AI approach
- Function calling support
- Safety filtering controls (safe_prompt parameter)
- OpenAI-compatible API format

### **`cohere_runner.py`** - Cohere Models
```python
# Supported Models:
- command-r-plus
- command-r
```

**Key Features:**
- Enterprise-focused RAG capabilities
- Unique chat history format (USER/CHATBOT roles)
- Preamble support for system prompts
- Specialized billing metrics

### **`together_runner.py`** - Together AI Models
```python
# Supported Models:
- meta-llama/Llama-3.3-70B-Instruct-Turbo
- meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo
```

**Key Features:**
- Access to latest open models
- Custom stop sequences support
- Extended timeouts for large models
- Repetition penalty controls

## 🔄 Common Patterns

### **Centralized Rate Limiting** ⚡ NEW!
All runners now inherit from `BaseAPIRunner` with unified rate limiting:

```python
class ExampleRunner(BaseAPIRunner):
    def __init__(self, model_name="example-model"):
        super().__init__("provider_name", api_key)
    
    def _count_tokens(self, text: str) -> int:
        return len(text) // 4  # Provider-specific estimation
    
    def _make_api_call(self, prompt: str, **kwargs) -> dict:
        # Provider-specific API call logic
        response = self.client.generate(prompt, **kwargs)
        return {"model_output": response.text, ...}
```

### **Rate Limiting Features**
- **Pre-emptive waiting** for strict providers (OpenAI, Mistral)
- **Reactive retries** for flexible providers (Google, Anthropic, etc.)
- **Token bucket algorithm** for smooth rate limiting
- **Provider-specific strategies** from `rate_limits.yaml`
- **Environment overrides** for different user tiers
- **Real-time monitoring** via `make rates`

### **Usage Tracking**
```python
# Standard usage format across all vendors:
{
    "prompt_tokens": 150,
    "completion_tokens": 75,
    "total_tokens": 225
}
```

## 📊 Model Registry

### **Configuration Location**: `app/config/config.py`

```python
MODEL_REGISTRY = {
    "claude-opus-4-20250514": {
        "vendor": "anthropic",
        "context_window": 1000000,
        "supports_thinking": True,
        "cost_per_million_tokens": {"input": 15.0, "output": 75.0}
    },
    # ... other models
}
```

### **Helper Functions**
```python
- resolve_model(name)      # Canonical name resolution
- get_model_vendor(name)   # Vendor identification  
- get_model_code(name)     # Short code for IDs
- list_available_models()  # All registered models
```

## 🚧 PHASE 1 CONSIDERATIONS

### **Evaluation Integration**
- LLM evaluator uses `anthropic_runner.py` or `google_runner.py`
- Proper system prompt handling critical for evaluation quality
- Error handling must support evaluation retry logic

### **Batch Processing**
- Async/parallel execution support
- Progress tracking integration
- Graceful degradation on API failures

## 🛡️ Security Patterns

### **API Key Management**
- Environment variable configuration
- No hardcoded credentials
- Secure key rotation support

### **Content Filtering**
- Input sanitization
- Output containment checks
- Safety settings per vendor

### **Rate Limiting**
- Respectful API usage
- Cost control mechanisms
- Automatic throttling

## 🎯 Design Principles

- **Vendor Agnostic**: Same interface regardless of underlying API
- **Resilient**: Graceful error handling and retry logic
- **Observable**: Rich logging and usage metrics
- **Efficient**: Optimal API usage patterns
- **Secure**: Safe credential handling and content filtering

## 📝 Development Protocol

**Git Workflow**: Commit after every minor change with descriptive messages
```bash
git add -A && git commit -m "Description of change"
```

---
*Part of RED_CORE by Cassius Oldenburg*