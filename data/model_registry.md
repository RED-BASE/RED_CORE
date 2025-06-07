# RED_CORE Model Registry

**Last Updated**: 2025-06-06 based on official SDK documentation

---

## 🤖 Available Models

### OpenAI Models

#### GPT-4.1 Series (Latest)
- **gpt-4.1** ⭐ *Default*
  - **Code**: G41
  - **Context**: 1,000,000 tokens
  - **Features**: Text, Vision
  - **Notes**: 54.6% on SWE-bench, 26% cost reduction vs GPT-4o
  - **Aliases**: `gpt-4-1`, `gpt4.1`, `openai-flagship`

- **gpt-4.1-mini**
  - **Code**: G41M
  - **Context**: 1,000,000 tokens
  - **Features**: Text
  - **Notes**: Matches GPT-4o intelligence, 50% faster, 83% cost reduction

- **gpt-4.1-nano**
  - **Code**: G41N
  - **Context**: 1,000,000 tokens
  - **Features**: Text
  - **Notes**: Fastest/cheapest, 80.1% MMLU, 9.8% Aider polyglot coding

#### GPT-4o Series
- **gpt-4o**
  - **Code**: GPT4O
  - **Context**: 128,000 tokens
  - **Features**: Text, Vision
  - **Aliases**: `gpt-4`, `gpt4`, `gpt-4o-latest`

- **gpt-4o-mini**
  - **Code**: G4OM
  - **Context**: 128,000 tokens
  - **Features**: Text
  - **Aliases**: `gpt-4o-mini-latest`

---

### Anthropic Claude Models

#### Claude 4 Series (Latest)
- **claude-opus-4-20250514**
  - **Code**: C4O
  - **Context**: 200,000 tokens
  - **Features**: Text, Thinking
  - **Notes**: World's best coding model, 72.5% SWE-bench, $15/$75 per million tokens
  - **Aliases**: `claude-4-opus`, `claude-opus-4`, `claude-4`

- **claude-sonnet-4-20250514**
  - **Code**: C4S
  - **Context**: 200,000 tokens
  - **Features**: Text, Thinking
  - **Notes**: 72.7% SWE-bench, $3/$15 per million tokens
  - **Aliases**: `claude-4-sonnet`, `claude-sonnet-4`

#### Claude 3.7 Series
- **claude-3-7-sonnet-20250219**
  - **Code**: C37S
  - **Context**: 200,000 tokens
  - **Features**: Text, Thinking
  - **Notes**: Extended thinking capability, Oct 2024 knowledge cutoff
  - **Aliases**: `claude-3-7-sonnet`, `claude-3-7-sonnet-latest`, `claude`

#### Claude 3.5 Series
- **claude-3-5-sonnet-20241022**
  - **Code**: C35S
  - **Context**: 200,000 tokens
  - **Features**: Text, Vision
  - **Notes**: Strongest vision model in 3.5 family
  - **Aliases**: `claude-3-5-sonnet`, `claude-3-5-sonnet-latest`

- **claude-3-5-haiku-20241022**
  - **Code**: C35H
  - **Context**: 200,000 tokens
  - **Features**: Text
  - **Aliases**: `claude-3-5-haiku`

#### Claude 3 Series
- **claude-3-opus-20240229**
  - **Code**: C3O
  - **Context**: 200,000 tokens
  - **Features**: Text
  - **Aliases**: `claude-3-opus`

---

### Google Gemini Models

#### Gemini 2.5 Series (Latest)
- **gemini-2.5-pro-preview-06-05**
  - **Code**: G25P
  - **Context**: 2,000,000 tokens
  - **Features**: Text, Vision, Thinking
  - **Notes**: #1 on LMArena, state-of-the-art reasoning
  - **Aliases**: `gemini-2.5-pro`, `gemini-2-5-pro`, `gemini`

- **gemini-2.5-flash-preview-05-20**
  - **Code**: G25F
  - **Context**: 1,000,000 tokens
  - **Features**: Text, Vision, Thinking
  - **Notes**: #2 on LMArena, balanced price/performance
  - **Aliases**: `gemini-2.5-flash`, `gemini-2-5-flash`

#### Gemini 2.0 Series
- **gemini-2.0-flash**
  - **Code**: G20F
  - **Context**: 1,000,000 tokens
  - **Features**: Text, Vision, Multimodal
  - **Notes**: Native image generation, TTS, experimental
  - **Aliases**: `gemini-2-0-flash`

- **gemini-2.0-flash-lite**
  - **Code**: G20L
  - **Context**: 1,000,000 tokens
  - **Features**: Text
  - **Notes**: Optimized for cost efficiency and low latency
  - **Aliases**: `gemini-2-0-flash-lite`

#### Gemini 1.5 Series (Deprecated)
- **gemini-1.5-pro** ⚠️ *Limited Availability*
  - **Code**: G15P
  - **Context**: 2,000,000 tokens
  - **Features**: Text, Vision
  - **Deprecated**: Not available for new projects
  - **Aliases**: `gemini-pro`

- **gemini-1.5-flash** ⚠️ *Limited Availability*
  - **Code**: G15F
  - **Context**: 1,000,000 tokens
  - **Features**: Text
  - **Deprecated**: Not available for new projects

---

### Local Models

- **mistral-7b-instruct-v0.1.Q4_K_M.gguf**
  - **Code**: M7B
  - **Context**: 4,096 tokens
  - **Features**: Text
  - **Notes**: Local execution via llama.cpp
  - **Aliases**: `mistral`, `mistral-7b`

---

## 🔧 Configuration Functions

```python
from app.config import (
    resolve_model,           # Get canonical name from alias
    get_model_code,          # Get experiment log code
    get_model_vendor,        # Get vendor (openai/anthropic/google/local)
    get_model_features,      # Get feature list
    get_model_context_window,# Get context window size
    is_model_deprecated,     # Check if deprecated
    list_available_models,   # Get all available models
    get_default_model        # Get default model
)
```

---

## 📊 Feature Tags

- **Text**: Standard text generation
- **Vision**: Image/document processing
- **Thinking**: Extended reasoning capabilities
- **Multimodal**: Multiple input/output modalities

---

## 🎯 Model Selection Guidelines

### For Production Experiments:
- **GPT-4.1**: Best overall performance, massive context
- **Claude Opus 4**: Best coding, highest intelligence
- **Gemini 2.5 Pro**: Largest context, excellent reasoning

### For Cost-Effective Testing:
- **GPT-4.1 Mini**: Fast, cheap, good performance
- **Claude Sonnet 4**: Great coding at lower cost
- **Gemini 2.5 Flash**: Balanced performance/price

### For Safety Research:
- **Claude models**: Strong safety training, refusal patterns
- **GPT-4.1 series**: Large context for complex scenarios
- **Gemini 2.5**: Thinking models for reasoning analysis

---

*Model specifications based on official SDK documentation as of June 6, 2025*