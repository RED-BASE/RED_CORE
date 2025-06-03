import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import time

# --- CONFIG IMPORT ---
try:
    from app.config import ANTHROPIC_API_KEY, MODEL_ALIASES
except ImportError:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    MODEL_ALIASES = {}

class AnthropicRunner:
    def __init__(self, model_name: str = "claude-3-opus"):
        self.api_key = ANTHROPIC_API_KEY
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set (check your .env or config.py)")

        self.model_name = MODEL_ALIASES.get(model_name, model_name)
        self._system_prompt = ""
        self.client = Anthropic(api_key=self.api_key)
        self.last_response = None

    def set_system_prompt(self, text: str):
        self._system_prompt = text.strip()

    def get_model_name(self):
        return self.model_name

    def generate(self, prompt: str, **kwds) -> dict:
        max_retries = 5
        backoff = 1
        for attempt in range(1, max_retries + 1):
            try:
                temp = kwds.get("temperature", 0.7)
                top_p = kwds.get("top_p", 0.95)
                max_tokens = kwds.get("max_tokens", 512)

                # Claude 3 API
                if self.model_name.startswith("claude-3-"):
                    messages = [{"role": "user", "content": prompt}]
                    response = self.client.messages.create(
                        model=self.model_name,
                        system=self._system_prompt or None,
                        messages=messages,
                        temperature=temp,
                        top_p=top_p,
                        max_tokens=max_tokens,
                    )
                    output = response.content[0].text.strip()

                    self.last_response = response
                    return {
                        "model_output": output,
                        "model_name": self.model_name,
                        "usage": {
                            "prompt_tokens": getattr(response.usage, "input_tokens", None),
                            "completion_tokens": getattr(response.usage, "output_tokens", None),
                            "total_tokens": getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0) if response.usage else None,
                        },
                        "raw_response": response,
                    }

                # Legacy Claude (2.x, 1.x)
                else:
                    system = f"{self._system_prompt}\n\n" if self._system_prompt else ""
                    claude_prompt = f"{system}{HUMAN_PROMPT} {prompt}\n\n{AI_PROMPT}"
                    response = self.client.completions.create(
                        model=self.model_name,
                        prompt=claude_prompt,
                        max_tokens_to_sample=max_tokens,
                        temperature=temp,
                        top_p=top_p,
                    )
                    output = response.completion.strip()

                    self.last_response = response
                    return {
                        "model_output": output,
                        "model_name": self.model_name,
                        "usage": {
                            "prompt_tokens": None,
                            "completion_tokens": None,
                            "total_tokens": None,
                        },
                        "raw_response": response,
                    }

            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "rate limit" in err_str.lower()) and attempt < max_retries:
                    print(f"[AnthropicRunner] Rate limit hit, retrying in {backoff}s (attempt {attempt}/{max_retries})...")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                print(f"[AnthropicRunner] request failed: {e}")
                return {
                    "model_output": f"[ERROR] {e}",
                    "model_name": self.model_name,
                    "usage": None,
                    "raw_response": None
                }
