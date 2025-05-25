import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# --- CONFIG IMPORT ---
try:
    from config import ANTHROPIC_API_KEY, MODEL_ALIASES
except ImportError:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    MODEL_ALIASES = {}

class AnthropicRunner:
    def __init__(self, model_name: str = "claude-3-opus"):
        self.api_key = ANTHROPIC_API_KEY
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set (check your .env or config.py)")

        # Use global alias mapping
        self.model_name = MODEL_ALIASES.get(model_name, model_name)
        self._system_prompt = ""
        self.client = Anthropic(api_key=self.api_key)

    def set_system_prompt(self, text: str):
        self._system_prompt = text.strip()

    def generate(self, prompt: str, **kwds) -> str:
        try:
            temp = kwds.get("temperature", 0.7)
            top_p = kwds.get("top_p", 0.95)
            max_tokens = kwds.get("max_tokens", 512)

            # Claude 3 (messages API)
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
                return response.content[0].text.strip()

            # Legacy Claude (completions API)
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
                return response.completion.strip()

        except Exception as e:
            print(f"[AnthropicRunner] request failed: {e}")
            return ""
