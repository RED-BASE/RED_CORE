# runners/openai_runner.py

from pathlib import Path
from typing import Dict, Any

import tiktoken
from openai import OpenAI

# ---- CONFIG IMPORT ----
try:
    from app.config import OPENAI_API_KEY
except ImportError:
    raise ImportError("config.py with OPENAI_API_KEY must be present in project root.")

class OpenAIRunner:
    """
    Thin wrapper around OpenAI ChatCompletion.
    • set_system_prompt() stores a system message for every call.
    • generate() accepts optional kwargs: temperature, max_tokens, etc.
    """

    def __init__(self, api_key: str | None = None, model_name: str = "gpt-4o"):
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set and no api_key provided.")
        self.model_name = model_name
        self._system_prompt: str = ""
        self.client = OpenAI(api_key=self.api_key)

    # ------------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------------
    def set_system_prompt(self, text: str) -> None:
        """Store the prompt so generate() can prepend it."""
        self._system_prompt = text.strip()

    def generate(self, prompt: str, **kwds) -> str:
        """
        Returns assistant content as str.
        Recognized kwds: temperature, max_tokens, top_p,
                         frequency_penalty, presence_penalty
        """
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self._build_kwargs(kwds),
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[OpenAIRunner] request failed: {e}")
            return ""

    def count_tokens(self, text: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def get_info(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "api_key_present": bool(self.api_key)}

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    @staticmethod
    def _build_kwargs(kwds: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {
            "temperature": 1.0,
            "max_tokens": 512,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        out = {}
        for k, default in allowed.items():
            out[k] = kwds.get(k, default)
        return out


if __name__ == "__main__":
    print("OpenAIRunner smoke-test OK")
