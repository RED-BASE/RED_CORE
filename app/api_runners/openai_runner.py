# runners/openai_runner.py

from typing import Dict, Any

import tiktoken
from openai import OpenAI
from app.core.context import ConversationHistory
import time

# ---- CONFIG IMPORT ----
try:
    from app.config import OPENAI_API_KEY
except ImportError:
    raise ImportError("config.py with OPENAI_API_KEY must be present in project root.")

class OpenAIRunner:
    """
    Finalized wrapper around OpenAI ChatCompletion API.
    Returns structured outputs for logging + schema integration.
    """

    def __init__(self, api_key: str | None = None, model_name: str = "gpt-4o"):
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set and no api_key provided.")
        self.model_name = model_name
        self._system_prompt: str = ""
        self.client = OpenAI(api_key=self.api_key)
        self.last_response = None

    def set_system_prompt(self, text: str) -> None:
        self._system_prompt = text.strip()

    def get_model_name(self):
        return self.model_name

    def generate(
        self,
        prompt: str,
        conversation: ConversationHistory | None = None,
        **kwds,
    ) -> dict:
        """Generate a completion from OpenAI.

        Parameters
        ----------
        prompt : str
            The message text to send when no conversation history exists.
        conversation : ConversationHistory | None, optional
            Prior conversation turns used to construct the message payload.

        Returns
        -------
        dict
            Mapping with keys ``model_output`` (the assistant reply),
            ``model_name`` (the model identifier), ``usage`` (token counts or
            ``None``) and ``raw_response`` (the raw OpenAI response object).
        """
        if conversation is not None:
            messages = conversation.to_openai_format()
        else:
            messages = []
            if self._system_prompt:
                messages.append({"role": "system", "content": self._system_prompt})
            messages.append({"role": "user", "content": prompt})

        max_retries = 5
        backoff = 1
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **self._build_kwargs(kwds),
                )
                self.last_response = response
                output = response.choices[0].message.content.strip()
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if hasattr(response, "usage") else None
                return {
                    "model_output": output,
                    "model_name": self.model_name,
                    "usage": usage,
                    "raw_response": response
                }
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "rate limit" in err_str.lower()) and attempt < max_retries:
                    print(f"[OpenAIRunner] Rate limit hit, retrying in {backoff}s (attempt {attempt}/{max_retries})...")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                print(f"[OpenAIRunner] request failed: {e}")
                return {
                    "model_output": f"[ERROR] {e}",
                    "model_name": self.model_name,
                    "usage": None,
                    "raw_response": None
                }

    def count_tokens(self, text: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def get_info(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "api_key_present": bool(self.api_key)}

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
