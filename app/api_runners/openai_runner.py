# runners/openai_runner.py

from typing import Dict, Any

import tiktoken
from openai import OpenAI
from app.core.context import ConversationHistory
from .base_runner import BaseAPIRunner
import time

# ---- CONFIG IMPORT ----
try:
    from app.config import OPENAI_API_KEY
except ImportError:
    raise ImportError("config.py with OPENAI_API_KEY must be present in project root.")

class OpenAIRunner(BaseAPIRunner):
    """
    Wrapper around OpenAI ChatCompletion API with centralized rate limiting.
    Returns structured outputs for logging + schema integration.
    """

    def __init__(self, api_key: str | None = None, model_name: str = "gpt-4o"):
        # Initialize base class with provider name
        super().__init__("openai", api_key or OPENAI_API_KEY)
        
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

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken for accurate OpenAI token counting."""
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def _make_api_call(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Make the actual API call to OpenAI.
        
        This is called by the base class after rate limit checks.
        """
        conversation = kwargs.get('conversation')
        
        if conversation is not None:
            messages = conversation.to_openai_format()
        else:
            messages = []
            if self._system_prompt:
                messages.append({"role": "system", "content": self._system_prompt})
            messages.append({"role": "user", "content": prompt})

        # Make API call
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self._build_kwargs(kwargs),
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

    def generate(
        self,
        prompt: str,
        conversation: ConversationHistory | None = None,
        **kwds,
    ) -> dict:
        """Generate a completion from OpenAI with centralized rate limiting.

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
        # Add conversation to kwargs for base class
        if conversation is not None:
            kwds['conversation'] = conversation
            
        # Use base class generate which handles rate limiting and retries
        return super().generate(prompt, **kwds)

    def count_tokens(self, text: str) -> int:
        """Public method for token counting."""
        return self._count_tokens(text)

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