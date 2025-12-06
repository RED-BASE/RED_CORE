# runners/together_runner.py

from typing import Dict, Any
import requests
from app.core.context import ConversationHistory
from app.api_runners.base_runner import BaseAPIRunner

# ---- CONFIG IMPORT ----
try:
    from app.config import TOGETHER_API_KEY
except ImportError:
    raise ImportError("config.py with TOGETHER_API_KEY must be present in project root.")

class TogetherRunner(BaseAPIRunner):
    """
    Wrapper around Together AI API for RED_CORE integration.
    Returns structured outputs for logging + schema integration.
    """

    def __init__(self, api_key: str | None = None, model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        api_key = api_key or TOGETHER_API_KEY
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not set and no api_key provided.")
        
        # Initialize base class with provider name for rate limiting
        super().__init__(provider_name="together", api_key=api_key)
        
        self.model_name = model_name
        self._system_prompt: str = ""
        self.base_url = "https://api.together.xyz/v1"
        self.last_response = None

    def set_system_prompt(self, text: str) -> None:
        self._system_prompt = text.strip()

    def get_model_name(self):
        return self.model_name

    def _count_tokens(self, text: str) -> int:
        """
        Estimate token count for Together AI models.
        Uses 4 characters per token as rough estimate since Together doesn't expose tokenizer in API.
        """
        return len(text) // 4

    def _make_api_call(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Make the actual Together AI API call.
        
        Args:
            prompt: The prompt to send
            **kwargs: Additional parameters including conversation, temperature, etc.
            
        Returns:
            Dict with 'model_output', 'model_name', 'usage', and 'raw_response'
        """
        conversation = kwargs.get('conversation')
        
        # Build messages
        if conversation is not None:
            messages = conversation.to_openai_format()  # Together uses OpenAI-compatible format
        else:
            messages = []
            if self._system_prompt:
                messages.append({"role": "system", "content": self._system_prompt})
            messages.append({"role": "user", "content": prompt})

        # Build request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            **self._build_kwargs(kwargs)
        }
        
        # Make API call
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120  # Together can be slower for large models
        )
        response.raise_for_status()
        
        data = response.json()
        self.last_response = data
        
        output = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        
        return {
            "model_output": output,
            "model_name": self.model_name,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            } if usage else None,
            "raw_response": data
        }

    def generate(
        self,
        prompt: str,
        conversation: ConversationHistory | None = None,
        **kwds,
    ) -> dict:
        """Generate a completion from Together AI.

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
            ``None``) and ``raw_response`` (the raw Together response object).
        """
        # Pass conversation as a keyword argument to _make_api_call
        kwds['conversation'] = conversation
        return super().generate(prompt, **kwds)

    def count_tokens(self, text: str) -> int:
        """
        Public method for token counting (backwards compatibility).
        """
        return self._count_tokens(text)

    def get_info(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "api_key_present": bool(self.api_key)}

    @staticmethod
    def _build_kwargs(kwds: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "stream": False,
            "stop": [],  # Together supports custom stop sequences
        }
        out = {}
        for k, default in allowed.items():
            out[k] = kwds.get(k, default)
        return out


if __name__ == "__main__":
    print("TogetherRunner smoke-test OK")