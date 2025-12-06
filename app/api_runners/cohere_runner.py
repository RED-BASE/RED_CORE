# runners/cohere_runner.py

from typing import Dict, Any
import requests
from app.core.context import ConversationHistory
from app.api_runners.base_runner import BaseAPIRunner

# ---- CONFIG IMPORT ----
try:
    from app.config import COHERE_API_KEY
except ImportError:
    raise ImportError("config.py with COHERE_API_KEY must be present in project root.")

class CohereRunner(BaseAPIRunner):
    """
    Wrapper around Cohere API for RED_CORE integration.
    Returns structured outputs for logging + schema integration.
    """

    def __init__(self, api_key: str | None = None, model_name: str = "command-r-plus"):
        api_key = api_key or COHERE_API_KEY
        if not api_key:
            raise ValueError("COHERE_API_KEY not set and no api_key provided.")
        
        # Initialize base class with provider name for rate limiting
        super().__init__(provider_name="cohere", api_key=api_key)
        
        self.model_name = model_name
        self._system_prompt: str = ""
        self.base_url = "https://api.cohere.ai/v1"
        self.last_response = None

    def set_system_prompt(self, text: str) -> None:
        self._system_prompt = text.strip()

    def get_model_name(self):
        return self.model_name

    def _count_tokens(self, text: str) -> int:
        """
        Estimate token count for Cohere models.
        Uses 4 characters per token as rough estimate since Cohere doesn't expose tokenizer in basic API.
        """
        return len(text) // 4

    def _make_api_call(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Make the actual Cohere API call.
        
        Args:
            prompt: The prompt to send
            **kwargs: Additional parameters including conversation, temperature, etc.
            
        Returns:
            Dict with 'model_output', 'model_name', 'usage', and 'raw_response'
        """
        conversation = kwargs.get('conversation')
        
        # Convert conversation to Cohere format
        if conversation is not None:
            # Build chat history for Cohere format
            chat_history = []
            messages = conversation.to_openai_format()
            
            for msg in messages[:-1]:  # All but the last message
                if msg["role"] == "user":
                    chat_history.append({"role": "USER", "message": msg["content"]})
                elif msg["role"] == "assistant":
                    chat_history.append({"role": "CHATBOT", "message": msg["content"]})
            
            # Last message should be the current prompt
            current_message = messages[-1]["content"] if messages else prompt
        else:
            chat_history = []
            current_message = prompt

        # Build request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "message": current_message,
            "chat_history": chat_history,
            **self._build_kwargs(kwargs)
        }
        
        # Add system prompt as preamble if provided
        if self._system_prompt:
            payload["preamble"] = self._system_prompt
        
        # Make API call
        response = requests.post(
            f"{self.base_url}/chat",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        data = response.json()
        self.last_response = data
        
        output = data["text"].strip()
        
        # Cohere provides token usage in a different format
        usage_info = data.get("meta", {}).get("billed_units", {})
        usage = {
            "prompt_tokens": usage_info.get("input_tokens", 0),
            "completion_tokens": usage_info.get("output_tokens", 0),
            "total_tokens": usage_info.get("input_tokens", 0) + usage_info.get("output_tokens", 0)
        } if usage_info else None
        
        return {
            "model_output": output,
            "model_name": self.model_name,
            "usage": usage,
            "raw_response": data
        }

    def generate(
        self,
        prompt: str,
        conversation: ConversationHistory | None = None,
        **kwds,
    ) -> dict:
        """Generate a completion from Cohere.

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
            ``None``) and ``raw_response`` (the raw Cohere response object).
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
            "temperature": 0.3,
            "max_tokens": 1024,
            "p": 0.75,  # Cohere uses 'p' instead of 'top_p'
            "k": 0,     # Top-k sampling
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        out = {}
        for k, default in allowed.items():
            out[k] = kwds.get(k, default)
        return out


if __name__ == "__main__":
    print("CohereRunner smoke-test OK")