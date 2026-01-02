import os
import time
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from app.core.context import ConversationHistory
from .base_runner import BaseAPIRunner

# --- CONFIG IMPORT ---
try:
    from app.config import ANTHROPIC_API_KEY, resolve_model
except ImportError:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    def resolve_model(name): return name

class AnthropicRunner(BaseAPIRunner):
    """
    Wrapper around Anthropic Claude API with centralized rate limiting.
    Supports both Claude 3+ (messages API) and legacy Claude (completions API).
    """
    
    def __init__(self, model_name: str = "claude-3-opus"):
        # Initialize base class with provider name
        super().__init__("anthropic", ANTHROPIC_API_KEY)
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set (check your .env or config.py)")

        self.model_name = resolve_model(model_name)
        self._system_prompt = ""
        self.client = Anthropic(api_key=self.api_key)
        self.last_response = None

    def set_system_prompt(self, text: str):
        self._system_prompt = text.strip()

    def get_model_name(self):
        return self.model_name

    def _count_tokens(self, text: str) -> int:
        """
        Estimate tokens for Anthropic models.
        
        Anthropic doesn't provide a public tokenizer, so we use
        a conservative estimate of ~4 characters per token.
        """
        # Anthropic models use ~4 chars/token on average
        return len(text) // 4

    def _make_api_call(self, prompt: str, **kwargs) -> dict:
        """
        Make the actual API call to Anthropic.
        
        This is called by the base class after rate limit checks.
        """
        conversation = kwargs.get('conversation')
        temp = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.95)
        max_tokens = kwargs.get("max_tokens", 512)

        # Claude 3+ API (messages format) - all modern Claude models
        # Handles: claude-3-*, claude-4-*, claude-opus-4*, claude-sonnet-4*, claude-haiku-4*
        if self.model_name.startswith(("claude-3", "claude-4", "claude-opus", "claude-sonnet", "claude-haiku")):
            if conversation is not None:
                conv = conversation.to_claude_format()
                system_prompt = conv["system"] or None
                messages = conv["messages"]
            else:
                system_prompt = self._system_prompt or None
                messages = [{"role": "user", "content": prompt}]

            # Validate messages have content (Anthropic requires non-empty)
            for i, msg in enumerate(messages):
                if not msg.get("content", "").strip():
                    raise ValueError(f"Message {i} has empty content. Messages: {messages}")

            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
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

        # Legacy Claude (2.x, 1.x) - completions format
        else:
            if conversation is not None:
                conv = conversation.to_claude_format()
                system = f"{conv['system']}\n\n" if conv['system'] else ""
                prompt_parts = []
                for turn in conv['messages']:
                    if turn['role'] == 'user':
                        prompt_parts.append(f"{HUMAN_PROMPT} {turn['content']}")
                    else:
                        prompt_parts.append(f"{AI_PROMPT} {turn['content']}")
                claude_prompt = system + "\n\n".join(prompt_parts) + f"\n\n{AI_PROMPT}"
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

    def generate(
        self,
        prompt: str,
        conversation: ConversationHistory | None = None,
        **kwds,
    ) -> dict:
        """Generate a completion from Anthropic models with centralized rate limiting.

        Parameters
        ----------
        prompt : str
            Text of the user request when no ``conversation`` is supplied.
        conversation : ConversationHistory | None, optional
            Prior turns that already include the latest user input.

        Returns
        -------
        dict
            Mapping with keys ``model_output``, ``model_name``, ``usage`` and
            ``raw_response``.
        """
        # Add conversation to kwargs for base class
        if conversation is not None:
            kwds['conversation'] = conversation
            
        # Use base class generate which handles rate limiting and retries
        return super().generate(prompt, **kwds)


if __name__ == "__main__":
    print("AnthropicRunner smoke-test OK")