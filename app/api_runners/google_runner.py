# runners/google_runner.py
# Finalized Gemini runner for structured harness use

import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types as gtypes
import time

from app.core.context import ConversationHistory
from .base_runner import BaseAPIRunner

# --- CONFIG IMPORT ---
try:
    from app.config import GOOGLE_API_KEY, MODEL_ALIASES
except ImportError:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    MODEL_ALIASES = {}

load_dotenv()

class GoogleRunner(BaseAPIRunner):
    """
    Wrapper for Google Gemini models with centralized rate limiting.
    Structured return format for use in harnesses and experiments.
    """

    def __init__(self, model_name: str = "gemini-pro"):
        api_key = GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set - define in .env or config.py")

        # Initialize base class with provider name
        super().__init__("google", api_key)

        genai.configure(api_key=api_key)
        canonical = MODEL_ALIASES.get(model_name, model_name)
        try:
            self.model = genai.GenerativeModel(canonical)
        except Exception as e:
            raise ValueError(f"Model id '{canonical}' not recognised by SDK: {e}")

        self._system_prompt = ""
        self.last_response = None  # Cached raw response for audit

    def set_system_prompt(self, text: str):
        self._system_prompt = text.strip()

    def get_model_name(self):
        return self.model.model_name

    def _count_tokens(self, text: str) -> int:
        """
        Estimate tokens for Google models.
        
        Google doesn't provide an easy tokenizer, so we use
        a conservative estimate of ~4 characters per token.
        """
        # Google models use ~4 chars/token on average
        return len(text) // 4

    def _make_api_call(self, prompt: str, **kwargs) -> dict:
        """
        Make the actual API call to Google Gemini.
        
        This is called by the base class after rate limit checks.
        """
        conversation = kwargs.get("conversation")
        
        if conversation:
            if os.getenv("DEBUG_GEMINI", "false").lower() == "true":
                print("\n🚨 Gemini Debug — Raw Conversation.turns:")
                for t in conversation.turns:
                    print(f"  - role: {t.role}, content: {repr(t.content)}")

            messages = conversation.to_gemini_format()

            if os.getenv("DEBUG_GEMINI", "false").lower() == "true":
                print("\n🚨 Gemini Debug — Formatted messages for Gemini:")
                for m in messages:
                    print(m)

            if not messages:
                raise ValueError("[GoogleRunner] Conversation format produced empty content.")
        else:
            turn_index = kwargs.get("turn_index", 1)
            if turn_index == 1 and self._system_prompt:
                full_prompt = f"{self._system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt

            if not full_prompt.strip():
                raise ValueError("[GoogleRunner] Refusing to send empty prompt to Gemini API.")

            messages = [{"role": "user", "parts": [{"text": full_prompt}]}]

        response = self.model.generate_content(
            contents=messages,
            generation_config=gtypes.GenerationConfig(
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.95),
                max_output_tokens=kwargs.get("max_tokens", 512),
            ),
        )

        self.last_response = response

        if hasattr(response, "text"):
            output_text = response.text.strip()
        elif hasattr(response, "candidates"):
            output_text = response.candidates[0].content.parts[0].text.strip()
        else:
            output_text = "[GeminiRunner] No valid text response returned."

        usage = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", None),
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", None),
                "total_tokens": getattr(response.usage_metadata, "total_token_count", None)
            }
            
        return {
            "model_output": output_text,
            "model_name": self.model.model_name,
            "usage": usage,
            "raw_response": response
        }

    def generate(self, prompt: str, **kw) -> dict:
        """Generate response from Google Gemini with centralized rate limiting."""
        # Use base class generate which handles rate limiting and retries
        return super().generate(prompt, **kw)


if __name__ == "__main__":
    print("GoogleRunner smoke-test OK")