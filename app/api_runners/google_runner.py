# runners/google_runner.py
# Finalized Gemini runner for structured harness use

import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types as gtypes
import time

from app.core.context import ConversationHistory

# --- CONFIG IMPORT ---
try:
    from app.config import GOOGLE_API_KEY, MODEL_ALIASES
except ImportError:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    MODEL_ALIASES = {}

load_dotenv()

class GoogleRunner:
    """
    Wrapper for Google Gemini models via the google-generativeai SDK.
    Structured return format for use in harnesses and experiments.
    """

    def __init__(self, model_name: str = "gemini-pro"):
        api_key = GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set - define in .env or config.py")

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

    def generate(self, prompt: str, **kw) -> dict:
        max_retries = 5
        backoff = 1
        for attempt in range(1, max_retries + 1):
            try:
                conversation = kw.get("conversation")
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
                    turn_index = kw.get("turn_index", 1)
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
                        temperature=kw.get("temperature", 0.7),
                        top_p=kw.get("top_p", 0.95),
                        max_output_tokens=kw.get("max_tokens", 512),
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
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "rate limit" in err_str.lower()) and attempt < max_retries:
                    print(f"[GoogleRunner] Rate limit hit, retrying in {backoff}s (attempt {attempt}/{max_retries})...")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                import traceback
                print(f"[GoogleRunner] request failed: {e}")
                traceback.print_exc()
                return {
                    "model_output": f"[ERROR] {e}",
                    "model_name": self.model.model_name,
                    "usage": None,
                    "raw_response": None
                }

