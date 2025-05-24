# runners/google_runner.py
# Lightweight wrapper for Google Gemini via google-generativeai SDK

import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types as gtypes

# --- CONFIG IMPORT ---
try:
    from config import GOOGLE_API_KEY, MODEL_ALIASES
except ImportError:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    MODEL_ALIASES = {}

load_dotenv()

class GoogleRunner:
    """
    Wrapper for Google Gemini models via the google-generativeai SDK.
    Supports single-turn generate_content calls with optional system prompt.
    """

    def __init__(self, model_name: str = "gemini-flash"):
        api_key = GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set - please define it in .env or config.py")

        genai.configure(api_key=api_key)
        # Use the global MODEL_ALIASES for all runner factories
        canonical = MODEL_ALIASES.get(model_name, model_name)
        try:
            self.model = genai.GenerativeModel(canonical)
        except Exception as e:
            raise ValueError(f"Model id '{canonical}' not recognised by SDK: {e}")

        self._system_prompt = ""

    def set_system_prompt(self, text: str):
        self._system_prompt = text.strip()

    def generate(self, prompt: str, **kw) -> str:
        try:
            conversation = kw.get("conversation")
            if conversation:
                # Debug flag: could also be in config
                if os.getenv("DEBUG_GEMINI", "false").lower() == "true":
                    print("\n🚨 Gemini Debug — Raw Conversation.turns:")
                    for t in conversation.turns:
                        print(f"  - role: {t['role']}, content: {repr(t['content'])}")

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

            if hasattr(response, "text"):
                return response.text.strip()
            elif hasattr(response, "candidates"):
                return response.candidates[0].content.parts[0].text.strip()
            else:
                return "[GeminiRunner] No valid text response returned."

        except Exception as e:
            import traceback
            print(f"[GoogleRunner] request failed: {e}")
            traceback.print_exc()
            return ""

# ---------------------------------------------------------------------
# ConversationHistory for Gemini and other runners
# ---------------------------------------------------------------------

class ConversationHistory:
    def __init__(self, system_prompt=""):
        self.turns = []
        self.system_prompt = system_prompt.strip()

    def append_user(self, msg):
        self.turns.append({"role": "user", "content": msg.strip()})

    def append_assistant(self, msg):
        self.turns.append({"role": "model", "content": msg.strip()})

    def to_gemini_format(self):
        messages = []
        for idx, turn in enumerate(self.turns):
            role = turn["role"]
            content = turn["content"].strip()
            if not content or role not in {"user", "model"}:
                continue
            if idx == 0 and self.system_prompt:
                content = f"{self.system_prompt}\n\n{content}"
            messages.append({
                "role": role if role == "user" else "model",
                "parts": [{"text": content}],
            })
        return messages
