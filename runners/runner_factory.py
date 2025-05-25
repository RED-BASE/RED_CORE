import os
from typing import Protocol
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the root .env file
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Import new registry helpers
from config import resolve_model, get_model_vendor

class BaseRunner(Protocol):
    def generate(self, prompt: str, **kwds) -> str: ...
    def set_system_prompt(self, text: str): ...

def get_runner(model_name: str):
    canonical_name = resolve_model(model_name)
    vendor = get_model_vendor(canonical_name)

    if vendor == "openai":
        from app.api_runners.openai_runner import OpenAIRunner
        return OpenAIRunner(model_name=canonical_name)
    elif vendor == "anthropic":
        from app.api_runners.anthropic_runner import AnthropicRunner
        return AnthropicRunner(model_name=canonical_name)
    elif vendor == "google":
        from app.api_runners.google_runner import GoogleRunner
        return GoogleRunner(model_name=canonical_name)
    elif vendor == "local":
        # Add your local runner here if desired
        from llama_cpp_runner import LlamaCppRunner
        return LlamaCppRunner(model_path=canonical_name)
    else:
        raise ValueError(f"No runner registered for model '{model_name}' (resolved as '{canonical_name}')")

