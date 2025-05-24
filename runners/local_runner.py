# runners/local_runner.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- CONFIG IMPORT ---
try:
    from config import LOCAL_MODEL_PATH, DEFAULT_CONTEXT_SIZE, LOCAL_ATTACK_YAML, LOCAL_ATTACK_LOG, LOCAL_ATTACK_TURNS
except ImportError:
    # Fallbacks if config.py is not fully defined yet
    LOCAL_MODEL_PATH = "/Users/redhat/llm_models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    DEFAULT_CONTEXT_SIZE = 4096
    LOCAL_ATTACK_YAML = "prompts_user/simple_prompt_injection.yaml"
    LOCAL_ATTACK_LOG = "logs/attack_logs/local_test_run.lrc"
    LOCAL_ATTACK_TURNS = 1

from llama_cpp_runner import LlamaCppRunner
from unused.attack_runner import run_attack

def main():
    model = LlamaCppRunner(model_path=LOCAL_MODEL_PATH, n_ctx=DEFAULT_CONTEXT_SIZE)
    run_attack(
        model_runner=model,
        attack_yaml=LOCAL_ATTACK_YAML,
        output_path=LOCAL_ATTACK_LOG,
        n_turns=LOCAL_ATTACK_TURNS,
    )

if __name__ == "__main__":
    main()
