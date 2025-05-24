import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# runners/interactive_runner.py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from llama_cpp_runner import LlamaCppRunner
from runners.openai_runner import OpenAIRunner
from unused.attack_runner import run_attack

ATTACKS_DIR = "prompts_user/"
DEFAULT_CONTEXT_SIZE = 4096

def get_model_runner(name: str):
    if name.lower() == "mistral":
        return LlamaCppRunner(
            model_path="/Users/redhat/llm_models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            n_ctx=DEFAULT_CONTEXT_SIZE
        )
    elif name.lower() == "gpt4":
        return OpenAIRunner(model_name="gpt-4")
    else:
        raise ValueError(f"Unknown model: {name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model to use: mistral or gpt4")
    parser.add_argument("--attack", type=str, required=True, help="Path to YAML attack profile under ./prompts_user/")
    parser.add_argument("--turns", type=int, default=3, help="Number of prompt turns (0 = manual)")
    parser.add_argument("--dry-run", action="store_true", help="Skip model call")
    parser.add_argument("--verbose", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    model_runner = get_model_runner(args.model)
    attack_path = os.path.join(ATTACKS_DIR, args.attack)

    if args.turns == 0:
        print


if __name__ == '__main__':
    print(f'Running {__file__}...')
