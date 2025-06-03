import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Optional config import for dynamic defaults
try:
    from app.config import (
        LOCAL_MODEL_PATH,
        DEFAULT_CONTEXT_SIZE,
        DEFAULT_GPU_LAYERS,
        DEFAULT_LLAMA_VERBOSE
    )
except ImportError:
    LOCAL_MODEL_PATH = "/Users/redhat/llm_models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    DEFAULT_CONTEXT_SIZE = 2048
    DEFAULT_GPU_LAYERS = 0
    DEFAULT_LLAMA_VERBOSE = False

# Optional dependency
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None  # Define Llama as None if import fails

class LlamaCppRunner:
    """Runner for GGUF models using llama-cpp-python."""

    def __init__(
        self,
        model_path: str = LOCAL_MODEL_PATH,
        n_gpu_layers: int = DEFAULT_GPU_LAYERS,
        n_ctx: int = DEFAULT_CONTEXT_SIZE,
        verbose: bool = DEFAULT_LLAMA_VERBOSE,
        **kwargs
    ):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("The 'llama-cpp-python' library is required. Please install it.")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.verbose = verbose
        self.model_kwargs = kwargs

        # Used in log metadata
        self.model_name = os.path.basename(model_path).split(".")[0]

        print(f"Initializing LlamaCppRunner with model: {self.model_path}")
        print(f"GPU Layers: {self.n_gpu_layers}, Context Size: {self.n_ctx}")

        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=self.verbose,
                **self.model_kwargs
            )
            print("Llama.cpp model loaded successfully.")
        except Exception as e:
            print(f"Error loading Llama.cpp model: {e}")
            raise

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.llm:
            print("Llama.cpp model not initialized.")
            return ""

        generation_params = {
            'max_tokens': kwargs.get('max_tokens', 512),
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.95),
            'top_k': kwargs.get('top_k', 40),
            'repeat_penalty': kwargs.get('repeat_penalty', 1.1),
            'stop': kwargs.get('stop', [])
        }
        generation_params = {k: v for k, v in generation_params.items() if v is not None}

        try:
            raw_result = self.llm(prompt, **generation_params)

            if raw_result and 'choices' in raw_result and raw_result['choices']:
                raw_output = raw_result['choices'][0]['text']
                if isinstance(raw_output, list):
                    return raw_output[0].strip() if raw_output else ""
                return str(raw_output).strip()
            return ""

        except Exception as e:
            print(f"Error during Llama.cpp generation: {e}")
            return ""

    def count_tokens(self, prompt: str) -> int:
        return len(self.llm.tokenize(prompt.encode("utf-8")))

    def get_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "n_gpu_layers": self.n_gpu_layers,
            "n_ctx": self.n_ctx,
            "verbose": self.verbose
        }


