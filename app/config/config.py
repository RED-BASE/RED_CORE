import os
from dotenv import load_dotenv

load_dotenv()  # Loads from .env at project root if available

# === API KEYS ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# === MODEL REGISTRY ===
# Based on official SDK documentation as of June 2025
MODEL_REGISTRY = {
    # OpenAI Models
    "gpt-4.1": {
        "aliases": ["gpt-4-1", "gpt4.1", "openai-flagship"],
        "code": "G41",
        "vendor": "openai",
        "features": ["text", "vision"],
        "context_window": 1000000,
        "default": True,
    },
    "gpt-4.1-mini": {
        "aliases": ["gpt-4-1-mini", "gpt4.1-mini"],
        "code": "G41M",
        "vendor": "openai",
        "features": ["text"],
        "context_window": 1000000,
    },
    "gpt-4.1-nano": {
        "aliases": ["gpt-4-1-nano", "gpt4.1-nano"],
        "code": "G41N",
        "vendor": "openai",
        "features": ["text"],
        "context_window": 1000000,
    },
    "gpt-4o": {
        "aliases": ["gpt-4", "gpt4", "gpt-4o-latest"],
        "code": "GPT4O",
        "vendor": "openai",
        "features": ["text", "vision"],
        "context_window": 128000,
    },
    "gpt-4o-mini": {
        "aliases": ["gpt-4o-mini-latest"],
        "code": "G4OM",
        "vendor": "openai",
        "features": ["text"],
        "context_window": 128000,
    },
    
    # Anthropic Claude Models
    "claude-opus-4-20250514": {
        "aliases": ["claude-4-opus", "claude-opus-4", "claude-4"],
        "code": "C4O",
        "vendor": "anthropic",
        "features": ["text", "thinking"],
        "context_window": 200000,
    },
    "claude-sonnet-4-20250514": {
        "aliases": ["claude-4-sonnet", "claude-sonnet-4"],
        "code": "C4S",
        "vendor": "anthropic",
        "features": ["text", "thinking"],
        "context_window": 200000,
    },
    "claude-3-7-sonnet-20250219": {
        "aliases": ["claude-3-7-sonnet", "claude-3-7-sonnet-latest", "claude"],
        "code": "C37S",
        "vendor": "anthropic",
        "features": ["text", "thinking"],
        "context_window": 200000,
    },
    "claude-3-5-sonnet-20241022": {
        "aliases": ["claude-3-5-sonnet", "claude-3-5-sonnet-latest"],
        "code": "C35S",
        "vendor": "anthropic",
        "features": ["text", "vision"],
        "context_window": 200000,
    },
    "claude-3-5-haiku-20241022": {
        "aliases": ["claude-3-5-haiku"],
        "code": "C35H",
        "vendor": "anthropic",
        "features": ["text"],
        "context_window": 200000,
    },
    "claude-3-opus-20240229": {
        "aliases": ["claude-3-opus"],
        "code": "C3O",
        "vendor": "anthropic",
        "features": ["text"],
        "context_window": 200000,
    },
    
    # Google Gemini Models
    "gemini-2.5-flash-preview-05-20": {
        "aliases": ["gemini-2.5-flash", "gemini-2-5-flash"],
        "code": "G25F",
        "vendor": "google",
        "features": ["text", "vision", "thinking"],
        "context_window": 1000000,
    },
    "gemini-2.5-pro-preview-06-05": {
        "aliases": ["gemini-2.5-pro", "gemini-2-5-pro", "gemini"],
        "code": "G25P",
        "vendor": "google",
        "features": ["text", "vision", "thinking"],
        "context_window": 2000000,
    },
    "gemini-2.0-flash": {
        "aliases": ["gemini-2-0-flash"],
        "code": "G20F",
        "vendor": "google",
        "features": ["text", "vision", "multimodal"],
        "context_window": 1000000,
    },
    "gemini-2.0-flash-lite": {
        "aliases": ["gemini-2-0-flash-lite"],
        "code": "G20L",
        "vendor": "google",
        "features": ["text"],
        "context_window": 1000000,
    },
    "gemini-1.5-pro": {
        "aliases": ["gemini-pro"],
        "code": "G15P",
        "vendor": "google",
        "features": ["text", "vision"],
        "context_window": 2000000,
        "deprecated": "Limited availability - not available for new projects",
    },
    "gemini-1.5-flash": {
        "aliases": [],
        "code": "G15F",
        "vendor": "google",
        "features": ["text"],
        "context_window": 1000000,
        "deprecated": "Limited availability - not available for new projects",
    },
    
    # Local Models
    "mistral-7b-instruct-v0.1.Q4_K_M.gguf": {
        "aliases": ["mistral", "mistral-7b"],
        "code": "M7B",
        "vendor": "local",
        "features": ["text"],
        "context_window": 4096,
    },
}

def resolve_model(name: str) -> str:
    """Return canonical model key for any alias or canonical name."""
    n = name.lower().strip()
    for canon, meta in MODEL_REGISTRY.items():
        if n == canon or n in [a.lower() for a in meta.get("aliases", [])]:
            return canon
    raise ValueError(f"Unknown model or alias: {name}")

def get_model_code(name: str) -> str:
    """Return the log code for a model (by alias or canonical)."""
    canon = resolve_model(name)
    return MODEL_REGISTRY[canon].get("code", "UNK")

def get_model_aliases(name: str) -> list:
    """Return all aliases for a canonical model name."""
    canon = resolve_model(name)
    return MODEL_REGISTRY[canon].get("aliases", [])

def get_model_vendor(name: str) -> str:
    canon = resolve_model(name)
    return MODEL_REGISTRY[canon].get("vendor", "unknown")

def get_model_features(name: str) -> list:
    canon = resolve_model(name)
    return MODEL_REGISTRY[canon].get("features", [])

def get_model_context_window(name: str) -> int:
    canon = resolve_model(name)
    return MODEL_REGISTRY[canon].get("context_window", 4096)

def is_model_deprecated(name: str) -> bool:
    canon = resolve_model(name)
    return MODEL_REGISTRY[canon].get("deprecated") is not None

def get_model_deprecation_notice(name: str) -> str:
    canon = resolve_model(name)
    return MODEL_REGISTRY[canon].get("deprecated", "")

def get_default_model() -> str:
    for name, meta in MODEL_REGISTRY.items():
        if meta.get("default", False):
            return name
    raise RuntimeError("No default model set in MODEL_REGISTRY.")

def list_available_models(include_deprecated: bool = False) -> list:
    """Return list of available model names, optionally including deprecated ones."""
    models = []
    for name, meta in MODEL_REGISTRY.items():
        if include_deprecated or not meta.get("deprecated"):
            models.append(name)
    return sorted(models)

# === FILESYSTEM DEFAULTS ===
LOG_DIR = os.getenv("LOG_DIR", "experiments/guardrail_decay/logs/")
SUMMARY_DIR = "logs/summary"
SYSTEM_PROMPT_DIR = "data/prompts/system"
USER_PROMPT_DIR = "data/prompts/user"

# === PROMPT & EXPERIMENT DEFAULTS ===
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", 0.7))
DEFAULT_SYS_TAG = "standard_guardrails:latest"
DEFAULT_MODE = os.getenv("DEFAULT_MODE", "audit")
DEFAULT_EXPERIMENT_CODE = os.getenv("DEFAULT_EXPERIMENT_CODE", "GRD")

# === THRESHOLDS ===
DRIFT_SCORE_THRESHOLD = float(os.getenv("DRIFT_SCORE_THRESHOLD", 0.7))
DRIFT_SPIKE_COUNT_THRESHOLD = int(os.getenv("DRIFT_SPIKE_COUNT_THRESHOLD", 2))
DRIFT_ANALYSIS_THRESHOLD = float(os.getenv("DRIFT_ANALYSIS_THRESHOLD", 0.20))
DEFAULT_DRIFT_THRESHOLD = DRIFT_ANALYSIS_THRESHOLD  # For clarity/legacy CLI

# === LOCAL MODEL CONFIG ===
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "/Users/redhat/llm_models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
DEFAULT_CONTEXT_SIZE = int(os.getenv("DEFAULT_CONTEXT_SIZE", 4096))
LOCAL_ATTACK_YAML = os.getenv("LOCAL_ATTACK_YAML", "data/prompts/user/simple_prompt_injection.yaml")
LOCAL_ATTACK_LOG = os.getenv("LOCAL_ATTACK_LOG", "logs/attack_logs/local_test_run.lrc")
LOCAL_ATTACK_TURNS = int(os.getenv("LOCAL_ATTACK_TURNS", 1))
DEFAULT_GPU_LAYERS = int(os.getenv("DEFAULT_GPU_LAYERS", 0))
DEFAULT_LLAMA_VERBOSE = os.getenv("DEFAULT_LLAMA_VERBOSE", "false").lower() == "true"

# === LOG HASHING ===
ENABLE_LOG_HASHING = os.getenv("ENABLE_LOG_HASHING", "true").lower() == "true"

def get_model_snapshot_id(model_vendor, canonical_model_name):
    # Placeholder: return None or a dummy string
    return None
