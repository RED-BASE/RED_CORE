import os
from dotenv import load_dotenv

load_dotenv()  # Loads from .env at project root if available

# === API KEYS ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# === MODEL REGISTRY ===
# Based on official SDK documentation as of January 2026
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

    # Anthropic Claude Models (Latest as of Jan 2026)
    "claude-opus-4-5-20251101": {
        "aliases": ["claude-opus-4.5", "claude-4.5-opus", "claude-opus", "claude-4.5"],
        "code": "C45O",
        "vendor": "anthropic",
        "features": ["text", "thinking", "vision", "agents"],
        "context_window": 200000,
    },
    "claude-sonnet-4-5-20250929": {
        "aliases": ["claude-sonnet-4.5", "claude-4.5-sonnet", "claude-sonnet", "claude"],
        "code": "C45S",
        "vendor": "anthropic",
        "features": ["text", "thinking", "vision", "agents"],
        "context_window": 200000,
    },
    "claude-haiku-4-5-20251001": {
        "aliases": ["claude-haiku-4.5", "claude-4.5-haiku", "claude-haiku"],
        "code": "C45H",
        "vendor": "anthropic",
        "features": ["text", "thinking"],
        "context_window": 200000,
    },
    "claude-opus-4-20250514": {
        "aliases": ["claude-4-opus", "claude-opus-4"],
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
    "claude-3-5-haiku-20241022": {
        "aliases": ["claude-3-5-haiku", "claude-3.5-haiku"],
        "code": "C35H",
        "vendor": "anthropic",
        "features": ["text"],
        "context_window": 200000,
    },
    "claude-3-7-sonnet-20250219": {
        "aliases": ["claude-3-7-sonnet", "claude-3.7-sonnet"],
        "code": "C37S",
        "vendor": "anthropic",
        "features": ["text", "thinking"],
        "context_window": 200000,
    },
    "claude-3-5-sonnet-20241022": {
        "aliases": ["claude-3-5-sonnet", "claude-3.5-sonnet"],
        "code": "C35S",
        "vendor": "anthropic",
        "features": ["text", "vision"],
        "context_window": 200000,
    },
    "claude-3-opus-20240229": {
        "aliases": ["claude-3-opus"],
        "code": "C3O",
        "vendor": "anthropic",
        "features": ["text"],
        "context_window": 200000,
        "deprecated": "Retiring Jan 5, 2026 - migrate to Opus 4.5",
    },

    # Google Gemini Models (Latest as of Jan 2026)
    "gemini-2.5-pro": {
        "aliases": ["gemini-2-5-pro", "gemini-pro", "gemini"],
        "code": "G25P",
        "vendor": "google",
        "features": ["text", "vision", "thinking"],
        "context_window": 2000000,
    },
    "gemini-2.5-flash": {
        "aliases": ["gemini-2-5-flash", "gemini-flash"],
        "code": "G25F",
        "vendor": "google",
        "features": ["text", "vision", "thinking"],
        "context_window": 1000000,
    },
    "gemini-3-pro": {
        "aliases": ["gemini-3", "gemini-3-pro-preview"],
        "code": "G3P",
        "vendor": "google",
        "features": ["text", "vision", "thinking", "agents"],
        "context_window": 2000000,
        "preview": True,
    },
    "gemini-3-flash": {
        "aliases": ["gemini-3-flash-preview"],
        "code": "G3F",
        "vendor": "google",
        "features": ["text", "vision", "thinking"],
        "context_window": 1000000,
        "preview": True,
    },
    "gemini-2.0-flash": {
        "aliases": ["gemini-2-0-flash"],
        "code": "G20F",
        "vendor": "google",
        "features": ["text", "vision", "multimodal"],
        "context_window": 1000000,
    },
    "gemini-2.0-flash-lite": {
        "aliases": ["gemini-2-0-flash-lite", "gemini-lite"],
        "code": "G20L",
        "vendor": "google",
        "features": ["text"],
        "context_window": 1000000,
    },
    "gemini-1.5-pro": {
        "aliases": ["gemini-1-5-pro"],
        "code": "G15P",
        "vendor": "google",
        "features": ["text", "vision"],
        "context_window": 2000000,
        "deprecated": "Not available for new projects - existing users only",
    },
    "gemini-1.5-flash": {
        "aliases": ["gemini-1-5-flash"],
        "code": "G15F",
        "vendor": "google",
        "features": ["text"],
        "context_window": 1000000,
        "deprecated": "Not available for new projects - existing users only",
    },

    # Local Models
    "mistral-7b-instruct-v0.1.Q4_K_M.gguf": {
        "aliases": ["mistral-local", "mistral-7b-local"],
        "code": "M7B",
        "vendor": "local",
        "features": ["text"],
        "context_window": 4096,
    },

    # XAI Models (Grok)
    "grok-beta": {
        "aliases": ["grok", "grok-1", "xai-grok"],
        "code": "GROK",
        "vendor": "xai",
        "features": ["text", "realtime"],
        "context_window": 131072,
    },

    # DeepSeek Models (Latest as of Jan 2026)
    "deepseek-chat": {
        "aliases": ["deepseek", "deepseek-v3", "deepseek-v3.2"],
        "code": "DSV3",
        "vendor": "deepseek",
        "features": ["text", "code", "reasoning"],
        "context_window": 128000,
    },
    "deepseek-reasoner": {
        "aliases": ["deepseek-r1", "deepseek-reasoning"],
        "code": "DSR1",
        "vendor": "deepseek",
        "features": ["text", "code", "reasoning"],
        "context_window": 128000,
    },

    # Mistral AI (API)
    "mistral-large-latest": {
        "aliases": ["mistral-large", "mistral"],
        "code": "MLRG",
        "vendor": "mistral",
        "features": ["text", "function-calling"],
        "context_window": 128000,
    },
    "mistral-small-latest": {
        "aliases": ["mistral-small"],
        "code": "MSML",
        "vendor": "mistral",
        "features": ["text"],
        "context_window": 128000,
    },
    "codestral-latest": {
        "aliases": ["codestral", "mistral-code"],
        "code": "CDRL",
        "vendor": "mistral",
        "features": ["text", "code", "function-calling"],
        "context_window": 128000,
    },

    # Cohere Models
    "command-r-plus": {
        "aliases": ["command-r+", "cohere-command-r+"],
        "code": "CRP",
        "vendor": "cohere",
        "features": ["text", "rag"],
        "context_window": 128000,
    },
    "command-r": {
        "aliases": ["cohere-command-r"],
        "code": "CR",
        "vendor": "cohere",
        "features": ["text", "rag"],
        "context_window": 128000,
    },

    # Together AI (Representative models)
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {
        "aliases": ["llama-3.3-70b", "llama-70b-together"],
        "code": "L33T",
        "vendor": "together",
        "features": ["text"],
        "context_window": 131072,
    },
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": {
        "aliases": ["llama-3.2-90b-vision", "llama-vision-together"],
        "code": "L32V",
        "vendor": "together",
        "features": ["text", "vision"],
        "context_window": 131072,
    },
}

def resolve_model(name: str) -> str:
    """Return canonical model key for any alias or canonical name."""
    n = name.lower().strip()
    for canon, meta in MODEL_REGISTRY.items():
        if n == canon.lower() or n in [a.lower() for a in meta.get("aliases", [])]:
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

# === CENTRALIZED CONFIGURATION ===
# R1.4: Replace scattered hardcoded values with environment-based config

import platform
from pathlib import Path

# === PROJECT PATHS (Portable) ===
def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent

def get_data_dir() -> Path:
    """Get data directory (configurable for academic environments)."""
    data_dir = os.getenv("RED_CORE_DATA_DIR")
    if data_dir:
        return Path(data_dir)
    return get_project_root() / "data"

def get_experiments_dir() -> Path:
    """Get experiments directory (configurable for shared lab environments)."""
    exp_dir = os.getenv("RED_CORE_EXPERIMENTS_DIR")
    if exp_dir:
        return Path(exp_dir)
    return get_project_root() / "experiments"

def get_models_dir() -> Path:
    """Get local models directory (platform-aware defaults)."""
    models_dir = os.getenv("RED_CORE_MODELS_DIR")
    if models_dir:
        return Path(models_dir)
    
    # Platform-aware defaults for academic environments
    if platform.system() == "Darwin":  # macOS
        return Path.home() / "llm_models"
    elif platform.system() == "Linux":
        return Path.home() / "models" 
    else:  # Windows
        return Path.home() / "Documents" / "llm_models"

# === FILESYSTEM PATHS (All configurable) ===
PROJECT_ROOT = get_project_root()
DATA_DIR = get_data_dir()
EXPERIMENTS_DIR = get_experiments_dir()
MODELS_DIR = get_models_dir()

# Derived paths
LOG_DIR = os.getenv("RED_CORE_LOG_DIR", str(EXPERIMENTS_DIR / "guardrail_decay" / "logs"))
SUMMARY_DIR = os.getenv("RED_CORE_SUMMARY_DIR", str(DATA_DIR / "logs" / "summary"))
SYSTEM_PROMPT_DIR = os.getenv("RED_CORE_SYSTEM_PROMPTS", str(DATA_DIR / "prompts" / "system"))
USER_PROMPT_DIR = os.getenv("RED_CORE_USER_PROMPTS", str(DATA_DIR / "prompts" / "user"))

# === EXPERIMENT CONFIGURATION ===
DEFAULT_TEMPERATURE = float(os.getenv("RED_CORE_TEMPERATURE", "0.7"))
DEFAULT_SYS_TAG = os.getenv("RED_CORE_SYSTEM_TAG", "standard_guardrails:latest")
DEFAULT_MODE = os.getenv("RED_CORE_MODE", "audit")
DEFAULT_EXPERIMENT_CODE = os.getenv("RED_CORE_EXPERIMENT_CODE", "GRD")

# === ANALYSIS THRESHOLDS ===
DRIFT_SCORE_THRESHOLD = float(os.getenv("RED_CORE_DRIFT_THRESHOLD", "0.7"))
DRIFT_SPIKE_COUNT_THRESHOLD = int(os.getenv("RED_CORE_DRIFT_SPIKES", "2"))
DRIFT_ANALYSIS_THRESHOLD = float(os.getenv("RED_CORE_DRIFT_ANALYSIS", "0.20"))
DEFAULT_DRIFT_THRESHOLD = DRIFT_ANALYSIS_THRESHOLD

# === LOCAL MODEL CONFIGURATION (Portable) ===
def get_default_local_model_path() -> str:
    """Get default local model path based on platform and available models."""
    models_dir = get_models_dir()
    
    # Common model filenames to search for
    common_models = [
        "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "llama-2-7b-chat.Q4_K_M.gguf", 
        "codellama-7b-instruct.Q4_K_M.gguf"
    ]
    
    # Try to find an existing model
    for model_file in common_models:
        model_path = models_dir / model_file
        if model_path.exists():
            return str(model_path)
    
    # Return default path (may not exist)
    return str(models_dir / "mistral-7b-instruct-v0.1.Q4_K_M.gguf")

LOCAL_MODEL_PATH = os.getenv("RED_CORE_LOCAL_MODEL", get_default_local_model_path())
DEFAULT_CONTEXT_SIZE = int(os.getenv("RED_CORE_CONTEXT_SIZE", "4096"))
LOCAL_ATTACK_YAML = os.getenv("RED_CORE_ATTACK_YAML", str(DATA_DIR / "prompts" / "user" / "simple_prompt_injection.yaml"))
LOCAL_ATTACK_LOG = os.getenv("RED_CORE_ATTACK_LOG", "logs/attack_logs/local_test_run.lrc") 
LOCAL_ATTACK_TURNS = int(os.getenv("RED_CORE_ATTACK_TURNS", "1"))
DEFAULT_GPU_LAYERS = int(os.getenv("RED_CORE_GPU_LAYERS", "0"))
DEFAULT_LLAMA_VERBOSE = os.getenv("RED_CORE_VERBOSE", "false").lower() == "true"

# === FEATURE FLAGS ===
ENABLE_LOG_HASHING = os.getenv("RED_CORE_LOG_HASHING", "true").lower() == "true"

# === VALIDATION FUNCTIONS ===
def validate_config() -> list:
    """Validate configuration and return list of issues."""
    issues = []
    
    # Check required directories exist or can be created
    for name, path in [
        ("Data directory", DATA_DIR),
        ("Experiments directory", EXPERIMENTS_DIR),
        ("System prompts", SYSTEM_PROMPT_DIR),
        ("User prompts", USER_PROMPT_DIR)
    ]:
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"{name} ({path}): Cannot create - {e}")
    
    # Check local model if configured
    if LOCAL_MODEL_PATH and not Path(LOCAL_MODEL_PATH).exists():
        issues.append(f"Local model not found: {LOCAL_MODEL_PATH}")
    
    # Check API keys for common providers
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        issues.append(f"Missing API keys: {', '.join(missing_keys)}")
    
    return issues

def get_config_summary() -> dict:
    """Get configuration summary for diagnostics."""
    return {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "experiments_dir": str(EXPERIMENTS_DIR),
        "models_dir": str(MODELS_DIR),
        "local_model_exists": Path(LOCAL_MODEL_PATH).exists(),
        "platform": platform.system(),
        "api_keys_configured": len([k for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"] if os.getenv(k)])
    }

def get_model_snapshot_id(model_vendor, canonical_model_name):
    # Placeholder: return None or a dummy string
    return None
