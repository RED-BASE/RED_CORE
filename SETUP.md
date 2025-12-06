# RED_CORE Setup Guide

**Quick start for university researchers and students** 🎓

## One-Command Setup

```bash
# Clone and setup RED_CORE
git clone <repository-url> RED_CORE
cd RED_CORE
./setup.sh
```

## Manual Setup

### 1. Install Dependencies

```bash
# Using Poetry (recommended)
pip install poetry
poetry install

# Or using pip
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
# Required for most experiments
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here  
GOOGLE_API_KEY=your_google_key_here

# Optional providers
XAI_API_KEY=your_xai_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
MISTRAL_API_KEY=your_mistral_key_here
```

### 3. Verify Setup

```bash
# Test configuration
poetry run python -c "from app.config.config import validate_config; print('Issues:', validate_config())"

# Run a quick test
make status
```

## Academic Lab Configuration

### Shared Environment Setup

```bash
# Set custom paths for shared lab environments
export RED_CORE_EXPERIMENTS_DIR="/shared/experiments"
export RED_CORE_DATA_DIR="/shared/red_core_data"
export RED_CORE_MODELS_DIR="/shared/models"
```

### Multi-User Batch Tracking

The system automatically handles multiple users running experiments simultaneously:
- Uses file locking to prevent race conditions
- Includes user identification in batch metadata  
- Supports configurable experiments directory

### Platform Compatibility

**macOS**: Works out of the box
- Default model path: `~/llm_models/`
- Full GUI support for progress displays

**Linux**: Academic server friendly
- Default model path: `~/models/`
- Headless operation supported

**Windows**: Student laptop compatible  
- Default model path: `~/Documents/llm_models/`
- PowerShell and Command Prompt support

## Configuration Options

All paths and settings can be customized via environment variables:

```bash
# Core paths
RED_CORE_EXPERIMENTS_DIR="/custom/experiments"
RED_CORE_DATA_DIR="/custom/data" 
RED_CORE_MODELS_DIR="/custom/models"

# Experiment settings
RED_CORE_TEMPERATURE="0.8"
RED_CORE_MODE="research"
RED_CORE_EXPERIMENT_CODE="LAB"

# Analysis thresholds  
RED_CORE_DRIFT_THRESHOLD="0.6"
RED_CORE_DRIFT_ANALYSIS="0.15"

# Local models
RED_CORE_LOCAL_MODEL="/path/to/your/model.gguf"
RED_CORE_CONTEXT_SIZE="8192"
RED_CORE_GPU_LAYERS="32"
```

## Quick Start Commands

```bash
# Create a new experiment
make exp

# Run experiments interactively  
make run

# Evaluate results
make eval

# Export data for analysis
make csv

# Check system status
make status
```

## Troubleshooting

### API Key Issues
```bash
# Check which keys are configured
poetry run python -c "from app.config.config import get_config_summary; print(get_config_summary())"
```

### Path Issues
```bash
# Validate all paths
poetry run python -c "from app.config.config import validate_config; [print(f'❌ {issue}') for issue in validate_config()]"
```

### Local Model Issues
```bash
# Check local model availability
ls -la ~/llm_models/
# or your custom model directory
ls -la $RED_CORE_MODELS_DIR
```

### Permission Issues (Linux/macOS)
```bash
# Fix permissions for shared lab environments
sudo chown -R $USER:$USER /shared/experiments
chmod -R 755 /shared/experiments
```

## Academic Use Cases

### Single Researcher
Default setup works perfectly - just add API keys and run experiments.

### Research Lab (Multiple Users)
Set `RED_CORE_EXPERIMENTS_DIR` to a shared directory and ensure all users have read/write access.

### University Course
Use Docker deployment for consistent environment across all student machines.

### Cross-Institutional Research
Share experiment configurations via YAML files, use consistent environment variable naming.

## Getting Help

1. **Configuration Issues**: Run `make status` for diagnostics
2. **Experiment Problems**: Check `experiments/*/logs/` for error details  
3. **API Limits**: Monitor rate limits with `make rates`
4. **Model Access**: Verify API keys and model availability

---

*RED_CORE is designed for easy academic adoption - if setup takes more than 5 minutes, we've done something wrong!*