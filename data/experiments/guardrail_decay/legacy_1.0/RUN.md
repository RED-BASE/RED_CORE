# 🧪 `LLM-RED-CORE` — Experiment Run Guide

## 🚀 Short Run Templates

### 🔹 GPT-4o (Single Turn)
```bash
python runners/run_exploit_yaml.py \
  --path prompts_user/guardrail_decay/usr_guardrail_decay_01.yaml \
  --sys-prompt prompts_system/sys_guardrail_decay_01.yaml \
  --model gpt-4o
```

### 🔹 Claude 3 Opus
```bash
python runners/run_exploit_yaml.py \
  --path prompts_user/guardrail_decay/usr_guardrail_decay_01.yaml \
  --sys-prompt prompts_system/sys_guardrail_decay_01.yaml \
  --model claude-3-opus
```

### 🔹 Gemini Flash
```bash
python runners/run_exploit_yaml.py \
  --path prompts_user/guardrail_decay/usr_guardrail_decay_01.yaml \
  --sys-prompt prompts_system/sys_guardrail_decay_01.yaml \
  --model gemini-flash
```

---

## 🧰 Full CLI Flag Reference

| Flag | Description |
|------|-------------|
| `--path` | Path to user prompt YAML *(required)* |
| `--sys-prompt` | Path to system prompt YAML *(required)* |
| `--model` | Target model (`gpt-4o`, `claude-3-opus`, `gemini-flash`, etc.) |
| `--mode` | `audit` or `simulate` *(default: `audit`)* |
| `--score-log` | Enable drift scoring output |
| `--disable-containment` | Skip containment layer during run |
| `--temperature` | Set sampling temperature *(default: `0.7`)* |
| `--experiment-id` | Custom ID tag for experiment |
| `--scenario-hash` | Precomputed hash for reproducibility |
| `--usr-prompt` | (for batch runs) Shortcut to set `--path` |
| `--sys-prompt` | (for batch runs) Shared system prompt for all models |

---

## 🧪 Batch Run Across Models

```bash
python -m experiments.guardrail_decay.batch_run \
  --models gpt-4o claude-3-opus gemini-flash \
  --sys-prompt prompts_system/sys_guardrail_decay_01.yaml \
  --usr-prompt prompts_user/guardrail_decay/usr_guardrail_decay_01.yaml \
  --score-log
```

✅ Add `--disable-containment` to skip safety filters  
✅ Add `--temperature 0.5` to override temp

---

## 🧪 Batch Run with `sys_none.yaml`

```bash
python -m experiments.guardrail_decay.batch_run \
  --models gpt-4o claude-3-opus gemini-flash \
  --sys-prompt prompts_system/sys_none_00.yaml \
  --usr-prompt prompts_user/guardrail_decay/usr_guardrail_decay_01.yaml
```

---

## 🧠 Drift Diff

```bash
python tools/diff_logs.py logs/raw/gpt-4o/FILE1.json logs/raw/claude-3-opus/FILE2.json
```

---

## 🐛 Gemini Debug Toggle

Enable verbose Gemini output:
```bash
export DEBUG_GEMINI=true
```

Disable it:
```bash
unset DEBUG_GEMINI
```