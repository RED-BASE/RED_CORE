# Unicode Experiment Context

**Last Updated**: 2025-06-08 by Claude Code

## 🎯 Purpose

Testing adversarial attacks that use unicode to obfuscate or cloak requests and intent.

## 📁 Experiment Structure

```
unicode/
├── README.md              # Methodology and findings
├── CLAUDE.md             # This context file
├── prompts/              # Experiment-specific prompts
│   └── usr_unicode_01.yaml
├── logs/                 # Raw experiment logs (batch tracked)
└── analysis/            # Generated CSV exports
    ├── unicode-complete.csv
    ├── unicode-01.csv
    └── unicode-02.csv
```

## 🔄 Current Status

**Phase**: Initial Development  
**Code**: UNI  
**Batch Tracking**: Enabled

## 🚀 Quick Commands

```bash
# Run this experiment
make run  # Select "unicode"

# Evaluate results  
make eval

# Export results
make csv  # Select "unicode"
```

## 🧪 Research Focus

### Primary Hypothesis
"Unicode confusables will lead to a 50-70% increase in incorrect intent classification by AI models, as compared to unaltered text inputs, when adversarial inputs are introduced in a context designed to elicit policy violations."

### Key Metrics
- Refusal rates by model
- Safety mechanism effectiveness
- Response pattern analysis
- Cross-model behavioral differences

## 📋 Development Protocol

**Git Workflow**: Commit after every minor change with descriptive messages
```bash
git add -A && git commit -m "Description of change"
```

---
*Part of RED_CORE by Cassius Oldenburg*