# Experiments Context

**Last Updated**: 2025-06-07 by Claude Code

## 🎯 Purpose

Experiment organization and batch workflow management. Each subdirectory represents a focused research investigation with its own prompts, logs, and analysis.

## 📁 Directory Structure

```
experiments/
├── demo/                    # Simple test experiment
├── 80k_hours_demo/         # Professional showcase experiment  
├── refusal_robustness/     # Refusal pattern testing
├── guardrail_decay/        # Safety degradation analysis
└── exp_template/           # Template for new experiments
```

### **Standard Experiment Layout**
```
{experiment_name}/
├── README.md              # Purpose, methodology, findings
├── prompts/               # Experiment-specific prompts
│   ├── usr_*.yaml        # User prompts with multiple turns
│   └── sys_*.yaml        # System prompts (if experiment-specific)
├── logs/                  # Raw experiment logs
└── analysis/             # Generated CSV exports, reports
```

## 🔄 Current Workflow Issues

### **Problems Identified:**
1. **Log Accumulation**: Multiple batch runs create confusing file collections
2. **No Batch Boundaries**: Can't distinguish between different test runs
3. **Re-evaluation Waste**: Evaluator processes all logs every time
4. **Poor Organization**: Hard to know what each batch was testing

## 🚧 PHASE 1 WORKFLOW REDESIGN

### **Enhanced Directory Structure**
```
{experiment_name}/
├── README.md
├── prompts/
├── logs/                  # All experiment logs
│   ├── {exp}-01-*.json   # Batch 1 logs
│   ├── {exp}-02-*.json   # Batch 2 logs  
│   └── {exp}-03-*.json   # Batch 3 logs
├── .batch_counter         # Simple integer: "3"
└── analysis/             # Generated exports (gitignored)
    ├── batch-01.csv      # CSV exports per batch
    └── summary.csv       # Combined analysis
```

### **Batch Tracking in Logs**
```json
{
  "isbn_run_id": "DMO-G20F-03-2326-447716",
  "_workflow": {
    "batch_id": "demo-03",
    "batch_purpose": "testing edge case prompts",
    "batch_created": "2025-06-07T23:26:00",
    "experiment_name": "demo"
  }
}
```

### **Smart Commands**
```bash
make exp                   # Create new experiment
make run                   # Enhanced with batch tracking  
make eval                 # Only incomplete logs
make csv                  # Interactive CSV export menu
make status               # Dashboard view
```

## 📋 Experiment Categories

### **Research Experiments**
- `refusal_robustness` - Testing prompt injection resistance
- `guardrail_decay` - Multi-turn safety degradation
- `universal_exploits` - Cross-model attack patterns

### **Demo Experiments**  
- `demo` - Simple test cases for development
- `80k_hours_demo` - Professional showcase for career discussions

### **Templates**
- `exp_template` - Scaffolding for new experiments

## 🎯 Experiment Lifecycle

### **1. Creation** (`make exp`)
```bash
🚀 Create New Experiment
━━━━━━━━━━━━━━━━━━━━━━
Experiment code: adversarial_poetry
Full name: Adversarial Poetry Generation  
Purpose: Testing creative prompt attacks on safety systems
```

### **2. Execution** (`make run`)
```bash
🎯 Select experiment: adversarial_poetry
📝 Batch purpose: testing haiku-based jailbreaks
✨ Starting batch: adversarial_poetry-01
```

### **3. Analysis** (`make eval`, `make csv`)
- Automatic evaluation of new logs only
- CSV export for statistical analysis
- Summary reports with key metrics

### **4. Review** (Red Score Mobile)
- Filter logs by review status
- Mobile-friendly log browsing
- Progress tracking per experiment

## 🏷️ Naming Conventions

### **Experiment Codes**
- Short, descriptive: `demo`, `refusal_robustness` 
- Snake_case for multi-word names
- No spaces or special characters

### **Batch IDs**
- Format: `{experiment}-{01,02,03...}`
- Auto-incrementing per experiment
- Purpose stored in log metadata

### **Log IDs** 
- Format: `{EXP}-{MODEL}-{BATCH}-{TIMESTAMP}-{HASH}`
- Example: `DMO-G20F-03-2326-447716`

## 📊 Current Experiments

### **Demo** (Simple Test Cases)
- **Purpose**: Development testing and workflow validation
- **Status**: Active, multiple batches run
- **Key Findings**: Workflow improvements needed

### **80K Hours Demo** (Professional Showcase)
- **Purpose**: Career discussion material, methodology demonstration  
- **Status**: Complete, publication-ready
- **Key Findings**: 5-turn experiment with 100% safety compliance

### **Refusal Robustness** (Research Focus)
- **Purpose**: Testing resistance to manipulation techniques
- **Status**: Large dataset, needs analysis
- **Key Findings**: Significant model differences in refusal patterns

## 📝 Development Protocol

**Git Workflow**: Commit after every minor change with descriptive messages
```bash
git add -A && git commit -m "Description of change"
```

---
*Part of RED_CORE by Cassius Oldenburg*