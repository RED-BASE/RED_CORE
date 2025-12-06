# CLI Interface Context

**Last Updated**: 2025-06-07 by Claude Code

## 🎯 Purpose

The CLI layer provides the primary user interface for RED_CORE, focusing on experiment orchestration and batch management.

## 📁 Key Files

### **`run_experiments.py`** - Main Orchestrator
```python
# Primary Functions:
- run_exploit_yaml()     # Single experiment execution
- main()                 # CLI argument parsing and batch orchestration
- configure_interactive() # Rich-based experiment selection
- generate_readable_run_id() # Unique experiment IDs
```

**Dependencies:**
- All API runners (anthropic, openai, google, llama_cpp)
- `app.core.*` (logging, schemas, context)
- `app.config.*` (model registry, scoring config)
- `safety.containment` (content filtering)
- Rich UI library for interactive configuration

**Security Constraint:** Only file allowed to write outputs

## 🎨 User Experience Patterns

### **Interactive Configuration**
- Color-coded experiment selection
- Model selection with context windows
- Progressive disclosure of options
- Professional progress indicators

### **Progress Display**
- Golden orange progress indicator with blinking symbol
- Real-time turn completion tracking
- Error count display during execution
- Clean completion summaries with success coloring

### **Error Handling**
- Graceful API failure recovery
- Detailed error logs written to experiment directories
- User-friendly error messages
- Systematic issue detection (all runs failing for a model)

## 🔄 Current Workflow

**Batch Execution Flow:**
1. Interactive experiment selection
2. Model and prompt configuration
3. Parallel execution with progress tracking
4. Automatic dual evaluation (if enabled)
5. Results summary with completion metrics

## 🚧 PHASE 1 CHANGES (Next Sprint)

### **Enhanced Batch Tracking**
```python
# Add to run_exploit_yaml():
log_output = SessionLog(
    # ... existing fields ...
    _workflow={
        "batch_id": "demo-03",
        "batch_purpose": "testing new refusal angles",
        "batch_created": datetime.now().isoformat(),
        "evaluations": {
            "llm": {"completed": False}
        }
    }
)
```

### **New Make Commands to Implement**
- `make exp` - Interactive experiment creator
- `make usr` - User prompt builder
- `make sys` - System prompt builder
- `make eval` - Smart evaluation (incomplete only)
- `make csv` - Interactive CSV export menu
- `make status` - Dashboard showing what needs attention

### **Batch Counter Logic**
```python
def get_next_batch_id(experiment_name):
    counter_file = Path(f"experiments/{experiment_name}/.batch_counter")
    if counter_file.exists():
        count = int(counter_file.read_text()) + 1
    else:
        count = 1
    counter_file.write_text(str(count))
    return f"{experiment_name}-{count:02d}"
```

## 🎯 Design Principles

- **Immediate Feedback**: Users see progress and results in real-time
- **Error Transparency**: Clear indication of what went wrong and why
- **Professional Feel**: Clean, corporate-appropriate interface
- **Efficiency Focus**: Minimize user input, maximize automation
- **Context Preservation**: Remember user choices for next session

## 📝 Development Protocol

**Git Workflow**: Commit after every minor change with descriptive messages
```bash
git add -A && git commit -m "Description of change"
```

**Strategic Comments** (high-impact only):
- `# PHASE 1:` markers on major changes
- `# RED SCORE:` integration points for mobile app
- `# BREAKING:` when removing/changing existing behavior
- `# DESIGN:` rationale for non-obvious choices

---
*Part of RED_CORE by Cassius Oldenburg*