# Core Engine Context

**Last Updated**: 2025-06-07 by Claude Code

## 🎯 Purpose

The core engine provides fundamental data structures, utilities, and patterns that power all RED_CORE operations.

## 📁 Key Files

### **`log_schema.py`** - Data Structure Definitions
```python
# Primary Classes:
- SessionLog      # Complete experiment log structure
- Turn           # Individual conversation turn
- ReproductionInfo # Experiment reproducibility metadata
- EvaluatorVersions # Tool version tracking
- LLMEvaluation  # LLM-based scoring results
```

**Current Schema Version**: 2.0.0 (enhanced for dual evaluation)

### **`log_utils.py`** - Logging Operations
```python
# Key Functions:
- generate_readable_run_id() # Creates unique experiment IDs
- log_session()             # Saves SessionLog to JSON
- log_session_from_dict()   # Handles schema validation errors
```

**ID Format**: `{experiment_code}-{model_code}-{prompt_tags}-{timestamp}-{unique_hash}`

### **`context.py`** - Conversation Management
```python
# Classes:
- ConversationContext   # Single turn context and metadata
- ConversationHistory   # Multi-turn conversation state
- MessageTurn          # Individual message in conversation
```

**Usage Pattern**: Standardized conversation handling across all API runners

### **`hash_utils.py`** - Content Hashing
```python
# Functions:
- hash_string()  # Consistent SHA-256 hashing for content
```

**Purpose**: Reproducibility verification, duplicate detection

### **`logger.py`** - Experiment Logging
```python
# Functions:
- get_experiment_logger()  # Configured logger for experiment tracking
```

## 🔄 Data Flow Patterns

### **Immutable Experiment Data**
```python
# Original experiment data never changes
experiment_data = {
    "model": "gpt-4.1",
    "turns": [...],  # Original, immutable
    "created_at": "2025-06-07T23:26:00"
}
```

### **Append-Only Evaluation Results**
```python
# Evaluations added without modifying original data
_workflow = {
    "evaluations": {
        "llm": {
            "completed": True,
            "date": "2025-06-07T23:27:30",
            "results": {...}
        }
    }
}
```

## 🚧 PHASE 1 SCHEMA CHANGES (Next Sprint)

### **Enhanced SessionLog Structure**
```python
@dataclass
class SessionLog:
    # ... existing fields ...
    
    # NEW: Workflow state tracking
    _workflow: Optional[Dict] = field(default_factory=dict)
    
    # Structure:
    # _workflow = {
    #     "batch_id": "demo-03",
    #     "batch_purpose": "testing new refusal angles",
    #     "batch_created": "2025-06-07T23:26:00",
    #     "evaluations": {
    #         "llm": {
    #             "completed": False,
    #             "date": None,
    #             "model": None,
    #             "error": None
    #         }
    #     }
    # }
```

### **Evaluation State Management**
```python
def needs_llm_evaluation(log: SessionLog) -> bool:
    """Check if log needs LLM evaluation"""
    workflow = log._workflow or {}
    llm_eval = workflow.get("evaluations", {}).get("llm", {})
    return not llm_eval.get("completed", False)

def mark_evaluation_complete(log: SessionLog, eval_type: str, model: str):
    """Mark evaluation as completed"""
    if not log._workflow:
        log._workflow = {"evaluations": {}}
    log._workflow["evaluations"][eval_type] = {
        "completed": True,
        "date": datetime.now().isoformat(),
        "model": model
    }
```

## 🛡️ Data Integrity

### **Checksums for Verification**
- All content hashed with SHA-256
- Original experiment data immutable
- Evaluation additions clearly separated

### **Schema Validation**
- Pydantic validation on all data structures
- Graceful degradation for validation errors
- Version tracking for schema evolution

## 🎯 Design Principles

- **Immutability**: Original experiment data never modified
- **Composability**: Small, focused utilities that combine well
- **Type Safety**: Strong typing with Pydantic models
- **Auditability**: Complete lineage tracking in data structures

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