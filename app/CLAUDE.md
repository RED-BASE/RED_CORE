# App Architecture Context

**Last Updated**: 2025-06-07 by Claude Code

## 🎯 App Overview

The `app/` directory contains RED_CORE's core execution engine - everything needed to run experiments, interface with AI models, and process results.

## 🏗️ Module Architecture

### **CLI Layer** (`cli/`)
- **Primary Interface**: Command-line tools for experiment execution
- **User Experience**: Rich interactive prompts, progress displays
- **Entry Points**: Main experiment runner, batch orchestration

### **Core Engine** (`core/`)
- **Data Structures**: Log schemas, conversation contexts
- **Utilities**: Hashing, logging, ID generation
- **Patterns**: Immutable data flows, type safety

### **Analysis Pipeline** (`analysis/`)
- **Evaluation**: Automated scoring + LLM-based assessment
- **Processing**: Batch evaluation, result aggregation
- **Output**: Structured analysis for human review

### **Model Interfaces** (`api_runners/`)
- **Abstraction**: Unified interface across AI vendors
- **Implementation**: Vendor-specific API handling
- **Reliability**: Retry logic, rate limiting, error handling

## 🔄 Data Flow

```
CLI → Core (schema) → API Runners → Models
 ↓                                    ↓
Analysis ← Core (logging) ← Results ←
```

## 🛡️ Security Patterns

- **File I/O Restriction**: Only CLI layer writes to filesystem
- **Input Validation**: All user inputs validated at boundaries
- **Containment**: Multi-layer content filtering on outputs

## 📋 Current Priorities

See main `/CLAUDE.md` for current sprint tasks affecting this module.

## 📝 Development Protocol

**Git Workflow**: Commit after every minor change with descriptive messages
```bash
git add -A && git commit -m "Description of change"
```

---
*Part of RED_CORE by Cassius Oldenburg*