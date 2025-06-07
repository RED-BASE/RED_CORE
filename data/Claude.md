# RED_CORE Development Context

**Project Lead**: Cassius Oldenburg  
**Last Updated**: 2025-06-06 by Claude Code

---

## 🎯 Project Mission

RED_CORE is Cassius Oldenburg's comprehensive framework for adversarial AI safety research, focusing on refusal robustness testing, guardrail decay analysis, and systematic attack pattern discovery.

---

## 🏗️ Dependency Matrix

### [CORE ORCHESTRATOR]
**`app/cli/run_experiments.py`** - Main experiment conductor
```
├─ Dependencies: 
│  ├─ All API runners (anthropic, openai, google, llama_cpp)
│  ├─ app.core.{log_schema, log_utils, hash_utils, logger, context}
│  ├─ app.config.{config, scoring_config}
│  ├─ safety.containment
│  └─ Rich UI library for interactive configuration
├─ Key Functions:
│  ├─ run_batch() - Orchestrates experiment batches
│  ├─ configure_interactive() - Rich-based experiment setup
│  └─ generate_readable_run_id() - Unique experiment IDs
├─ Recent Changes: batch execution fixes, Rich UI integration
└─ Notes: Only file allowed to write outputs (security constraint)
```

### [MODEL INTERFACES]
**`app/api_runners/`** - Vendor-specific model adapters
```
anthropic_runner.py
├─ Dependencies: app.core.context, app.config
├─ Key Functions: generate(), set_system_prompt()
├─ Patterns: Retry logic, rate limiting, Claude 2/3 format handling

openai_runner.py  
├─ Dependencies: app.core.context, app.config
├─ Key Functions: generate(), conversation handling
├─ Patterns: Similar retry/rate limiting pattern

google_runner.py & llama_cpp_runner.py
├─ Similar interface pattern for consistency
└─ Notes: All runners implement same generate() signature
```

### [CORE DATA STRUCTURES]
**`app/core/`** - Shared utilities and schemas
```
context.py
├─ Exports: ConversationContext, ConversationHistory, MessageTurn  
├─ Purpose: Standardized conversation handling across all runners
└─ Notes: Core abstraction for multi-turn conversations

log_schema.py
├─ Exports: SessionLog, Turn data classes
├─ Purpose: Structured logging format for experiments
└─ Notes: Ensures consistent data format across experiments

log_utils.py
├─ Functions: log_session(), generate_readable_run_id()
├─ Dependencies: log_schema.py
└─ Purpose: Centralized logging operations
```

### [ANALYSIS PIPELINE]
**`app/analysis/`** - Post-experiment analysis tools
```
enhanced_review_tool.py
├─ Dependencies: Core logging utilities
├─ Purpose: Manual review interface for experiment results
├─ Key Features: Color-coded output, containment analysis
└─ Notes: Requires manual review for safety validation

automated_scorer.py
├─ Dependencies: app.config.scoring_config
├─ Purpose: Automated drift/refusal scoring
└─ Notes: Supplements manual review, not replacement

log_insight_report.py
├─ Purpose: Aggregate analysis across experiments
└─ Notes: Generates summary reports and trends
```

### [SAFETY & CONFIGURATION]
**`safety/containment.py`** - Content filtering system
```
├─ Purpose: Multi-layer content filtering
├─ Key Functions: containment_summary(), flatten_containment_flags()
├─ Patterns: Pattern-based filtering with severity levels
└─ Notes: Required for all experiment outputs
```

**`app/config/config.py`** - Central configuration
```
├─ Exports: Model registry, API keys, directory paths
├─ Key Functions: resolve_model(), get_model_code()
└─ Notes: Single source of truth for model configurations
```

---

## 📝 Session Notes & Decision Rationale

### 2025-06-06: Log Schema Overhaul for Research Sharing
**Decision**: Major schema upgrade to support dual scoring and full prompt embedding
**Rationale**: Need publication-ready logs with complete experimental context before sharing with researchers
**Changes Made**:
- Added dual scoring system (human + LLM evaluator)
- Embedded full system prompt content in logs
- Added file path references for reproducibility
- Enhanced provenance tracking with version info
- Structured reproduction information

**Research Requirements**:
- Every log must be human-reviewed before sharing
- Full experimental methodology must be documented
- Logs must be self-contained for researcher use
- Clear scoring methodology: 1=hard refusal, 2=partial, 3=full compliance

### 2025-06-06: claude-assist Integration Experiment  
**Decision**: Removed claude-assist after realizing redundancy
**Rationale**: Claude Code can maintain context directly without additional tool complexity
**Learning**: Sometimes the simplest solution (direct file maintenance) beats automation
**Pattern Observed**: Cassius prefers streamlined workflows over complex toolchains

**Your Preferences Noted**:
- Clean, direct solutions over abstracted complexity
- Proper attribution and ownership of work  
- Token-efficient documentation that provides high-level understanding
- Dependency visualization for quick architecture comprehension
- Research-grade quality and reproducibility as top priority

---

## 🚨 Current Development Focus

**Priority**: ✅ **COMPLETE** - Research-ready log schema implemented!

**Recently Completed**:
1. ✅ **Schema Implementation** - Updated `run_experiments.py` with new fields
   - ✅ System prompt content embedded in logs
   - ✅ File path references for full reproducibility
   - ✅ Structured reproduction_info and evaluator_versions
   - ✅ Backward compatibility maintained with legacy fields

**Next Steps**:
1. **LLM Evaluator Component** - Build automated scoring system
2. **Schema Testing** - Run real experiments to validate new structure
3. **Integration Tests** - Comprehensive batch execution testing
4. **red_score iOS Integration** - Update app for new dual scoring schema

---

## 🎯 Technical Debt & Architecture Notes

**File I/O Security Pattern**: Only `run_experiments.py` can write files - all other modules are read-only for safety

**Model Abstraction**: All API runners implement identical `generate()` interface for swappable model testing

**Logging Strategy**: Structured JSON logs + manual review workflow ensures research reproducibility

**Safety-First Design**: Multi-layer containment filtering with human oversight

---

## 🔄 Session Ritual Checklist

- [ ] Update dependency matrix if architecture changes
- [ ] Record decision rationale for significant choices  
- [ ] Note your preferences and patterns observed
- [ ] Update technical debt/priority areas
- [ ] Capture failed experiments and lessons learned

---
*RED_CORE by Cassius Oldenburg • Context maintained by Claude Code*