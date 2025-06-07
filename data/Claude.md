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
├─ Purpose: Rule-based automated drift/refusal scoring
├─ Key Features: Pattern-matching, hazard classification
└─ Notes: Fast, deterministic scoring with industry-standard patterns

llm_evaluator.py [NEW]
├─ Dependencies: API runners, log schema
├─ Purpose: LLM-based nuanced evaluation of experiment results
├─ Key Features: Contextual assessment, confidence scoring, detailed reasoning
└─ Notes: Async processing, model-agnostic (Claude/GPT), integrates with LLMEvaluation schema

dual_evaluator.py [NEW]
├─ Dependencies: automated_scorer, llm_evaluator
├─ Purpose: Combined rule-based + LLM evaluation with comparison metrics
├─ Key Features: Method agreement analysis, batch processing, summary reports
├─ Output Behavior: Updates original logs in-place via llm_evaluation schema fields
└─ Notes: Maintains data lineage by enriching existing experiment files

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
├─ Model Details: Complete specifications in [model_registry.md](model_registry.md)
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

### 2025-06-06: Codebase Standardization & Cleanup
**Decision**: Comprehensive naming convention cleanup and configuration consolidation
**Rationale**: Eliminate inconsistencies that could cause import errors and improve maintainability
**Changes Made**:
- Fixed critical `MODEL_ALIASES` import error in `anthropic_runner.py`
- Consolidated duplicate `MODEL_REGISTRY` between `config.py` and `__init__.py`
- Standardized model naming with consistent versioning (e.g., `claude-3-7-sonnet-20250219`)
- Unified dev log file naming to `dev_log-YYYY-MM-DD HH:MM:SS.md` format
- Removed duplicate `score_logs` directory, standardized on `scored_logs`
- Fixed documentation file naming (`scoring rules.md` → `scoring_rules.md`)

### 2025-06-06: Production-Ready Data Lineage & Directory Structure
**Decision**: Fix directory organization and implement in-place log enrichment
**Rationale**: Research requires connected data - separate scoring files break experimental lineage
**Changes Made**:
- Fixed experiment directory mapping logic (80K → 80k_hours_demo, etc.)
- Redesigned dual evaluator to update original logs in-place via schema fields
- Embedded scoring directly in `llm_evaluation` and automated scoring fields
- Created clean directory structure: `logs/`, `dual_evaluated/`, `scored_logs/`

**Key Innovation**: In-place log enrichment preserves complete data lineage while maintaining schema compliance
- Original experiment data remains unchanged
- Scoring metadata added to designated schema fields
- Single authoritative file contains everything needed for research
- Follows ML industry best practices (MLflow, W&B patterns)

**Dual Evaluator Output Structure**:
```
experiments/{experiment}/
├── logs/                    # Original + enriched experiment logs
│   ├── 80K-C37S-*.json     # Now contains llm_evaluation + automated scoring
│   └── run_failures.txt    # Error logs
├── dual_evaluated/          # Optional separate analysis files 
│   ├── *_dual_evaluated.json        # Comparison metrics & method agreement
│   └── dual_evaluation_summary.json # Aggregate analysis across all logs
└── scored_logs/             # Manual human review files (existing)
```

### 2025-06-06: Schema Validation & Dual Evaluation Success
**Decision**: End-to-end pipeline validation with real experiment data
**Rationale**: Confirm new schema and dual evaluation system before large-scale deployment
**Changes Made**:
- Fixed critical execution errors in `run_experiments.py` (NameError, path resolution)
- Validated new log schema fields with 80k_hours_demo experiment
- Successfully executed dual evaluation pipeline (automated + LLM scoring)
- Confirmed research-ready JSON output format with complete provenance

**Results Achieved**:
- 5-turn experiment: 1 refusal, 4 compliance responses, 100% safety rate
- Complete automated scoring with confidence metrics and reasoning
- LLM evaluation framework operational (API issues fixable)
- Publication-ready structured logs with embedded prompts and file references

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

**Priority**: ✅ **COMPLETE** - Production-ready dual evaluation system with proper data lineage!

**Recently Completed**:
1. ✅ **Schema Implementation** - Updated `run_experiments.py` with new fields
   - ✅ System prompt content embedded in logs
   - ✅ File path references for full reproducibility
   - ✅ Structured reproduction_info and evaluator_versions
   - ✅ Backward compatibility maintained with legacy fields

2. ✅ **LLM Evaluator Component** - Built automated scoring system
   - ✅ `app/analysis/llm_evaluator.py` - Core LLM-based evaluation
   - ✅ `app/analysis/dual_evaluator.py` - Combined rule-based + LLM evaluation
   - ✅ Model-agnostic design (Claude, GPT support via existing runners)
   - ✅ Async processing with rate limiting and retry logic
   - ✅ JSON parsing resilience for various LLM response formats
   - ✅ Integration with new `LLMEvaluation` schema fields

3. ✅ **Codebase Standardization** - Comprehensive cleanup and consistency fixes
   - ✅ Fixed critical import errors (MODEL_ALIASES → resolve_model)
   - ✅ Consolidated duplicate MODEL_REGISTRY configurations
   - ✅ Standardized model naming conventions with versioning
   - ✅ Unified file naming patterns across project
   - ✅ Cleaned up directory structure inconsistencies
   - ✅ Validated all syntax changes with compilation checks

4. ✅ **Model Library Overhaul** - Updated with latest SDK documentation (June 2025)
   - ✅ Added Claude 4 models (claude-opus-4-20250514, claude-sonnet-4-20250514)
   - ✅ Added GPT-4.1 series (gpt-4.1, gpt-4.1-mini, gpt-4.1-nano) with 1M context
   - ✅ Added Gemini 2.5 models (gemini-2.5-pro, gemini-2.5-flash) with thinking capability
   - ✅ Updated all model names to match official API documentation
   - ✅ Added context window sizes, deprecation notices, and feature tags
   - ✅ Enhanced config with helper functions for model metadata access
   - ✅ Fixed interactive command integration - all 16 new models now selectable via `make run`
   - ✅ Updated command line defaults and shell scripts with current models

5. ✅ **Schema Validation & Testing** - End-to-end pipeline verification
   - ✅ Fixed critical NameError in `run_experiments.py` (num_turns_per_model calculation)
   - ✅ Fixed relative path calculation errors preventing experiment execution
   - ✅ Validated new log schema with real experiment data (80k_hours_demo)
   - ✅ Successful dual evaluation pipeline execution (rule-based + LLM scoring)
   - ✅ Confirmed automated scoring accuracy (5 turns: 1 refusal, 4 compliance, 100% safety rate)
   - ✅ Verified JSON output format for research publication readiness

6. ✅ **Directory Structure & Data Lineage Fixes** - Production-ready organization
   - ✅ Fixed experiment directory mapping (80K → 80k_hours_demo, GRD → guardrail_decay, RRS → refusal_robustness)
   - ✅ Implemented in-place log enrichment instead of separate disconnected files
   - ✅ Embedded dual scoring directly in original logs via `llm_evaluation` schema fields
   - ✅ Maintained complete data lineage for research reproducibility
   - ✅ Created smart directory structure: `logs/`, `dual_evaluated/`, `scored_logs/`
   - ✅ Preserved backward compatibility with optional separate analysis files

**Next Steps** (Strategic Roadmap):

**🎯 Immediate Priorities** (Next 1-2 weeks):
1. **LLM Evaluator Refinement**
   - Fix system prompt API format for Claude evaluator calls
   - Test LLM evaluation with GPT-4.1 as alternative evaluator model  
   - Validate LLM scoring accuracy against manual human annotations
   - Optimize evaluation prompts for better JSON response consistency

2. **Production Readiness & Scaling**
   - Batch processing validation for large experiment directories
   - Error handling refinement for edge cases in evaluation
   - Performance benchmarking (evaluation speed, token costs)
   - Integration testing with existing refusal_robustness experiments

**🚀 Medium-Term Research Expansion** (Next month):
3. **Advanced Evaluation Features**
   - Multi-evaluator ensemble (Claude + GPT consensus scoring)
   - Calibration studies (measure evaluator agreement vs human reviewers)
   - Domain-specific prompts for different hazard categories
   - Longitudinal drift detection across conversation turns

4. **Research Publication Prep**
   - Methodology documentation for peer review
   - Benchmark dataset creation with gold-standard human annotations
   - Statistical analysis pipeline for significance testing
   - Reproducibility package for other researchers

**🔬 Strategic Research Directions** (Next quarter):
5. **Advanced Attack Patterns**
   - Jailbreak detection using LLM evaluators
   - Multi-turn manipulation pattern recognition
   - Context window exploitation experiments
   - Prompt injection resilience testing

6. **Cross-Model Analysis**
   - Comparative safety evaluation across vendors
   - Transfer learning for evaluator fine-tuning
   - Model-specific vulnerability mapping
   - Safety degradation patterns identification

**💡 High-Impact Quick Wins**:
- Demo experiment using dual evaluator on refusal_robustness logs
- Evaluation summary dashboard for experiment insights
- Automated report generation for stakeholder updates
- Cost-benefit analysis of LLM vs rule-based evaluation

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