# RED_CORE TODO

## Async Transition (In Progress)
- [x] `experiment_runner_async.py` - refactored with clean architecture
- [x] `run_experiments.py` - updated to call async runner
- [x] Callback handling fixed (sync/async detection)
- [ ] End-to-end testing needed - was stable before async transition
- [ ] Clean up dead imports in `run_experiments.py` (threading.Lock, ThreadPoolExecutor)
- [ ] Test with actual API calls

## Model Updates Needed

### API Runners
- [ ] Update model names to latest versions (Claude, GPT, Gemini, etc.)
- [ ] Verify API compatibility with latest SDKs
- [ ] Check rate limits against current provider docs

### Model Registry
- [ ] Update `data/model_registry.md` with current model IDs
- [ ] Remove deprecated models
- [ ] Add any new models from providers

## Low Priority
- [ ] Clean up experiment log files (decide what to keep)
- [ ] Remove remaining CLAUDE.md files if not needed
