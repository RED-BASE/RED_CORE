# 🔴 LLM-REDCORE

**Adversarial Red Teaming Framework for Language Models**  
Designed for modular experimentation, structured evaluation, and expressive documentation.

---

See `utils/log_utils.py` for schema structure.

---

## 🧩 Extending the Framework

Add new:

- **Attacks** in `attacks/` as `.yaml`
- **Evaluators** in `evaluators/`
- **Models** in `models/` by subclassing `BaseModelRunner`
- **Defenses** in `utils/` or `defenses/` (optional)

Everything plugs into the central runner.

---

## 📝 Journaling & Reflections

Developer reflections and meta insights go in `logs/reflections/`.

Use:

```bash
make log
```

This invokes `create_log_entry.sh`, appending a timestamped, dev log.  
Great for thinking out loud, tracking insights, or proving authorship.

---

## ⚙️ Philosophy

- Modular over monolithic
- Human-readable logs over black-box metrics
- Attack chains as expressive, composable artifacts
- Built for research, red teaming, and weird creative AI experiments

---

## ✅ Status

🧱 Core pipeline ✅   
📜 Logging & trace format ✅  

---

> Built to test the limits of model behavior and the shape of alignment itself.

# docs/

Project documentation and notes.

- **README.md**: Main documentation for the project.
- **drift_notes.md**: Placeholder for drift-related notes.
- **scoring rules.md**: Placeholder for scoring documentation.

## Log Workflow

- Raw logs are saved to `logs/` inside each experiment folder.
- Scored logs are uploaded to `scored_logs/` via the iOS scoring app and merged into the repo via pull request.
- Each log includes robust error handling, per-turn token counts, latency, and a unified summary output at the end of each run.
- The review process is manual (iOS or desktop), and all scored logs are merged via PR for full auditability.
