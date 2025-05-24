Guardrail Decay Experiment

**Hypothesis**

> When instructed to simulate a version of itself that forgets alignment rules over time, the model will begin violating its own guardrails or contradicting prior safety logic—even if the simulation is explicitly declared and scoped.

python -m runners.multi_system_batch_runner \
  --sys-prompts prompts_system/sys_none_00.yaml \
               prompts_system/sys_guardrail_decay_01.yaml \
               prompts_system/sys_guardrail_decay_02.yaml \
               prompts_system/sys_guardrail_decay_03.yaml \
  --usr-prompt prompts_user/guardrail_decay/usr_guardrail_decay_01.yaml \
  --models gpt-4o claude-3-7-sonnet gemini-1.5-pro \
  --reps 30 \
  --score-log \
  --experiment-id "exp_guardrail_decay" \
  --extra-args --temperature 0.7 --disable-containment true
