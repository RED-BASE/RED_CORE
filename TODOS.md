#to do list

##“Drift/scoring is not implemented. When ready, will be based on literature, not rule-based code.”

5. tools/ folder with hook scripts
Presumably Git hooks or pre-commit enforcement scripts.

Are they active and enforced?

Are their rules up to date?

Risk: Outdated or unused hooks are dead weight.

Action:

Validate active use in your repo setup and CI.

6. Multiple README.md scattered everywhere
Good for local context.

But inconsistent or outdated READMEs breed confusion.

Action:

Audit for conflicts or outdated info.

Standardize critical README info (e.g., experiment lifecycle, folder usage).

7. Potential for dead code in app/analysis/
Several analysis scripts exist.

Do you have test coverage or usage for all?

Unused analysis code is latent rot.

Action:

Identify coverage and utility.

Remove or refactor unused scri