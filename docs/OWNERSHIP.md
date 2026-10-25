# Owners & repository working agreement

**A (Lead / final decision):** GitHub repo, protection settings, experiment specification, baseline training pipeline, compute resource management, model artifacts, review all research claims. PR labels: `lead`, `baseline`.

**B (Dataset & evaluation):** data license and download guides, prepare canonical 19 trainId masks, split manifests, audit 20 samples per domain visually, metrics/tests, paper matrix, experimental tables. PR labels: `data`, `evaluation`.

**Approval gates:** G0 repo created and CI green; G1 official datasets and labels verified; G2 baseline train/eval recorded; G3 proposal reviewed before any changes. Nothing merged without A review. A also opens PRs and requests B review for own changes.

**Rules:** 1 issue = 1 task = 1 branch = 1 PR. Branches `feat/a-*`, `feat/b-*`, `fix/*`, `exp/*`. Do not commit personal access tokens, datasets, checkpoints or external licenses. Log seeds, configuration, hashes and outputs. Never use validation as test repeatedly; keep the final test set untouched.
