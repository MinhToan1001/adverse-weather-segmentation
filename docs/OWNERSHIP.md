# Owners & repository working agreement

**A (Lead / final decision, @khangcules10):** GitHub repo and protection settings, experiment specification, baseline training pipeline, compute resource management, proposed model, model artifacts, review all research claims. PR labels: `lead`, `baseline`, `owner/a`.

**B (Dataset & evaluation, @MinhToan1001):** data license and download guides, prepare canonical 19 trainId masks, split manifests, audit 20 samples per domain visually, metrics/tests, paper matrix, per-weather tables, plots and demo support. PR labels: `data`, `evaluation`, `owner/b`.

**Approval gates (A–E):** A scope and DG protocol frozen; B official datasets, labels and manifests verified; C baseline trained and reproduced; D one-shot ACDC evaluation and controlled ablation; E deliverables, release and handoff. Gate criteria live in `docs/experiment_protocol.md` and map 1:1 to GitHub milestones. @khangcules10 is the Leader and approves merges. @MinhToan1001 reviews data/evaluation work; A requests B review for A's own changes.

**Rules:** 1 issue = 1 task = 1 branch = 1 PR. Branches `feat/a-*`, `feat/b-*`, `fix/*`, `exp/*`. Do not commit personal access tokens, datasets, checkpoints or external licenses. Log seeds, configuration, hashes and outputs. Never use validation as test repeatedly; keep the final test set untouched.
