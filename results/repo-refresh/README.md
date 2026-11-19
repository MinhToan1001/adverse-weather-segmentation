# Repository refresh · 2026-11-19

[Verification record](verification.json) · [Project overview](../../README.md) · [Reading guide](../../docs/READING_GUIDE.md)

The repository now presents an application capstone: B2 is the main model, with shared data/training/evaluation and a Node.js UI. DeepLab remains optional. B0 is excluded from new CLI/campaign/notebook/UI choices; its implementation moved to `src/models/legacy/` for historical checkpoint compatibility. Existing weights and result reports are preserved. The historical manuscript and bibliography moved from `docs/paper/` to `research/archive/`.

Validation: **141 Python tests passed**, including cached pretrained checks with network downloads disabled; **1 Node HTTP test passed**; **1 real Chrome end-to-end test passed** using B2 source/adapt checkpoints. Browser checks cover before/after inference, per-stage cache, day/night samples, uploads, invalid images, downloads and mobile layout. Downloaded trainId/color PNGs are native 2048×1024 with exact palette correspondence. Two active notebooks regenerate and their code parses. Local Markdown destinations were checked; no broken links remained.

Full logs, screenshots and dataset-derived images are local under `outputs/repo-refresh/` and `outputs/web-studio/`, outside Git. Loopback tests run outside the filesystem/network sandbox because sandboxed HTTP requests cannot reach the local server. No new GPU training or quality benchmark is claimed. These changes do not implement the proposed published-Cityscapes-checkpoint baseline or alter the existing B2 training recipe.
