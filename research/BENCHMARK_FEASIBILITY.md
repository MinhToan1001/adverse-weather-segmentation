# DG benchmark feasibility decision

Reviewed: 2026-11-01. Decision: retain SegFormer-B0 as the executable baseline;
VLTSeg is a conditional DG comparator, not yet evaluated or reproduced.

VLTSeg's author project links an unofficial implementation. Its README requires
a separate Python 3.8/CUDA 12.1 stack and provides Zenodo checkpoint links.
Code is MIT; checkpoints are CC-BY-NC-4.0. This local environment is CPU-only,
Python 3.11 with torch 2.10; installing into it would not reproduce that stack.

Select only Cityscapes-source weights for our benchmark, excluding ACDC-trained
or additional-source weights. Reconcile inference/TTA and label mapping first.
Do not substitute published scores for our single-scale ACDC-val evaluation.

Blockers: GPU environment, checkpoint/config audit and real data. The initial
matrix records the screening; no benchmark score is claimed. Keep a separate
future environment and record exact dependencies/checkpoint hash before use.

Sources checked:
- https://vltseg.github.io/
- https://github.com/VLTSeg/VLTSeg
- https://github.com/lhoyer/HRDA
- https://github.com/brdav/refign

A historical arXiv v2 has a different title/setup from the ACCV project page.
Full paper reviews must identify the exact version; they remain pending.
