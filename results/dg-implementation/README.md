# Shared DG implementation verification · 2026-11-19

[Method](../../docs/DG_METHOD.md) · [Training guide](../../docs/TRAINING_GUIDE.md) · [Verification record](verification.json)

Implemented baseline / source-style two-view CE / two-view CE+detached-teacher KL for B2 and DeepLabV3+ ResNet50. RGB style preserves one shared geometry and mask; local RNG isolates appearance and pair scheduling from geometric/dropout RNG. Sequential clean/styled backward uses the original minibatch accumulation/update count. C1/C2 have the same forward/BatchNorm order.

**173 Python tests passed**, including cached pretrained checks, teacher detachment, void exclusion, joint-vs-sequential gradient agreement, BN buffers, source-only guards, cohort tamper detection and actual notebook resume after Kaggle nested ZIP expansion for both models/all three arms. GPU/AMP timing and convergence remain unverified until hardware is available.

Real CPU QA used4Cityscapes train/2val,64×96,random initialization,2batches/epoch for each model. Both completed2epochs with interruption/resume and matched continuous-run weights bitwise. All three arm initial safetensors hashes matched within each architecture; optimizer updates remained1per epoch and paired KL was finite. Tiny random-model metrics are not reported as quality evidence.

All six model/arm configs passed DG guards against real Cityscapes train2975/val500 manifests and file checks. No ACDC images/masks were used in these tests or QA. Two notebooks default to PROFILE_ONLY=True; the GPU pilot discards its weights and produces timing/VRAM estimates that exclude checkpoint/ZIP/setup costs.

The cohort evaluator requires5completed20epoch candidates, equal source split/code/seed/processing, equal initialization/recipe within each model, aligned DG settings and immutable artifact hashes. Existing generic historical evaluation tools are not the new DG reporting entrypoint.

No GPU run, DG mIoU improvement, paper reproduction or statistical superiority is established. Candidate defaults p_pair0,5 and λ0,1 require full GPU/source-validation review. Source-image preview, full logs and large probe checkpoints stay local/Git-ignored; only this compact evidence is committed.
