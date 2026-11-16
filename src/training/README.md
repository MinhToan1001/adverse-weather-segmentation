# Shared training

| Module | Responsibility |
| --- | --- |
|losses.py|CE,fit-weighted CE,Focal,CE+Dice;void-safe,architecture-independent|
|optim.py|AdamW,encoder/head LR groups|
|strategies.py|BN mode and optional data-policy composition|
|checkpoints.py|Staging/SHA/previous generations,recovery and log rollback|
|train.py|Data loaders,AMP,accumulation,scheduler,validation,best/last,RNG state|

Current source/adaptation recipes each20epochs,no early stopping. Source uses batch2/accum4,encoderLR6e-5/head×10,BN stats update. Adapt uses batch1/accum8,LR1e-5 uniform,BN stats frozen and affine learnable. Both use weight decay0.01,warm-up20updates,poly power1,FP16AMP onGPU. SegFormer encoder LayerNorm differs from BN.

Model notebooks call scripts.run_research_campaign with one model;it reuses scripts.run_research for source/adapt and exports one atomic workspace ZIP per completed epoch. Interrupted epochs repeat. A model run may span several sessions. [Guide](../../docs/TRAINING_GUIDE.md).
