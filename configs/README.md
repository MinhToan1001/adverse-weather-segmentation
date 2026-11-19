[DG method đã triển khai và kiểm thử CPU](../docs/DG_METHOD.md) · baseline / aug / consistency dùng chung B2 và DeepLab. Full GPU results pending.

# Shared experiment configuration

| Component | Location |
| --- | --- |
| Processing contract |preprocessing/shared_512.yaml|
| Source /adaptation policies |recipes/source_20epoch.yaml,adaptation_20epoch.yaml|
| Objectives |losses/ce.yaml,weighted_ce.yaml,focal.yaml,ce_dice.yaml|
| Model architecture /encoder pins |src/models/ registry and separate model files|
| Run-bound config |Generated train.yaml,hashed and frozen inside outputs|

src/experiments/config.py composes policies once. Do not copy data/training recipes into each model file. Both stages20epochs;CE controls first. Altering frozen config/source/data during resume is rejected.
