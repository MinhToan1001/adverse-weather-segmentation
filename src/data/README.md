# Shared data pipeline

partitions.py creates grouped Cityscapes fit/monitor and ACDC fit/dev without using model scores. preparation.py handles explicit labelIds→trainIds conversion and image-mask manifests. validation.py checks files,dimensions,labels and disjoint paths. dataset.py applies paired geometry and deterministic evaluation. preprocessing.py defines the common canvas/normalization contract.

class_weights.py computes bounded fit-only pixel priors. replay.py balances source/target draws. augmentations/ contains RGB-only gamma/photometric transforms and label-guided class crop. These components do not depend on B0/B2/DeepLab classes.

Use scripts.prepare_research_data once per model workspace. Generated manifests/derived masks stay outside Git. [Contract](../../docs/DATA_CONTRACT.md).
