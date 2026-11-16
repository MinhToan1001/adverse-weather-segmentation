# Current evidence and pending results

[Software release verification](repository-release/README.md) · [Run guide](../docs/TRAINING_GUIDE.md)

| Fresh model | Source20 | Adapt20 | Official Cityval500 | Official ACDCval406 |
| --- | --- | --- | --- | --- |
|[B0](segformer_b0_seed42/README.md)|Complete: best epoch 19, monitor 65.13%|Complete: best epoch 20, dev 51.65%|Pending|Pending|
|B2|Pending|Pending|Pending|Pending|
|DeepLabV3+R50|Pending|Pending|Pending|Pending|

The B0 scores are internal selection metrics, not official validation results. Source monitoring uses 188 Cityscapes images; adaptation selection uses 402 ACDC dev images. B0 training is complete, with official validation still pending. B2 and DeepLab have not yet been received under this fresh protocol.

No old project score fills these cells. Each notebook trains one model from an ImageNet encoder/new head with transactional checkpointing. Software tests and tiny real-image CPU probes are not accuracy benchmarks. Large checkpoints/ZIPs/raw data and local legacy archives are ignored by Git; only compact verification evidence belongs here.
