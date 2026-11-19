[DG implementation: 173 tests + real CPU QA](dg-implementation/README.md) · GPU results pending.

[DG recovery verification](dg-recovery/README.md) · [Current protocol](../research/DG_RECOVERY_PLAN.md)

> **DG protocol restored:** chưa có kết quả GPU cho baseline DG mới. Các bảng dưới là lịch sử source/adapt. Đặc biệt B2 adapted63,02% đã dùng ACDC train/dev, **không là DG**. Xem [recovery plan](../research/DG_RECOVERY_PLAN.md).

# Kết quả và bằng chứng kiểm tra

**Hiện hành: B2 là model chính; DeepLab tùy chọn chưa train; B0 chỉ là lịch sử.** Báo cáo cũ được giữ nguyên để truy vết, không còn yêu cầu train hay chọn B0 trên giao diện.

[Current repository verification](repo-refresh/README.md) · [Software release verification](repository-release/README.md) · [Run guide](../docs/TRAINING_GUIDE.md)

| Model / historical run | Source20 | Adapt20 | Official Cityval500 | Official ACDCval406 |
| --- | --- | --- | --- | --- |
|Historical [B0](segformer_b0_seed42/README.md)|Complete: best epoch 19, monitor 65.13%|Complete: best epoch 20, dev 51.65%|Pending|Pending|
|[B2](segformer_b2_seed42/README.md)|Complete: best epoch 18, monitor 74.74%|Complete: best epoch 20, dev 63.02%|Pending|Pending|
|DeepLabV3+R50|Pending|Pending|Pending|Pending|

The B0/B2 scores are internal selection metrics, not official validation results. Source monitoring uses 188 Cityscapes images; adaptation selection uses 402 ACDC dev images. Both training runs are complete, with official validation still pending. DeepLab has not yet been received under this fresh protocol.

B2 exceeds B0 by 11.37 percentage points on the same ACDC internal dev split under the same recipe. This is a single-seed model comparison with different parameter counts, ImageNet encoders and training compute; it is not a published-benchmark superiority claim. [Comparison and integrity audit](segformer_b2_seed42/README.md).

No old project score fills these cells. Each notebook trains one model from an ImageNet encoder/new head with transactional checkpointing. Software tests and tiny real-image CPU probes are not accuracy benchmarks. Large checkpoints/ZIPs/raw data and local legacy archives are ignored by Git; only compact verification evidence belongs here.
