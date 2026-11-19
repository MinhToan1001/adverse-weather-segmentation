[Kế hoạch hoàn thành B2 + DeepLab với 30 giờ GPU](PROJECT_COMPLETION_PLAN.md)

# Tài liệu DG và lịch sử

**Hiện hành:** [DG recovery plan](DG_RECOVERY_PLAN.md). Cityscapes train/selection, ACDC chỉ evaluation sau freeze.

[VLTSeg](paper_reviews/VLTSeg_2024.md) là ứng viên paper DG; [SegFormer](paper_reviews/SegFormer_2021.md) là nguồn kiến trúc. ISSA/ALDM và hướng consistency được phân tích trong recovery plan. HRDA/Refign nguyên thiết lập là UDA, không đối chứng DG trực tiếp.

[BASELINE_PROTOCOL.md](BASELINE_PROTOCOL.md) từng chọn supervised ACDC FT20 và nay đã bị thay thế. [Kế hoạch cũ](EXPERIMENT_PLAN.md) và [archive](archive/README.md) chỉ dùng truy vết. Không đổi các kết quả supervised thành DG.
