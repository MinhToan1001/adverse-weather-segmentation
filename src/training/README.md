# Training dùng chung cho B2 và DeepLab

| Module | Vai trò |
| --- | --- |
| `train.py` | Vòng train, validation, best/last, log và resume |
| `dg.py` | Baseline, hai view CE và consistency KL |
| `losses.py` | CE dùng trong DG; các loss khác giữ cho thử nghiệm/tương thích |
| `optim.py` | AdamW, nhóm LR encoder/head |
| `strategies.py` | Chế độ BatchNorm và tiện ích data-policy |
| `checkpoints.py` | Ghi checkpoint theo giao dịch, xác minh hash, khôi phục generation trước |

Recipe hiện hành: 20 epoch, không early stopping, batch thật2, accumulation4, encoder LR6e-5/head×10, weight decay0,01, warmup20updates, poly power1, FP16 AMP khi có GPU. BN cập nhật từ ảnh nguồn; không nhận ACDC. SegFormer encoder dùng LayerNorm, DeepLab có BatchNorm nên cần giữ batch thật ít nhất2.

[run_dg_suite.py](../../scripts/run_dg_suite.py) tự chạy ba chiến thuật độc lập từ ImageNet; xuất ZIP sau mỗi epoch hoàn thành. Checkpoint giữ model/optimizer/scheduler/scaler/RNG; ngắt giữa epoch thì lặp lại epoch đó. [Phương pháp](../../docs/DG_METHOD.md) · [Chạy/resume](../../docs/TRAINING_GUIDE.md).

Trainer vẫn đọc config supervised lịch sử qua opt-in riêng; notebook DG không gọi đường này và protocol guard chặn nó mặc định.
