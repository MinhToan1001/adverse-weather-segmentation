# Phương pháp hiện hành của đồ án

[Tổng quan](../README.md) · [Lộ trình đọc](READING_GUIDE.md) · [Training / resume](TRAINING_GUIDE.md)

Tên file được giữ để các liên kết cũ vẫn hoạt động. Nội dung này mô tả **code đang chạy**, tách khỏi [protocol public-checkpoint dự kiến](../research/BASELINE_PROTOCOL.md). Đồ án hiện tập trung ứng dụng B2 cho ảnh đường phố trong thời tiết xấu; không bắt buộc đề xuất kiến trúc mới.

## 1. Dữ liệu và cách chia

| Nguồn | Phần dùng để học | Phần dùng chọn checkpoint | Kiểm tra sau khi chốt |
| --- | --- | --- | --- |
| Cityscapes official train 2.975 ảnh | 2.787 fit | 188 monitor, giữ lại theo nhóm city/sequence | Official val 500, chưa nghiệm thu cho lượt B2 hiện tại |
| ACDC official train 1.600 ảnh | 1.198 fit | 402 dev, chia nhóm recording, có đủ bốn thời tiết | Official val 406, chưa nghiệm thu cho lượt B2 hiện tại |

ACDC được dùng **có nhãn** khi học tiếp. Ảnh reference trời quang được kiểm kê nhưng chưa tham gia objective. Chia nhóm token giúp giảm trùng cảnh gần nhau, chưa chứng minh độc lập địa lý. Các official-val từng được xem trong lịch sử phát triển nên chưa được gọi là test độc lập.

Nhãn dùng 19 trainIds 0–18; ignore 255 bỏ khỏi loss/metrics. Cityscapes labelIds được chuyển rõ ràng, không đoán kiểu mã hóa từ giá trị pixel. [Data contract](DATA_CONTRACT.md), [partition code](../src/data/partitions.py).

## 2. Xử lý và tăng cường dữ liệu

- Khung ảnh **512 × 1024**, giữ aspect ratio, padding thay vì kéo méo hình. ImageNet mean/std dùng chuẩn hóa RGB.
- Train: random scale 0,5–2, crop/pad, lật ngang xác suất 0,5. Ảnh và mask dùng cùng biến đổi hình học; RGB bilinear, mask nearest, vùng mask padding là 255.
- Dev/monitor/evaluation: fit ảnh vào khung theo cách xác định, không random augmentation. Logits được bỏ padding rồi nội suy về kích thước nhãn gốc.
- Gamma/photometric, class-aware crop và sampling theo lớp có code, **không bật mặc định** trong kết quả CE hiện tại.

## 3. Model và training

B2 chính có khoảng 27,36M tham số; DeepLabV3+ ResNet50 tùy chọn khoảng 26,70M. Mỗi kiến trúc có file riêng, đầu ra cùng dạng logits B×19×H×W. Registry chọn model; trainer/data/evaluator dùng lại. B0 đã chuyển sang [legacy](../src/models/legacy/README.md).

| Thiết lập | Source: Cityscapes | Adapt: ACDC có nhãn + replay |
| --- | --- | --- |
| Khởi tạo | Encoder ImageNet, head 19 lớp mới | Source CE `best/` của cùng model sau đủ 20 epoch |
| Epoch / early stopping | 20 / tắt | 20 / tắt |
| Bộ mẫu một epoch | Cityscapes fit | 1.198 ACDC + 1.198 lượt lấy mẫu Cityscapes |
| Batch / gradient accumulation | 2 / 4, drop_last | 1 / 8 |
| AdamW LR encoder / head | 6e-5 / 6e-4 | 1e-5 / 1e-5 |
| Weight decay | 0,01 | 0,01 |
| Scheduler | Warm-up 20 bước cập nhật, polynomial power 1 | Tương tự |
| BatchNorm buffers / affine | Cập nhật / học | Đóng băng thống kê / vẫn học affine |
| Precision / GPU | FP16 AMP, một GPU | Tương tự |
| Chọn best | Cityscapes monitor mIoU | ACDC dev mIoU với source drop ≤ 1 điểm phần trăm |

Nếu không candidate nào qua source gate, code xếp hạng fallback theo mức giữ nguồn; phải đọc cờ gate thay vì mặc định best luôn đạt gate. Một epoch replay không có nghĩa đi qua toàn bộ ảnh Cityscapes; target đi đủ một vòng, source lấy mẫu xen kẽ. Cùng số epoch giữa model không đồng nghĩa cùng compute.

SegFormer encoder dùng LayerNorm; head có BatchNorm. Tích lũy gradient tạo update từ nhiều microbatch, **không gộp thống kê BatchNorm**. Nhánh pooled BatchNorm của DeepLab cần batch source thực tế ≥ 2. Recipe dùng LR head ×10 nhưng khác warm-up/weight decay/resolution của bài SegFormer, vì vậy không gọi là tái lập nguyên paper.

Recipe thực tế: [source_20epoch.yaml](../configs/recipes/source_20epoch.yaml), [adaptation_20epoch.yaml](../configs/recipes/adaptation_20epoch.yaml). [config.py](../src/experiments/config.py) ghép chúng với model/data/loss.

## 4. Loss và các tùy chọn chưa chứng minh hiệu quả

[losses.py](../src/training/losses.py) nội suy logits đến nhãn, tính FP32, bỏ vùng 255. CE là control đang dùng. Low IoU không tự chứng minh lớp thiếu dữ liệu hoặc CE không phù hợp.

| Loss | Cách hiểu | Thiết lập và giới hạn |
| --- | --- | --- |
| CE | Phạt dự đoán sai ở từng pixel | Trung bình cross-entropy pixel hợp lệ; vùng lớn có thể chi phối |
| Weighted CE | Tăng chú ý lớp ít pixel | Inverse-square-root priors từ fit masks, chặn tỷ lệ max/min ≤ 5; không lấy trọng số từ điểm dev |
| Focal | Giảm trọng số pixel dễ | Gamma 2; pixel khó có thể là nhãn nhiễu |
| CE + Dice | Vừa đúng pixel vừa khớp vùng | CE + 0,5 × macro Dice loss trên lớp có GT trong batch; smoothing 1e-6 |

All-void batch trả loss/gradient 0 hữu hạn. Không dùng train Focal trừ dev CE để đo overfit vì khác objective. Weighted CE chưa tính được mọi thay đổi phân bố sau crop.

`--class-crop` là tùy chọn: xác suất 0,5 chọn lớp đang có trong ảnh theo inverse-square-root pixel count, thử tối đa 10 vị trí để giữ ít nhất 64 pixel lớp đó. Không đạt thì giữ crop tốt nhất và ghi log. Crop không khôi phục chi tiết đã mất khi giảm độ phân giải. Thử từng yếu tố riêng, cùng source checkpoint/split/seed trước khi kết hợp.

## 5. Khôi phục và truy vết

[checkpoints.py](../src/training/checkpoints.py) ghi thư mục tạm, kiểm hash rồi thay best/last; giữ thế hệ previous. Bản lưu chứa model, optimizer, scheduler, AMP scaler, RNG và epoch. Bộ điều phối xác minh config/source/manifest để không trộn thí nghiệm khi resume. ZIP kết quả thay atomically sau mỗi epoch hoàn thành; mất giữa epoch thì chạy lại epoch đó. Notebook có cửa sổ phiên 10 giờ, không giới hạn tổng giờ của cả thí nghiệm. [Training guide](TRAINING_GUIDE.md).

## 6. Đánh giá và giới hạn

Evaluator cộng ma trận nhầm lẫn trên toàn split trước khi tính IoU/mIoU, trả thêm per-class, per-weather, calibration và panel ảnh/GT/prediction/lỗi. Inference trả về kích thước gốc, FP32, không test-time augmentation. Xem [runner.py](../src/evaluation/runner.py).

Kết quả B2 đã nhận: City monitor 74,74%; ACDC dev 48,44% → 63,02%; City monitor sau thích nghi 74,53%. Đây là **một seed, tập chọn checkpoint**, chưa phải bằng chứng vượt paper hay test độc lập. Night vẫn là nhóm khó. [Báo cáo đã kiểm tra](../results/segformer_b2_seed42/README.md).

## 7. Tham khảo dùng thế nào?

[SegFormer 2021](../research/paper_reviews/SegFormer_2021.md) cung cấp kiến trúc và mốc Cityscapes. [DeepLabV3+](https://arxiv.org/abs/1802.02611), [Focal](https://arxiv.org/abs/1708.02002) và [V-Net / Dice](https://arxiv.org/abs/1606.04797) là nguồn kỹ thuật, không phải đóng góp mới của repo. [Ghi nhận nguồn](../THIRD_PARTY_NOTICES.md).

Hướng đã chọn tiếp theo là checkpoint B2 Cityscapes công bố → fine-tune ACDC 20 epoch, rồi đối chứng replay cùng lượt thấy ACDC và bổ sung control cùng số update. **Chưa có notebook triển khai protocol mới đó**; không thay nhãn kết quả hiện tại thành public-checkpoint baseline. [Chi tiết quyết định](../research/BASELINE_PROTOCOL.md).
