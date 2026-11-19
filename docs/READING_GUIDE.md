# Đọc repo theo giao thức DG

## 1. Hiểu câu hỏi trước

Đọc [README](../README.md) và [DG recovery plan](../research/DG_RECOVERY_PLAN.md). Câu hỏi: học Cityscapes rồi có xử lý được thời tiết xấu chưa học hay không? Vì vậy **không train/chọn checkpoint bằng ACDC**. Điểm supervised63,02% cũ không trả lời câu hỏi này.

## 2. Theo một ảnh qua dữ liệu

[labels.py](../src/data/labels.py): 19 mã lớp0–18, ignore255. [preparation.py](../src/data/preparation.py): ghép ảnh–mask và chuyển labelIds. [preprocessing.py](../src/data/preprocessing.py): giữ tỷ lệ, padding, normalization. [dataset.py](../src/data/dataset.py): tensor ảnh/nhãn, train geometry đồng bộ; validation xác định. [validation.py](../src/data/validation.py): lỗi ghép cặp/split/nhãn.

DG launcher chuẩn bị toàn Cityscapes train2975/val500. Các module [partitions.py](../src/data/partitions.py), [replay.py](../src/data/replay.py), [acdc.py](../src/data/acdc.py) phục vụ lịch sử supervised hoặc công cụ dataset; **không dùng target partitions/replay trong DG**. Không nhầm sự tồn tại của module với việc đang bật nó.

## 3. Xem model và training

[segformer_b2.py](../src/models/segformer_b2.py) là kiến trúc chính; [registry.py](../src/models/registry.py) chọn/nạp đúng model. [config.py](../src/experiments/config.py) ghép recipe. [protocol.py](../src/experiments/protocol.py) chặn target train/selection và checkpoint ACDC trong DG v1.

[run_dg.py](../scripts/run_dg.py) là entrypoint hiện hành: chuẩn bị Cityscapes, khóa config/manifest/source, chạy20epoch và xuất ZIP. [train.py](../src/training/train.py) dùng chung: forward → CE → backward → optimizer → Cityscapes val → best/last. [optim.py](../src/training/optim.py), [strategies.py](../src/training/strategies.py), [checkpoints.py](../src/training/checkpoints.py) tách optimizer, BatchNorm và lưu/khôi phục. [Notebook DG](../notebooks/README.md) chỉ gọi các phần đó.

Ba arm đã cài trong [dg.py](../src/training/dg.py), [SourceStyle](../src/data/augmentations/source_style.py) và [DG composer](../src/experiments/dg.py). C0=baseline, C1=aug, C2=consistency. Xem [phương pháp chính xác và kiểm thử](DG_METHOD.md); CPU QA đã qua, full GPU còn chờ. Các loss/class crop khác không bật trong DG controls.

## 4. Đánh giá và sử dụng

[metrics.py](../src/evaluation/metrics.py): ma trận nhầm lẫn → IoU từng lớp → mIoU. [runner.py](../src/evaluation/runner.py): native output, per-weather, độ tin cậy và panel lỗi. [evaluate_research.py](../scripts/evaluate_research.py) là công cụ đánh giá riêng; chỉ dùng ACDC sau khi khóa phương pháp/checkpoint. Không chạy vòng lặp chọn tham số bằng điểm ACDC.

[inference.py](../src/inference.py) xử lý một ảnh. [server.mjs](../web/server.mjs) nhận request; [worker.py](../web/worker.py) chạy PyTorch; [app.js](../web/public/app.js), [index.html](../web/public/index.html), [styles.css](../web/public/styles.css) trình bày ảnh/mask/overlay. Demo hiện chứa checkpoints lịch sử, chưa có model DG mới.

## 5. Kiểm chứng và tham khảo

[tests/test_dg_protocol.py](../tests/test_dg_protocol.py) kiểm target selection, adapted initialization, override và resume. Các [tests](../tests) khác kiểm metrics/model/loss/checkpoint; [web tests](../web/tests) kiểm UI/API. Test phần mềm không thay benchmark model.

[SegFormer](../research/paper_reviews/SegFormer_2021.md) là nguồn kiến trúc. [VLTSeg](../research/paper_reviews/VLTSeg_2024.md) là ứng viên đối chứng DG; ISSA và SHADE được nối với giả thuyết trong [DG plan](../research/DG_RECOVERY_PLAN.md). HRDA/Refign nguyên thiết lập là UDA. [Archive](../research/archive/README.md) giữ các hướng cũ; đọc sau cùng khi cần truy vết.

## Từ vựng ngắn

| Từ | Nghĩa |
| --- | --- |
| DG | Học miền nguồn, đánh giá miền chưa dùng để học/chọn model |
| Supervised adaptation | Học tiếp bằng ảnh và nhãn miền mới; khác DG |
| Source / target | Cityscapes / ACDC trong giao thức này |
| Consistency | Cùng nội dung dưới ngoại hình khác thì dự đoán nên nhất quán |
| Ablation | Tắt một thành phần để đo tác dụng riêng |
| Manifest | Danh sách ảnh–mask/split/domain |
| Epoch / batch | Vòng theo bộ mẫu / số ảnh xử lý cùng lúc |
| CE / logits | Hàm lỗi pixel / điểm thô của19lớp |
| Best / last / resume | Bản tốt nhất / epoch đã chốt gần nhất / tiếp tục trạng thái lưu |
| mIoU | Trung bình mức chồng khớp dự đoán/nhãn của các lớp |
| Normalization / augmentation | Chuẩn hóa pixel / tăng đa dạng ảnh nguồn |
| Pretraining | Kiến thức ban đầu từ dữ liệu được công bố rõ, ở đây ImageNet |
