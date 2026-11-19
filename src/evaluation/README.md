# Đánh giá dùng chung

- `metrics.py`: ma trận nhầm lẫn → IoU từng lớp → mIoU; bỏ ignore255. Lớp có union0 không có IoU xác định; chỉ có false-positive thì IoU0.
- `semantic.py`: nội suy logits, bỏ pad và phục hồi kích thước ảnh gốc.
- `runner.py`: chạy inference, gộp confusion, per-weather/per-class, xuất ảnh RGB/GT/prediction/confidence/error.
- `calibration.py`: ECE và reliability diagnostics; không phải chứng nhận an toàn triển khai.

Entry point hiện hành là [scripts.evaluate_dg](../../scripts/evaluate_dg.py). Nó khóa sáu candidate hoàn chỉnh trước khi đánh giá ACDC; thay checkpoint sau freeze sẽ bị từ chối. Dùng canvas512×1024, FP32, native output, không TTA (không cộng nhiều phiên bản ảnh khi dự đoán).

Overall mIoU tính từ confusion của toàn tập; không trung bình cơ học bốn mIoU thời tiết. Confidence của ảnh upload không phải độ chính xác có nhãn thật. Khoảng tin cậy theo nhóm và xác nhận nhiều seed chưa triển khai trong luồng bàn giao hiện tại. [Hướng dẫn](../../docs/TRAINING_GUIDE.md).
