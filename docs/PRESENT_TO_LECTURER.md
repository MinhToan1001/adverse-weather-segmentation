# Trình bày giao thức DG

“Em kiểm tra khả năng phân đoạn đường phố trong thời tiết xấu khi chỉ được học Cityscapes. ACDC được giữ để đánh giá sau khi chốt model. Baseline B2 mới chọn checkpoint bằng Cityscapes val; hướng cải tiến dự kiến là tăng đa dạng style ảnh nguồn và ràng buộc nhất quán dự đoán.”

Mở lần lượt: [DG plan](../research/DG_RECOVERY_PLAN.md) → [labels](../src/data/labels.py) → [dataset](../src/data/dataset.py) → [B2](../src/models/segformer_b2.py) → [protocol guard](../src/experiments/protocol.py) → [DG launcher](../scripts/run_dg.py) → [trainer](../src/training/train.py) → [evaluator](../src/evaluation/runner.py) → [web](../web/README.md).

Phải nói rõ: kết quả source/adapt cũ là lịch sử khác protocol. Điểm63,02% đã dùng nhãn ACDC để học/chọn best nên **không chứng minh DG**. Baseline DG mới và proposed chưa có kết quả GPU; consistency là đề xuất chưa triển khai. B2 thay B0 là đổi backbone so với kế hoạch gốc, không phải tự chứng minh một cải tiến DG. ACDC đã được xem lịch sử nên không nhận là test độc lập mới.

Cần bàn giao: baseline source-only đúng protocol, paper DG khả thi hoặc blocker, proposed+ablation, bảng per-class/per-weather/compute, demo và báo cáo truy vết. Chỉ có demo ứng dụng chưa đủ các mục tiêu thực nghiệm trong file gốc.
