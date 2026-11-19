# Gợi ý trình bày đồ án

“Nhóm xây dựng hệ thống phân đoạn 19 lớp đường phố. Model chỉ học Cityscapes; sau khi chốt model, dùng ACDC để kiểm tra trong đêm, sương, mưa và tuyết. Nhóm so sánh SegFormer-B2 và DeepLabV3+ ResNet50, cùng ba chiến thuật: baseline, tăng đa dạng ảnh nguồn, rồi thêm ràng buộc dự đoán nhất quán.”

## Thứ tự show code

1. [Sơ đồ dữ liệu và split](DATA_CONTRACT.md).
2. [Dataset](../src/data/dataset.py) và [source style](../src/data/augmentations/source_style.py): ảnh đổi ngoại hình nhưng mask không lệch.
3. [B2](../src/models/segformer_b2.py) và [DeepLab](../src/models/deeplabv3plus_resnet50.py): kiến trúc riêng, pipeline dùng chung.
4. [DG loss](../src/training/dg.py), [recipe](../configs/recipes/source_20epoch.yaml) và [trainer](../src/training/train.py).
5. [Suite](../scripts/run_dg_suite.py), [checkpoint](../src/training/checkpoints.py) và [guard](../src/experiments/protocol.py).
6. [Evaluator](../src/evaluation/runner.py), [kết quả](../results/README.md), rồi [demo](../web/README.md).

## Phải nói đúng trạng thái

Đã có code và kiểm thử CPU; sáu lượt train GPU DG, bảng kết quả và tích hợp weights DG còn chờ. Điểm B2 adapted 63,02% là kết quả supervised trên ACDC dev402, không chứng minh DG. Demo hiện sử dụng weights lịch sử đó.

Kiến trúc và ý tưởng style/consistency dựa trên công trình công bố; chưa nhận là phát minh mới hay vượt paper. Đóng góp dự kiến của đồ án là pipeline dùng chung, thực nghiệm có đối chứng và hệ thống demo; hiệu quả của chiến thuật phải được kiểm chứng. B2 thay B0 là lựa chọn kiến trúc, không tự chứng minh cải tiến. ACDC đã từng được xem trong phát triển, cần công bố giới hạn này.

[Cơ sở bài báo](../research/README.md) · [Phương pháp chính xác](DG_METHOD.md) · [Các việc còn lại](../research/PROJECT_COMPLETION_PLAN.md)
