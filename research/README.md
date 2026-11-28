# Cơ sở phương pháp và kế hoạch

[Vòng DG v2 ngày 26/11](DG_V2_PLAN.md): ma trận weighting × augmentation trên source-dev đã triển khai riêng; [Kiểm thử và chuẩn bị Cityscapes thật](../results/dg_v2_implementation/README.md) đã qua. [Notebook05/06 và code ZIP mới](../docs/DG_V2_GUIDE.md) sẵn sàng; chưa có kết quả GPU v2, không thay kết quả v1.

Project dùng **kiến trúc đã công bố + pipeline dùng chung + so sánh có đối chứng**. Không nhận là sáng tạo ra SegFormer/DeepLab hay tái lập đầy đủ một bài DG. [Phương pháp đang cài](../docs/DG_METHOD.md) · [Việc còn lại](PROJECT_COMPLETION_PLAN.md).

## Đọc bài báo để trả lời điều gì?

| Nguồn | Dùng để hiểu/chọn gì? | Giới hạn so sánh |
| --- | --- | --- |
| [SegFormer, NeurIPS2021](paper_reviews/SegFormer_2021.md) | Kiến trúc B2 và cách dùng encoder pretrained | Recipe20epoch của project không tái lập nguyên kết quả paper |
| [DeepLabV3+, ECCV2018](https://arxiv.org/abs/1802.02611) | Kiến trúc CNN để đối chiếu với B2 | ResNet50 là biến thể project; không gán điểm Xception của paper cho model này |
| [ISSA, WACV2023](https://arxiv.org/abs/2210.10175) | Động cơ tăng đa dạng style từ nguồn | Đổi màu/gamma ở đây không phải ISSA/StyleGAN |
| [SHADE, ECCV2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880530.pdf) | Động cơ style + consistency | Khác miền/thiết lập; project không cài đầy đủ SHADE |
| [VLTSeg, ACCV2024](paper_reviews/VLTSeg_2024.md) | Tham khảo DG Cityscapes→ACDC | Backbone/pretraining/inference khác; chưa có đối chứng GPU hoàn chỉnh trong luồng mới |
| [HRDA](paper_reviews/HRDA_2022.md), [Refign](paper_reviews/Refign_2023.md) | Hiểu hướng UDA: học thêm ảnh target không nhãn | UDA có quyền dùng target khác DG, không so trực tiếp như cùng protocol |

Luận điểm cần kiểm tra: đa dạng ngoại hình ảnh nguồn và dự đoán nhất quán có giúp model giảm phụ thuộc ánh sáng/màu không? `aug` so `baseline` kiểm tăng cường; `consistency` so `aug` kiểm riêng ràng buộc nhất quán. Lặp lại trên B2 và DeepLab để xem hiệu quả có chuyển kiến trúc.

[comparison_matrix.csv](comparison_matrix.csv) ghi vai trò từng nguồn. Số paper phải ghi là số công bố và nêu split/model/recipe; số project phải do evaluator đo. Không dùng điểm supervised ACDC cũ để nhận là DG hoặc vượt paper.

## Những hướng đã bỏ

Hướng học ảnh/nhãn ACDC và kế hoạch public Cityscapes checkpoint→ACDC fine-tune không còn là luồng hiện hành. Kế hoạch và notebook đó đã gỡ khỏi cây làm việc để tránh chạy nhầm; lịch sử Git và snapshot trong các ZIP cũ giữ thông tin truy vết. [Kết quả lịch sử](../results/README.md) còn nguyên, không đổi nhãn.
