# Tham khảo và kế hoạch

Đây là tài liệu nền, không phải phần bắt buộc để chạy ứng dụng. [Tổng quan đồ án](../README.md), [phương pháp đang triển khai](../docs/RESEARCH_FRAMEWORK.md) và [results](../results/README.md) là nguồn trạng thái hiện hành.

| Tài liệu | Đọc để làm gì? | Trạng thái |
| --- | --- | --- |
| [SegFormer 2021](paper_reviews/SegFormer_2021.md) | Kiến trúc và mốc Cityscapes của bài gốc | Nguồn model B2 |
| [Protocol baseline mới](BASELINE_PROTOCOL.md) | B2 Cityscapes công bố → ACDC supervised FT20, đối chứng replay | Đã thống nhất hướng; chưa triển khai |
| [HRDA](paper_reviews/HRDA_2022.md), [Refign](paper_reviews/Refign_2023.md) | Hiểu hướng thích nghi không nhãn, khác quyền dùng dữ liệu hiện tại | Tham khảo, chưa tích hợp |
| [VLTSeg](paper_reviews/VLTSeg_2024.md) | Hướng model khác từng tìm hiểu | Tham khảo |
| [Ma trận bài báo](comparison_matrix.csv) | Phân biệt dataset/protocol/label access | Tra cứu, không bảng thắng-thua |
| [Kế hoạch thí nghiệm cũ](EXPERIMENT_PLAN.md) | Lịch sử thiết kế B0/B2/DeepLab | Lưu để truy vết |
| [Bản thảo lưu trữ](archive/README.md), [bibliography](archive/references.bib) | Tài liệu nghiên cứu trước khi chốt hướng đồ án ứng dụng | Không dùng làm báo cáo kết quả hiện tại |

Code Kaggle hiện tại vẫn khởi tạo encoder ImageNet, học Cityscapes rồi ACDC có nhãn + replay. Việc chuyển sang checkpoint Cityscapes công bố cần triển khai thành thí nghiệm mới, không sửa nguồn gốc của lượt B2 đã có.
