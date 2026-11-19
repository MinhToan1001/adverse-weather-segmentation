# Trình bày đồ án với giảng viên

## Giới thiệu trong một phút

“Em xây dựng ứng dụng phân đoạn ảnh đường phố trong điều kiện thiếu sáng và thời tiết xấu. Model SegFormer-B2 học Cityscapes, sau đó học tiếp ảnh và nhãn ACDC. Khi học ACDC, em xen ảnh Cityscapes để giảm quên cảnh trời quang. Ứng dụng Node.js cho phép đưa ảnh vào, xem 19 lớp, so sánh trước/sau thích nghi và tải kết quả. Em đánh giá chất lượng bằng nhãn chuẩn, không chỉ quan sát màu.”

## Mở code theo thứ tự

| Bước | File | Ý cần giải thích |
| --- | --- | --- |
| 1 | [labels.py](../src/data/labels.py) | Các lớp cần nhận diện, 0–18 và vùng bỏ qua 255 |
| 2 | [dataset.py](../src/data/dataset.py) | Ảnh/mask đi cùng nhau; biến đổi hình học đồng bộ |
| 3 | [segformer_b2.py](../src/models/segformer_b2.py) | Model chính: encoder và decoder; kế thừa kiến trúc đã công bố |
| 4 | [registry.py](../src/models/registry.py) | Thay model mà tái sử dụng phần còn lại; DeepLab là tùy chọn chưa train |
| 5 | [config.py](../src/experiments/config.py) | Recipe source/adapt, 20 epoch mỗi stage, loss CE |
| 6 | [train.py](../src/training/train.py), [replay.py](../src/data/replay.py) | Cách học, ôn dữ liệu nguồn và chọn best |
| 7 | [checkpoints.py](../src/training/checkpoints.py) | Lưu/khôi phục khi phiên Kaggle ngắt |
| 8 | [runner.py](../src/evaluation/runner.py) | mIoU từng lớp/thời tiết và hình minh họa lỗi |
| 9 | [web guide](../web/README.md) | Node nhận ảnh, Python chạy model, browser hiện kết quả |

## Trình diễn

Dùng cùng một ảnh cho B2 source và B2 adapted. Chỉ ra vùng tối, người/xe nhỏ và ranh giới dễ sai. Nếu không có mask chuẩn của ảnh, chỉ mô tả khác biệt quan sát được, không gán điểm chính xác cho ảnh đó.

## Báo cáo đúng phạm vi

- Đã có B2 source20/adapt20, seed42; ACDC dev nội bộ tăng từ 48,44% lên 63,02% mIoU. Đây là tập chọn checkpoint 402 ảnh, chưa là official-val/test độc lập.
- Giao diện và pipeline là phần ứng dụng tự tích hợp; SegFormer là kiến trúc của tác giả, không nhận là model mới do đồ án phát minh.
- Điểm bài gốc trên Cityscapes dùng làm tham khảo nguồn; không so trực tiếp với điểm ACDC dev của project.
- B0 là lịch sử, DeepLab và baseline public-checkpoint mới chưa có kết quả hoàn chỉnh. Các loss/augmentation tùy chọn chưa chứng minh tăng chất lượng.

[Từ vựng và lộ trình đọc đầy đủ](READING_GUIDE.md) · [Chứng cứ B2](../results/segformer_b2_seed42/README.md) · [Phương pháp](RESEARCH_FRAMEWORK.md)
