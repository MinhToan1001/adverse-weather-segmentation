# Pipeline dữ liệu dùng chung

| Module | Vai trò trong DG |
| --- | --- |
| `labels.py` | 19 trainIds và ignore255 |
| `preparation.py` | Tạo manifest, chuyển labelIds → trainIds |
| `validation.py` | Kiểm cặp file, nhãn, kích thước và đường dẫn trùng |
| `preprocessing.py` | Giữ tỷ lệ, resize/pad, chuẩn hóa |
| `dataset.py` | Hình học ảnh–mask đồng bộ, clean/style views, eval xác định |
| `augmentations/source_style.py` | Đổi RGB nguồn; cùng mask, RNG riêng |

Suite DG chỉ dùng Cityscapes train2975/val500; ACDC val dành cho evaluator sau freeze. [Contract](../../docs/DATA_CONTRACT.md).

`partitions.py`, `replay.py`, `sampling.py`, `class_weights.py` và augmentation cũ còn giữ cho checkpoint/config lịch sử và kiểm thử tiện ích. Chúng không tham gia chiến thuật DG đã chốt. Manifest và mask sinh ra nằm trong workspace local, không commit.
