# Notebook Kaggle

| Notebook | Vai trò |
| --- | --- |
| [02_train_segformer_b2.ipynb](02_train_segformer_b2.ipynb) | Model chính B2 |
| [03_train_deeplabv3plus_resnet50.ipynb](03_train_deeplabv3plus_resnet50.ipynb) | Mở rộng CNN tùy chọn, chưa có kết quả GPU đầy đủ |

Mỗi notebook chạy **một model**, ImageNet encoder → Cityscapes 20 epoch → ACDC có nhãn + Cityscapes replay 20 epoch. Đây là recipe đang triển khai, chưa phải [protocol public-checkpoint mới](../research/BASELINE_PROTOCOL.md). B0 đã rút khỏi luồng chạy mới; giữ số thứ tự 02/03 để không đổi tên notebook đang được chia sẻ.

- Chạy mới: code ZIP mới + Cityscapes + ACDC.
- Resume: ZIP kết quả của đúng model và cùng dataset/mount; không thay config của lượt đang chạy.
- Sinh lại bằng `python -m scripts.build_kaggle_notebooks`; output code ZIP ở `outputs/kaggle/project-code.zip`.
- Sửa [template chung](../scripts/build_research_notebooks.py), không sửa lặp logic trainer trong notebook.

[Hướng dẫn từng bước và checkpoint](../docs/TRAINING_GUIDE.md) · [Đọc code trước khi chạy](../docs/READING_GUIDE.md)
