# Ba notebook Kaggle hiện hành

[Hướng dẫn chạy và resume](../docs/TRAINING_GUIDE.md)

| Notebook | Model | Hai giai đoạn |
| --- | --- | --- |
| [01_train_segformer_b0.ipynb](01_train_segformer_b0.ipynb) | B0 baseline | Cityscapes 20 epoch → ACDC 20 epoch |
| [02_train_segformer_b2.ipynb](02_train_segformer_b2.ipynb) | B2 | Cityscapes 20 epoch → ACDC 20 epoch |
| [03_train_deeplabv3plus_resnet50.ipynb](03_train_deeplabv3plus_resnet50.ipynb) | DeepLabV3+ ResNet50 | Cityscapes 20 epoch → ACDC 20 epoch |

Các notebook được sinh từ template chung [build_research_notebooks.py](../scripts/build_research_notebooks.py). Code xử lý nằm trong `src/` và `scripts/`; mỗi notebook chọn model và đường dẫn output riêng.

- Chạy mới: code ZIP mới + Cityscapes + ACDC.
- Resume: thêm ZIP kết quả mới nhất của chính model đó, giữ cùng dataset và đường dẫn mount.

Notebook cũ được lưu trữ local, ngoài Git. Mỗi lượt hiện hành có tổng cộng **40 epoch/model**, gồm hai giai đoạn 20 epoch.
