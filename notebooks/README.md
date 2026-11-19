# Notebook hiện hành

[01_dg_segformer_b2_baseline.ipynb](01_dg_segformer_b2_baseline.ipynb): B2 encoder ImageNet → Cityscapes train2975, chọn best bằng val500, 20epoch. Chỉ một stage, không có ảnh/nhãn ACDC trong train hoặc selection. Chưa có GPU benchmark mới, chưa có module proposed.

Sinh bằng `python -m scripts.build_kaggle_notebooks`; template [build_dg_notebook.py](../scripts/build_dg_notebook.py). Upload code ZIP DG mới và Cityscapes; không dùng ZIP supervised làm resume. Checkpoint/ZIP sau mỗi epoch, cùng code snapshot và mount khi resume. [Hướng dẫn](../docs/TRAINING_GUIDE.md).

Hai notebook B2/DeepLab source→ACDC cũ chuyển sang [research/archive/notebooks](../research/archive/notebooks). Đó là lịch sử supervised, không phải DG.
