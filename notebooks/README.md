# Hai notebook DG hiện hành

| File | Model | ARM |
| --- | --- | --- |
| [01_dg_segformer_b2_baseline.ipynb](01_dg_segformer_b2_baseline.ipynb) | B2 | baseline / aug / consistency |
| [02_dg_deeplabv3plus_resnet50.ipynb](02_dg_deeplabv3plus_resnet50.ipynb) | DeepLabV3+ ResNet50 | baseline / aug / consistency |

B2 giữ tên file cũ để bảo toàn liên kết, nhưng hỗ trợ cả ba arm. Mỗi phiên một model/arm,20epoch; trainCityscapes2975/chọnbestval500. Không có ACDC input. First GPU run mặc định `PROFILE_ONLY=True` để đo thời gian/VRAM, bỏ weights pilot; `False` mới train chính. CPU implementation đã được kiểm thử, full GPU quality còn chờ.

`python -m scripts.build_kaggle_notebooks` sinh notebook từ [template](../scripts/build_dg_notebook.py) và code ZIP mới. Dùng **cùng bundle** cho5lượt chính. Resume cần cùng model/arm/seed/source/data mount; không dùng best baseline làm đầu vào proposed.

[Hướng dẫn và freeze/evaluate](../docs/TRAINING_GUIDE.md) · [Phương pháp](../docs/DG_METHOD.md) · [Lịch sử supervised](../research/archive/notebooks)
