# Kiến trúc model

| Vai trò hiện tại | File | Tham số |
| --- | --- | ---: |
| B2, model chính | [segformer_b2.py](segformer_b2.py) | 27.361.235 |
| DeepLabV3+ ResNet50, mở rộng tùy chọn chưa train đầy đủ | [deeplabv3plus_resnet50.py](deeplabv3plus_resnet50.py) | 26.695.619 |
| B0, chỉ tương thích lịch sử | [legacy/segformer_b0.py](legacy/segformer_b0.py) | 3.719.027 |

[registry.py](registry.py) chọn kiến trúc, xác minh pretrained và nạp checkpoint. [segformer.py](segformer.py) giữ các helper chung. Mỗi model trả logits 19 lớp; loss, dữ liệu, optimizer và evaluation nằm ở module dùng chung, không copy vào từng model.

B2 tái sử dụng các khối encoder/decoder Transformers với class/config/forward riêng. DeepLab dùng backbone TorchVision ResNet50, separable ASPP/decoder, output stride 16. Đây là triển khai dựa trên kiến trúc đã công bố, không phải model mới do đồ án phát minh. [Nguồn và giấy phép](../../THIRD_PARTY_NOTICES.md).

B0 không xuất hiện ở CLI/campaign/notebook/UI cho lượt mới. Loader và test hồi quy vẫn hỗ trợ đọc checkpoint cũ; [lý do giữ legacy](legacy/README.md).
