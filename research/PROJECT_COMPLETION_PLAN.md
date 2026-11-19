# Kế hoạch hoàn thành project

Cập nhật19/11/2026. GPU dự kiến hồi khoảng21/11 với quota30giờ. Chỉ học Cityscapes; ACDC chỉ đánh giá sau khi khóa cả sáu candidate. [Phương pháp](../docs/DG_METHOD.md) · [Cách chạy](../docs/TRAINING_GUIDE.md).

## Đã có

- Hai kiến trúc riêng B2 và DeepLab; data/trainer/loss/evaluator dùng chung.
- Ba chiến thuật baseline/aug/consistency, cùng20epoch và khởi tạo độc lập.
- Hai notebook tự chuẩn bị dữ liệu, profile, chạy ba chiến thuật, xuất ZIP và resume qua phiên mới.
- Guard chống dùng target sai protocol; freeze/cohort kiểm config, source và hash checkpoint.
- Kiểm thử CPU, probe nhỏ trên ảnh thật và Node.js demo bằng weights lịch sử.

## Ma trận cần chạy GPU

| Model | baseline | aug | consistency |
| --- | --- | --- | --- |
| SegFormer-B2 | 20epoch | 20epoch | 20epoch |
| DeepLabV3+ ResNet50 | 20epoch | 20epoch | 20epoch |

Trong từng model: cùng encoder ImageNet đã pin, head mới cùng seed42, train2975/val500, preprocessing/optimizer. Baseline không truyền weights cho nhánh tiếp theo. C1/C2 tốn thêm forward nên phải báo compute riêng, không gọi đây là đối chứng cùng chi phí.

## Các bước tiếp theo

1. **Trước GPU:** dùng code ZIP mới, kiểm Cityscapes đủ train/val và notebook đúng model. Không thêm ACDC vào input train.
2. **Chạy B2:** Run All để profile nguồn rồi chạy ba nhánh. Theo dõi timing/VRAM/quota; giữ ZIP sau mỗi phiên.
3. **Chạy DeepLab:** cùng quy trình, cùng recipe, đủ ba nhánh. Có thể tiếp tục qua nhiều phiên; không giảm epoch hay bỏ nhánh khi chưa hoàn thành.
4. **Nghiệm thu:** mỗi nhánh đủ log1–20, checkpoint best/last, initialization/config/source/manifests hashes. Chỉ khi đủ hai suite mới freeze cohort.
5. **Đánh giá:** checkpoint đã khóa trên Cityscapes val500 và ACDC val406, cùng FP32/native/noTTA; báo overall/per-weather/per-class. Không tune quay lại bằng điểm ACDC.
6. **Phân tích:** loss/mIoU curves, ảnh lỗi, delta C1−C0 và C2−C1, params/giờ train/VRAM. Nếu không tăng, báo kết quả âm và giới hạn.
7. **Bàn giao:** tích hợp weights DG vào Node.js UI, kiểm upload/overlay/download, quay video; hoàn thiện báo cáo/slide và README kết quả.

Đối chứng VLTSeg: xác minh checkpoint chỉ học Cityscapes, class mapping, preprocessing và GPU trước khi evaluate. Nếu không chạy được, ghi blocker và để số công bố ở bảng riêng. Không nhận là đã tái lập paper.

## Thời gian và checkpoint

Chưa đo GPU cho recipe DG hai view và DeepLab, nên chưa cam kết sáu lượt vừa30giờ. Profile chỉ là dự trù, chưa bao gồm hết setup/ZIP/checkpoint. Hai notebook không đọc quota chung Kaggle; xem quota thực tế và giữ dự phòng cho đánh giá.

Cửa sổ mỗi phiên tối đa10giờ; gần hết suite tự lưu/dừng. Phiên mới gắn ZIP và đặt RESUME_INPUT. Epoch đã chốt được kiểm tra và bỏ qua, epoch dở lặp lại. Hard-kill có thể bỏ qua finally; luôn giữ ZIP hoàn chỉnh gần nhất. Không tự sửa recipe để vượt giới hạn thời gian rồi gọi là cùng thí nghiệm.

## Giới hạn cần báo cáo

Một seed chưa chứng minh ổn định thống kê. Cityscapes val là selection, ACDC đã từng được xem lịch sử; không nhận là test độc lập mới. Không so chéo DG/UDA/supervised như cùng quyền dữ liệu, không hứa trước vượt paper và không gán hiệu quả cho consistency nếu C2 không hơn C1.
