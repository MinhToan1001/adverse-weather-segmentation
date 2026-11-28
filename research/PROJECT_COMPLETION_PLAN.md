# Kế hoạch hoàn thành project

Cập nhật 26/11/2026. **Training đủ sáu lượt × 20 epoch và 12 phép đánh giá Cityscapes/ACDC đã được nghiệm thu.** [Kết quả target](../results/dg_evaluation_seed42/README.md). Chỉ học Cityscapes. [Phương pháp](../docs/DG_METHOD.md) · [Cách chạy](../docs/TRAINING_GUIDE.md) · [Biên bản cohort](../results/dg_cohort_seed42/README.md).

## Đã có

- Hai kiến trúc riêng B2 và DeepLab; data/trainer/loss/evaluator dùng chung.
- Ba chiến thuật baseline/aug/consistency, cùng20epoch và khởi tạo độc lập.
- Hai notebook tự chuẩn bị dữ liệu, profile, chạy ba chiến thuật, xuất ZIP và resume qua phiên mới.
- Guard chống dùng target sai protocol; freeze/cohort kiểm config, source và hash checkpoint.
- Kiểm thử CPU, probe nhỏ trên ảnh thật và Node.js demo bằng weights lịch sử.

## Ma trận cần chạy GPU

| Model | baseline | aug | consistency |
| --- | --- | --- | --- |
| SegFormer-B2 | Đủ 20; 73,98% | Đủ 20; 74,37% | Đủ 20; 74,44% |
| DeepLabV3+ ResNet50 | Đủ 20; 71,82% | Đủ 20; 72,08% | Đủ 20; 71,58% |

Điểm trong bảng là Cityscapes val500 dùng chọn checkpoint, chưa phải ACDC. Giữ cả sáu candidate khi đánh giá; không bỏ DeepLab consistency vì điểm nguồn thấp hơn aug.

Trong từng model: cùng encoder ImageNet đã pin, head mới cùng seed42, train2975/val500, preprocessing/optimizer. Baseline không truyền weights cho nhánh tiếp theo. C1/C2 tốn thêm forward nên phải báo compute riêng, không gọi đây là đối chứng cùng chi phí.

## Trạng thái các bước

Bước 1–3 đã hoàn tất ngày 26/11; còn bước 4 phân tích lỗi/chi phí và bước 5 tích hợp demo/báo cáo. Giữ cohort hiện tại; nếu nghiên cứu cải tiến tiếp, mở một vòng thí nghiệm riêng.

1. **Đã xong:** B2 và DeepLab đủ ba nhánh; log 1–20, checkpoint/source/config/dataset hashes được nghiệm thu, sáu best checkpoint được khóa.
2. **Đánh giá tiếp:** đưa hai ZIP suite hoàn chỉnh `(1)` vào cùng môi trường có Cityscapes và ACDC. Khóa lại cohort tại mount mới bằng evaluator, giữ nguyên danh tính/hash best đã nghiệm thu; chỉ remap đường dẫn phục vụ evaluation, không sửa artifact train.
3. **Đánh giá:** sáu checkpoint đã khóa trên Cityscapes val500 và ACDC val406, cùng FP32/native/noTTA; báo overall/per-weather/per-class. Không tune quay lại bằng điểm ACDC.
4. **Phân tích:** loss/mIoU curves, ảnh lỗi, delta C1−C0 và C2−C1, params/giờ train/VRAM. Nếu không tăng, báo kết quả âm và giới hạn.
5. **Bàn giao:** tích hợp weights DG vào Node.js UI, kiểm upload/overlay/download, quay video; hoàn thiện báo cáo/slide và README kết quả.

Đối chứng VLTSeg: xác minh checkpoint chỉ học Cityscapes, class mapping, preprocessing và GPU trước khi evaluate. Nếu không chạy được, ghi blocker và để số công bố ở bảng riêng. Không nhận là đã tái lập paper.

## Thời gian và checkpoint

Đã có timing GPU từ sáu lượt. B2 train+validation tổng khoảng 12,44 giờ; DeepLab khoảng 11,22 giờ. Đây không phải tổng quota tiêu thụ: chưa gồm toàn bộ setup, profile, ZIP/checkpoint và các invocation bị dừng. Giữ dự phòng GPU cho đánh giá sáu checkpoint; không dựa vào timing CPU smoke để ước tính đánh giá đầy đủ.

Cửa sổ mỗi phiên tối đa10giờ; gần hết suite tự lưu/dừng. Phiên mới gắn ZIP và đặt RESUME_INPUT. Epoch đã chốt được kiểm tra và bỏ qua, epoch dở lặp lại. Hard-kill có thể bỏ qua finally; luôn giữ ZIP hoàn chỉnh gần nhất. Không tự sửa recipe để vượt giới hạn thời gian rồi gọi là cùng thí nghiệm.

## Giới hạn cần báo cáo

Một seed chưa chứng minh ổn định thống kê. Cityscapes val là selection, ACDC đã từng được xem lịch sử; không nhận là test độc lập mới. Không so chéo DG/UDA/supervised như cùng quyền dữ liệu, không hứa trước vượt paper và không gán hiệu quả cho consistency nếu C2 không hơn C1.
