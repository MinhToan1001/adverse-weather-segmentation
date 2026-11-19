# Training B2 trên Kaggle và model mở rộng

[Danh sách notebook](../notebooks/README.md) · [Phương pháp](RESEARCH_FRAMEWORK.md)

## Mỗi lượt chạy làm gì?

Chọn notebook B2 chính hoặc DeepLab tùy chọn. B0 đã rút khỏi luồng chạy mới. Đây là recipe ImageNet → Cityscapes → ACDC hiện có, chưa phải protocol checkpoint Cityscapes công bố trong research/BASELINE_PROTOCOL.md. Model khởi tạo encoder từ ImageNet, đầu phân đoạn 19 lớp mới; train Cityscapes 20 epoch rồi tự dùng checkpoint source tốt nhất để train ACDC với source replay thêm 20 epoch. Mỗi model có workspace, bản source đóng băng và ZIP kết quả riêng.

| Model | Notebook | Thời gian trên một T4 |
| --- | --- | --- |
| B2 | [02](../notebooks/02_train_segformer_b2.ipynb) | Đã ghi nhận 7,77 giờ cho lượt seed 42 |
| DeepLabV3+ ResNet50 | [03](../notebooks/03_train_deeplabv3plus_resnet50.ipynb) | 12–24 giờ; chưa có đo GPU trực tiếp |

Các số trên tính cho **40 epoch/model**, gồm hai giai đoạn 20 epoch. Thời gian B2 lấy từ training invocation thực tế, gồm validation và nén kết quả trong lượt chạy; chưa tính cài dependencies và chuẩn bị dữ liệu. B2 có median khoảng 8,36 phút/epoch source và 10,24 phút/epoch adaptation, nhanh hơn dự trù bảo thủ 16–24 giờ trước đây. Đây không phải cam kết thời gian cho mọi phiên Kaggle. [báo cáo B2](../results/segformer_b2_seed42/README.md). DeepLab vẫn là ước lượng sơ bộ, chưa đo GPU trực tiếp; [dự trù ban đầu](../results/repository-release/runtime_budget.json).

Sau những epoch đầu, log và `status.json` cập nhật ETA từ thời gian thực tế của **giai đoạn đang chạy**. Cần cộng thêm thời gian setup, nén ZIP và giai đoạn sau chưa đo. Notebook dùng cửa sổ phiên 10 giờ. B2 đã hoàn thành trong một invocation ở các lượt ghi nhận, nhưng vẫn cần giữ checkpoint để tiếp tục nếu phiên khác chậm hơn; DeepLab có thể cần nhiều phiên. Trainer hiện dùng một GPU, kể cả khi giao diện chọn T4 × 2.

## Bắt đầu một lượt mới

1. Tạo bộ code: `python -m scripts.build_kaggle_notebooks`.
2. Import notebook của model cần chạy; thêm `outputs/kaggle/project-code.zip`, Cityscapes và ACDC làm input. Cập nhật dataset chứa code khi đổi code.
3. Bật GPU và Internet. Giữ `RESUME_INPUT=None` để bắt đầu mới.
4. Dataset được tự tìm theo cấu trúc. Nếu gắn nhiều bản code, đặt `CODE_INPUT` đến đúng dataset code. Các biến đường dẫn Cityscapes/ACDC cũng có thể khai báo cụ thể.
5. Chọn **Save Version → Save & Run All**. Notebook tự chạy source rồi adaptation; không cần thêm checkpoint `best/` của project cũ.

## Checkpoint lưu những gì?

Sau mỗi epoch hoàn thành, `last/` và `best/` lưu weights, config, optimizer, scheduler, AMP scaler, trạng thái ngẫu nhiên và epoch đã chọn. File được ghi vào thư mục tạm, kiểm tra hash rồi mới thay thế bản chính. `last.previous/` và `best.previous/` là bản dự phòng. Khôi phục đối chiếu checkpoint với log; log của epoch chưa ghi xong được loại bỏ.

Toàn bộ workspace được nén vào ZIP tạm rồi mới thay ZIP hoàn chỉnh trước đó. Nếu ngắt giữa epoch, resume chạy lại epoch đó. **Cần giữ output của phiên Kaggle hoặc tải ZIP về**: file trong runtime không tự đồng bộ sang Drive và có thể mất nếu runtime bị xóa trước khi lưu output.

## Tiếp tục sau khi ngắt

1. Giữ ZIP hoàn chỉnh mới nhất `<model>_seed42.zip` và output của phiên Kaggle.
2. Mở phiên mới bằng **notebook của cùng model**, cùng phiên bản dataset và đường dẫn mount.
3. Thêm ZIP kết quả hoặc dataset đã giải nén của model đó; đặt `RESUME_INPUT` đến vị trí tương ứng.
4. Chạy notebook. Nó khôi phục source, config và trạng thái training ban đầu; tiếp tục phần chưa hoàn thành. Các giai đoạn đã xong được bỏ qua.

ZIP của các model không dùng thay cho nhau. B0 lịch sử cần source/notebook đã đóng băng của lượt cũ. Resume dùng bản source đã đóng băng trong kết quả, không tự áp dụng thay đổi code mới vào thí nghiệm đang chạy.

## Đọc kết quả

`training_complete` nghĩa là cả hai giai đoạn 20 epoch đã hoàn thành cho model đó. Điểm monitor/dev phục vụ chọn checkpoint; đây chưa phải kết quả official-val. Sau khi chốt checkpoint, dùng `scripts.evaluate_research` để đánh giá Cityscapes val 500 và ACDC val 406. Giữ checkpoint và manifest của thí nghiệm để có thể kiểm tra lại.
