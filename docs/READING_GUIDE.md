# Đọc và hiểu toàn bộ repo

Đọc theo **luồng một ảnh đi qua hệ thống**, rồi mới xem các tùy chọn mở rộng. Không cần bắt đầu bằng notebook dài hoặc các ZIP kết quả.

## Vòng 1 · Hiểu hệ thống trước khi đọc thuật toán

1. [README gốc](../README.md): nói lại được đầu vào là ảnh đường phố, đầu ra là 19 lớp cho từng pixel, mục tiêu là thời tiết xấu.
2. [Kết quả B2](../results/segformer_b2_seed42/README.md): phân biệt model source với model đã thích nghi ACDC; biết số nào là dev, số nào chưa có.
3. [Web guide](../web/README.md): chạy giao diện khi có weights. Thử cùng một ảnh ở B2 trước/sau ACDC, quan sát người/xe nhỏ, ranh giới và vùng tối. Không suy ra độ chính xác chỉ từ diện tích màu.

Sau vòng này cần trả lời: **hệ thống giúp việc gì, đã chạy được gì, còn thiếu bằng chứng gì?**

## Vòng 2 · Dữ liệu và một mẫu train

| Thứ tự | Đọc | Điều cần hiểu |
| --- | --- | --- |
| 1 | [labels.py](../src/data/labels.py), [data contract](DATA_CONTRACT.md) | 19 trainIds 0–18; 255 là vùng bỏ qua, không phải lớp thứ 20 |
| 2 | [preparation.py](../src/data/preparation.py), [acdc.py](../src/data/acdc.py) | Tìm ảnh và nhãn tương ứng; khác tên thư mục Cityscapes/ACDC |
| 3 | [partitions.py](../src/data/partitions.py) | Chia theo nhóm chuỗi/recording để giảm ảnh gần nhau lọt sang cả fit và dev |
| 4 | [prepare_research_data.py](../scripts/prepare_research_data.py) | Lệnh tạo manifest và chuyển nhãn; `data.json` nối các phần lại |
| 5 | [preprocessing.py](../src/data/preprocessing.py), [shared_512.yaml](../configs/preprocessing/shared_512.yaml) | Giữ tỷ lệ, resize, padding và chuẩn hóa; ảnh/mask dùng kiểu nội suy khác nhau |
| 6 | [dataset.py](../src/data/dataset.py) | Một hàng manifest biến thành tensor ảnh và nhãn; train có random geometry, dev không có |
| 7 | [replay.py](../src/data/replay.py) | Xen mẫu ACDC và Cityscapes khi thích nghi, cùng luật chọn checkpoint giữ kiến thức nguồn |
| 8 | [validation.py](../src/data/validation.py) | Kiểm tra ghép cặp, nhãn, split và lỗi dữ liệu trước khi train |

**Ví dụ:** ảnh sương mù và mask của nó cùng bị lật ngang; mask dùng nearest để không tạo mã lớp lạ. Nếu crop ảnh mà không crop mask cùng tọa độ, model học sai ngay cả khi code vẫn chạy.

## Vòng 3 · Model và quy trình học dùng chung

| Thứ tự | Đọc | Điều cần hiểu |
| --- | --- | --- |
| 1 | [segformer_b2.py](../src/models/segformer_b2.py) | Encoder lấy đặc trưng; decoder dự đoán điểm cho 19 lớp. Đây là triển khai dựa trên SegFormer, không phải kiến trúc tự phát minh |
| 2 | [registry.py](../src/models/registry.py), [segformer.py](../src/models/segformer.py) | Chọn kiến trúc, khởi tạo và nạp đúng checkpoint; loader chung vẫn đọc được B0 lịch sử |
| 3 | [config.py](../src/experiments/config.py), [configs](../configs/README.md) | Ghép tên model + stage + preprocessing + loss + đường dẫn thành cấu hình hoàn chỉnh |
| 4 | [losses.py](../src/training/losses.py), [optim.py](../src/training/optim.py), [strategies.py](../src/training/strategies.py) | Tính lỗi, cập nhật tham số, đóng băng thống kê BatchNorm; CE đang là mặc định |
| 5 | [train.py](../src/training/train.py) | `main`: tạo dataset/model → forward → loss → backward → optimizer → validation → best/last |
| 6 | [checkpoints.py](../src/training/checkpoints.py) | Lưu có xác minh, giữ thế hệ trước, khôi phục trạng thái khi ngắt |
| 7 | [run_research.py](../scripts/run_research.py), [run_research_campaign.py](../scripts/run_research_campaign.py) | Điều phối từng epoch và hai stage; đóng băng config/source và tránh trộn các lượt chạy |
| 8 | [build_research_notebooks.py](../scripts/build_research_notebooks.py) | Template sinh notebook B2/DeepLab; notebook chỉ gọi code dùng chung |

Đọc `train.py` theo các mốc `main`, `evaluate_model`, `optimizer`, `for epoch`, `is_best`, checkpoint. Không cần thuộc toàn bộ file. Khi đổi model qua registry, dataset/loss/evaluator vẫn được tái sử dụng.

## Vòng 4 · Đánh giá và giao diện

1. [metrics.py](../src/evaluation/metrics.py): ma trận nhầm lẫn → IoU từng lớp → mIoU; loại pixel 255.
2. [runner.py](../src/evaluation/runner.py), [calibration.py](../src/evaluation/calibration.py): suy luận về kích thước nhãn gốc, điểm theo thời tiết, độ tin cậy và hình lỗi. Độ tự tin cao vẫn có thể đoán sai.
3. [evaluate_research.py](../scripts/evaluate_research.py), [plot_research_run.py](../scripts/plot_research_run.py): CLI đánh giá checkpoint đã chọn và vẽ log; không train lại trong bước này.
4. [inference.py](../src/inference.py): cùng xử lý ảnh cho inference, logits → mask native-size và màu lớp.
5. [server.mjs](../web/server.mjs) → [worker.py](../web/worker.py) → [app.js](../web/public/app.js): browser gửi ảnh cho Node; Python chạy PyTorch; browser hiện mask/overlay. [index.html](../web/public/index.html) và [styles.css](../web/public/styles.css) phụ trách bố cục.
6. [tests](../tests) và [web tests](../web/tests): xem các ví dụ nhỏ chứng minh code xử lý đúng. Test phần mềm không thay thế kết quả model trên dataset.

## Vòng 5 · Tham khảo và mở rộng sau cùng

- Đọc [SegFormer 2021](../research/paper_reviews/SegFormer_2021.md) để biết kiến trúc và kết quả Cityscapes của tác giả. So sánh cần cùng dataset, split và cách evaluate.
- Đọc [protocol B2 công bố](../research/BASELINE_PROTOCOL.md): **kế hoạch tiếp theo**, chưa phải recipe đang chạy. Mốc Cityscapes của bài báo và phần học ACDC của đồ án phải báo riêng.
- [DeepLabV3+](../src/models/deeplabv3plus_resnet50.py) là model tùy chọn cùng đầu ra 19 lớp. Augmentation trong [augmentations](../src/data/augmentations), [class_weights.py](../src/data/class_weights.py) và [sampling.py](../src/data/sampling.py) là công cụ để làm thí nghiệm có đối chứng, không bật hết cùng lúc.
- HRDA/Refign trong [research](../research/README.md) là tài liệu nền về hướng khác; không phải thành phần đang được tích hợp trong hệ thống.
- [Legacy B0](../src/models/legacy/README.md), báo cáo cũ và [bản thảo lưu trữ](../research/archive/README.md) chỉ đọc khi cần truy vết lịch sử.

## Từ vựng tối thiểu

| Từ | Nói ngắn gọn |
| --- | --- |
| Source / target | Miền ban đầu Cityscapes / miền cần thích nghi ACDC |
| Fine-tune / adaptation | Học tiếp weights đã có để phù hợp dữ liệu mới |
| Supervised / UDA | Học có nhãn / thích nghi với ảnh đích không có nhãn; project hiện dùng có nhãn |
| Fit / dev / official-val | Fit để học; dev chọn checkpoint; official-val kiểm tra sau khi chốt |
| Manifest | Danh sách đường dẫn ảnh, mask, split và nhóm; không phải dataset mới |
| Epoch / batch | Một vòng theo bộ lấy mẫu / số ảnh xử lý cùng lúc |
| CE / loss | Hàm tính mức sai để điều chỉnh model; CE là cross-entropy |
| Logits / softmax | Điểm thô từng lớp / đổi điểm thành phân bố xác suất |
| Checkpoint / best / last | Bản lưu / tốt nhất theo luật chọn / epoch hoàn thành gần nhất |
| Replay / forgetting | Ôn ảnh cũ / quên khả năng cũ khi học miền mới |
| mIoU / điểm phần trăm | Chồng khớp trung bình các lớp / ví dụ 50% lên 60% là tăng 10 điểm phần trăm |
| Augmentation / normalization | Biến đổi ảnh để tăng đa dạng / chuẩn hóa giá trị pixel theo quy ước model |
| Overfitting | Học tốt dữ liệu đã thấy nhưng kém trên dữ liệu giữ lại |

Khi thuyết trình, dùng [walkthrough ngắn](PRESENT_TO_LECTURER.md). Khi vận hành, dùng [training guide](TRAINING_GUIDE.md); không copy recipe từ bản thảo lịch sử.
