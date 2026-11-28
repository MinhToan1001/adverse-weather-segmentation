# Bàn giao DG v2 cho Toàn

Nhánh chia sẻ: **`khang`**. Dùng mã commit Khang gửi để bắt đầu từ đúng snapshot; `git rev-parse HEAD` cho biết phiên bản đang checkout. Ngày28/11/2026: code/CPU QA và chuẩn bị Cityscapes đã có; **chưa có kết quả train GPU DG v2**. Ngân hàng style/SHADE chưa được triển khai trong nhánh này.

## Một model, bốn thí nghiệm rõ tên

| Tên hiện tại | Mã cũ, chỉ để đối chiếu | Loss | Augmentation RGB |
| --- | --- | --- | --- |
| `baseline` | V0 | CE | Chính sách cơ bản đã có |
| `class_weights` | V1 | Weighted CE | Như baseline |
| `weather_aug` | V2 | CE | Mixture cơ bản50% / low-light30% / haze20% |
| `weather_aug_class_weights` | V3 | Weighted CE | Như weather_aug |

Baseline đã có tăng cường RGB cơ bản; đây là mốc cho **hai thay đổi mới** của Khang. Bốn lượt khởi tạo độc lập, không học nối từ checkpoint lượt khác. Ma trận giúp tách tác dụng của weights, augmentation và tương tác của chúng. Chưa chốt lượt nào là phương pháp tốt nhất.

## Manifest dùng chung, không chứa đường dẫn máy cá nhân

- [fit.csv](../configs/dg_v2/splits/fit.csv): 2.380 cặp ảnh/mask huấn luyện.
- [dev.csv](../configs/dg_v2/splits/dev.csv): 595 cặp chỉ dùng chọn checkpoint, không backward.
- [split.json](../configs/dg_v2/splits/split.json): seed chia tập42, giữ nguyên thành phố; dev gồm bochum, erfurt, hamburg, weimar.

CSV công khai chỉ chứa tên/đường dẫn tương đối từ gốc Cityscapes và metadata, không chứa ảnh/nhãn. Cột `split=train` nghĩa là thuộc **official Cityscapes train**, kể cả các ảnh được giữ riêng làm dev nội bộ. Mask kỳ vọng là trainIds0–18/ignore255; nếu input chỉ có labelIds, launcher chuyển sang thư mục working, không sửa dataset gốc.

`prepare` tạo `data/fit.csv` và `data/dev.csv` có đường dẫn đúng môi trường chạy, rồi đối chiếu từng sample với CSV chung. `data/fit_portable.csv` và `dev_portable.csv` lưu danh sách không phụ thuộc mount. Nhóm và training seed không làm thay đổi split seed42.

## Cấu hình B2 và recipe thực sự được code đọc

| Thành phần | File nguồn |
| --- | --- |
| B2,19lớp,ImageNet encoder | [segformer_b2.yaml](../configs/dg_v2/segformer_b2.yaml) |
| Model/encoder revision và SHA256 | [registry.py](../src/models/registry.py), hàm `build_model` và `ENCODERS['b2']` |
|20epoch,AdamW,batch,LR,AMP | [source_20epoch.yaml](../configs/recipes/source_20epoch.yaml) |
| Resize/aspect ratio/normalization/ignore255 | [shared_512.yaml](../configs/preprocessing/shared_512.yaml) |
| Bốn thí nghiệm và selection | [development.yaml](../configs/dg_v2/development.yaml) |
| Ghép config và kiểm protocol | [dg_v2.py](../src/experiments/dg_v2.py), `compose`/`validate` |

Encoder `nvidia/mit-b2` revision `3bb39e8739149c3777d0325349b2a6c32c6413db`; head19lớp mới. Không dùng checkpoint đã học Cityscapes/ACDC. Recipe:20epoch cố định, không early stopping, batch2 × tích lũy4; AdamW LR6e-5,head×10,weight_decay0,01; linear polynomial,warmup20updates; AMPfloat16; BN running stats không đóng băng. Crop scale0,5–2 và flip0,5, canvas512×1024, cùng biến đổi hình học cho ảnh/mask.

Config đầy đủ cho từng lượt nằm ở `jobs/<tên>/train.yaml` sau `--prepare-only`, nên có thể kiểm trước khi dùng GPU. Dùng cùng config nguồn cho hai tài khoản; không sửa các YAML đã khóa trong kết quả khi resume.

## Augmentation và class weights

- Cơ bản: [paired_source.yaml](../configs/dg/paired_source.yaml) → [source_style.py](../src/data/augmentations/source_style.py).
- Mở rộng và corruption dev: [appearance.yaml](../configs/dg_v2/appearance.yaml) → [weather_v2.py](../src/data/augmentations/weather_v2.py).
- Biến đổi RGB **sau crop/pad/flip dùng chung**, mask không đổi. Đây là chi tiết cần giữ hoặc làm đối chứng riêng khi đổi sang K1/native-image hoặc augmentation ở feature space.
- Xác suất paired batch0,5 giống nhau giữa các lượt. Paired: trung bình hai loss clean/styled; không paired: clean CE. Mixture50/30/20 chỉ áp dụng cho **styled view**. Ma trận Khang không có KL/teacher/style bank/SHADE.
- [class_weights_reference.json](../configs/dg_v2/class_weights_reference.json):19giá trị, pixel counts, số ảnh chứa lớp, công thức, hash portable-fit. Công thức `clip(sqrt(median(f)/(f+1e-12)),0.75,2.0)`, từ fit labels gốc, bỏ255.
- Runtime tính lại thống kê từ fit, đối chiếu reference rồi ghi `data/class_weights.json` với hash manifest mount thật. File reference không phải artifact đưa thẳng vào loss vì runtime hash khác nhau theo đường dẫn.

## Chọn checkpoint và đánh giá

Sau mỗi epoch, chạy cùng595ảnh dev ở bốn dạng: sạch, low-light, noise, haze. Ba corruption dùng seed cố định theo ảnh, không đổi theo epoch. Score chọn best:

`0.5 * clean_dev_mIoU + 0.5 * mean(low_light_mIoU, noise_mIoU, haze_mIoU)`.

Điểm bằng nhau giữ checkpoint xuất hiện trước. Không dùng loss train hoặc điểm ACDC để chọn best. Dev là proxy nguồn; ACDC đã được xem ở v1 nên không gọi là test hoàn toàn chưa từng biết. Cityscapes official val500/test và ACDC không tham gia bốn lượt phát triển này.

## Phát triển style bank/SHADE trên cùng nền

Toàn có thể tạo nhánh riêng bắt đầu từ commit bàn giao, giữ nguyên split, khởi tạo, preprocessing,20epoch,seed và selection/evaluator. Bank/style statistics chỉ lấy **fit**, không lấy dev hay ACDC. Ghi rõ dùng CE hay weighted CE; SHADE+CE nên đối chiếu baseline+CE để tách tác dụng của SHADE. SHADE+weighted CE là lượt kết hợp riêng nếu muốn kiểm tra thêm.

Thêm module/config/entry point riêng cho SHADE và kiểm protocol tương ứng; không nới guard v2 để âm thầm đổi bốn lượt đối chứng đã chốt. Nếu khác augmentation ở đâu, pretrained teacher, số forward/update hoặc dữ liệu bổ sung, phải ghi lại và báo chi phí tính toán. Không nhận phép vận dụng Cityscapes→ACDC là tái lập nguyên thiết lập synthetic→real của bài SHADE.

Hai nhóm chạy song song: notebook05 (`baseline_weights`) và notebook06 (`weather_weights`). [Hướng dẫn chạy GPU và khôi phục](DG_V2_GUIDE.md).
