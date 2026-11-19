# K1 — Bộ tạo ảnh thời tiết có tham số

K1 là nền dữ liệu cho phần K2 FiLM/condition head. Nhánh này hiện có cả [K2 và notebook train](K2_TRAINING.md); notebook03 vẫn chỉ preview K1. Recipe của sáu lượt B2/DeepLab hiện hành được giữ riêng.

## Chạy thử trên Kaggle

Import [03_k1_weather_preview.ipynb](../notebooks/03_k1_weather_preview.ipynb), bật Internet, chọn CPU là đủ rồi Run All. Notebook clone nhánh `feat/k1-weather-augmentation`; có thể dùng code ZIP upload thay cho clone qua `CODE_INPUT`.

1. Tạo lưới bốn điều kiện × năm mức trên một cảnh QA vẽ bằng code với khoảng cách tổng hợp đã biết. Cảnh này kiểm tra phần mềm, không đại diện cho Cityscapes.
2. Nếu đã gắn Cityscapes, chọn tối đa ba ảnh **train** bằng seed cố định và tạo preview. Không tự chọn ảnh đẹp, không đọc ACDC.
3. Tự tìm disparity/camera theo cùng tên ảnh. Nếu thiếu, bỏ riêng sương và in lý do; vẫn xem được đêm/mưa/tuyết.
4. Xuất `k1_preview.zip`, trong đó mỗi thư mục có ảnh, `preview.png` và `metadata.json` ghi tham số/seed/provenance/checksum. Không có model training trong notebook này.

Nếu dùng stereo chưa hoàn thiện, notebook đặt rõ `INVALID_POLICY='far'`: disparity không hợp lệ được gán 200m và tỷ lệ gán được báo trong metadata. Có thể đổi thành `'error'` để dừng khi có giá trị thiếu. Far-fill chỉ là xấp xỉ, không thay thế bước hoàn thiện độ sâu của Foggy Cityscapes.

Ảnh, nhãn, disparity/camera phải do người dùng tải theo điều khoản dữ liệu. Không có dataset hoặc weights trong nhánh Git này. Ảnh real/preview giữ ở `outputs/` hoặc Kaggle Output, không commit.

## API và nhãn điều kiện

```python
from src.data.augmentations.weather import WeatherGenerator

sample = WeatherGenerator().apply(
    rgb_pil, 'fog', severity=3, seed=42,
    distance_m=aligned_camera_ray_distance,
)
augmented_rgb = sample.image
condition_id = sample.condition_id
parameters = sample.metadata
```

Thứ tự cố định: `clean=0`, `fog=1`, `night=2`, `rain=3`, `snow=4`. Mỗi sample có RGB cùng kích thước, condition ID, severity1–5 và metadata. Nhãn này mô tả phép biến đổi đã áp dụng, không phải nhãn thời tiết thật hoặc chứng nhận model nhận biết được thời tiết.

Không đổi mask. Khi vào dataset, hiệu ứng được áp dụng lên RGB native trước crop/resize/pad/flip. Sau đó hình học của ảnh và mask vẫn dùng pipeline đồng bộ cũ. Padding không bị phủ thời tiết. RNG riêng theo seed/epoch/index không làm đổi RNG hình học hoặc dropout toàn cục.

## Cơ sở kỹ thuật và giới hạn

| Hiệu ứng | Thực hiện | Giới hạn |
| --- | --- | --- |
| Sương | `I = J*t + A*(1-t)`, `t=exp(-beta*d)` trong RGB tuyến tính; d là khoảng cách tia nhìn theo mét, beta là m⁻¹ | Phụ thuộc chất lượng d; airlight cố định; chưa có depth denoising/completion/edge-aware filtering như Foggy Cityscapes |
| Đêm | Chuyển sRGB sang tuyến tính, giảm exposure/gamma/saturation, Poisson shot noise và Gaussian read noise; bloom từ vùng sáng sẵn có | Không có RAW/camera ISP, không sinh đèn đường/phản xạ thực, không tái lập đầy đủ Unprocessing |
| Mưa | Vệt theo hướng, độ dài/mật độ/opacity, blur và lớp giảm tương phản nhẹ | Lớp phủ 2D, không có hình học mưa 3D, đường ướt hoặc giọt trên ống kính |
| Tuyết | Ba lớp hạt với kích thước/blur khác nhau và haze nhẹ | Lớp phủ 2D, không có tuyết tích tụ trên mặt đất/vật thể hoặc tương tác hình học |

Các beta sương mức1–5: **0,005 / 0,01 / 0,02 / 0,03 / 0,05 m⁻¹**. Ba mức đầu tham khảo mức công bố của Foggy Cityscapes; hai mức cuối là mức stress-test do project chọn. Severity không phải độ khó tương đương giữa bốn thời tiết và không được hiệu chỉnh theo ACDC.

Đây là cài đặt mô hình ảnh đã biết và một số xấp xỉ có tham số; không tuyên bố là bộ sinh mới hoặc ảnh giống thời tiết thật chỉ dựa trên preview đẹp.

## Độ sâu: không dùng relative depth như số mét

Hàm `cityscapes_distance(disparity, camera, invalid_policy='error')`:

```text
disparity_px = (stored_uint16 - 1) / 256
Z = baseline_m * fx / disparity_px
d = Z * sqrt(1 + ((u-u0)/fx)^2 + ((v-v0)/fy)^2)
```

`stored=0` hoặc disparity không dương là invalid. Calibration phải tương ứng ảnh/disparity native; không resize disparity rồi dùng fx gốc. `d` được chặn bởi `max_distance_m` khai báo. Nguồn dữ liệu mặc định là disparity/camera gốc Cityscapes; code kiểm tra kích thước với RGB. Khi dùng NPY tự chuẩn bị, phải cung cấp **camera-ray distance theo mét**, không phải Z-depth hoặc độ sâu tương đối chưa hiệu chuẩn.

## Chạy local

```powershell
python -m scripts.preview_weather --output outputs/k1-demo
python -m scripts.preview_weather --image path/to/source.png --distance path/to/distance_m.npy --output outputs/k1-source
python -m scripts.preview_weather --image path/to/source.png --disparity path/to/disparity.png --camera path/to/camera.json --invalid-policy far --output outputs/k1-stereo
python -m scripts.preview_weather --image path/to/source.png --skip-fog --output outputs/k1-without-depth
python -m pytest tests/test_weather_k1.py -q
```

Không ghi đè thư mục preview có nội dung. Thêm `--mask` nếu muốn kiểm tra cặp ảnh-mask và hash mask không đổi. CLI mặc định reject invalid stereo, không tự far-fill.

## Tích hợp vào một thí nghiệm mới

K1 chưa tự mở một lượt GPU hoặc thay cohort DG v1. Bật bằng `data.augment.weather` trong config **mới**. Ví dụ chưa có độ sâu:

```yaml
data:
  augment:
    min_scale: 0.5
    max_scale: 2.0
    horizontal_flip: 0.5
    weather:
      conditions: [clean, night, rain, snow]
      severities: [1, 2, 3, 4, 5]
      probabilities: [0.25, 0.25, 0.25, 0.25]
```

Để có đủ năm nhãn, dùng [weather_k1.yaml](../configs/augmentations/weather_k1.yaml), điền `cityscapes_root` chứa cả `disparity/train/<city>` và `camera/train/<city>`; hoặc `distance_root` chứa `<original-image-stem>.npy`, kèm `distance_unit: meters`. Nếu chưa điền nguồn độ sâu mà bật fog, config sẽ bị từ chối. Mỗi ảnh phải có sidecar đúng tên; thiếu file sẽ dừng, không tự đổi phân bố điều kiện.

Dataset phát thêm `condition_labels` dạng int64 và `weather_severity`. Trainer cũ có thể bỏ qua các trường này để train CE với augmentation; K2 sau đó có thể dùng condition_labels cho loss phụ. Không bật đồng thời K1 với `paired_style`, `gamma_darkening` hoặc `photometric`: phép trộn sẽ làm mơ hồ nhãn điều kiện. Dataset chỉ cho K1 trên Cityscapes/synthetic **train**, không val hay target train.

## Trước khi gọi là cải tiến nghiên cứu

- Xem nhiều ảnh nguồn chọn ngẫu nhiên ở đủ mức, nhất là vùng biên/đối tượng nhỏ và stereo invalid. Local hiện chỉ có ảnh QA tổng hợp, chưa kiểm ảnh Cityscapes thật trong lần triển khai này.
- So sánh baseline, RGB augmentation có sẵn và K1 với cùng backbone/recipe; tách từng loại hiệu ứng. Profile chi phí CPU/dataloader trên ảnh native trước khi chốt GPU budget.
- Khóa policy bằng nguồn. Đánh giá trên thời tiết thật và/hoặc generator khác; không dùng chính generator train làm bằng chứng tổng quát hóa duy nhất.
- Chưa có số mIoU cải thiện, timing Kaggle hoặc xác nhận ảnh tổng hợp khớp phân bố ACDC.

## Nguồn gốc ý tưởng

- Sakaridis, Dai, Van Gool, *Semantic Foggy Scene Understanding with Synthetic Data*, IJCV2018: [trang tác giả và dữ liệu](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/). Project chỉ cài phương trình tán xạ với khoảng cách được cung cấp, không tái lập toàn bộ pipeline.
- [Cityscapes Scripts README](https://github.com/mcordts/cityscapesScripts): encoding disparity `(p-1)/256`, invalid0, camera calibration và cấu trúc dataset.
- Brooks et al., *Unprocessing Images for Learned Raw Denoising*, CVPR2019: [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Brooks_Unprocessing_Images_for_Learned_Raw_Denoising_CVPR_2019_paper.html), [code tác giả](https://github.com/timothybrooks/unprocessing). Tham khảo mô hình shot/read noise; không copy code hoặc nhận là tái lập RAW unprocessing.

Mưa/tuyết là heuristic project, không gán là cài đặt chính xác một paper vật lý đã công bố.
