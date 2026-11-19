# Train K1 + K2 trên Kaggle

Nhánh Git: **`feat/k1-weather-augmentation`**. K1 và K2 nằm chung nhánh; không cần lấy nhánh K2 WIP cũ.

## Bắt đầu

1. Tải/import [04_k2_weatherseg_b0_train.ipynb](../notebooks/04_k2_weatherseg_b0_train.ipynb) vào Kaggle, bật **GPU + Internet**.
2. Gắn Cityscapes gồm `leftImg8bit/train`, `leftImg8bit/val`, `gtFine/train`, `gtFine/val`. Root có bốn thư mục này được tự tìm; điền `CITYSCAPES_ROOT` nếu có nhiều ứng viên.
3. Run All. Notebook lấy code cùng nhánh, cài dependencies phụ, chạy smoke CUDA/AMP nhỏ, chuẩn bị nhãn trainIds, kiểm dữ liệu rồi train.
4. Mặc định **một lượt `ARM='k2'`, seed42, 20epoch, batch2/accumulation4**, canvas512×1024. Mỗi epoch lưu checkpoint và xuất ZIP. Model chọn best bằng **Cityscapes val500 mIoU**.

Không gắn ACDC để train. Các checkpoint supervised lịch sử không được dùng làm khởi tạo. Encoder ImageNet `nvidia/mit-b0` được pin revision và kiểm SHA-256; decoder, FiLM và condition head là phần học cho thí nghiệm này. PyTorch/TorchVision CUDA của Kaggle được giữ nguyên; yêu cầu PyTorch>=2.6 và TorchVision tương thích.

Nếu không muốn clone, upload `outputs/kaggle/project-code.zip` từ `python -m scripts.build_kaggle_notebooks`, rồi đặt `CODE_INPUT` tới ZIP hoặc thư mục đã giải nén. Chỉ ZIP code mới có cả K1/K2 mới dùng được.

## Sương mù và dữ liệu độ sâu

| `WEATHER_MODE` | Hành vi |
| --- | --- |
| `'auto'` mặc định | Có đầy đủ độ sâu thì dùng năm điều kiện; không có nguồn độ sâu thì dùng clean/night/rain/snow và in rõ fog bị loại |
| `'full'` | Bắt buộc đủ clean/fog/night/rain/snow; thiếu hoặc sai độ sâu thì dừng trước train |
| `'no-fog'` | Chủ động chỉ train clean/night/rain/snow |

Để train đủ năm điều kiện:

```python
WEATHER_MODE = 'full'
DEPTH_ROOT = '/kaggle/input/your-cityscapes-depth-root'
# DEPTH_ROOT chứa disparity/train/<city>/*.png và camera/train/<city>/*.json
# Nếu nằm cùng CITYSCAPES_ROOT thì DEPTH_ROOT có thể để None.
```

Hoặc đặt `DISTANCE_ROOT` tới thư mục có `<original-image-stem>.npy` là **camera-ray distance theo mét**. Không dùng relative depth hay Z-depth chưa chuyển đổi. Chỉ chọn một nguồn độ sâu.

Launcher kiểm tất cả sidecar train, geometry và encoding trước khi train. Khi đã khai báo nguồn độ sâu, thiếu một file cũng dừng; không tự bỏ ảnh hoặc thay điều kiện ngẫu nhiên. Preflight stereo phải giải mã nhiều PNG native nên có thể mất thời gian CPU.

`INVALID_POLICY='far'` trong notebook là lựa chọn được khai báo: stereo invalid được gán200m; đây là xấp xỉ của K1, không phải hoàn thiện depth như Foggy Cityscapes. Chọn `'error'` nếu muốn reject ảnh có disparity invalid.

Head luôn có năm logits theo ID `clean0/fog1/night2/rain3/snow4`. Khi chạy không fog, **không có nhãn dương fog**; không nhận là đã huấn luyện nhận dạng đủ năm điều kiện. `weather_preflight.json` ghi điều kiện thực sự bật. Loss/accuracy condition chỉ đo nhận dạng phép biến đổi K1, không phải thời tiết thật.

## K2 thực hiện điều gì?

Sau mỗi stage của MiT-B0:

```text
statistics = concat(mean_spatial(feature), variance_spatial(feature))
(delta_gamma, beta) = MLP(statistics)
feature_out = (1 + delta_gamma) * feature + beta
```

- Feature sau FiLM đi vào stage tiếp theo và decoder. Không chỉ sửa các skip feature sau encoder.
- Projection cuối MLP được khởi tạo zero: FiLM ban đầu là identity. Copy cùng weights B0 thì logits/feature ban đầu khớp chính xác trên CPU QA.
- Stage4 global pooling -> linear head5logits. Nhãn condition lấy trực tiếp từ K1, không dùng nhãn ACDC.
- Loss mặc định: **CE phân đoạn + 0,1 × CE điều kiện**. Ignore255 chỉ áp dụng semantic mask; condition CE ở cấp ảnh.
- FiLM và condition head dùng nhóm learning rate head×10; backbone dùng LR6e-5. Thống kê mean/variance tính FP32 khi dùng AMP.
- Model có **3.817.720 tham số**; B0 gốc có3.719.027, tăng98.693 (~2,65%). Không có teacher EMA/K3 hoặc test-time adaptation trong phần này.

Giao diện `model(pixel_values=...).logits`, `save_pretrained` và registry/evaluator/inference được giữ. Output mới có thêm `condition_logits`; có thể truyền `labels` và `condition_labels` để lấy loss. Checkpoint có marker `project_model=weatherseg_b0` để tránh nạp nhầm như B0 thuần và làm rơi weights FiLM.

## Các nhánh ablation trong cùng notebook

Chọn `ARM` trước khi bắt đầu một workspace mới; notebook không tự chạy cả bốn lượt.

| ARM | Mô hình | Ảnh K1 | Loss điều kiện |
| --- | --- | --- | --- |
| `baseline` | B0 thuần | Không | Không |
| `aug` | B0 thuần | Có | Không |
| `film` | B0 + FiLM + head5logits | Có | Tắt; head phụ không được huấn luyện |
| `k2` | B0 + FiLM + head5logits | Có | CE×0,1 |

Các arm độc lập bắt đầu từ encoder ImageNet và segmentation head mới, cùng seed, dữ liệu/recipe. Không học nối từ baseline sang k2. Registry copy weights B0 vào K2 và cô lập RNG tạo tham số mới để không vô tình đổi khởi tạo backbone/head hoặc RNG dropout của control.

Giữ cùng `WEATHER_MODE`/depth/severity giữa `aug`, `film`, `k2`. Điều kiện thời tiết được chọn online bằng RNG theo seed/epoch/index. Bảng này là thiết kế ablation, chưa có bằng chứng tăng mIoU.

## ZIP, thời gian phiên và resume

- Tệp tải về: `/kaggle/working/k2_k2_seed42.zip` khi ARM=k2. Checkpoint best/last, optimizer/scheduler/scaler/RNG, config, manifest và source snapshot đều nằm trong ZIP.
- Trong phiên mới, gắn ZIP cũ làm input, đặt `RESUME_INPUT` tới ZIP hoặc thư mục Kaggle đã giải nén; giữ `ARM`, `SEED`, `EPOCHS`, `BATCH_SIZE` như ban đầu.
- Notebook tự khôi phục source snapshot và weather settings của run cũ. Gắn Cityscapes/depth ở **cùng đường dẫn mount**; không thay nội dung input giữa phiên.
- Runner tiếp tục từ epoch đã commit. Epoch dở bị lặp lại; không tự rút20epoch hoặc đổi scheduler để kịp phiên.
- Cửa sổ notebook tự đặt9giờ trừ thời gian setup, runner dùng timing gần đây để ước lượng có đủ một epoch tiếp không. Đây không phải đọc quota Kaggle và không đảm bảo tránh mọi hard-kill. Giữ ZIP hoàn chỉnh mới nhất trước khi kết thúc phiên.

Không resume bằng code ZIP mới hoặc sửa config trong archive. Config/source/manifest/preflight có hash để phát hiện thay đổi. Hash này không phải full content hash của toàn bộ raw images/depth; người dùng phải giữ nguyên phiên bản dữ liệu.

Log `history.jsonl` có tổng train loss, Cityscapes val mIoU/CE và `condition_train` gồm CE/accuracy/counts cho năm ID. `status.json` cho biết run đã đủ epoch hay đang chờ phiên tiếp theo. `frozen_candidate.json` xuất khi train hoàn thành; không tự evaluate ACDC hoặc nhập vào cohort sáu lượt B2/DeepLab cũ.

## Local QA và giới hạn

```powershell
python -m scripts.verify_k2 --output outputs/k2-smoke.json
python -m pytest tests/test_weatherseg_k2.py tests/test_k2_notebook.py -q
python -m scripts.build_kaggle_notebooks
```

Local chỉ có CPU: đã kiểm identity, gradient, loss, K1 input và checkpoint/resume bằng ảnh nguồn giả lập. CUDA/AMP smoke sẽ chạy trên GPU Kaggle trước lượt chính; full GPU training, hội tụ, VRAM/latency và mIoU chưa được đo ở local.

Tham khảo FiLM: Perez et al., [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871), AAAI2018. Project dùng self-conditioning từ thống kê ảnh, không phải tái lập bài visual reasoning gốc. [K1 và giới hạn mô phỏng](K1_WEATHER.md).
