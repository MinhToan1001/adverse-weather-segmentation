# Nghiệm thu ngày 02

Ngày thực hiện: 31/10/2026. Ngày trong kế hoạch gốc: 27/10/2026.
Người thực hiện đảm nhận cả nhiệm vụ Khang và Toàn. Nhánh làm việc: `khang`.

Hai task ngày 02 đã đạt tiêu chí kỹ thuật. Bộ test riêng ngày 02: **18 passed**;
toàn bộ test của checkout hiện tại: **32 passed**, không skip, không failure.
Đây là kiểm chứng code bằng tensor tổng hợp và checkpoint ImageNet, chưa phải
kết quả huấn luyện hoặc mIoU trên Cityscapes/ACDC.

## Khang task02

- Dùng chính `src.models.segformer.make_model` của pipeline để tạo model.
- Ghim checkpoint `nvidia/mit-b0` tại revision
  `80983a413c30d36a39c20203974ae7807835e2b4` trong config baseline;
  model factory chuyển revision này cho Hugging Face.
- So sánh tất cả tensor encoder với encoder nạp riêng từ cùng checkpoint: bằng nhau;
  encoder tham chiếu không có missing/mismatched keys.
- Forward input `(1,3,512,512)` cho logits `(1,19,128,128)`;
  bilinear upsample cho `(1,19,512,512)`, mọi giá trị hữu hạn.
- Input chữ nhật `(1,3,64,96)` cho logits `(1,19,16,24)`.
- `semantic_loss_ignore_index=255`; loss của model bằng cross entropy
  chỉ trên pixel hợp lệ khi nửa mask là nhãn 255.
- Default CI có kiểm tra forward B0 offline. Test trọng số thật được bật bằng
  `RUN_PRETRAINED_TESTS=1` và đã chạy thành công trong lần nghiệm thu này.

MiT-B0 cung cấp encoder pretrained ImageNet; decoder/head 19 lớp được khởi tạo
mới. Cảnh báo Hugging Face về head chưa được huấn luyện là phù hợp.
Nguồn: [model card chính thức](https://huggingface.co/nvidia/mit-b0).

## Toàn task02

`tests/test_metrics.py` có 13 trường hợp kiểm tra:

- Confusion matrix tính tay và mIoU = 0.5 với một pixel ignore.
- Dự đoán hoàn hảo đủ 19 lớp cho mIoU = 1.
- Lớp bị bỏ sót hoàn toàn vẫn có IoU = 0.
- Lớp chỉ xuất hiện do false positive vẫn có IoU = 0.
- Thay dự đoán trên pixel 255 bằng 0, 18, 255 hoặc -1 không đổi metric.
- Toàn bộ nhãn là 255: tổng confusion = 0, mIoU = NaN, per-class = None.
- Cộng confusion toàn split trước khi tính mIoU, không trung bình mIoU từng ảnh.
- Dự đoán -1, 19 hoặc 255 trên pixel hợp lệ bị từ chối.

Đã làm rõ mục 2.3 của protocol để khớp hành vi metric hiện có: chỉ loại lớp
khi union bằng 0. Không đổi thuật toán hoặc kết quả metric.

## Trạng thái repository và dữ liệu

- GitHub remote đúng: `MinhToan1001/adverse-weather-segmentation`.
- Khi kiểm tra, remote `main` ở `5854acf`, remote `khang` ở `b5ccbac`.
  Local `khang` đã fast-forward tới `5854acf`; remote chưa được cập nhật.
- Checkout có nhiều thay đổi chưa commit từ trước ở data/training/evaluation.
  Các thay đổi đó được giữ lại; kết quả 32 test áp dụng cho toàn checkout này.
- Thống kê **tên file** tại `data/raw`: Cityscapes có 2.975 labelIds train
  và 500 labelIds val; chưa có ảnh `leftImg8bit`, chưa có thư mục `acdc`.
  Số file nhãn chưa chứng minh tính toàn vẹn hoặc chất lượng dữ liệu.
- Chưa có manifest dữ liệu thật. Việc tải, chuyển trainId và QA dataset thuộc
  các bước tiếp theo; ngày 02 không phụ thuộc chúng.
- Môi trường: Python 3.11.5, torch 2.10.0+cpu, transformers 4.57.6,
  pytest 9.2.0. CUDA không khả dụng trong môi trường hiện tại.

## Lệnh chạy lại trong PowerShell

Chạy từ thư mục repository. Checkpoint đã có trong Hugging Face cache trên máy.

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:RUN_PRETRAINED_TESTS = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
.venv/Scripts/python.exe -m pytest tests/test_metrics.py tests/test_pretrained_forward.py -v
.venv/Scripts/python.exe -m pytest -v
```

Trên máy chưa có checkpoint, bỏ biến `HF_HUB_OFFLINE` để Hugging Face tải trọng số
ở revision đã ghim. Nếu không bật `RUN_PRETRAINED_TESTS`, bốn trường hợp kiểm tra
checkpoint thật sẽ skip; lượt chạy đó không đủ nghiệm thu phần pretrained.

## Bằng chứng

- `results/day02/day02-tests.txt` và `.xml`: 18 passed trong 15.97 giây.
- `results/day02/full-suite.txt` và `.xml`: 32 passed trong 10.86 giây.
- `results/day02/environment.json`: môi trường, seed, revision và SHA-256 trọng số.
- `results/day02/source-sha256.json`: hash code/config/test của checkout đã kiểm tra.

Có thể chuyển hai thẻ Khang-task02 và Toàn-task02 sang Done với các bằng chứng
trên. Trạng thái bảng công việc chưa được cập nhật từ chat này.
