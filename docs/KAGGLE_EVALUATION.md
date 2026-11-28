# Đánh giá sáu checkpoint trên Kaggle

Notebook: [04_evaluate_dg_cohort.ipynb](../notebooks/04_evaluate_dg_cohort.ipynb). Chỉ inference và chấm điểm; không train thêm. Dùng source và best checkpoint đã khóa trong hai suite, không cần code ZIP riêng.

## 1. Import notebook

Tạo notebook Kaggle mới → **File → Import Notebook → Upload** → chọn file `notebooks/04_evaluate_dg_cohort.ipynb` trong repo local. Đặt tên dễ nhận biết, ví dụ `dg-evaluation-b2-deeplab`.

## 2. Thêm đúng bốn input

| Input | Cần có |
| --- | --- |
| Kết quả B2 hoàn chỉnh | `dg_segformer_b2_suite_seed42 (1).zip`, hoặc thư mục kết quả đã giải nén; đủ baseline/aug/consistency 20/20/20 |
| Kết quả DeepLab hoàn chỉnh | `dg_deeplabv3plus_resnet50_suite_seed42 (1).zip`, hoặc thư mục kết quả đã giải nén; đủ 20/20/20 |
| Cityscapes | `leftImg8bit/val` và `gtFine/val`, đủ 500 ảnh và nhãn; giữ nguyên input đang có |
| ACDC | `rgb_anon_trainvaltest/rgb_anon/{fog,night,rain,snow}/val` và `gt_trainval/gt/{fog,night,rain,snow}/val`, đủ 406 cặp |

Nhấn **Add Input** ở cột phải. Với kết quả model, có thể chọn output của hai notebook train qua tab Notebooks, đúng version đã hoàn thành, thay vì upload lại ZIP lớn. Nếu dùng ZIP local, upload làm dataset riêng tư rồi thêm vào notebook. Notebook tự tìm được ZIP hoặc thư mục đã giải nén, kể cả nested `source.zip` đã thành `source/`.

Không dùng ZIP DeepLab epoch4 trên GitHub Release, ZIP B0, hoặc checkpoint ACDC supervised lịch sử. Cần **toàn bộ suite**, không chỉ thư mục `best/`, vì evaluator kiểm cả metadata/source/khởi tạo của ba nhánh.

## 3. Đường dẫn

Ở cell cấu hình:

```python
B2_INPUT = None
DEEPLAB_INPUT = None
CITYSCAPES_ROOT = None
ACDC_ROOT = None
RESUME_EVAL_INPUT = None
```

`None` ở bốn dòng đầu nghĩa là **tự tìm trong input**, không phải bỏ qua dataset. Mỗi model phải tìm thấy đúng một suite hoàn chỉnh. Nếu có nhiều bản, bấm Copy path bên input rồi thay đúng dòng:

```python
B2_INPUT = Path('/kaggle/input/<input-B2-hoan-chinh>')
DEEPLAB_INPUT = Path('/kaggle/input/<input-DeepLab-hoan-chinh>')
```

Hai đường dẫn dataset đã dùng trong project, nếu mount vẫn giữ nguyên:

```python
CITYSCAPES_ROOT = Path('/kaggle/input/datasets/kavithak1388/cityscapes/Cityscape')
ACDC_ROOT = Path('/kaggle/input/datasets/khangcules10/acdcdataset/ACDC')
```

ACDC_ROOT là thư mục cha chứa cả hai gói RGB và GT; không trỏ riêng vào `gt_trainval`. Không đoán tên mount cho hai kết quả mới: dùng auto-find hoặc Copy path thực tế.

## 4. Chạy

1. Chọn GPU trong Notebook settings, bật Internet để cài dependencies; giữ notebook Private.
2. Chạy tới hết mục **2. Kiểm input ảnh–mask và tạo manifest đánh giá**. Phải thấy `PASS: all six audited best checkpoints frozen` và `PASS: files/labels and five-domain geometry smoke`.
3. Chọn **Save Version → Save & Run All** để chạy nền trong phiên mới. Không cần đổi model/arm hoặc chạy từng notebook model.
4. Luồng tự chạy 12 job: sáu checkpoint × Cityscapes val500 và ACDC val406. Giữ FP32, input512×1024 giữ tỷ lệ, output native, ignore255 và không TTA.

Save & Run All chạy từ đầu trong phiên mới; bước kiểm thử tương tác không làm nó bắt đầu từ cell giữa. [Tài liệu Kaggle](https://www.kaggle.com/docs/notebooks).

Notebook kiểm source/weights/config/split hash trước target. Không học target, không cập nhật BatchNorm và không chọn lại best theo ACDC. Chưa có timing cho toàn bộ 12 job; không cam kết vừa một phiên. ZIP được cập nhật sau mỗi job hoàn chỉnh, không phải mỗi ảnh.

## 5. Nhận kết quả hoặc tiếp tục nếu bị dừng

Khi đủ 12 job, log báo `COMPLETE: 12 evaluations`. Tải `/kaggle/working/dg_evaluation_seed42.zip` ở tab Output. Gói chứa:

- `summary.csv`: mIoU chung và timing của 12 job.
- `per_weather.csv`, `per_class.csv`, `strategy_deltas.csv`: từng thời tiết, từng lớp và chênh lệch phương pháp.
- `comparison.png`: biểu đồ so sánh.
- `evaluations/`: metrics/calibration, confusion matrix từng ảnh, panel RGB/GT/dự đoán/độ tự tin/lỗi.
- `evaluation_plan.json`, `cohort.json`, manifests và `evaluation_status.json` để kiểm chứng.

Nếu chưa đủ 12 job, giữ ZIP gần nhất. Phiên mới thêm lại cùng bốn input và ZIP đánh giá đó, đặt `RESUME_EVAL_INPUT` theo Copy path. Job đã hoàn tất chỉ được bỏ qua khi hash/số lượng khớp; job dở chạy lại từ đầu, dữ liệu dở giữ trong `incomplete_attempts/`. Dùng cùng mount dataset để reuse cache; khác mount sẽ dừng thay vì trộn kết quả.

Giữ kết quả riêng tư vì chứa panel từ dataset. Gửi lại ZIP đánh giá để nghiệm thu. Source-val scores tính native ở đây có thể khác điểm monitor dùng chọn best khi train; giữ tên bảng và cách tính rõ ràng.

[Kiểm thử notebook local](../results/dg_evaluation_notebook/README.md) · [Danh tính sáu checkpoint](../results/dg_cohort_seed42/README.md).
