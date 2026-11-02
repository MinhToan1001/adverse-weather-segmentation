# Nghiệm thu task03 của Khang và Toàn

Ngày ghi nhận: 01/11/2026. Ngày trong kế hoạch gốc: 28/10/2026.
Nhánh: `khang`. Cả hai phần triển khai đã hoàn thành và được kiểm thử bằng dữ liệu
tổng hợp. Toàn bộ checkout đạt **74 passed**, không skip/failure, trong 49.17 giây.

**Dataset thật chưa sẵn sàng theo xác nhận của người dùng.** Lần làm task03 này
không tải, giải nén hay đọc pixel Cityscapes/ACDC thật, không tạo manifest full
cho dữ liệu thật và không chứng nhận Gate B đã hoàn tất.

## Khang task03

- Preprocessing giữ tỉ lệ; ảnh bilinear, mask nearest; giữ class 0 và ignore 255.
- Train scale/crop/pad/flip đồng bộ giữa ảnh và mask; normalize ImageNet.
- Validation deterministic, không lấy số ngẫu nhiên; hỗ trợ mask gốc và fit box
  để evaluator loại padding trước khi đưa logits về kích thước gốc.
- Kiểm tra augmentation hữu hạn, khoảng scale hợp lệ và xác suất flip `0..1`;
  từ chối option bị viết sai.
- Loader dùng chung quy tắc CSV/path với validator; BOM được hỗ trợ.
- Training từ chối source val và toàn bộ target-domain rows. Trainer kiểm tra
  tồn tại file và trùng ảnh/mask giữa train/val trước khi tạo model.

## Toàn task03

- Generator Cityscapes và ACDC ghép đúng ID, thứ tự ổn định, hỗ trợ mask root
  tách biệt với raw và `--split` để tạo riêng train hoặc val.
- Full manifest yêu cầu đủ số cặp theo protocol; subset cần bật rõ
  `--allow-subset`. Thiếu mask hoặc mask không có ảnh tương ứng đều báo lỗi.
- Generator kiểm tra tất cả split yêu cầu trước khi ghi; bảo vệ CSV đã có bằng
  flag `--overwrite`.
- Validator dùng chung được qua CLI và Python: CSV malformed/field trống,
  domain/split, ID ghép sai theo tên chuẩn, đường dẫn trùng, thiếu file,
  kích thước lệch, mask màu/palette và trainId không hợp lệ.
- ACDC chỉ kiểm tra metadata/tồn tại file trước frozen evaluation; thao tác
  `--check-labels` trên ACDC bị chặn trước khi mở pixel.
- Hướng dẫn đầy đủ: [DATA_PREPARATION.md](DATA_PREPARATION.md).

## Kiểm thử và bằng chứng

- `tests/test_data_task03.py`: 42 trường hợp mới, gồm chạy CLI tạo manifest rồi
  validate trong subprocess, kiểm tra ghi đè và preflight nhiều split.
- Cùng các test dataset/preparation có sẵn: tổng cộng 52 kiểm tra dữ liệu.
- `results/day03/full-suite.txt` và `full-suite.xml`: 74 passed cho toàn repo,
  bao gồm các test pretrained thật bật bằng `RUN_PRETRAINED_TESTS=1`.
- `results/day03/data-tests.txt` và `data-tests.xml`: lượt trung gian đầu tiên,
  50 passed; lượt full-suite sau đó gồm thêm hai regression test và bản sửa cuối.
- `results/day03/environment.json`: môi trường, trạng thái dataset và kết quả.
- `results/day03/source-sha256.json`: fingerprint code, config và test đã kiểm tra.

Lệnh chạy full suite từ repository root:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:RUN_PRETRAINED_TESTS = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
.venv/Scripts/python.exe -m pytest -v
```

## Phần chờ dữ liệu thật

Sau khi tải/giải nén xong: đối chiếu layout và encoding với tài liệu đi kèm gói,
chuyển nhãn Cityscapes bằng mapping chính thức, tạo manifest thật, chạy validator
và QA dữ liệu nguồn. ACDC vẫn chỉ QA metadata theo protocol trước frozen pass.
Số lượng file hoặc giá trị nhãn hợp lệ riêng lẻ không đủ chứng minh dữ liệu đúng;
không suy diễn trạng thái tải từ các file cũ có sẵn trong workspace.

Các thay đổi vẫn lưu local, chưa commit/push. Những thay đổi có sẵn từ trước
được giữ nguyên; kết quả full-suite áp dụng cho toàn bộ checkout hiện tại.
