# Chuẩn bị dữ liệu và preprocessing

Task03 hoàn thiện code và kiểm thử bằng dữ liệu tổng hợp. Dataset thật được xem
là **chưa sẵn sàng** theo xác nhận của người dùng. Các lệnh dưới đây chỉ chạy sau
khi tải và giải nén xong vào đúng cấu trúc; chúng không tự tải dataset.

## Preprocessing dùng chung

`src/data/dataset.py` đọc CSV bằng cùng quy tắc với validator. Đường dẫn ảnh/mask
trong CSV là đường dẫn tuyệt đối hoặc tương đối với **repository root**, không
phải thư mục chứa CSV. Bản thân đường dẫn CSV và các đối số CLI tương đối vẫn
được truyền từ thư mục chạy lệnh. Nên chạy các lệnh dưới đây từ repository root.

| Bước | Train Cityscapes | Validation |
|---|---|---|
| Kiểm tra đầu vào | Mask integer một kênh, IDs `0..18/255`, cùng kích thước ảnh | Như train |
| Resize | Giữ tỉ lệ; fit vào canvas rồi nhân scale ngẫu nhiên `0.5..2.0` | Fit vào canvas, giữ tỉ lệ |
| Nội suy | Ảnh bilinear; mask nearest | Như train |
| Crop/pad | Crop ngẫu nhiên nếu vượt canvas; đặt ngẫu nhiên nếu nhỏ hơn canvas | Pad ở giữa, không random crop |
| Flip | Lật ngang ảnh và mask đồng thời, xác suất `0.5` | Không flip |
| Chuẩn hóa | RGB `/255`, trừ mean `[0.485,0.456,0.406]`, chia std `[0.229,0.224,0.225]` | Như train |
| Đầu ra | Image float32 CHW, labels int64 HW | Image cùng canvas; evaluator nhận labels gốc và `fit_box` |

Canvas baseline là `[height, width] = [512, 1024]`. Padding ảnh dùng màu mean
ImageNet làm tròn về uint8, nên sau normalize gần bằng 0; padding mask dùng 255.
Không giảm label IDs: lớp road vẫn là 0. Không dùng color jitter hoặc augmentation
từ ACDC. Seed được đặt trong trainer, bao gồm seed cho các DataLoader worker.

`native_labels=True` chỉ dùng cho evaluation: giữ nguyên kích thước mask gốc.
Evaluator đưa logits lên canvas, bỏ vùng padding theo `fit_box`, rồi nội suy
về kích thước nhãn gốc trước argmax. Collate hỗ trợ nhiều kích thước mask trong
cùng batch. Không đặt `native_labels=True` khi train.

## Cityscapes

Cấu trúc raw và processed tách biệt:

```text
data/raw/cityscapes/
  leftImg8bit/{train,val}/{city}/{id}_leftImg8bit.png
  gtFine/{train,val}/{city}/{id}_gtFine_labelIds.png
data/processed/cityscapes/
  gtFine/{train,val}/{city}/{id}_gtFine_labelTrainIds.png
```

Quy ước tên lấy từ [Cityscapes Scripts chính thức](https://raw.githubusercontent.com/mcordts/cityscapesScripts/master/README.md).
Không dùng color mask, instanceIds hoặc raw labelIds làm training target.
Script chuyển nhãn hiện có viết ra cây processed và giữ nguyên raw.

```powershell
.venv/Scripts/python.exe scripts/convert_labels.py --root data/raw/cityscapes --output-root data/processed/cityscapes
.venv/Scripts/python.exe scripts/make_manifest.py --dataset cityscapes --root data/raw/cityscapes --mask-root data/processed/cityscapes
.venv/Scripts/python.exe scripts/validate_manifest.py --manifest data/manifests/cityscapes_train.csv data/manifests/cityscapes_val.csv --check-files --check-labels
```

Generator mặc định yêu cầu đủ 2.975/500 cặp train/val và ghép theo ID, không ghép
hai danh sách chỉ bằng vị trí. Mask thiếu ảnh hoặc ảnh thiếu mask đều báo lỗi.
CSV sắp xếp ổn định, gồm `image,mask,domain,split`. Mọi split yêu cầu được kiểm
tra trước khi ghi CSV đầu tiên. Lệnh train cũng kiểm tra tồn tại file và trùng
đường dẫn giữa train/val trước khi tạo model.

Có thể dùng `--split train` hoặc `--split val` để tạo riêng một split.
`--allow-subset` dành cho bộ dữ liệu nhỏ có chủ đích, vẫn yêu cầu ghép 1–1.
Với subset, cây ảnh và cây mask phải cùng chứa đúng subset đó. Lưu CSV subset
vào thư mục riêng qua `--output-dir`; không dùng nó như manifest full baseline.
Generator từ chối ghi đè CSV đã có, trừ khi truyền rõ `--overwrite` sau khi xem lại.

## ACDC

Adapter hiện hỗ trợ layout sau; cần đối chiếu README đi kèm gói tải thật khi
dữ liệu sẵn sàng. Bộ test chỉ mô phỏng cấu trúc này, chưa xác minh gói trên máy.

```text
data/raw/acdc/
  rgb_anon/{fog,night,rain,snow}/val/{sequence}/{id}_rgb_anon.png
  gt/{fog,night,rain,snow}/val/{sequence}/{id}_gt_labelTrainIds.png
```

```powershell
.venv/Scripts/python.exe scripts/make_manifest.py --dataset acdc --root data/raw/acdc
.venv/Scripts/python.exe scripts/validate_manifest.py --manifest data/manifests/acdc_val.csv --check-files
```

Mặc định kiểm tra 406 cặp, gồm fog 100, night 106, rain 100, snow 100.
Generator và `--check-files` chỉ đọc tên/đường dẫn và kiểm tra tồn tại file.
`--check-labels` với ACDC bị từ chối trước khi mở pixel. Theo protocol hiện tại,
ACDC chỉ được mở để đánh giá sau khi baseline đã đóng băng; không đưa ACDC vào
training hoặc chọn hyperparameter. Xem [protocol](experiment_protocol.md).

## Validator kiểm tra gì

- Không có flag: kiểm tra CSV, field bắt buộc, domain/split hợp lệ, đường dẫn
  trùng, image=mask, tên ID ghép sai và split/domain mâu thuẫn trong cây chuẩn.
- `--check-files`: thêm kiểm tra file tồn tại, không giải mã pixel.
- `--check-labels`: thêm đọc ảnh/mask nguồn, kiểm tra giải mã được, cùng kích
  thước, mask integer một kênh và giá trị `0..18/255`.
- Có thể truyền nhiều CSV sau `--manifest` để phát hiện ảnh hoặc mask dùng lặp
  giữa các split. UTF-8 có BOM cũng được hỗ trợ.

Path validation không phát hiện hai file khác tên nhưng trùng nội dung. Giá trị
mask hợp lệ cũng không chứng minh ánh xạ nhãn đúng: raw IDs bị đổi tên vẫn có
thể chứa toàn giá trị nhỏ. Nguồn nhãn và QA dữ liệu thật phải được xác minh riêng.

## Chạy kiểm thử không cần dataset

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
.venv/Scripts/python.exe -m pytest tests/test_data_task03.py tests/test_dataset.py tests/test_preparation.py -v
```

Test tự tạo ảnh, mask và cây thư mục tạm. Không đọc, tải hoặc giải nén Cityscapes
hay ACDC thật. Test ACDC cố tình dùng file không phải PNG và chặn `Image.open`
để chứng minh quá trình tạo/kiểm tra manifest không giải mã pixel.

## Cấu trúc đã xác minh trên máy

Hai gói ACDC thực tế được giải nén riêng:

```text
data/raw/ACDC/rgb_anon_trainvaltest/rgb_anon/{condition}/{split}/{sequence}/
data/raw/ACDC/gt_trainval/gt/{condition}/{split}/{sequence}/
```

README đi kèm xác nhận `labelTrainIds` là Cityscapes trainIDs. `labelIds`,
`labelColor`, `invIds` và `invGray` là các encoding khác; không dùng thay mask
semantic trainId. `invIds` là nhãn invalid 0/1, không phải nhãn semantic 19 lớp.
Không tự chuyển các giá trị của invalid mask thành training classes.

Lệnh tạo ACDC manifest với layout hiện tại, chạy khi CSV chưa tồn tại:

```powershell
.venv/Scripts/python.exe scripts/make_manifest.py --dataset acdc --root data/raw/ACDC/rgb_anon_trainvaltest --mask-root data/raw/ACDC/gt_trainval
.venv/Scripts/python.exe scripts/verify_dataset_readiness.py --out results/readiness-repeat.json
```

Lệnh verify tái tạo CSV trong thư mục tạm, so từng byte với ba manifest hiện tại,
kiểm tra split/file, dung lượng và metadata provenance. Nó chặn Image.open trong
suốt pha kiểm tra metadata. Không gọi `dataset[i]` trên ACDC trước frozen pass.

Thông tin provenance hiện có là README/license đi kèm; chưa có checksum từ nhà
cung cấp hoặc release ID xác thực. Giới hạn này được giữ trong báo cáo. Điều
khoản gốc vẫn áp dụng khi dùng trên dịch vụ GPU; đặt Private không tự cấp quyền
lưu trữ bên thứ ba. Không đưa raw data hoặc visual QA lên repo/code ZIP.
