# Chạy B2 DG v2 trên hai tài khoản Kaggle

**Một model B2, bốn đối chứng, chia hai notebook độc lập.** Mỗi lượt20epoch, khởi tạo mới từ ImageNet encoder và head19lớp mới; không dùng weights cũ, không học ACDC. Chưa có kết quả GPU v2.

| Tài khoản | Notebook import | Hai lượt tự chạy | ZIP nhận về |
| --- | --- | --- | --- |
|1 | [05_dg_v2_segformer_b2.ipynb](../notebooks/05_dg_v2_segformer_b2.ipynb) | `baseline` → `class_weights` | `dg_v2_segformer_b2_baseline_weights_seed42.zip` |
|2 | [06_dg_v2_b2_weather.ipynb](../notebooks/06_dg_v2_b2_weather.ipynb) | `weather_aug` → `weather_aug_class_weights` | `dg_v2_segformer_b2_weather_weights_seed42.zip` |

Notebook đã chốt nhóm tương ứng; không sửa ARM/loss/GROUP để chuyển lượt. Cả hai tài khoản dùng cùng **code ZIP mới** và cùng dataset Cityscapes. Hai thư mục working/ZIP tách biệt; không ghi chồng khi ghép kết quả.

## 1. File và input cần chuẩn bị

1. Pull nhánh `khang`. Import notebook05 vào tài khoản1; notebook06 vào tài khoản2.
2. Upload **`outputs/kaggle/project-code-dg-v2.zip` mới** làm dataset code riêng tư, thêm vào cả hai notebook. Có thể dùng cùng dataset code được chia sẻ riêng tư trong nhóm. ZIP có src/scripts/configs/tests, manifest và weights metadata; không có weights model hoặc ảnh.
3. Thêm **Cityscapes**, đủ2.975ảnh `leftImg8bit/train` và mask `gtFine/train` (labelIds hoặc trainIds). Không cần thêm ACDC, output B2/DeepLab cũ hay ZIP DG-v1.
4. Chọn GPU và bật Internet để cài dependency/tải encoder đã pin. Giữ notebook và dữ liệu kết quả có ảnh ở chế độ Private. Trainer hiện dùng một CUDA device; không có DDP để dùng đồng thời hai T4 trong một notebook.

Nếu member chỉ clone code và chưa có ZIP local:

```powershell
python -m scripts.build_dg_v2_notebook
```

Lệnh này tạo lại hai notebook và `outputs/kaggle/project-code-dg-v2.zip`; ZIP được bỏ qua bởi Git.

## 2. Đường dẫn ở cell đầu

```python
from pathlib import Path
CODE_INPUT = Path('/kaggle/input')
CITYSCAPES_ROOT = None
RESUME_INPUT = None
```

- `CODE_INPUT`: tự tìm code có đủ marker DG-v2 mới. Nếu có nhiều code input, dùng Copy path của dataset code rồi đặt `Path('...')`.
- `CITYSCAPES_ROOT=None`: tự tìm đúng một thư mục chứa cả `leftImg8bit/train` và `gtFine/train`. Nếu cần chỉ định, Copy path đến **Cityscape**, không đến thư mục train hoặc file ZIP dataset.
- `RESUME_INPUT=None`: bắt đầu mới. Chỉ đổi khi chạy tiếp một ZIP DG-v2 của **đúng notebook/nhóm này**.

Đường dẫn Cityscapes từng dùng trong project là `/kaggle/input/datasets/kavithak1388/cityscapes/Cityscape`; chỉ dùng nếu input hiện tại thực sự có đường dẫn đó. None là chế độ tự tìm, không phải bỏ qua dữ liệu.

## 3. Kiểm tra trước khi treo GPU

Chạy các cell tới `INPUT CHECK PASS`, kiểm tra:

- Model `segformer_b2`; nhóm đúng bảng ở đầu tài liệu, chỉ có hai tên thí nghiệm tương ứng.
- Fit2.380ảnh, dev595ảnh, tách theo thành phố; dev gồm bochum, erfurt, hamburg, weimar.
- Có bảng19class weights và preview ảnh sạch/low-light/noise/haze. Ảnh tổng hợp còn thấy đối tượng.
- Có dòng GPU/Torch và file cấu hình `jobs/<tên>/train.yaml`. Không đọc official-val/test/ACDC ở bước này.

Sau đó **Save Version → Save & Run All** để chạy nền. Hai notebook chạy độc lập, có thể bắt đầu cùng lúc. Theo dõi trang Logs/Output của đúng version đang chạy. Không cần giữ trình duyệt mở sau khi phiên chạy nền đã được khởi động.

## 4. Khi hoàn thành hoặc tự dừng gần hết phiên

Sau mỗi epoch committed, notebook lưu best/last/previous, optimizer/scheduler/AMP scaler/RNG và xuất lại ZIP nguyên nhóm. `suite_status.json` ghi tiến độ hai lượt. File ZIP được ghi tạm rồi đổi tên để giữ bản hoàn chỉnh trước đó nếu export bị ngắt.

- `suite_complete` + cả hai `epochs=20`: nhóm đó đã xong.
- `session_checkpoint_ready`: nhóm còn dở; tải ZIP và tiếp tục từ checkpoint. Không chạy mới từ đầu.
- Cửa sổ mỗi phiên khoảng10giờ do launcher chủ động giữ dư thời gian; đây không phải giới hạn tổng số giờ project. Chưa có timing GPU v2 để cam kết hai lượt xong trong một phiên.
- Hard timeout hoặc kill có thể bỏ qua finally; lấy ZIP hoàn chỉnh gần nhất trong Output, epoch đang dở có thể phải chạy lại.

Gửi lại **cả hai ZIP** trong bảng để nghiệm thu và ghép bảng bốn đối chứng. Mỗi nhóm có `suite_summary.csv` và `learning_curves.png` khi đã có epoch committed; score là source-dev, chưa phải điểm ACDC.

## 5. Chạy tiếp khi bị ngắt

1. Tải ZIP của nhóm đang dở về và thêm làm Kaggle Input riêng tư ở phiên mới.
2. Import **cùng notebook**05 hoặc06; thêm lại đúng Cityscapes với mount/version trước đó.
3. Đặt `RESUME_INPUT=Path('đường_dẫn_Copy_path_tới_ZIP_hoặc_thư_mục_kết_quả')`. Source đã khóa nằm trong kết quả, không cần code dataset riêng để resume.
4. Run All/Save & Run All. Notebook kiểm nhóm, source/config/manifest/mask/weights; bỏ qua epoch committed. Không đổi seed/epoch/loss hoặc sửa file kết quả để vượt kiểm hash.

Rerun riêng cell bootstrap trong cùng phiên được hỗ trợ khi input không đổi; checkpoint hiện có được giữ nguyên. Nếu thay đường dẫn rồi chạy lại cell thiết lập, notebook sẽ yêu cầu giữ kết quả và dùng phiên mới để tránh ghi chồng.

## Phương pháp đang chạy

Baseline có RGB augmentation cơ bản; lượt `class_weights` chỉ thay loss. Hai lượt `weather_aug*` dùng mixture50% cơ bản/30% low-light/20% haze trên styled view, paired batch0,5; loss clean/styled được trung bình, không có KL/SHADE. Weighted CE tính từ fit labels bằng `clip(sqrt(median(f)/(f+eps)),0.75,2.0)`.

Best được chọn bằng `0.5 clean-dev mIoU + 0.5 mean(low-light/noise/haze-dev mIoU)`. Đọc [bản bàn giao kỹ thuật](DG_V2_HANDOFF.md) để xem manifest, config B2, mức augmentation và19weights cụ thể. Hai nhóm phải cùng snapshot code và dữ liệu; kết quả cũ DG-v1 không thay được baseline của split mới.

## CLI tương đương

```powershell
python -m scripts.run_dg_v2_suite --city data/raw/cityscapes --root outputs/b2_pair1 --archive outputs/b2_pair1.zip --group baseline_weights --prepare-only
python -m scripts.run_dg_v2_suite --city data/raw/cityscapes --root outputs/b2_pair2 --archive outputs/b2_pair2.zip --group weather_weights --prepare-only
```

Trên máy có GPU, bỏ `--prepare-only` để train; giữ nguyên root/nhóm khi tiếp tục. Notebook01–04 là DG-v1 đã hoàn tất, không phải launcher mới này.
