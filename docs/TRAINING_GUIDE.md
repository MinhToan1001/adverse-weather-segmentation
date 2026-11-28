# Chạy lại DG v1 trên Kaggle

**Đang phát triển B2 DG v2:** dùng [hướng dẫn notebook05/06](DG_V2_GUIDE.md), mỗi notebook hai đối chứng20epoch. Phần dưới giữ recipe DG v1 đã có kết quả.

[Hai notebook](../notebooks/README.md) · [Phương pháp](DG_METHOD.md)

## Luồng sử dụng

Mỗi model có một notebook. Cả B2 và DeepLab đều tự chạy **profiling → baseline20 → aug20 → consistency20**. Không chọn ARM và không đổi PROFILE_ONLY. Các arm khởi tạo độc lập từ cùng ImageNet encoder/head seed trong từng model; không học nối từ baseline.

1. Chạy `python -m scripts.build_kaggle_notebooks`; upload `outputs/kaggle/project-code.zip` mới vào Kaggle.
2. Import notebook B2 hoặc DeepLab. Gắn cùng bộ Cityscapes có train/val, bật GPU/Internet. Không cần ACDC cho training.
3. CODE_INPUT mặc định auto-find; đặt rõ dataset code nếu có nhiều bản. CITYSCAPES_ROOT=None tự tìm theo leftImg8bit/gtFine train/val. Lượt mới RESUME_INPUT=None.
4. Save Version → Save & Run All. Notebook chuẩn bị dữ liệu một lần, profile model một lần; khi profile đạt thì tự chạy tiếp cả ba chiến thuật. Không dừng để bắt đổi cờ.
5. Giữ ZIP `dg_<model>_suite_seed42.zip`. suite_complete nghĩa là model đó đủ3×20epoch; hai suite hoàn chỉnh tạo đủ6candidate.

`None` có hai nghĩa theo biến: CITYSCAPES_ROOT=None yêu cầu tự tìm dataset; RESUME_INPUT=None bắt đầu lượt mới. Khi có nhiều bản code/dataset, đặt đường dẫn rõ ràng, ví dụ:

```python
CODE_INPUT = Path('/kaggle/input/datasets/khangcules10/codizip')
CITYSCAPES_ROOT = Path('/kaggle/input/datasets/kavithak1388/cityscapes/Cityscape')
RESUME_INPUT = None
```

Đây là ví dụ; dùng nút copy path của **input thực tế**. CODE_INPUT nhận ZIP hoặc thư mục đã giải nén chứa src/scripts/configs; CITYSCAPES_ROOT phải chứa leftImg8bit và gtFine, đủ train/val. Cùng code ZIP dùng cho hai notebook.

## Nếu phiên bị ngắt hoặc hết thời gian

Mỗi epoch hoàn thành cập nhật một ZIP chung. Gần hết cửa sổ10giờ, suite lưu trạng thái session_checkpoint_ready. Phiên mới: thêm ZIP suite của đúng model, giữ cùng dataset version/mount, chỉ đặt RESUME_INPUT và Run All.

Profile đã hoàn thành được kiểm hash và không chạy lại. Epoch đã chốt được xác minh, không train lại; epoch đang dở chạy lại. Nếu source.zip được Kaggle giải nén, notebook tái tạo từ các file đã kiểm hash. Không dùng ZIP từng arm hoặc ZIP supervised lịch sử làm resume suite mới.

Kaggle hard-kill có thể không chạy finally; cần giữ ZIP hoàn chỉnh gần nhất. Quota30giờ chung và cửa sổ mỗi phiên là hai giới hạn khác nhau. Notebook không đọc được quota chung giữa hai phiên/model và chưa bảo đảm sáu lượt vừa30giờ. Nó giữ nguyên matrix và20epoch, không tự bỏ chiến thuật để báo hoàn thành.

## Theo dõi kết quả

- suite_status.json: tiến độ từng arm, epoch, best và Cityscapes val mIoU dùng selection.
- gpu_profile.json/profile_seal.json: timing/VRAM và hash; weights pilot bị bỏ, không dùng train chính.
- jobs/baseline, jobs/aug, jobs/consistency: config, logs, initial/best/last/previous của từng lượt.
- data/: converted masks dùng chung một lần trong suite; mỗi arm giữ CSV đã khóa của nó.

Profile dự trù train/val, chưa gồm toàn bộ checkpoint/ZIP/setup hoặc biến thiên runtime. Đọc ETA và quota thực tế của Kaggle; không dùng CPU QA timings làm GPU ETA. λ0,1 và p_pair0,5 là recipe ứng viên cố định, chưa có bằng chứng tối ưu chất lượng.

## CLI

```powershell
python -m scripts.run_dg_suite --city data/raw/cityscapes --model segformer_b2 --root outputs/dg/b2_suite --archive outputs/dg_b2_suite.zip
python -m scripts.run_dg_suite --city data/raw/cityscapes --model deeplabv3plus_resnet50 --root outputs/dg/dl_suite --archive outputs/dg_dl_suite.zip
```

run_dg vẫn là API/CLI từng arm cho phát triển và QA; notebook thông thường không yêu cầu người dùng gọi từng arm.

## Freeze và evaluate sau cả hai suite

**Cách chạy trên Kaggle bằng một lần Run All:** [notebook04](../notebooks/04_evaluate_dg_cohort.ipynb), thêm hai suite hoàn chỉnh + Cityscapes + ACDC. [Hướng dẫn import/input/kết quả](KAGGLE_EVALUATION.md). Các lệnh CLI dưới đây phục vụ chạy thủ công.

Đưa hai suite hoàn chỉnh vào cùng môi trường. Lệnh tự tìm sáu arm. Những đường dẫn trong dấu `<...>` dưới đây cần thay bằng đường dẫn thực tế:

```powershell
python -m scripts.evaluate_dg freeze --suites <B2_suite_root> <DeepLab_suite_root> --out outputs/dg_cohort.json
python -m scripts.evaluate_dg evaluate --cohort outputs/dg_cohort.json --run-id segformer_b2:consistency --manifest <ACDC_val406.csv> --scope official-acdc --out outputs/evaluation/b2_consistency
```

Chỉ sau khi đủ sáu candidate đã khóa mới evaluate ACDC. Không dùng ACDC để học/chọn best/tune. Dùng official val406, không dev402 lịch sử. Điểm target đã từng xem phải khai báo, không nhận là test độc lập mới.

Sau freeze, tạo manifest ACDC với cấu trúc local đã mô tả trong [data contract](DATA_CONTRACT.md):

```powershell
python -m scripts.make_manifest --dataset acdc --split val --root data/raw/ACDC/rgb_anon_trainvaltest --mask-root data/raw/ACDC/gt_trainval --output-dir outputs/evaluation/manifests
python -m scripts.validate_manifest --manifest outputs/evaluation/manifests/acdc_val.csv --check-files --check-labels
```

Dùng CSV đó làm `--manifest`, lặp lệnh evaluate cho cả sáu run-id theo dạng `model:arm`. Đổi `--scope official-city` và dùng Cityscapes val CSV nếu đánh giá nguồn. Không trộn CSV từ máy/mount khác nếu chưa kiểm và tái khóa đường dẫn.
