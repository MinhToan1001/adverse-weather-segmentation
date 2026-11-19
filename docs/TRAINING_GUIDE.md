# Chạy DG tự động trên Kaggle

[Hai notebook](../notebooks/README.md) · [Phương pháp](DG_METHOD.md)

## Luồng sử dụng

Mỗi model có một notebook. Cả B2 và DeepLab đều tự chạy **profiling → baseline20 → aug20 → consistency20**. Không chọn ARM và không đổi PROFILE_ONLY. Các arm khởi tạo độc lập từ cùng ImageNet encoder/head seed trong từng model; không học nối từ baseline.

1. Chạy `python -m scripts.build_kaggle_notebooks`; upload `outputs/kaggle/project-code.zip` mới vào Kaggle.
2. Import notebook B2 hoặc DeepLab. Gắn cùng bộ Cityscapes có train/val, bật GPU/Internet. Không cần ACDC cho training.
3. CODE_INPUT mặc định auto-find; đặt rõ dataset code nếu có nhiều bản. CITYSCAPES_ROOT=None tự tìm theo leftImg8bit/gtFine train/val. Lượt mới RESUME_INPUT=None.
4. Save Version → Save & Run All. Notebook chuẩn bị dữ liệu một lần, profile model một lần; khi profile đạt thì tự chạy tiếp cả ba chiến thuật. Không dừng để bắt đổi cờ.
5. Giữ ZIP `dg_<model>_suite_seed42.zip`. suite_complete nghĩa là model đó đủ3×20epoch; hai suite hoàn chỉnh tạo đủ6candidate.

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

Đưa hai suite hoàn chỉnh vào cùng môi trường. Lệnh tự tìm sáu arm:

```powershell
python -m scripts.evaluate_dg freeze --suites <B2_suite_root> <DeepLab_suite_root> --out outputs/dg_cohort.json
python -m scripts.evaluate_dg evaluate --cohort outputs/dg_cohort.json --run-id segformer_b2:consistency --manifest <ACDC_val406.csv> --scope official-acdc --out outputs/evaluation/b2_consistency
```

Chỉ sau khi đủ sáu candidate đã khóa mới evaluate ACDC. Không dùng ACDC để học/chọn best/tune. Dùng official val406, không dev402 lịch sử. Điểm target đã từng xem phải khai báo, không nhận là test độc lập mới.
