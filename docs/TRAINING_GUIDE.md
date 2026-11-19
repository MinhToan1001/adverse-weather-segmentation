# Chạy các arm DG trên Kaggle

[Phương pháp chính xác](DG_METHOD.md) · [Ma trận5lượt/30giờ](../research/PROJECT_COMPLETION_PLAN.md)

## Chuẩn bị

`python -m scripts.build_kaggle_notebooks` sinh code ZIP `outputs/kaggle/project-code.zip` và hai notebook:

- [B2](../notebooks/01_dg_segformer_b2_baseline.ipynb): tên file giữ để không phá liên kết cũ, hỗ trợ đủ ba ARM.
- [DeepLabV3+ ResNet50](../notebooks/02_dg_deeplabv3plus_resnet50.ipynb).

Mỗi Saved Version chạy **một model/arm**,20epoch. Gắn code ZIP mới và Cityscapes có train/val; không gắn ACDC vào training. Giữ đúng cùng code ZIP cho5lượt. Không dùng các notebook source→ACDC trong archive.

## Khi GPU hồi

1. Bật GPU/Internet, import notebook đúng model. `CODE_INPUT` đặt rõ dataset code nếu auto-find có nhiều bản; `CITYSCAPES_ROOT=None` tự tìm theo leftImg8bit/gtFine train/val.
2. Giữ `PROFILE_ONLY=True` ở lần GPU đầu. Notebook chỉ chuẩn bị nguồn và đo batch/VRAM cho C0/C1/C2 của kiến trúc đó; bỏ weights pilot. Chạy một lần cho B2, một lần cho DeepLab.
3. Đọc `gpu_profile.json`. ETA là extrapolation nhỏ, **chưa tính checkpoint/ZIP/setup**. Chốt p_pair/λ/style và ngân sách trước lượt chính; không suy ra GPU ETA từ CPU QA.
4. Để train chính, đặt `PROFILE_ONLY=False`, `ARM='baseline'`, `'aug'` hoặc `'consistency'`. Mỗi arm khởi tạo độc lập từ ImageNet; không nạp best baseline để học thêm20epoch.
5. Matrix: B2baseline → DeepLabbaseline → B2aug → B2consistency → DeepLabconsistency. Sau khi chốt thiết kế, không thay recipe giữa các arm hoặc tune theo ACDC.
6. Save Version → Save & Run All. ZIP riêng có tên `dg_<model>_<arm>_seed42.zip`. Cần giữ output/ZIP khi kết thúc phiên.

Mặc định ứng viên: p_pair0,5; λ0,1 chỉ có hiệu lực ở consistency. Các giá trị này chưa có bằng chứng tối ưu chất lượng. Nếu thay, phải áp cùng giữa các arm tương ứng và giữ nguyên khi resume.

## Resume đúng lượt

Phiên mới: cùng dataset version/mount, cùng notebook MODEL/ARM/SEED; `RESUME_INPUT` trỏ ZIP đầy đủ hoặc workspace đã giải nén. Notebook khôi phục source snapshot, kiểm hash, model/arm/seed và settings. Kaggle tự giải nén source.zip vẫn được hỗ trợ bằng tái đóng gói từ file đã xác minh.

Có thể resume ZIP profiling để bắt đầu train **cùng model/arm** bằng PROFILE_ONLY=False; pilot không có checkpoint train, nên model thật vẫn khởi tạo mới. Không dùng ZIP baseline để resume aug/consistency. Muốn arm mới: phiên/workspace mới, RESUME_INPUT=None.

Sau mỗi epoch hoàn thành: best/last/previous, optimizer/scheduler/scaler/RNG và ZIP. Ngắt giữa epoch sẽ chạy lại epoch đó; hard-kill không bảo đảm finally chạy nên giữ ZIP hoàn chỉnh gần nhất. Trainer dùng một GPU. Cửa sổ phiên10giờ khác tổng quota30giờ.

## CLI và CPU QA

```powershell
python -m scripts.run_dg --city data/raw/cityscapes --model segformer_b2 --arm consistency --root outputs/dg/b2_consistency --archive outputs/dg_b2_consistency.zip --prepare-only
# Bỏ --prepare-only khi đã có GPU để train/resume cùng workspace.
python -m scripts.verify_dg --data-root outputs/dg/b2_consistency/data --out outputs/my_dg_cpu_qa
```

verify_dg chỉ probe2epoch nhỏ bằng random weights, không là benchmark. Full DG guard không được nới để chạy QA subset.

## Đánh giá sau khi khóa cohort

Sau khi có5workspace hoàn chỉnh, đưa chúng vào cùng môi trường đánh giá, giữ đường dẫn đúng rồi seal:

```powershell
python -m scripts.evaluate_dg freeze --workspaces <B2-C0> <B2-C1> <B2-C2> <DL-C0> <DL-C2> --out outputs/dg_cohort.json
python -m scripts.evaluate_dg evaluate --cohort outputs/dg_cohort.json --run-id segformer_b2:consistency --manifest <ACDC_val406.csv> --scope official-acdc --out outputs/evaluation/b2_consistency
```

Manifest evaluation được tạo riêng từ official val bằng script manifest hiện có; không dùng ACDC dev402 lịch sử. Evaluator chỉ nhận checkpoint thuộc cohort, kiểm hash/model/arm/split/init/code và đủ20epoch. Không mở target để chọn best/hyperparameter. Số ACDC đã xem trong lịch sử không phải test độc lập mới.
