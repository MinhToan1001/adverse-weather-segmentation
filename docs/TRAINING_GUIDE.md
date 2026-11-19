# Train và khôi phục baseline DG B2

[Protocol khôi phục](../research/DG_RECOVERY_PLAN.md) · [Notebook](../notebooks/01_dg_segformer_b2_baseline.ipynb)

## Quy tắc

Chỉ Cityscapes train 2.975 ảnh để học, Cityscapes val500 để chọn best; **không ACDC input**. Encoder ImageNet được pin, head mới, 20 epoch, CE, không early stopping. Giữ recipe source chung: khung512×1024 giữ tỷ lệ, scale0,5–2/crop/flip đồng bộ, batch2/accum4, AdamW encoder6e-5/head×10, weight_decay0,01, warm-up20updates/poly1, FP16AMP. Đây là baseline mới, chưa có module DG proposed hoặc kết quả GPU.

## Kaggle

1. Chạy `python -m scripts.build_kaggle_notebooks` để sinh notebook DG và code ZIP mới.
2. Upload `outputs/kaggle/project-code.zip`; import notebook DG; gắn Cityscapes có cả train/val. Bật GPU/Internet. Không thêm ACDC cho training.
3. `CODE_INPUT` trỏ tới code mới; `CITYSCAPES_ROOT=None` tự tìm theo cấu trúc, hoặc đặt cụ thể khi có nhiều bản. `RESUME_INPUT=None` là lượt mới.
4. Chạy setup kiểm tra, rồi Save Version → Save & Run All. Nó chỉ chạy một stage Cityscapes, **không tự học ACDC**.
5. Giữ ZIP `dg_b2_baseline_seed42.zip` và output. Chỉ sau khi khóa phương pháp/checkpoint mới lập lượt đánh giá ACDC riêng.

## Resume

Sau mỗi epoch hoàn thành, trainer lưu weights/config/optimizer/scheduler/scaler/RNG, best/last và thế hệ previous; launcher cập nhật ZIP. Epoch dở chạy lại. Cửa sổ phiên tối đa10giờ, không tổng cap. Nếu bị Kaggle hard-kill, `finally` không được bảo đảm chạy, nên giữ bản ZIP hoàn chỉnh gần nhất và raw checkpoint nếu có.

Phiên mới: cùng dataset/mount, `RESUME_INPUT` trỏ kết quả DG; notebook khôi phục code snapshot và kiểm hash. Không dùng ZIP source→ACDC cũ, không thay recipe giữa lượt. Không mở lại workspace cũ để chạy từ đầu; tạo workspace mới.

## CLI và chuẩn bị trên CPU

```powershell
python -m scripts.run_dg --city data/raw/cityscapes --root outputs/dg/b2 --archive outputs/dg_b2_seed42.zip --prepare-only
# Khi có GPU, cùng lệnh bỏ --prepare-only để train/resume:
python -m scripts.run_dg --city data/raw/cityscapes --root outputs/dg/b2 --archive outputs/dg_b2_seed42.zip
```

`--prepare-only` chuyển nhãn và tạo manifest/config, không train hoặc mở dữ liệu ACDC. `dg_plan.json` ghi split, source/hash/seed; `frozen_candidate.json` chỉ sinh sau khi đủ20epoch, ghi hash best và quy tắc selection. Launcher không evaluate ACDC tự động.

## Lịch sử

Các notebook B2/DeepLab source20→adapt20 đã chuyển sang [archive](../research/archive/notebooks). `scripts.run_research_campaign` và stage adapt yêu cầu `--allow-supervised-acdc`, để người chạy xác nhận đó là supervised ngoài DG. Resume lịch sử phải dùng code snapshot tương ứng, không sửa weights hoặc nhãn protocol của run cũ.
