# Đọc code theo một đường đi

Nên đọc [README](../README.md) trước. Luồng hiện hành là **Cityscapes → train → chọn best bằng Cityscapes val → khóa sáu candidate → đánh giá ACDC**.

## 1. Dữ liệu vào model như thế nào?

- [preparation.py](../src/data/preparation.py): ghép ảnh với mask và chuyển mã nhãn khi cần. Manifest là file CSV liệt kê các cặp ảnh–mask.
- [labels.py](../src/data/labels.py): 19 lớp và mã bỏ qua 255.
- [preprocessing.py](../src/data/preprocessing.py): giữ tỷ lệ, resize/pad, normalization.
- [dataset.py](../src/data/dataset.py): áp dụng cùng hình học cho ảnh và mask.
- [source_style.py](../src/data/augmentations/source_style.py): tạo thêm phiên bản chỉ thay ngoại hình RGB, giữ nhãn.

[DATA_CONTRACT](DATA_CONTRACT.md) giải thích split và cấu trúc thư mục. Các file `partitions.py`, `replay.py` phục vụ luồng supervised lịch sử; suite DG không dùng chúng để chia/học ACDC.

## 2. Hai kiến trúc nằm ở đâu?

Mở [segformer_b2.py](../src/models/segformer_b2.py) và [deeplabv3plus_resnet50.py](../src/models/deeplabv3plus_resnet50.py). Đây là hai file model riêng. [registry.py](../src/models/registry.py) chọn/nạp model theo tên; [segformer.py](../src/models/segformer.py) giữ helper SegFormer dùng chung. Cả hai trả logits — điểm dự đoán 19 lớp trước khi chọn lớp cho từng pixel.

## 3. Chiến thuật và vòng train nằm ở đâu?

- [dg.py trong experiments](../src/experiments/dg.py) ghép config model + chiến thuật + đường dẫn.
- [protocol.py](../src/experiments/protocol.py) chặn ACDC train/selection và config sai DG.
- [dg.py trong training](../src/training/dg.py) thực hiện baseline/aug/consistency.
- [train.py](../src/training/train.py) quản lý batch, loss, optimizer, validation và chọn best.
- [checkpoints.py](../src/training/checkpoints.py) lưu/khôi phục cả trạng thái train.

Thông số tập trung ở [source_20epoch.yaml](../configs/recipes/source_20epoch.yaml), [shared_512.yaml](../configs/preprocessing/shared_512.yaml), [paired_source.yaml](../configs/dg/paired_source.yaml). Đọc [DG_METHOD](DG_METHOD.md) trước khi thay chúng.

## 4. Notebook điều phối gì?

[build_dg_notebook.py](../scripts/build_dg_notebook.py) sinh hai notebook từ một template. [run_dg_suite.py](../scripts/run_dg_suite.py) chuẩn bị Cityscapes một lần, đo GPU rồi chạy ba chiến thuật. [run_dg.py](../scripts/run_dg.py) chạy một chiến thuật, tiếp tục từ epoch đã lưu và xuất kết quả. Notebook không chứa bản sao kiến trúc hay trainer.

## 5. Đọc kết quả và giao diện

[evaluate_dg.py](../scripts/evaluate_dg.py) khóa đủ sáu candidate rồi cho phép đánh giá checkpoint đã khóa. [runner.py](../src/evaluation/runner.py) xuất metrics và ảnh lỗi; [metrics.py](../src/evaluation/metrics.py) tính IoU từ ma trận nhầm lẫn. [plot_research_run.py](../scripts/plot_research_run.py) vẽ lịch sử train.

[web/server.mjs](../web/server.mjs) phục vụ giao diện/API; [web/worker.py](../web/worker.py) nạp weights; [inference.py](../src/inference.py) dùng preprocessing/model chung. Demo B2 hiện là **lịch sử supervised**, chưa phải kết quả DG mới. Xem [results](../results/README.md) trước khi trích số liệu.
