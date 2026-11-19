# Dữ liệu local

Dataset không được đẩy lên Git. Mỗi thành viên chuẩn bị dữ liệu hợp lệ của mình.

```text
data/raw/cityscapes/
  leftImg8bit/train, val
  gtFine/train, val
data/raw/ACDC/
  rgb_anon_trainvaltest/rgb_anon/{fog,rain,snow,night}/
  gt_trainval/gt/{fog,rain,snow,night}/
```

DG hiện hành chỉ học Cityscapes; không dùng ACDC train hoặc nhãn ACDC để chọn best. ACDC val chỉ evaluate sau khi khóa model. Quy ước train/val và mã nhãn xem [data contract](../docs/DATA_CONTRACT.md). Thư mục mount trên Kaggle có thể khác; không cần đổi source để khớp đường dẫn máy người khác.

Chạy `python -m scripts.run_dg --city data/raw/cityscapes --root outputs/dg/b2 --archive outputs/dg_b2_seed42.zip --prepare-only` để tạo danh sách ảnh–mask và nhãn đã chuyển. Manifest sinh ra chứa đường dẫn máy hiện tại và hash, không commit dữ liệu sinh tự động. Không sửa trực tiếp mask gốc của dataset.

[Hướng dẫn training](../docs/TRAINING_GUIDE.md) · [Dữ liệu đi qua code thế nào](../docs/READING_GUIDE.md)
