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

Cityscapes dùng học nền tảng và replay; ACDC train dùng học thích nghi có nhãn. Chia fit/dev/monitor và mã nhãn xem [data contract](../docs/DATA_CONTRACT.md). Thư mục mount trên Kaggle có thể khác; không cần đổi source để khớp đường dẫn máy người khác.

Chạy `python -m scripts.prepare_research_data --city data/raw/cityscapes --acdc data/raw/ACDC --out outputs/data` để tạo danh sách ảnh–mask và nhãn đã chuyển. Manifest sinh ra chứa đường dẫn máy hiện tại và hash, không commit dữ liệu sinh tự động. Không sửa trực tiếp mask gốc của dataset.

[Hướng dẫn training](../docs/TRAINING_GUIDE.md) · [Dữ liệu đi qua code thế nào](../docs/READING_GUIDE.md)
