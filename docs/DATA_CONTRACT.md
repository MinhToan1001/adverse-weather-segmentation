# Quy ước dữ liệu hiện hành

## Split và quyền sử dụng

| Tập | Số ảnh | Được dùng làm gì? |
| --- | ---: | --- |
| Cityscapes official train | 2.975 | Học, tạo style nguồn |
| Cityscapes official val | 500 | Chọn best, theo dõi loss/mIoU |
| ACDC official val | 406 | Đánh giá sau khi khóa sáu candidate |

ACDC val gồm fog 100, night 106, rain 100, snow 100. Không dùng ACDC train, ảnh reference, nhãn, style statistics, pseudo-labels hoặc batch statistics để học/chọn model. Cityscapes val là tập chọn model, không phải kiểm thử độc lập.

Split Cityscapes fit2787/monitor188 và ACDC fit1198/dev402 chỉ xuất hiện ở [kết quả lịch sử](../results/README.md), không dùng trong suite DG. ACDC từng được xem trong phát triển nên phải khai báo giới hạn khi báo cáo kết quả mới.

## Cấu trúc local

```text
data/raw/cityscapes/
  leftImg8bit/{train,val}/<city>/*_leftImg8bit.png
  gtFine/{train,val}/<city>/*_gtFine_labelIds.png

data/raw/ACDC/
  rgb_anon_trainvaltest/rgb_anon/{fog,night,rain,snow}/val/...
  gt_trainval/gt/{fog,night,rain,snow}/val/...
```

Kaggle có mount khác; notebook tự tìm hoặc nhận đường dẫn tới thư mục chứa `leftImg8bit/` và `gtFine/`. Hai thư mục ACDC RGB/GT có thể truyền riêng lúc tạo manifest đánh giá. Dataset không được commit hoặc phát lại như một phần source.

## Nhãn và biến đổi

- Model dự đoán **trainIds 0–18**. Mã **255** là vùng bỏ qua khi tính loss và metrics.
- Cityscapes raw `labelIds` được chuyển sang `labelTrainIds` trong workspace sinh ra; không ghi đè nhãn gốc. Nếu đã đủ trainIds hợp lệ thì tái sử dụng.
- Canvas đầu vào **512×1024**; giữ tỷ lệ. Ảnh nội suy bilinear, mask nearest; vùng pad của mask là 255.
- Train dùng scale ngẫu nhiên 0,5–2, crop/pad và lật ngang xác suất 0,5. Ảnh–mask dùng chung hình học.
- RGB chuẩn hóa theo ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`.
- View style chỉ thay RGB sau hình học. Evaluation không augmentation, bỏ pad và khôi phục dự đoán về kích thước ảnh gốc.

## Manifest và dữ liệu sinh ra

Manifest là CSV có `image`, `mask`, `domain`, `split`. Suite tạo một bộ converted masks dùng chung; ba chiến thuật giữ các bản CSV có nội dung giống nhau. Config và CSV được ghi hash để phát hiện thay đổi khi resume. Đường dẫn mount/dataset version phải giữ nguyên khi tiếp tục phiên.

[run_dg.py](../scripts/run_dg.py) chuẩn bị và kiểm tra đủ Cityscapes train/val. Chỉ cần kiểm tra trước khi có GPU:

```powershell
python -m scripts.run_dg --city data/raw/cityscapes --root outputs/dg_preflight --archive outputs/dg_preflight.zip --prepare-only
```

Dùng một output mới; không sửa file của run đã khóa. [validation.py](../src/data/validation.py) kiểm tra file, kích thước, nhãn và đường dẫn trùng. [protocol.py](../src/experiments/protocol.py) kiểm ranh giới DG. [Train/resume/evaluate](TRAINING_GUIDE.md).
