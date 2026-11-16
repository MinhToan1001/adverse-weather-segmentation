# Trình bày code với giảng viên

[Method](RESEARCH_FRAMEWORK.md) · [Project](../README.md)

## Câu mở đầu

“Em giữ B0 làm baseline nhỏ. Em so B2 với DeepLabV3+ResNet50 gần nhau về số tham số. Dữ liệu,loss và evaluation tách khỏi model nên thay kiến trúc không phải viết lại pipeline. Mục tiêu là học tốt các lớp yếu và hạn chế quên trời quang khi thích nghi ACDC.”

## Mở code theo thứ tự

| Bước | File | Nói gì |
| --- | --- | --- |
|1|[segformer_b0.py](../src/models/segformer_b0.py)|Baseline3,72M;encoder/decoder riêng|
|2|[segformer_b2.py](../src/models/segformer_b2.py)|B2 lớn hơn27,36M,decoder768;không copy dataset/loss vào model|
|3|[deeplabv3plus_resnet50.py](../src/models/deeplabv3plus_resnet50.py)|CNN26,70M;ResNet trích feature,ASPP xem ngữ cảnh nhiều tầm,decoder kết hợp feature nông|
|4|[registry.py](../src/models/registry.py)|Đổi tên model,cùng19-classlogits,checkpoint đúng kiến trúc|
|5|[dataset.py](../src/data/dataset.py)|Ảnh/mask cùng geometry,void255;dev không randomaugment|
|6|[losses.py](../src/training/losses.py)|CEcontrol,weightedCE cho lớp ít pixel,Focal cho pixel khó,Dice cho overlap|
|7|[config.py](../src/experiments/config.py)|Model+recipe+loss,data/optimizer dùng chung,source/adapt rõ ràng|
|8|[train.py](../src/training/train.py)|AMP,accumulation,BN,optimizer,scheduler,best/last,resume|
|9|[evaluate_research.py](../scripts/evaluate_research.py)|IoU từng lớp/thời tiết,confidence,errorpanels,ECE|
|10|[checkpoints.py](../src/training/checkpoints.py)|Lưu staging+SHA,giữ thế hệ trước,rollback epoch chưa chốt|
|11|[three notebooks](../notebooks/README.md)|Mỗi notebook tự chạy source20→adapt20 của một model;dùng cùng template và trainer|

## Từ chuyên ngành

- **Baseline:** mốc ban đầu để biết cải tiến có giúp hay không.
- **Logits:** điểm thô của19lớp tại mỗi pixel;softmax chuyển thành xác suất.
- **Backbone/encoder:** trích đặc trưng;decoder biến đặc trưng thành bản đồ lớp.
- **ASPP:** nhiều nhánh convolution nhìn vùng rộng hẹp khác nhau để hiểu ngữ cảnh.
- **Class imbalance:** road có nhiều pixel,rider ít;model nhận tín hiệu học không đều.
- **Focal:** bớt chú ý pixel dễ;không tự phân biệt pixel khó với nhãn nhiễu.
- **Dice:** vùng dự đoán trùng vùng thật bao nhiêu.
- **Replay:** xen dữ liệu Cityscapes cũ khi học ACDC để giảm quên.
- **BatchNorm:** ổn định feature bằng thống kê;batch nhỏ có thể làm stats dao động.
- **Accumulation:** cộng gradient nhiều microbatch;không biến BNbatch1 thànhBNbatch8.
- **Epoch:** một vòng theo sampler;replay không hứa thấy mọi ảnh nguồn mỗi vòng.
- **Ablation:** thay một yếu tố để xác định tác dụng.
- **Retention gate:** chỉ nhận candidate nếu source-monitor giảm không quá1điểm phần trăm.
- **Dev/val/test:** dev chọn model,val kiểm tra candidate đã chốt,test độc lập hiện chưa có.

## Minh họa dùng lại

```bash
python -m scripts.run_research --model segformer_b2 --stage source --loss ce --data outputs/shared_data/data.json --output outputs/b2_source
python -m scripts.run_research --model deeplabv3plus_resnet50 --stage source --loss ce --data outputs/shared_data/data.json --output outputs/deeplab_source
```

Đổi tên model và output,giữ data/loss/trainer. Thử loss bằng job mới với `--loss`;không sửa forward của3model.

## Nói đúng mức kết quả

“Ba model đã qua data/loss/save/resume/evaluation với ảnh thật trên CPU. Đó là kiểm tra phần mềm. Thiết kế mới chưa có kết quả GPU và chưa chứng minh vượt paper. Em sẽ làm ablation,đọc IoU từng lớp/source retention và xác nhận nhiều seed trước khi kết luận.”
