# B2 DG · seed42 · đủ ba nhánh20epoch

Nghiệm thu22/11/2026 từ **`dg_segformer_b2_suite_seed42 (1).zip`**, thay cho ZIP B2 trước chỉ đủ baseline/aug. [Record kiểm tra](verification.json) · [CSV đường học](learning_curves.csv) · [IoU từng lớp](per_class_best.csv).

| Nhánh | Epoch hoàn thành | Best epoch | Cityscapes val500 mIoU | Train + validation ghi trong log |
| --- | ---: | ---: | ---: | ---: |
| baseline | 20/20 | 18 | 73,98% | 3,27giờ |
| aug | 20/20 | 18 | 74,37% | 4,52giờ |
| consistency | 20/20 | 18 | 74,44% | 4,65giờ |

Aug hơn baseline **0,394 điểm phần trăm**. Consistency hơn aug **0,071 điểm phần trăm**, hơn baseline **0,465 điểm phần trăm**. Consistency tăngIoU7/19lớp so với aug; rider giảm khoảng0,223điểm phần trăm. Đây là chênh lệch nhỏ trên tập chọn checkpoint, một seed; chưa có bằng chứng chắc chắn consistency tốt hơn, cải thiện trên thời tiết xấu hoặc vượt bài báo. Không dùng Cityscapes val như test độc lập.

![Learning curves](learning_curves.png)

## Đường chạy và tính toàn vẹn

ZIP mới ghi `suite_complete`,3/3nhánh20epoch và không có `failure.json`. Phiên trước dừng an toàn sau baseline/aug. Phiên resume ghi khoảng6,00giờ của invocation và hoàn thành consistency; median epoch consistency13,98phút trên T4, peak allocated khoảng5,29GiB. Thời gian train+validation không bao gồm đầy đủ process startup/ZIP/setup; không gọi thời gian invocation là compute thuần của model.

- Cả3.590entry ZIP qua CRC,36hash file thuộc12generation checkpoint và272hash source ở bốn snapshot khớp.
- Ba history/resources đều đủ1–20; best epoch khớp mIoU cao nhất và global_step khớp tổng optimizer updates.
- Baseline/aug giữ nguyên weights, trạng thái train, log và báo cáo của ZIP trước. `recovery.json` cập nhật epoch19→20 khi kiểm tra lại run là thay đổi dự kiến; không phải train lại.
- Manifest đủ2.975train/500val, chỉ Cityscapes; ba nhánh có cùng hash initial ImageNet/head và seed42. Consistency không học nối từ aug.
- Aug/consistency có cùng lịch paired minibatch; KL hữu hạn,λ0,1. Config/plan/best/profile và source hashes khớp.
- Cả ba nhánh qua `inspect_candidate` của evaluator hiện hành. Audit không chạy source nhúng hoặc deserialize training_state. Cả ba best checkpoint nạp strict và qua CPU smoke với input0 kích thước64×96, logits19lớp hữu hạn; không phải đánh giá độ chính xác.

Dataset/weights/full ZIP vẫn ở local ngoài Git; bảng số, biểu đồ và bằng chứng nhỏ được lưu tại đây. Giữ nguyên ZIP cũ để truy vết, dùng ZIP mới `(1)` làm artifact B2 hoàn chỉnh cho bước đánh giá.

## Bước tiếp theo

**B2 đã xong training; không cần resume B2 thêm.** Theo artifact DeepLab nhận gần nhất, baseline/aug đủ20epoch và consistency4/20. Cần kết quả DeepLab đủ20epoch cho nhánh cuối, rồi khóa cả sáu candidate trước khi evaluate ACDC val406.

Giữ phương pháp/config/checkpoint hiện tại; chưa thay loss, tăng epoch hoặc chọn mô hình bằng ACDC. [DeepLab và cách resume](../dg_deeplabv3plus_resnet50_seed42/README.md) · [Freeze/evaluate](../../docs/TRAINING_GUIDE.md).
