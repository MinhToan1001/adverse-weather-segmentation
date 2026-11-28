# Sáu checkpoint DG đã khóa · seed42

**Cập nhật 26/11/2026:** đã đánh giá đủ sáu checkpoint trên Cityscapes val500 và ACDC val406. [Kết quả đã nghiệm thu](../dg_evaluation_seed42/README.md). Các bước/chứng cứ bên dưới ghi lại thời điểm trước đánh giá.


Ngày 24/11/2026, đã nghiệm thu hai ZIP suite hoàn chỉnh `(1)` của SegFormer-B2 và DeepLabV3+ ResNet50. Cả sáu lượt độc lập đều đủ 20 epoch. [Biên bản khóa](verification.json) lưu hash từng candidate, weights, khởi tạo và danh tính tập dữ liệu; không chứa weights hoặc ảnh dataset.

| Model | baseline | aug | consistency | Best epoch |
| --- | ---: | ---: | ---: | ---: |
| SegFormer-B2 | 73,98% | 74,37% | 74,44% | 18 cho cả ba |
| DeepLabV3+ ResNet50 | 71,82% | 72,08% | 71,58% | 20 cho cả ba |

Đây là mIoU trên **Cityscapes val500 dùng chọn checkpoint**, không phải ACDC và chưa phải đánh giá cuối với output native. Giữ đủ sáu checkpoint, kể cả DeepLab consistency có điểm nguồn thấp hơn aug. Một seed chưa chứng minh khác biệt có ý nghĩa thống kê.

## Đã kiểm tra

Sáu candidate qua `inspect_candidate`, `seal_cohort` và kiểm tra lại `resolve_frozen`. Chúng dùng cùng source snapshot, split train2975/val500, seed42 và preprocessing. Trong mỗi model, ba nhánh có cùng khởi tạo và recipe train; các nhánh aug/consistency có cùng style/pair schedule. Chưa mở dữ liệu target hoặc chạy inference ACDC trong bước khóa này.

Cohort thực thi local nằm ở `outputs/reviews/dg_cohort_2026-11-24.json`. File đó chứa đường dẫn workspace local nên không dùng trực tiếp trên Kaggle. Khi chuyển môi trường, chạy lại lệnh freeze từ **cùng hai ZIP hoàn chỉnh** để gắn đường dẫn mới và kiểm lại hashes. Không sửa artifact train để đổi mount; chỉ tạo manifest evaluation mới.

## Tiếp theo: đánh giá, không train thêm

Kaggle: dùng [notebook04](../../notebooks/04_evaluate_dg_cohort.ipynb) và làm theo [hướng dẫn bốn input](../../docs/KAGGLE_EVALUATION.md). Source lấy từ snapshot train đã khóa, không cần upload code ZIP riêng.

1. Giữ hai ZIP hoàn chỉnh: `dg_segformer_b2_suite_seed42 (1).zip` và `dg_deeplabv3plus_resnet50_suite_seed42 (1).zip`.
2. Đưa chúng vào cùng môi trường GPU có Cityscapes và ACDC; dùng source/evaluator đã khóa, kiểm hash và freeze sáu candidate tại mount mới.
3. Tạo manifest evaluation Cityscapes val500 và ACDC val406: fog100, night106, rain100, snow100. Không dùng ACDC train.
4. Chạy đủ sáu best checkpoint với cùng FP32, input canvas512×1024 giữ tỷ lệ, trả output về kích thước nhãn gốc, ignore255, không TTA.
5. Tổng hợp mIoU gộp, từng lớp, từng thời tiết, ảnh lỗi và độ tự tin. So aug−baseline, consistency−aug trong từng model; không chọn lại best hoặc tune bằng điểm ACDC.

[Lệnh freeze/evaluate](../../docs/TRAINING_GUIDE.md#freeze-và-evaluate-sau-cả-hai-suite) · [Bằng chứng B2](../dg_segformer_b2_seed42/README.md) · [Bằng chứng DeepLab](../dg_deeplabv3plus_resnet50_seed42/README.md).

ACDC từng được xem trong các thử nghiệm lịch sử ngoài DG; đánh giá này cần công bố giới hạn đó, không nhận là một tập test độc lập chưa từng xem. Khóa cohort giữ cố định sáu candidate hiện tại, không xóa được ảnh hưởng của phản hồi lịch sử.
