# DeepLabV3+ ResNet50 DG · seed42 · chưa đủ ba nhánh

Nghiệm thu22/11/2026 từ `dg_deeplabv3plus_resnet50_suite_seed42.zip`. [Record kiểm tra](verification.json) · [CSV đường học](learning_curves.csv) · [Hướng dẫn train/resume](../../docs/TRAINING_GUIDE.md).

| Nhánh | Epoch đã hoàn thành | Best epoch hiện có | Cityscapes val500 mIoU |
| --- | ---: | ---: | ---: |
| baseline | 20/20 | 20 | 71,82% |
| aug | 20/20 | 20 | 72,08% |
| consistency | 4/20 | 4 | 61,38% — kết quả tạm, chưa so như run20epoch |

Aug hơn baseline **0,263 điểm phần trăm** trên tập Cityscapes val dùng chọn checkpoint. Không đánh giá ACDC trong suite; chưa có kết luận về hiệu quả thời tiết xấu. Best của hai run đầy đủ nằm ở epoch20; điều này chưa đủ để kết luận cần kéo dài training hoặc tăng LR. Giữ recipe20epoch đã chốt để hoàn thành ma trận trước.

![Learning curves; consistency is partial](learning_curves.png)

## Trạng thái và checkpoint

ZIP ghi `session_checkpoint_ready` sau khoảng9,55giờ của invocation, không có `failure.json`. Baseline/aug hoàn thành; consistency có checkpoint last epoch4 và log1–4. Lượt Save & Run All kết thúc không đồng nghĩa suite đã đủ20epoch ở mọi nhánh.

Kiểm tra toàn bộ3.589entry ZIP qua CRC,36hash file trong12generation checkpoint, hash initial/config/frozen candidate/profile,272hash source ở bốn snapshot, và manifest train2975/val500. Ba nhánh có cùng hash initial weights; KL của consistency hữu hạn với λ0,1 và lịch chọn paired batch bốn epoch đầu khớp aug. Không chạy source trong ZIP, deserialize training_state hoặc thực hiện inference mới trong bước audit.

## Đối chiếu với B2 đã nghiệm thu

| Cùng Cityscapes val500,20epoch | B2 | DeepLab | B2 − DeepLab |
| --- | ---: | ---: | ---: |
| baseline | 73,98% | 71,82% | +2,16điểm phần trăm |
| aug | 74,37% | 72,08% | +2,29điểm phần trăm |

Source snapshot giống nhau; manifest khớp sau chuẩn hóa prefix workspace và3.475mask đã chuyển nhãn giống từng byte. Recipe train/loss/DG/selection/preprocessing của các nhánh tương ứng khớp. Khác kiến trúc và encoder ImageNet; không quy chênh lệch chỉ cho số tham số hoặc một module. Một seed và tập selection chưa chứng minh ổn định thống kê, cũng không chứng minh B2 hơn trên ACDC.

## Tiếp tục từ epoch5

1. Import [02_dg_deeplabv3plus_resnet50.ipynb](../../notebooks/02_dg_deeplabv3plus_resnet50.ipynb) vào phiên Kaggle mới.
2. Thêm chính ZIP DeepLab này và Cityscapes cùng version/mount `kavithak1388/cityscapes/Cityscape`. Không thêm ACDC hay dùng ZIP B2 thay cho DeepLab.
3. Đặt `RESUME_INPUT=Path('/kaggle/input/<input-ket-qua-DeepLab>')` theo Copy path thực tế; bật GPU/Internet rồi Save & Run All.
4. Notebook xác minh/bỏ qua baseline và aug; consistency khôi phục checkpoint epoch4 rồi chạy epoch5–20. Giữ snapshot/config/seed gốc.

Bốn epoch consistency đã chạy có median khoảng12,22phút/epoch trên T4. Ngoại suy16epoch còn lại khoảng3,26giờ train+validation, chưa gồm startup/setup/ZIP/checkpoint hoặc biến động runtime. Khi có `suite_complete` và3/3nhánh20/20, tải ZIP mới để nghiệm thu. Cohort chỉ freeze sau khi cả B2 và DeepLab đủ sáu candidate20epoch.
