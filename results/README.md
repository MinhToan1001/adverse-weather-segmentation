# Trạng thái kết quả và bằng chứng

## Luồng DG hiện hành

**Đã nghiệm thu năm lượt đủ20epoch**, theo ZIP nhận ngày22/11. B2 đã đủ ba nhánh; còn chờ DeepLab consistency đủ20epoch trước khi khóa sáu candidate và đánh giá ACDC.

| Model | baseline20 | aug20 | Consistency trong ZIP đã nhận |
| --- | ---: | ---: | --- |
| [B2](dg_segformer_b2_seed42/README.md) | 73,98% | 74,37% | 20/20epoch;74,44% |
| [DeepLabV3+R50](dg_deeplabv3plus_resnet50_seed42/README.md) | 71,82% | 72,08% | 4/20epoch; resume từepoch5 |

Các điểm trên là Cityscapes val500 dùng chọn checkpoint; không phải điểm ACDC hoặc test độc lập. Chỉ so các lượt đủ20epoch; không dùng điểm DeepLab consistency epoch4 như kết quả hoàn chỉnh. B2 consistency hơn aug0,071điểm phần trăm, chênh nhỏ và một seed chưa chứng minh ưu thế. Điểm tiny CPU probe không được dùng làm độ chính xác của model.

| Bằng chứng | Phạm vi |
| --- | --- |
| [Suite tự động](dg-suite-automation/README.md) | Queue ba chiến thuật, resume, bootstrap notebook, profile seal và source preflight |
| [DG CPU QA](dg-implementation/README.md) | Loss/gradient/geometry; hai model chạy thử nhỏ và save/resume khớp chạy liền |

Test record ghi rõ phiên bản và thời điểm kiểm tra. Kết quả tests xác nhận phần mềm ở phạm vi đã thử, không thay thế mIoU hay timing GPU. [Kế hoạch còn lại](../research/PROJECT_COMPLETION_PLAN.md).

## Kết quả lịch sử — ngoài bảng DG

Các run dưới từng học ảnh và nhãn ACDC. Giữ nguyên báo cáo/weights để truy vết và phục vụ demo; không sử dụng điểm đó để chứng minh DG. B0 đã bỏ khỏi lượt train mới.

| Run | Cityscapes monitor188, source best | ACDC dev402, adapted best | Official validation của run này |
| --- | ---: | ---: | --- |
| [B0 lịch sử](segformer_b0_seed42/README.md) | 65,13% | 51,65% | Chưa nghiệm thu |
| [B2 lịch sử](segformer_b2_seed42/README.md) | 74,74% | 63,02% | Chưa nghiệm thu |

Đây là điểm chọn checkpoint trên split nội bộ, không phải ACDC val406. Demo Node.js hiện dùng B2 source/adapted của các run này. ACDC đã được xem trong phát triển là giới hạn phải công bố của đánh giá DG tiếp theo.

## Quy tắc lưu kết quả

- Git giữ config/tài liệu, bảng số nhỏ và báo cáo kiểm chứng.
- Dataset, weights, optimizer/RNG state, ZIP, ảnh dataset/panel và log lớn nằm local trong `data/`, `outputs/` hoặc input Kaggle; không commit.
- Giữ ZIP gốc để resume, cả snapshot code của run. Không thay config/đổi tên kết quả cũ thành protocol mới.
- Các log XML/TXT và báo cáo phần mềm cũ đã thay thế được gỡ khỏi cây hiện hành; xem lịch sử Git khi cần.
