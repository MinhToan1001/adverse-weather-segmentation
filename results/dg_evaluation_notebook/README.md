# Notebook đánh giá sáu checkpoint DG

**Cập nhật 26/11/2026:** đã đánh giá đủ sáu checkpoint trên Cityscapes val500 và ACDC val406. [Kết quả đã nghiệm thu](../dg_evaluation_seed42/README.md). Các bước/chứng cứ bên dưới ghi lại thời điểm trước đánh giá.


Ngày 24/11/2026. [Notebook04](../../notebooks/04_evaluate_dg_cohort.ipynb) · [Hướng dẫn import và bốn input](../../docs/KAGGLE_EVALUATION.md) · [Biên bản kiểm tra thực tế](verification.json).

Notebook tự chạy 12 job: sáu best checkpoint đã khóa × Cityscapes val500 và ACDC val406. Source evaluator lấy nguyên snapshot train, kiểm 68 hash; không sửa trainer/model/loss hoặc học thêm target. Kết quả xuất CSV, biểu đồ, confusion matrix từng ảnh và panel trực quan.

## Đã kiểm tra local

- **15 test đạt**, chạy actual code cells/helpers: tìm suite trong ZIP/thư mục, lọc đúng model và suite hoàn chỉnh, chặn nhiều kết quả nhập nhằng, path/source bị đổi, cấu trúc ACDC nested, cache bị thay, vòng lặp 12 job, bỏ qua job đã xong và export khi có lỗi.
- Cell manifest thực tế qua dữ liệu fixture có đủ 500+406 cặp; giữ nguyên guard của generic validator chặn quét nhãn target ngoài luồng đánh giá.
- Đọc hai ZIP kết quả thật, chỉ stage metadata và initial/best weights; bỏ optimizer/last/mask để giảm chép dữ liệu. Khôi phục source đã khóa và chạy cell freeze, chấp nhận đúng sáu candidate đã nghiệm thu.
- Tạo manifest mới trên dữ liệu local thật: Cityscapes val500 và ACDC val406, kiểm cặp/file tồn tại. Đọc một mẫu thật từ mỗi domain qua dataset đánh giá, kiểm kích thước/nhãn và tensor ảnh hữu hạn.

Kiểm thử thực tế chạy CPU; bước cài packages và kiểm tra GPU được stub để kiểm luồng setup/freeze. Không chạy target inference hoặc train trong bước này. **Chưa có kết quả ACDC mới và chưa chạy đủ 12 job trên Kaggle.** Mỗi ảnh còn lại được kiểm nhãn/kích thước khi evaluator đọc trong lượt inference thật.

## Phạm vi resume

ZIP được xuất sau mỗi job hoàn chỉnh và khi Python báo lỗi. Job đã xong chỉ được skip nếu weights/manifest/metrics/confusion hashes, số lượng domain, FP32/noTTA và canvas khớp. Job dở được giữ riêng và chạy lại từ đầu. Không có resume từng ảnh; hard timeout có thể bỏ qua finally. Dùng cùng mount input khi tiếp tục.

Hai notebook train và notebook03 giữ nguyên; notebook04 là wrapper riêng. Source/config sáu lượt đã khóa không đổi. Giữ notebook/output Kaggle riêng tư vì panel chứa ảnh dataset.
