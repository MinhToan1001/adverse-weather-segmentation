# Hướng dẫn làm việc trong nhóm

## Bắt đầu

Tạo môi trường ảo, cài `requirements.txt`, rồi chạy `python -m pytest -q`. Các kiểm tra pretrained tùy chọn dùng `RUN_PRETRAINED_TESTS=1`; `HF_HUB_OFFLINE=1` ngăn tải weights. GPU training chạy trên Kaggle qua một trong ba notebook.

## Đặt thay đổi ở đâu?

- Kiến trúc model: file riêng trong `src/models/`, đăng ký tên và loader trong registry.
- Dataset, augmentation và nhãn: `src/data/`.
- Loss, optimizer và checkpoint: `src/training/`.
- Ghép cấu hình thí nghiệm: `src/experiments/` và `configs/`.
- Đánh giá và độ tin cậy: `src/evaluation/`.
- Notebook: sửa generator chung, rồi tạo lại cả ba notebook. Trainer được tái sử dụng từ module chung.
- Giao diện Node.js và API: `web/`; chạy `npm test`. Kiểm thử trình duyệt bằng `npm run test:e2e` khi có checkpoint và dataset local, xem `web/README.md`.

## Trước khi commit hoặc tạo PR

1. Chạy test liên quan; chạy `python -m pytest -q` khi đổi pipeline.
2. Chạy `python -m scripts.build_kaggle_notebooks`. Kiểm tra notebook sinh ra không chứa output hay trạng thái thực thi.
3. Xem Git diff. Commit source, config, tài liệu, test và ba notebook. Dataset, manifest sinh tự động, weights, ZIP kết quả và archive local được Git bỏ qua.
4. Mô tả vấn đề, thay đổi hành vi và cách kiểm thử. Dùng nhánh riêng cho thay đổi của nhóm; tránh force-push ghi đè công việc của thành viên khác.
5. Ghi config, source và seed trước khi chạy GPU. Khi resume, giữ nguyên thí nghiệm. Phân biệt kết quả source-only, adaptation có nhãn và UDA khi báo cáo.

Dataset local đặt trong `data/raw/`; trên Kaggle dùng input mount. Mỗi thành viên có thể chọn đường dẫn khác nhau và tạo manifest bằng `scripts.prepare_research_data`, thay vì commit đường dẫn tuyệt đối trên máy cá nhân. Archive cũ trong `outputs/` không được đẩy lên Git và không cần thiết để clone/chạy project.
