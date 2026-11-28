# Làm việc trong nhóm

## Quy tắc chung

- Giữ DG: học Cityscapes nguồn; ACDC đánh giá sau khi chốt model. V2 dùng city-disjoint fit/dev và score sạch+corruption; v1 đã hoàn tất chọn bằng official Cityscapes val. [Bản bàn giao v2](DG_V2_HANDOFF.md). Không trộn điểm supervised lịch sử vào DG.
- Mỗi model có file riêng; data, augmentation, loss, trainer và evaluator dùng chung.
- Thay chiến thuật ở config/module chung; không copy vòng train vào notebook.
- Dùng nhánh riêng cho thay đổi của thành viên, mô tả hành vi và cách kiểm tra trong PR; tránh force-push.

## Đặt thay đổi ở đâu?

| Nội dung | Vị trí |
| --- | --- |
| Kiến trúc/nạp model | `src/models/`, `registry.py` |
| Dataset, nhãn, augmentation | `src/data/` |
| Loss, optimizer, checkpoint | `src/training/` |
| Recipe/preprocessing | `configs/` |
| Ghép config, guard và khóa kết quả | `src/experiments/` |
| Điều phối suite/đánh giá | `scripts/` |
| Giao diện Node.js | `web/` |

## Trước khi bàn giao

```powershell
python -m pytest -q
python -m scripts.build_kaggle_notebooks
npm test
```

DG v2 dùng `python -m scripts.build_dg_v2_notebook` sinh notebook05/06 và code ZIP riêng. Sửa notebook qua generator tương ứng; không commit output thực thi. `npm test` chạy kiểm thử API, không cần weights. Kiểm thử trình duyệt cần checkpoint/dataset và dev dependencies; xem [web](../web/README.md).

Test pretrained chỉ chạy khi đặt `RUN_PRETRAINED_TESTS=1`; `HF_HUB_OFFLINE=1` dùng cache, không tải weights. Tiny CPU QA kiểm tra đường đi dữ liệu/gradient/lưu–khôi phục, không phải đánh giá độ chính xác.

Commit code/config/tài liệu/tests/notebook và báo cáo nhỏ. Không commit dataset, đường dẫn máy cá nhân, weights, ZIP, ảnh dataset hay outputs lớn. Những file này nằm trong `data/raw/`, `outputs/` và được Git bỏ qua.

Resume phải dùng snapshot code/config/dataset gốc. Đổi code/recipe là thí nghiệm mới; không sửa config trong ZIP cũ để vượt kiểm tra hash. Lưu kết quả và các giới hạn vào [results](../results/README.md); chỉ đánh dấu train hoàn thành khi đủ log và checkpoint 20 epoch của từng chiến thuật.
