# Các lệnh điều phối

Logic model/data/training nằm trong `src/`; `scripts/` ghép chúng thành các thao tác chạy được. Tên lệnh `research` được giữ để tương thích, hiện phục vụ cả đồ án ứng dụng.

| Giai đoạn | File / lệnh | Vai trò |
| --- | --- | --- |
| Chuẩn bị dataset | [extract_dataset.py](extract_dataset.py) | Giải nén archive, kiểm tra đường dẫn |
| Ghép cặp và kiểm kê | [make_manifest.py](make_manifest.py), [validate_manifest.py](validate_manifest.py), [audit_labels.py](audit_labels.py) | Danh sách ảnh–mask, xác minh dữ liệu và mã nhãn |
| Chuyển nhãn | [convert_labels.py](convert_labels.py) | LabelIds → trainIds |
| Chuẩn bị toàn pipeline | [prepare_research_data.py](prepare_research_data.py) | Chia fit/dev/monitor, manifest và data.json chung |
| Sinh notebook/code ZIP | [build_kaggle_notebooks.py](build_kaggle_notebooks.py), [build_research_notebooks.py](build_research_notebooks.py) | Hai notebook hiện hành B2/DeepLab từ một template |
| Input Kaggle | [kaggle_bootstrap.py](kaggle_bootstrap.py) | Tìm code, kiểm tra và copy/giải nén an toàn |
| Một stage | [run_research.py](run_research.py) | Ghép config, train/resume theo epoch, giữ config/source |
| Hai stage một model | [run_research_campaign.py](run_research_campaign.py) | Source20 → adapt20; xuất ZIP sau epoch |
| Vẽ lịch sử | [plot_research_run.py](plot_research_run.py) | Loss, mIoU và diagnostics từ log |
| Evaluate | [evaluate_research.py](evaluate_research.py) | Checkpoint + manifest → metrics và ảnh lỗi, không train |
| Kiểm tra pipeline nhỏ | [verify_pipeline.py](verify_pipeline.py) | CPU probe để kiểm plumbing, không phải benchmark chất lượng |

Ví dụ CLI hiện hành:

```powershell
python -m scripts.prepare_research_data --city data/raw/cityscapes --acdc data/raw/ACDC --out outputs/data
python -m scripts.run_research_campaign --models segformer_b2 --data outputs/data/data.json --root outputs/runs/segformer_b2 --archive outputs/segformer_b2_seed42.zip
```

Luôn chọn rõ `--models segformer_b2` nếu chỉ muốn B2; gọi campaign không có `--models` sẽ xếp cả B2 và DeepLab tùy chọn. Không ghi đè workspace cũ để khởi động lại. [Hướng dẫn Kaggle/resume](../docs/TRAINING_GUIDE.md).
