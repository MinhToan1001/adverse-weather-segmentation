# Lệnh của project

Luồng chính: **build notebook → run suite → freeze → evaluate**. Logic học nằm trong `src/`; các script chỉ điều phối và chuẩn bị artifact.

| Công việc | File |
| --- | --- |
| Sinh hai notebook + code ZIP | [build_kaggle_notebooks.py](build_kaggle_notebooks.py), [build_dg_notebook.py](build_dg_notebook.py) |
| Tìm/copy/giải nén code Kaggle | [kaggle_bootstrap.py](kaggle_bootstrap.py) |
| Tự chạy ba chiến thuật của một model | [run_dg_suite.py](run_dg_suite.py) |
| Chuẩn bị Cityscapes, train/resume một chiến thuật | [run_dg.py](run_dg.py) |
| Đo timing/VRAM bằng pilot nguồn | [profile_dg.py](profile_dg.py) |
| Khóa sáu candidate, đánh giá checkpoint đã khóa | [evaluate_dg.py](evaluate_dg.py) |
| Vẽ loss, Cityscapes mIoU và DG diagnostics | [plot_research_run.py](plot_research_run.py) |
| QA CPU nhỏ, không phải benchmark | [verify_dg.py](verify_dg.py) |
| Tạo/kiểm tra danh sách ảnh–mask | [make_manifest.py](make_manifest.py), [validate_manifest.py](validate_manifest.py) |
| Kiểm tra/chuyển mã nhãn | [audit_labels.py](audit_labels.py), [convert_labels.py](convert_labels.py) |
| Giải nén dataset an toàn | [extract_dataset.py](extract_dataset.py) |

```powershell
python -m scripts.build_kaggle_notebooks
python -m scripts.run_dg_suite --city data/raw/cityscapes --model segformer_b2 --root outputs/dg/b2_suite --archive outputs/dg_b2_suite.zip
```

Lệnh thứ hai cần GPU và có thể chạy qua nhiều phiên; người dùng Kaggle dùng notebook thay vì gọi CLI thủ công. [Hướng dẫn](../docs/TRAINING_GUIDE.md).

Các launcher source→ACDC supervised và generator notebook cũ đã gỡ. Snapshot code trong ZIP kết quả cũ vẫn giữ cách chạy của chính run đó; không resume run cũ bằng code hiện tại.
