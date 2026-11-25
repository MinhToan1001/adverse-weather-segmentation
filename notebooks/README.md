# Notebook DG tự động cho hai model

| Notebook | Quy trình tự chạy |
| --- | --- |
| [01_dg_segformer_b2.ipynb](01_dg_segformer_b2.ipynb) | Profile GPU → baseline20 → aug20 → consistency20 |
| [02_dg_deeplabv3plus_resnet50.ipynb](02_dg_deeplabv3plus_resnet50.ipynb) | Profile GPU → baseline20 → aug20 → consistency20 |
| [03_k1_weather_preview.ipynb](03_k1_weather_preview.ipynb) | K1 độc lập: CPU preview thời tiết mức1–5; không train |
| [04_k2_weatherseg_b0_train.ipynb](04_k2_weatherseg_b0_train.ipynb) | Một arm B0/K2 mỗi phiên; mặc định baseline; ZIP đầy đủ + ZIP nhẹ |
| [05_k2_balanced_train_eval.ipynb](05_k2_balanced_train_eval.ipynb) | Recipe ứng viên40epoch,640×1280; khóa best rồi đánh giá Cityscapes/ACDC và vẽ hình |
| [06_k2_evaluate_report.ipynb](06_k2_evaluate_report.ipynb) | Đọc ZIP checkpoint đã hoàn thành để chỉ evaluate và xuất PNG/SVG; không train lại |
| [07_k2_aug_legacy_train_eval.ipynb](07_k2_aug_legacy_train_eval.ipynb) | Đối chứng B0+K1, legacy20epoch/seed42; tự audit với B0 rồi đánh giá |
| [08_k2_film_legacy_train_eval.ipynb](08_k2_film_legacy_train_eval.ipynb) | Đối chứng B0+K1+FiLM, không loss điều kiện; cùng recipe20epoch |

K1 là phần mở rộng trên nhánh riêng. Notebook03 chỉ cần CPU + Internet, có demo với khoảng cách tổng hợp đã biết và preview ảnh Cityscapes train nếu được gắn vào input. Fog cần metric distance hoặc disparity/camera; thiếu độ sâu thì bỏ riêng fog. [Cách chạy và giới hạn](../docs/K1_WEATHER.md).

Notebook04 nằm **cùng nhánh K1** và dùng trực tiếp generator K1. Mặc định train một arm `baseline` trong20epoch; chọn `film_clean`, `aug`, `film`, `k2` trong các phiên độc lập để làm ablation. `WEATHER_MODE=no-fog` giữ bốn điều kiện như pilot đã chạy; đặt `full` và cung cấp depth để train đủ năm nhãn trong một đợt thí nghiệm riêng. Chỉ gắn Cityscapes để train/chọn best; ACDC đánh giá sau khi khóa candidate. K3/K4 chưa nằm trong notebook này. [Cách chuẩn bị dữ liệu, ablation và resume](../docs/K2_TRAINING.md).

Notebook05/06 là luồng K2 riêng, không thay cohort B2/DeepLab. Notebook06 dùng được ngay với checkpoint20epoch cũ. Notebook05 có recipe ứng viên mới, chưa chứng minh tăng mIoU; cần baseline cùng recipe40epoch để đối chứng. ACDC được gắn làm input cho bước **đánh giá sau train**, không truyền vào loss hoặc dùng chọn checkpoint. [Cách chạy, khác biệt validation/test và các biểu đồ](../docs/K2_OPTIMIZATION.md).

**Hai đối chứng còn thiếu cho pilot cũ:** dùng notebook07 (`aug`) và08 (`film`), mỗi notebook một phiên riêng. Gắn Cityscapes, ACDCdataset và output **Version2** của notebook `nguyenminhtoan/k2-frozen-evaluation-cityscapes-acdc`, trong đó có `k2_baseline_comparison_bundle.zip`. Chọn GPU + Internet rồi **Save Version → Save & Run All**. Không đổi ARM,20epoch,seed42,batch2,canvas512×1024 hoặc K1 no-fog/severity1–5; không dùng recipe40epoch để đối chiếu với B0/K2 cũ. Checkpoint best/last và full/share ZIP lưu sau mỗi epoch. ACDC chỉ đọc sau khi train đủ horizon và audit cùng nguồn/khởi tạo B0 thành công. Nếu không đủ9giờ phiên, tiếp tục bằng full ZIP của đúng arm; không cần train lại B0/K2. K3/K4 chưa chạy.

DeepLab có đủ aug như B2. Không còn phải đổi ARM hoặc PROFILE_ONLY trong notebook. Chỉ khai báo CODE_INPUT/CITYSCAPES_ROOT khi auto-find không rõ và RESUME_INPUT nếu tiếp tục kết quả trước. Run All sẽ lần lượt chạy ba chiến thuật độc lập của model đó, mỗi chiến thuật20epoch; không dùng weights baseline để học nối proposed.

Mỗi suite chuẩn bị/chuyển nhãn Cityscapes một lần, cùng train2975/val500, không ACDC input. Một ZIP chung lưu profile, source, dữ liệu đã chuẩn bị và checkpoint cả ba arm sau mỗi epoch. Gần hết cửa sổ phiên10giờ thì dừng an toàn; mở phiên mới, gắn ZIP và đặt RESUME_INPUT rồi Run All. Profile/epoch đã chốt được kiểm tra và tái sử dụng.

Hai model có tổng6lượt. Quota30giờ chung không được đọc tự động từ Kaggle; chưa bảo đảm đủ cả6lượt. Queue không tự giảm20epoch hoặc bỏ aug nếu chưa xong. Khi hết quota cần tiếp tục từ ZIP lúc có tài nguyên.

Sinh lại bằng `python -m scripts.build_kaggle_notebooks`, dùng **cùng code ZIP mới** cho hai notebook. [Hướng dẫn](../docs/TRAINING_GUIDE.md) · [Phương pháp](../docs/DG_METHOD.md)
