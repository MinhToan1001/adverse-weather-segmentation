# Notebook DG tự động cho hai model

| Notebook | Quy trình tự chạy |
| --- | --- |
| [01_dg_segformer_b2.ipynb](01_dg_segformer_b2.ipynb) | Profile GPU → baseline20 → aug20 → consistency20 |
| [02_dg_deeplabv3plus_resnet50.ipynb](02_dg_deeplabv3plus_resnet50.ipynb) | Profile GPU → baseline20 → aug20 → consistency20 |
| [03_k1_weather_preview.ipynb](03_k1_weather_preview.ipynb) | K1 độc lập: CPU preview thời tiết mức1–5; không train |
| [04_k2_weatherseg_b0_train.ipynb](04_k2_weatherseg_b0_train.ipynb) | Train K1 + FiLM B0 + condition head; GPU, ZIP/resume |

K1 là phần mở rộng trên nhánh riêng. Notebook03 chỉ cần CPU + Internet, có demo với khoảng cách tổng hợp đã biết và preview ảnh Cityscapes train nếu được gắn vào input. Fog cần metric distance hoặc disparity/camera; thiếu độ sâu thì bỏ riêng fog. [Cách chạy và giới hạn](../docs/K1_WEATHER.md).

Notebook04 nằm **cùng nhánh K1** và dùng trực tiếp generator K1. Mặc định train một arm `k2` trong20epoch. `WEATHER_MODE=auto` loại fog khi không có độ sâu; đặt `full` và cung cấp depth để train đủ năm nhãn. [Cách chuẩn bị dữ liệu, ablation và resume](../docs/K2_TRAINING.md).

DeepLab có đủ aug như B2. Không còn phải đổi ARM hoặc PROFILE_ONLY trong notebook. Chỉ khai báo CODE_INPUT/CITYSCAPES_ROOT khi auto-find không rõ và RESUME_INPUT nếu tiếp tục kết quả trước. Run All sẽ lần lượt chạy ba chiến thuật độc lập của model đó, mỗi chiến thuật20epoch; không dùng weights baseline để học nối proposed.

Mỗi suite chuẩn bị/chuyển nhãn Cityscapes một lần, cùng train2975/val500, không ACDC input. Một ZIP chung lưu profile, source, dữ liệu đã chuẩn bị và checkpoint cả ba arm sau mỗi epoch. Gần hết cửa sổ phiên10giờ thì dừng an toàn; mở phiên mới, gắn ZIP và đặt RESUME_INPUT rồi Run All. Profile/epoch đã chốt được kiểm tra và tái sử dụng.

Hai model có tổng6lượt. Quota30giờ chung không được đọc tự động từ Kaggle; chưa bảo đảm đủ cả6lượt. Queue không tự giảm20epoch hoặc bỏ aug nếu chưa xong. Khi hết quota cần tiếp tục từ ZIP lúc có tài nguyên.

Sinh lại bằng `python -m scripts.build_kaggle_notebooks`, dùng **cùng code ZIP mới** cho hai notebook. [Hướng dẫn](../docs/TRAINING_GUIDE.md) · [Phương pháp](../docs/DG_METHOD.md)
