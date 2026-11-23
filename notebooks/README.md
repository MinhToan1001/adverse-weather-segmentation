# Notebook DG tự động cho hai model

| Notebook | Quy trình tự chạy |
| --- | --- |
| [01_dg_segformer_b2.ipynb](01_dg_segformer_b2.ipynb) | Profile GPU → baseline20 → aug20 → consistency20 |
| [02_dg_deeplabv3plus_resnet50.ipynb](02_dg_deeplabv3plus_resnet50.ipynb) | Profile GPU → baseline20 → aug20 → consistency20 |

DeepLab có đủ aug như B2. Không còn phải đổi ARM hoặc PROFILE_ONLY trong notebook. Chỉ khai báo CODE_INPUT/CITYSCAPES_ROOT khi auto-find không rõ và RESUME_INPUT nếu tiếp tục kết quả trước. Run All sẽ lần lượt chạy ba chiến thuật độc lập của model đó, mỗi chiến thuật20epoch; không dùng weights baseline để học nối proposed.

Mỗi suite chuẩn bị/chuyển nhãn Cityscapes một lần, cùng train2975/val500, không ACDC input. Một ZIP chung lưu profile, source, dữ liệu đã chuẩn bị và checkpoint cả ba arm sau mỗi epoch. Gần hết cửa sổ phiên10giờ thì dừng an toàn; mở phiên mới, gắn ZIP và đặt RESUME_INPUT rồi Run All. Profile/epoch đã chốt được kiểm tra và tái sử dụng.

Hai model có tổng6lượt. Quota30giờ chung không được đọc tự động từ Kaggle; chưa bảo đảm đủ cả6lượt. Queue không tự giảm20epoch hoặc bỏ aug nếu chưa xong. Khi hết quota cần tiếp tục từ ZIP lúc có tài nguyên.

Sinh lại bằng `python -m scripts.build_kaggle_notebooks`, dùng **cùng code ZIP mới** cho hai notebook. [Hướng dẫn](../docs/TRAINING_GUIDE.md) · [Phương pháp](../docs/DG_METHOD.md)

## Thành viên chạy tiếp DeepLab consistency

Dùng **[03_resume_deeplab_consistency.ipynb](03_resume_deeplab_consistency.ipynb)** cho ZIP DeepLab đã nghiệm thu có baseline20, aug20 và consistency4. Đây là wrapper resume độc lập; hai notebook phía trên phục vụ chạy suite mới hoặc resume thủ công.

1. Pull repo rồi import notebook03 vào một notebook Kaggle mới.
2. Thêm đúng Cityscapes với mount `/kaggle/input/datasets/kavithak1388/cityscapes/Cityscape`, cùng version đã dùng, có ảnh và gtFine train/val. Không cần ACDC hoặc code ZIP riêng.
3. Checkpoint `20/20/4` được chia thành các file dưới2GiB tại [GitHub Release](https://github.com/MinhToan1001/adverse-weather-segmentation/releases/tag/deeplab-dg-seed42-resume-e4). Giữ `RESUME_INPUT=None`: nếu không có input kết quả DeepLab, notebook tự tải các phần, kiểm SHA256 và ghép ZIP. Git pull chỉ lấy code/notebook; weights được tải lúc chạy, không nằm trong lịch sử Git.
4. Nếu đã có ZIP `dg_deeplabv3plus_resnet50_suite_seed42.zip` hoặc thư mục kết quả đã giải nén trong input, wrapper ưu tiên bản đó. Có nhiều bản DeepLab thì đặt RESUME_INPUT theo Copy path; không chọn tùy ý. Tải/kiểm hash thất bại sẽ dừng, không bắt đầu train mới.
5. Bật GPU/Internet và Save Version → Save & Run All. Với ZIP hiện tại, log phải báo tiếp tục consistency từepoch5 đến20; baseline/aug được kiểm tra và bỏ qua.
6. Khi có `suite_complete`, ba nhánh20/20, tải ZIP DeepLab mới và gửi lại để nghiệm thu.

Gói Release giữ weights/optimizer/RNG/log/config/source, loại ảnh và3.475mask Cityscapes khỏi gói công khai. Notebook tạo lại mask từ input Cityscapes của người chạy rồi kiểm SHA256 của từng bản đồ pixel, giữ nguyên manifest/config đã khóa. Điều kiện [Cityscapes](https://www.cityscapes-dataset.com/license/) hạn chế phân phối lại dữ liệu và phiên bản sửa đổi; checkpoint được bàn giao riêng với dataset.

Notebook lấy source snapshot đã khóa trong ZIP và kiểm hash, giữ nguyên config/LR/seed/20epoch. Wrapper03 được lưu trực tiếp trong repo, không thay source training và không sinh lại bằng generator của hai notebook train. [Chi tiết checkpoint và thời gian dự kiến](../results/dg_deeplabv3plus_resnet50_seed42/README.md).
