# Bằng chứng CPU của phương pháp DG

[Phương pháp](../../docs/DG_METHOD.md) · [Record gốc](verification.json) · [Suite tự động hiện hành](../dg-suite-automation/README.md)

Record này giữ kết quả kiểm tra phương pháp ở lần triển khai trước đợt dọn repo. Nó gồm gradient clean/style, teacher detach, bỏ ignore255, RNG, BN, khởi tạo và tính toàn vẹn cohort. Số test trong JSON là số của lần kiểm tra đó, không phải số test hiện tại.

CPU QA thật dùng4ảnh Cityscapes train/2val,64×96,random init,2batches/epoch. B2 và DeepLab đều chạy2epoch: ngắt/resume cho weights khớp từng giá trị với chạy liền. Hash weights ban đầu của ba chiến thuật giống nhau trong mỗi kiến trúc. Config sáu nhánh qua guard trên manifest Cityscapes train2975/val500.

JSON gốc còn mô tả giao diện manual-arm trước đây; **hiện dùng hai notebook tự chạy đủ ba chiến thuật và khóa sáu candidate**, theo [training guide](../../docs/TRAINING_GUIDE.md). Không chạy theo các cờ notebook trong record lịch sử.

Không có full GPU timing, AMP/convergence benchmark hoặc bằng chứng tăng mIoU trong QA này. Checkpoint probe, preview ảnh và full log nằm local/Git-ignored dưới `outputs/dg-implementation/`.
