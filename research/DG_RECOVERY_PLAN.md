[Kế hoạch hoàn thành B2 + DeepLab với 30 giờ GPU](PROJECT_COMPLETION_PLAN.md)

# Khôi phục giao thức DG từ kế hoạch gốc

Ngày chốt lại: 2026-11-19. Tài liệu này **thay thế hướng supervised ACDC trong BASELINE_PROTOCOL.md**. Câu hỏi chính: chỉ học đường phố Cityscapes, model có phân đoạn tốt ảnh thời tiết xấu chưa dùng để học hay không?

## 1. Ranh giới dữ liệu

- Train: Cityscapes official train 2.975 ảnh và nhãn của nó.
- Chọn checkpoint/hyperparameters: Cityscapes official val 500. Đây là tập selection, không gọi điểm Cityscapes này là independent test.
- ACDC: không dùng ảnh, nhãn, reference frames, style statistics, pseudo-labels, gradient hoặc target batch statistics khi train/chọn model. Chỉ đánh giá sau khi khóa phương pháp, config và checkpoint. Inference dùng eval(), không test-time adaptation.
- ImageNet pretraining được phép như kế hoạch ban đầu; ghi rõ nguồn/revision/hash. Luồng DG v1 hiện bắt đầu từ encoder ImageNet được pin, head mới. Không khởi tạo từ model đã học ACDC rồi train Cityscapes lại để gọi là DG.
- Giữ B2 theo quyết định đổi model gần đây. Đây là thay đổi backbone so với B0 trong file gốc, phải ghi trong báo cáo; B0 không được tự khôi phục vào luồng mới.

## 2. Cứu phần nào, loại phần nào khỏi bảng DG?

| Thành phần | Cách dùng |
| --- | --- |
| Dataset/labels, geometry đồng bộ, metrics, native evaluator, checkpoint/resume, Node UI | Tái sử dụng |
| B2 source/best cũ | Mốc source-only tham khảo; fit2787/monitor188 khác giao thức gốc train2975/val500; chưa phải baseline DG v1 mới |
| B2 adapt/best và các điểm ACDC dev 63,02% | Lịch sử supervised ACDC; loại khỏi bảng DG; không xóa weights hay sửa nhãn thành DG |
| Bản kế hoạch public Cityscapes checkpoint → ACDC FT20 | Hủy vai trò kế hoạch hiện hành; lưu lịch sử |
| Baseline DG mới | Chạy source-only đúng split bằng scripts.run_dg; chưa có kết quả GPU |

Local audit xác nhận config source cũ là source_only, train/selection đều Cityscapes; adapt là supervised_acdc_replay, train/selection dùng ACDC. Điều này không đảo ngược được bằng đổi tên checkpoint. Bằng chứng metadata/hash được lưu riêng, không đổi nguồn gốc run cũ.

## 3. Cơ sở phương pháp

1. [ISSA, WACV 2023](https://arxiv.org/abs/2210.10175), [code tác giả](https://github.com/boschresearch/ISSA): tăng đa dạng style từ dữ liệu nguồn, cố giữ semantic layout, áp dụng được CNN/Transformer. ISSA dùng StyleGAN2 inversion/masked-noise encoder; gamma hoặc color jitter đơn giản **không phải tái lập ISSA**. Dùng làm luận cứ cho giả thuyết về style, kiểm tra chi phí trước khi chọn baseline paper.
2. [VLTSeg, ACCV 2024](https://vltseg.github.io/), [paper Table 6](https://openaccess.thecvf.com/content/ACCV2024/papers/Hummer_Strong_but_simple_A_Baseline_for_Domain_Generalized_Dense_Perception_ACCV_2024_paper.pdf): downstream train Cityscapes và đánh giá ACDC không train ACDC. Giữ là ứng viên đối chứng paper theo file gốc. Kiểm tra checkpoint downstream Cityscapes, không dùng checkpoint trained ACDC hay nhầm val/test/TTA. Backbone/pretraining/compute khác B2 phải công bố, không nhận là ablation cùng kiến trúc.
3. [ALDM, ICLR 2024](https://yumengli007.github.io/ALDM/), [code](https://github.com/boschresearch/ALDM): sinh ảnh theo layout/nhãn nguồn để tăng đa dạng và kiểm tra DG Cityscapes→ACDC. Là hướng mở rộng tốn compute; chưa triển khai, không tải generator hoặc chạy sinh ảnh hàng loạt ở bước khôi phục.

4. [SHADE, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880530.pdf), [code tác giả](https://github.com/HeliosZhao/SHADE): nguồn tham khảo trực tiếp cho style diversification + consistency trong DG segmentation. Bài gốc synthetic→real, khác Cityscapes→ACDC; proposal B2 hai view dưới đây không phải tái lập đầy đủ SHM/SC/RC của SHADE và không được so điểm chéo như cùng benchmark.

Không dùng HRDA/Refign nguyên thiết lập UDA làm đối chứng DG trực tiếp. Không cần chạy thêm DeepLab để sửa sai giao thức.

## 4. Hướng cải tiến B2 được đề xuất để kiểm chứng

Giả thuyết: B2 phụ thuộc nhiều vào màu/độ sáng của Cityscapes. Khi thay ngoại hình nhưng giữ nội dung và ràng buộc dự đoán ổn định, model có thể bền vững hơn trước dịch chuyển miền.

Tạo hai view **cùng một ảnh Cityscapes và cùng hình học**: một view thường, một view thay ngoại hình RGB. Bước nhẹ có thể dùng gamma/brightness/contrast/saturation, noise/blur được kiểm soát; mô phỏng sương/mưa/tuyết chỉ bổ sung khi kiểm tra giữ nhãn hợp lý. Gamma đơn lẻ không mô phỏng đủ đêm/sương/mưa. Không lấy style hay điều chỉnh cường độ bằng ảnh/điểm ACDC.

| Arm | Cách học | Vai trò |
| --- | --- | --- |
| C0 | B2 + CE, augmentation hình học chuẩn | Baseline DG source-only |
| C1 | CE trung bình trên view thường và view đổi style | Đối chứng augmentation và chi phí hai forward |
| C2 | C1 + consistency loss giữa hai dự đoán, nhánh tham chiếu stop-gradient | Proposed candidate; so C2/C1 để tách hiệu quả consistency |

Consistency nói nôm na: cùng một con đường, dù đổi màu/độ sáng thì model nên dự đoán cùng lớp. Công thức dự kiến: 0,5 CE(clean,y) + 0,5 CE(styled,y) + λ KL(stopgrad(p_clean) || p_styled), chỉ trên pixel nhãn hợp lệ. Đây là **đề xuất chưa triển khai/chưa chứng minh**, không được nhận là phương pháp hoàn toàn mới. Kiểm tra văn liệu consistency cụ thể, chốt λ/mức augmentation bằng dữ liệu nguồn, rồi đóng băng thiết kế trước khi train C1/C2.

Giữ cùng ImageNet init, split, seed, resolution, source-image sampling và 20 epoch. C1/C2 có hai forward; C0 một forward nên phải báo compute riêng. Không quy mọi chênh lệch C0/C2 cho module consistency. C2 so C1 là ablation chính. Không tăng epoch/LR hoặc đổi loss theo điểm ACDC trong quá trình thử.

## 5. Thứ tự thực hiện và điểm dừng

1. Chặn launcher supervised mặc định; tách notebook/kết quả lịch sử, sửa nhãn demo. **Đã triển khai.**
2. Chuẩn bị DG baseline Cityscapes train2975/val500; xác minh dataset/config/checkpoint policy trên CPU. Notebook và launcher source-only đã tạo; chưa có kết quả GPU.
3. Khi có GPU, chạy C0 20 epoch với checkpoint/resume. Chốt best bằng Cityscapes val; không tự evaluate ACDC mỗi epoch.
4. Hoàn thiện nguồn tham khảo consistency, chốt config và test semantics của C1/C2, sau đó chạy ablation cùng protocol. C1/C2 hiện chưa có code training hoàn chỉnh.
5. Khóa danh sách candidate/config/hash, sau đó evaluate cùng ACDC val406: mIoU tổng, fog/night/rain/snow, per-class, latency/params và ảnh lỗi. Không dùng kết quả này để quay lại chọn candidate hoặc tune vòng tiếp.
6. Đối chứng VLTSeg checkpoint đúng source nếu khả thi; nếu không, ghi blocker và số công bố ở bảng riêng. Không điền số paper như kết quả project tự chạy.

## 6. Giới hạn không thể xóa bằng khởi động lại

ACDC dev và các kết quả val đã từng được xem trong quá trình phát triển. Khởi động model mới không làm lịch sử đó biến mất. Luồng mới có thể **không dùng target trong huấn luyện và selection**, nhưng ACDC val chưa trở thành test hoàn toàn độc lập. Ghi lịch sử này, đăng ký trước cấu hình, và dùng official test qua evaluator hoặc một tập ngoài chưa xem để tăng độ tin cậy khi điều kiện cho phép. Không đưa một phần ACDC đã từng xem thành “test mới” chỉ bằng chia lại.

Mục tiêu là kiểm chứng khái quát hóa thật theo DG; không hứa trước vượt paper hoặc vượt model supervised đã học ACDC.
