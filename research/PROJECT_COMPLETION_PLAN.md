# Kế hoạch hoàn thành DG với B2 và DeepLabV3+ ResNet50

Lập ngày 19/11/2026. GPU dự kiến hồi khoảng **21/11**, ngân sách người dùng cung cấp **30 giờ**. Hai ngày chờ dành cho code/QA và khóa thiết kế; chưa chạy GPU hoặc nhận là kỹ thuật mới đã có hiệu quả. Kế hoạch này mở rộng [DG recovery plan](DG_RECOVERY_PLAN.md) để đưa DeepLab trở lại so sánh có kiểm soát.

## 1. Câu hỏi và điều kiện hoàn thành

- Học Cityscapes có giúp B2 khái quát sang thời tiết xấu chưa học được đến đâu?
- Thêm đa dạng ngoại hình nguồn và consistency có cải thiện so với source-only không?
- Áp cùng kỹ thuật cho DeepLabV3+ ResNet50 có tác dụng ra sao? So sánh **mức tăng trong từng model** và điểm cuối giữa hai model.
- Hoàn thành khi có checkpoint tái nạp được, cùng protocol và số liệu của bốn lượt chính, một ablation B2, bảng per-class/per-weather/compute, hình lỗi, demo Node.js và tài liệu truy vết.
- Nếu đủ nguồn lực, thêm ablation DeepLab, seed bổ sung và đối chứng paper. Không điền kết quả chưa chạy; không yêu cầu trước một ngưỡng mIoU hay thắng paper.

## 2. Giao thức dữ liệu khóa trước training

Cityscapes official train2975 để học; Cityscapes official val500 để chọn best. ACDC không dùng ảnh/nhãn/reference/style statistics, pseudo-label hoặc batch statistics trong training/selection. Sau khi khóa toàn bộ candidate và config mới evaluate ACDC val406. Không dùng kết quả ACDC để chọn giữa C0/C1/C2 hay tune vòng tiếp. Official test hoặc tập ngoài chưa xem là bước xác nhận thêm khi có điều kiện; phải khai báo lịch sử ACDC feedback.

Mỗi lượt train **20 epoch**, CE control, cùng 512×1024, 19 trainIds/ignore255, geometry và normalization. Tất cả các arm của **cùng kiến trúc** khởi tạo từ cùng ImageNet encoder và cùng seed cho head mới; xác minh hash initial weights. **Không train baseline20 rồi lấy best train DG thêm20 để so với baseline20.** Cách đó thành40vs20epoch, không tách được tác dụng phương pháp.

MiT-B2 và ResNet50 có pretraining ImageNet riêng, phải ghi nguồn; không gọi là weights ban đầu giống nhau giữa hai kiến trúc. Recipe chung là đối chứng kiểm soát, không chứng minh từng model đã được tối ưu riêng. Nếu cần pilot LR phải dùng Cityscapes, cùng ngân sách tìm kiếm cho hai kiến trúc, chốt trước các lượt chính và giữ nguyên giữa các arm của mỗi model.

## 3. Kỹ thuật ưu tiên: tăng đa dạng ngoại hình + consistency

Tạo hai view của cùng ảnh Cityscapes sau **cùng geometry**: view thường và view biến đổi RGB, cùng mask. Thay đổi độ sáng/gamma/contrast/saturation, noise/blur ở mức còn giữ nhãn hữu ích. Bản v1 không bắt buộc diffusion/StyleGAN hoặc mô phỏng đủ cả bốn thời tiết. Nếu bổ sung fog/rain/snow, kiểm tra occlusion/semantic validity bằng ảnh nguồn, không điều chỉnh theo ACDC. Đổi độ sáng đơn lẻ không được coi là mô phỏng đầy đủ đêm thật.

**Consistency**: buộc dự đoán của hai view cùng nội dung gần nhau. Đề xuất nhẹ: clean prediction làm tham chiếu stop-gradient cho KL, pixel255 bị bỏ qua. Đó là hypothesis để kiểm chứng, không phải công thức bảo đảm tăng mIoU. View clean vẫn có gradient từ CE của chính nó; stop-gradient chỉ áp cho vai trò teacher trong KL.

Để cân đối30giờ, đề xuất **p_pair=0,5**: một nửa minibatch dùng cặp view, phần còn lại học như control. C1/C2 dùng đúng cùng lịch lấy mẫu/cặp view. Trên minibatch ghép cặp:

- C1: L = 0,5 CE(clean,y) + 0,5 CE(styled,y).
- C2: L = L_C1 + λ KL(stopgrad(p_clean) || p_styled).
- C0: CE với augmentation hình học chuẩn, không extended RGB style hoặc KL.

λ và cường độ style phải được chốt bằng nguồn trước lượt chính; chưa có giá trị tối ưu được xác minh. Khi profiling, nếu dự trù5lượt vượt phần giờ train, có thể giảm p_pair đồng loạt cho C1/C2 **trước khi train bất kỳ lượt chính nào**, ghi decision log; không thay theo kết quả ACDC hay giữa lượt. Không tự giảm20epoch hoặc resolution khi hết quota.

C1 là control cùng số view/forward để xem tác dụng thêm của KL trong C2. C0/C1 khác chi phí và số view mỗi mẫu nên phải báo riêng; cùng epoch không là cùng compute. Sequential forward/backward có thể giảm bộ nhớ nhưng phải giữ đúng tổng loss, điểm cập nhật optimizer và detached teacher.

Nguồn tham khảo:

- [ISSA, WACV2023](https://arxiv.org/abs/2210.10175): style diversity từ nguồn và giữ layout; bản nhẹ của project không phải tái lập StyleGAN2/masked-noise encoder của ISSA.
- [SHADE, ECCV2022](https://arxiv.org/abs/2204.02548), [code tác giả](https://github.com/HeliosZhao/SHADE): kết hợp style và consistency trong DG segmentation; paper gốc synthetic→real, không là benchmark cùng split với Cityscapes→ACDC của mình. Không gọi công thức KL hai view là tái lập đầy đủ SHADE.
- [VLTSeg, ACCV2024](https://vltseg.github.io/): ứng viên đối chứng paper DG ngoài cặp model nội bộ. Chỉ đánh giá checkpoint downstream Cityscapes khi xác minh được dữ liệu/checkpoint/inference. Model/pretraining/compute khác, bảng paper reference riêng.

## 4. Ma trận chạy

| ID | Model | Arm | Epoch | Ưu tiên / mục đích |
| --- | --- | --- | ---: | --- |
| B2-C0 | SegFormer-B2 | CE source-only | 20 | Bắt buộc, baseline B2 |
| DL-C0 | DeepLabV3+ ResNet50 | CE source-only | 20 | Bắt buộc, baseline CNN để đo mức tăng của nó |
| B2-C1 | B2 | Hai view + CE, không KL | 20 | Ablation chính để cô lập consistency |
| B2-C2 | B2 | C1 + KL consistency | 20 | Bắt buộc, B2 với kỹ thuật DG |
| DL-C2 | DeepLabV3+ ResNet50 | C1 + KL consistency, cùng thiết kế | 20 | Bắt buộc, chuyển kỹ thuật sang CNN |
| DL-C1 | DeepLabV3+ ResNet50 | Hai view + CE, không KL | 20 | Mở rộng nếu còn giờ; chưa có thì không quy riêng lợi ích KL trên DeepLab |

Mục tiêu đợt30giờ: **5lượt đầu**. Bốn lượt C0/C2 tạo bảng so sánh hai kiến trúc; B2-C1 là ablation thêm. Chênh lệch chính: B2-C2−B2-C0, DL-C2−DL-C0, B2-C2−B2-C1. Một seed42 là lượt chính; chưa đủ chứng minh độ ổn định qua seed. Seed bổ sung thuộc phần mở rộng, không thay kết quả còn thiếu bằng số cũ.

## 5. Các phần dùng chung và riêng

Giữ riêng file model B2 và DeepLab hiện có. Viết augmentation source-only, paired-view dataset/collate, CE/KL, checkpoint và report dùng chung. Config chọn `model`, `arm`, seed và output; không copy trainer vào từng model.

Cập nhật triển khai CPU ngày19/11: run_dg và hai notebook đã hỗ trợ B2/DeepLab, đủ baseline/aug/consistency. CPU QA đã qua, chưa có full GPU run. Xem [phương pháp](../docs/DG_METHOD.md). Chuẩn bị hai notebook cố định model, mỗi lượt chọn ARM; mỗi run có tên/archive riêng, không dùng source→ACDC notebook lịch sử. Resume cần model/arm/protocol/config/source/hash trùng nhau.

DeepLab có BatchNorm, khác LayerNorm ở encoder B2. Giữ batch thật tối thiểu2 cho nhánh pooled BN, cùng chính sách BN và thứ tự clean/styled giữa C1/C2. Gradient accumulation không biến batch1 thành batch2 cho BN. Mọi thống kê chỉ lấy từ Cityscapes/view nguồn; inference eval(), không cập nhật theo ACDC. Không đổi BN thành GN chỉ ở một arm rồi quy chênh lệch cho DG.

## 6. Lịch CPU trong hai ngày chờ GPU

**19–20/11 — chưa dùng GPU:**

1. Hoàn thiện source-only paired-view pipeline và CLI chọn model/arm. Khóa semantics mask/geometry, seed và lịch p_pair; không quét ACDC.
2. Cài CE/KL FP32 trên pixel hợp lệ; kiểm tra KL teacher detached, all-ignore an toàn, spatial alignment, valid labels. Tắt KL phải tái hiện C1. Chế độ test tắt dropout để kiểm tra hai view giống nhau cho KL gần0.
3. Kiểm thử forward/backward/save/resume cả hai model bằng ảnh Cityscapes nhỏ. Kiểm optimizer-update count, AMP/scaler, BN và equality khi ngắt; không nhận đây là kiểm tra hội tụ.
4. Làm evaluator freeze-only và report cho5lượt; kiểm checkpoint hash và không có target trong selection. Chuẩn bị manifest ACDC phục vụ evaluation riêng sau freeze, không trộn vào gói train.
5. Sinh notebook/code ZIP cuối; khóa run matrix, recipe, augmentation và loss. Chuẩn bị checklist Kaggle và xuất kết quả tự động. Chưa chạy thử nhiều cấu hình bằng điểm ACDC.

**Khoảng21/11 — GPU có lại:** profile cả hai model và C0/C2 trên **Cityscapes train**, kể cả validation500 và ZIP overhead; pilot không dùng chọn model theo target. Khóa recipe sau pilot và tạo run mới từ init đúng; không lấy pilotcheckpoint nhiều bước hơn làm arm chính. Sau đó chạy theo ma trận, theo dõi lỗi/NaN/ETA, giữ best theo Cityscapes val.

**Sau khi training xong:** freeze danh sách candidate trước ACDC evaluation, tổng hợp bảng/ảnh/video, cập nhật Node UI bằng weights DG thật và hoàn thiện báo cáo/slide. Nếu vẫn giữ deadline24/11 trong file gốc, cần kết thúc training sớm để dành ngày cuối cho kiểm tra và bàn giao.

## 7. Phân bổ30giờ và quy tắc dừng

Các mốc dưới là **phân bổ dự kiến**, không phải ETA đã đo cho pipeline mới:

| Khoản | Giờ dự kiến |
| --- | ---: |
| Setup + profiling hai kiến trúc/paired views | 1,0 |
| B2-C0 | 3,5 |
| B2-C1 | 5,0 |
| B2-C2 | 5,0 |
| DL-C0 | 4,0 |
| DL-C2 | 6,0 |
| Evaluate, latency, export cuối | 3,0 |
| Dự phòng lỗi/resume | 2,5 |
| Tổng | 30,0 |

Dữ kiện duy nhất đã đo: B2 source cũ median8,36phút/epoch trên train2787/monitor188, tức khoảng2,79giờ cho20epoch nếu nhân median, chưa tính setup/ZIP. DG mới có train2975/val500 và paired views nên không dùng con số cũ làm lời hứa. DeepLab chưa có ETA train thật.

Sau profiling, tính lại cả5lượt và phần đánh giá/xuất file, giữ reserve. Không khởi chạy job nếu dự báo tổng vượt giờ còn lại trừ reserve. DL-C1 và seed thêm ưu tiên sau5lượt chính. Nếu5lượt vẫn không vừa, chốt phần phải hoãn và ghi rõ thiếu thí nghiệm; không cắt epoch rồi đưa vào bảng đủ20, không lựa chọn cắt bỏ theo điểm ACDC. Quota30giờ và thời lượng tối đa một session là hai giới hạn riêng; lưu mỗi epoch và chia phiên khi cần. Trainer hiện chỉ dùng một GPU; tính ngân sách theo tài nguyên thực tế được cấp.

## 8. Evaluate và kết luận được phép

Bảng chính có Cityscapes val, ACDC overall/fog/night/rain/snow, per-class IoU, số tham số, peakVRAM, giờ train và latency/FPS cùng input/hardware. Overall mIoU tính từ ma trận nhầm lẫn toàn tập, không trung bình cơ học bốn điểm thời tiết. Tách load model và inference time; warm-up GPU, đồng bộ đo và báo batch1/resolution/noTTA rõ ràng.

Vẽ learning curves, per-weather/per-class deltas, RGB/GT/pred/error panels và các trường hợp model giảm chất lượng. Độ tự tin/ECE bổ sung, không thay mIoU. Confidence trên ảnh upload không phải độ chính xác có GT.

Nếu C2 tốt hơn C0 nhưng không hơn C1, không kết luận consistency là nguyên nhân. Nếu DeepLab không tăng, báo giới hạn chuyển kiến trúc. Nếu một seed chênh nhỏ, không tuyên bố vượt trội chắc chắn. ACDC val đã được xem trong lịch sử, phải khai báo; không gọi test độc lập mới. Đối chứng paper evaluate được thì báo nguồn/checkpoint/inference; không evaluate được thì dùng số công bố ở bảng riêng và nêu blocker.

## 9. Gói bàn giao

- 5run chính và config/source/initial/checkpoint hashes, hoặc danh sách phần chưa hoàn thành có lý do.
- DG guard và tests; model files riêng, data/loss/trainer/evaluator dùng chung.
- Bảng kết quả, ablation, thời gian/params/VRAM, biểu đồ và phân tích ảnh lỗi.
- Node.js demo: B2/DeepLab source-only vs DG, nhãn protocol rõ, ảnh mẫu/upload/mask/overlay/latency, video ngắn.
- README tái lập, protocol, related work đúng vai trò, báo cáo và slide. Giữ lịch sử supervised riêng, không sử dụng làm bằng chứng DG.
