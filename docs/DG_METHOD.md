# Phương pháp DG đang triển khai

[Training guide](TRAINING_GUIDE.md) · [Cơ sở bài báo](../research/README.md) · [Kế hoạch](../research/PROJECT_COMPLETION_PLAN.md)

## Ranh giới dữ liệu và trạng thái

Chỉ Cityscapes train2975 để học, Cityscapes val500 để chọn best. Không dùng ACDC images/labels/reference/style statistics hoặc target BN. ACDC chỉ evaluate sau khi khóa cohort candidate. Theo các ZIP nghiệm thu22/11, [B2 đủ baseline/aug/consistency20](../results/dg_segformer_b2_seed42/README.md) đạt73,98%/74,37%/74,44% và [DeepLab baseline/aug20](../results/dg_deeplabv3plus_resnet50_seed42/README.md) đạt71,82%/72,08% Cityscapes val500; artifact DeepLab gần nhất có consistency4/20epoch. **Chưa có bằng chứng cải thiện mIoU trên ACDC**. CPU QA không thay thế benchmark GPU.

## Recipe train dùng chung

| Nội dung | Giá trị hiện hành / ý nghĩa |
| --- | --- |
| Epoch và chọn best | 20 epoch, không early stopping; best theo Cityscapes val mIoU |
| Khởi tạo | Encoder ImageNet đã pin, head mới; seed42 cho từng lượt |
| Batch / tích lũy gradient | Batch thật2, tích lũy4 minibatch rồi update; drop_last bỏ minibatch thiếu cuối epoch |
| Optimizer / LR | AdamW; encoder6e-5, head×10; weight decay0,01 |
| Scheduler | Warmup20 optimizer updates rồi poly power1 |
| Số học | FP16 AMP trên GPU; scaler nằm trong checkpoint; KL tính FP32 |
| Normalization | RGB theo ImageNet; BN cập nhật từ nguồn, SegFormer encoder dùng LayerNorm |
| Input và hình học | Canvas512×1024; giữ tỷ lệ, scale0,5–2/crop/pad/flip0,5 đồng bộ ảnh–mask |
| Loss | CE ở cả ba nhánh; consistency thêm KL theo bảng dưới |
| Checkpoint | best/last/previous theo epoch, kèm optimizer/scheduler/scaler/RNG; ZIP sau epoch hoàn thành |

BatchNorm (BN) chuẩn hóa bằng thống kê batch; accumulation chỉ cộng gradient nên không thay batch thật cho BN. AMP giảm chi phí số học; scheduler điều chỉnh LR theo tiến độ học. Recipe ở [source_20epoch.yaml](../configs/recipes/source_20epoch.yaml), preprocessing ở [shared_512.yaml](../configs/preprocessing/shared_512.yaml). Không thay recipe hoặc model giữa một run đang resume.

## Ba chế độ dùng chung

| ARM | Loss trên minibatch ghép cặp | Vai trò |
| --- | --- | --- |
| `baseline` | CE một view thường; không ghép cặp | C0 |
| `aug` | 0,5 CE(clean,y) + 0,5 CE(styled,y) | C1, control hai view |
| `consistency` | C1 + λ KL(stopgrad(p_clean) || p_styled) | C2 |

Trên minibatch không ghép cặp, cả ba dùng CE một view. `p_pair=0,5` chọn bằng RNG riêng theo seed/epoch/batch; C1/C2 có cùng lịch. Style dùng RNG riêng theo seed/epoch/index, không thay đổi RNG hình học hay dropout. Vẫn tạo styled tensor trong worker cho C1/C2 kể cả minibatch không ghép; chi phí CPU đó phải được tính khi profile.

[paired_source.yaml](../configs/dg/paired_source.yaml) là **recipe ứng viên**, chưa tối ưu bằng GPU: λ=0,1; brightness/contrast0,6–1,4; saturation0,5–1,5; gamma0,6–1,8; blur radius0–1px; Gaussian noise std0–0,015 trên RGB[0,1]. Không giả lập vật lý đủ bốn thời tiết và không gọi color jitter là tái lập ISSA/SHADE.

## Đồng bộ hình học và gradient

[dataset.py](../src/data/dataset.py) tạo clean view sau toàn bộ resize/crop/pad/flip, rồi [SourceStyle](../src/data/augmentations/source_style.py) chỉ đổi RGB. Hai view dùng một mask, mã0–18 và ignore255. Normalization giống nhau.

[backward_batch](../src/training/dg.py) forward clean, tính CE và giữ teacher log-probability đã detach, backward clean rồi giải phóng graph; sau đó forward/backward styled. **Optimizer chỉ step theo gradient accumulation của minibatch gốc**, không step thêm vì có hai view. KL tính FP32 sau nội suy đúng kích thước nhãn, chỉ trên pixel hợp lệ. Clean vẫn nhận gradient CE; chỉ nhánh tham chiếu của KL bị detach. All-ignore không tạo NaN.

B2/DeepLab không thêm layer hoặc teacher model thường trú; inference vẫn một forward kiến trúc ban đầu. C1/C2 có cùng thứ tự forward, dropout và BN updates; khác phần KL. C0 ít view/compute hơn, phải báo riêng. Trong DeepLab, giữ batch thật2/drop_last để nhánh pooled BN hợp lệ. Tích lũy gradient không thay thế batch thật. BN chỉ thấy nguồn hoặc view nguồn, không ACDC.

## Khởi tạo và khôi phục

Mỗi arm độc lập bắt đầu từ ImageNet encoder đã pin và head mới cùng seed, không train thêm từ baseline20epoch. `run/initialization.json` ghi hash safetensors ban đầu. Cohort yêu cầu hash này giống giữa các arm của cùng model; không yêu cầu B2 và ResNet cùng weights.

Model, optimizer, scheduler, scaler, RNG và epoch lưu trong checkpoint transactional. Config model/arm/p_pair/λ/style/source/data được đóng băng và đối chiếu khi resume; ngắt giữa epoch thì chạy lại epoch đó. Hai notebook cố định model tự chạy profile và cả ba arm; mỗi model một ZIP suite, từng arm có checkpoint riêng.

## Kiểm chứng và đánh giá

CPU QA: 4ảnh Cityscapes train/2val,64×96,random init,2batches/epoch; cả hai model chạy2epoch, ngắt/resume và chạy liền cho weights khớp từng giá trị. C0/C1/C2 có cùng hash khởi tạo trong từng kiến trúc. Đây là plumbing/serialization QA, không phải hội tụ/benchmark.

[profile_dg.py](../scripts/profile_dg.py) chỉ chạy khi có CUDA, đo nguồn và extrapolate Cityscapes val; weights pilot bị bỏ, không dùng cho lượt chính. Ước lượng giờ chưa gồm checkpoint/ZIP/setup và biến thiên runtime, cần giữ reserve30giờ. Profiler chưa được chạy trên GPU trong đợt triển khai CPU.

Sau đủ20epoch, launcher tạo frozen_candidate.json. [evaluate_dg.py](../scripts/evaluate_dg.py) yêu cầu cohort gồm6lượt chính B2-C0/C1/C2 và DeepLab-C0/C1/C2. Nó kiểm checkpoint, source split/code/seed, initial hash và recipe; candidate đổi sau seal sẽ bị từ chối trước khi đọc target. Đánh giá dùng FP32/native-size/noTTA; báo mIoU tổng/per-weather/per-class, calibration và ảnh lỗi. Historical ACDC feedback vẫn phải khai báo, không nhận test độc lập mới.
