# Research method: matched models and class-aware learning

[Project](../README.md) · [Code walkthrough](PRESENT_TO_LECTURER.md) · [Three notebooks](../notebooks/README.md)

**Current execution:** [three separate model runs](TRAINING_GUIDE.md),each source20→adapt20 from an ImageNet encoder/new head. Old project checkpoints/scores are excluded. Transactional checkpoints retain previous generations;atomic resultZIP updates every epoch.

## Research question and evidence

**Trong cùng pipeline và quyền dùng nhãn, học tốt các lớp yếu có tăng ACDC mIoU mà vẫn giữ Cityscapes không? Tác dụng có lặp trên Transformer và CNN gần nhau về số tham số không?**

B0 là baseline; B2 là hướng chính. DeepLabV3+ResNet50 có26.695.619 tham số, B2 có27.361.235, chênh−2,43%. Bản project dựa trên [DeepLabV3+](https://arxiv.org/abs/1802.02611), không tái lập nguyên recipe/checkpoint của tác giả. Cặp B2/CNN và vấn đề forgetting được tham khảo [Kalb & Beyerer, CVPR2023](https://arxiv.org/abs/2303.14115); supervised replay gộp thời tiết của project khác thí nghiệm domain-incremental của paper.

## Chẩn đoán trước khi đổi loss

CE tính lỗi từng pixel; road/building có diện tích lớn có thể chi phối gradient. mIoU lại cho mỗi lớp trọng số tương đương. Vì vậy cần kiểm tra mất cân bằng, không suy ra CE yếu chỉ từ công thức đơn giản.

| Quan sát | Giả thuyết cần đo | Can thiệp kiểm thử |
| --- | --- | --- |
| Ít fit pixels, ít crop chứa lớp | Mất cân bằng/đối tượng bị cắt | WeightedCE; class-aware crop |
| Pixel đủ nhưng hay nhầm cùng lớp | Đặc trưng khó/thiếu ngữ cảnh | B2 vs CNN; confusion/panels |
| Vật nhỏ biến mất sau resize | Giới hạn độ phân giải | Audit ảnh/nhãn sau geometry; giữ resolution chung ở controls |
| Train tốt hơn, dev xấu đi | Overfit/domain shift | Dev CE,mIoU,source retention,best đã định trước |
| Focal tập trung ranh giới tối | Hard pixel có thể là nhiễu | Xem error panels; không tự tăng gamma |

Nhật ký `crop_exposure` ghi số lần chọn lớp, số crop đạt ngưỡng và số pixel giữ được. Đây là exposure, không là mIoU. Không phải lớp điểm thấp nào cũng ít xuất hiện.

## Loss presets

Model trả logits B×19×H×W: điểm dự đoán cho19lớp ở từng pixel. [losses.py](../src/training/losses.py) nội suy bilinear đến nhãn, tính FP32, bỏ void255.

| Preset | Thiết kế | Nghĩa nôm na | Rủi ro |
| --- | --- | --- | --- |
|`ce`|−mean(log p_y)|Phạt pixel dự đoán sai|Lớp nhiều pixel chi phối|
|`weighted_ce`|−Σw_y log p_y /Σw_y|Lớp ít xuất hiện được coi trọng hơn|Tăng cả nhiễu của lớp hiếm|
|`focal`|−mean((1−p_y)^2 log p_y)|Bớt học pixel dễ, tập trung pixel khó|Hard không đồng nghĩa hữu ích|
|`ce_dice`|CE+0.5×macro(1−Dice)|Vừa đúng pixel vừa khớp vùng từng lớp|Nhạy với lớp nhỏ/crop thiếu lớp|

WeightedCE: trung bình priors pixel của các fit manifest (nguồn/đích1:1), inverse-square-root, chặn max/min≤5, chuẩn hóa trung bình lớp có mặt=1; lớp không có trong fit nhận0. Chỉ tính full fit masks, **không dùng IoU dev/val**. Không phải effective-number Class-Balanced Loss. Weights/manifest có SHA; full-mask priors chưa phản ánh đúng mọi crop.

Dice: overlap xác suất/one-hot trên pixel hợp lệ, trung bình các lớp có GT trong minibatch, smoothing1e-6. Lớp vắng GT không tham gia Dice, CE vẫn phạt đoán sai vào lớp đó. Kế thừa ý tưởng overlap từ [V-Net](https://arxiv.org/abs/1606.04797), không tái lập công thức volumetric. Focal kế thừa giảm trọng số ví dụ dễ từ [Lin et al.](https://arxiv.org/abs/1708.02002), áp dụng multiclass segmentation; chưa chứng minh trên ACDC.

All-void batch trả loss0/gradient0 hữu hạn. Không trừ train Focal/Dice với dev CE để đo overfit vì khác objective.

## Class-aware crop and processing

Control: scale0.5–2, crop/pad512×1024, flip0.5; ảnh/mask cùng tọa độ, RGB bilinear/mask nearest; ImageNet mean/std. Dev/monitor không augment ngẫu nhiên.

`--class-crop`: xác suất0.5 chọn lớp có trong mask theo inverse-square-root pixel count **của ảnh đó**; tối đa10 vị trí crop giữ≥64pixel lớp đã chọn. Nếu không đủ thì giữ crop tốt nhất và ghi `met=false`; không có lớp hợp lệ thì dùng crop control. Nhãn chỉ từ train. Crop không tạo lại thông tin đã mất khi downsample.

Photometric/gamma cũ vẫn có module nhưng không bật mặc định ở controls mới. CE+crop thử riêng trước khi ghép với loss thắng.

## Data and fairness

- Cityscapes official-train2.975 → grouped fit2.787/monitor188. Monitor không học trong source mới/replay.
- ACDC official-train1.600 → recording/scene-token fit1.198/dev402. Clear-reference đã kiểm kê nhưng chưa dùng trong objective.
- Official-val500City/406ACDC kiểm tra candidate đã khóa. Lịch sử feedback khiến đây chưa là independent test.
- Source mới: ImageNet1k encoder, head19lớp mới, cùng20epochs Cityscapes. MiT pin revision+SHA; TorchVisionR50 `IMAGENET1K_V1` filename hash. Recipe ImageNet từng encoder vẫn khác.
- Adapt dùng đúng source CE `best/` sau20epochs, đúng model và fit/monitor; mọi ablation cùng model cùng checkpoint.
- Cùng epoch/updates không là cùng FLOPs/giờGPU. Counted forwardFLOPs batch1,512×1024: B035,51G,B2258,33G,DeepLab150,71G;đếm conv/matmul hỗ trợ bằng Torch FlopCounterMode. DeepLab−41,7% soB2,cùng bậc quy mô nhưng không compute-matched. VRAM/timing còn chờ GPU. Recipe chung có thể chưa tối ưu từng model;model-specific LR search cần nghiên cứu/budget riêng. [Profile](../results/repository-release/complexity.json).

## Training policy

| Setting | Source | Adapt |
| --- | --- | --- |
|Horizon / early stopping|20 /Off|20 /Off|
|AdamW /weight decay|6e-5 /0.01|1e-5 /0.01|
|Head LR multiplier|10|1|
|Scheduler|Warm-up20updates,poly power1|Như source|
|Batch /accumulation|2 /4,drop_last|1 /8|
|BN running stats|Update|Frozen|
|BN affine parameters|Learn|Learn|
|Precision|GPU AMPfloat16|GPU AMPfloat16|
|Execution|Single GPU|Single GPU|

LayerNorm encoder SegFormer không có running mean/variance kiểu BN. Accumulation cộng gradient nhưng không gộp thống kê BN. DeepLab global pooling BN1×1 cần batch thật≥2. Source batch2 phải kiểm tra VRAM T4; chưa qua fullGPU run.

Source head19lớp khởi tạo mới nên dùng LR6e-4,encoder6e-5: tỷ lệ×10 lấy từ [config SegFormerB2 chính thức](https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B2/segformer.b2.1024x1024.city.160k.py), cũng là hướng từng kiểm tra trên B0. Áp cùng tỷ lệ lên DeepLab là control thống nhất, chưa bảo đảm tối ưu CNN. Adapt dùng multiplier1 vì head đã học. Project giữ warm-up20updates,weight decay0.01 cả norm như controls cũ;khác warm-up1500/normalization-decay0 trong recipe nguồn của tác giả,không gọi là paper reproduction. Norm-decay0/warm-up fraction là ablation optimizer tiếp theo nếu source diagnostics cho thấy cần.

Source:1.393minibatches/epoch,349planned optimizer updates;20epochs=6.980. Adapt:2.396draws,300updates/epoch,6.000planned. Batch hữu hiệu cuối có thể nhỏ hơn8;AMP overflow có thể bỏ update, logs ghi số thực tế.

Lưu `initial/`, `best/`, `last/`;last cóoptimizer/scheduler/scaler/RNG/config. Best được chọn trong20epochs, không mặc định epoch20. Adapt source gate giảm≤1pp; nếu không epoch nào đạt thì best là fallback, phải báo failed gate.

Session controller: max3epoch gần nhất×1.5+15phút reserve,window10h. Stop giữa epoch,resume cùng source/config/mount. Thời gian bất thường vẫn có thể vượt;hard-kill không chạy ZIP finally. Runner chỉ nhận20epoch hoàn tất khi checkpoint/config/log commits khớp.

## Predeclared experiment matrix

| Phase | Jobs | Mục đích |
| --- | --- | --- |
|Baselines|B0/B2/DeepLab sourceCE20|B0 mốc nhỏ,source công bằng cho2model phát triển;không trộn B0 lịch sử30 vào bảng20|
|Architecture control|B0/B2/DeepLab adaptCE20|Đo kiến trúc trước can thiệp loss|
|Loss ablation|B2 weightedCE,Focal,CE+Dice,mỗi job20|So B2CE,cùngsource seed42|
|Crop ablation|B2 CE+classcrop20|Tách tác dụng crop khỏi loss|
|Combination|Dev audit trước,loss thắng+crop sau|Kiểm tra tương tác,không cộng gain giả định|
|Transfer|DeepLab cùng can thiệp đã chọn|Kiểm tra không chỉ có tác dụng trênB2|
|Repetition|Seeds43/44 cho control/candidate cả2model|Đo dao động;chưa tự xếp hàng toàn matrix|
|Frozen evaluation|Cityval500+ACDCval406|19IoU,4thời tiết,retention,ECE,paired groupCI|

Không chọn loss bằng official-val. Không sửa gamma/weights khi resume. Báo selected-best và cùng epoch20; không nhầm selected-best gain với cùng-training-time gain.

## Contribution and limits

Thay CE bằng Focal/Dice,tăng B0 lênB2,dùng crop đều là kỹ thuật có trước. Đóng góp tiềm năng cần chứng minh: **chiến lược có đối chứng giúp lớp yếu, giữ source và lặp trên2kiến trúc**, dưới compute/resolution công bố. Không đặt mốc80% nếu paper dùng protocol khác.

Distillation,Lovasz/IoU loss,boundary-aware loss,reference-image learning,photometric/weather augment còn là hướng sau. Không ghép hết vì sẽ không biết yếu tố nào giúp. HRDA/Refign có quyền dữ liệu và recipe khác;related work không là bảng superiority.

CPU QA:4fit/2monitor Cityscapes thật,64×96,2batches/epoch,stop1→resume2 dưới scheduler20. Ba model forward/backward/save/load;DeepLab resumed weights khớp bitwise continuous2;native evaluator tạo confusion/calibration/panels. Đây là plumbing, không20epoch convergence/accuracy. MiT-B2 và DeepLab ImageNet initialization,memory,walltime,loss/crop gains còn chờ Kaggle. [Evidence](../results/repository-release/README.md).
