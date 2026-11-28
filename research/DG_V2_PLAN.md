# Đề xuất vòng DG v2: cải thiện lớp yếu và khả năng tổng quát

Ngày 28/11/2026. **Đợt1 đã triển khai thành luồng v2 riêng; notebook05/06 mỗi notebook hai thí nghiệm, có thể chạy song song. Chưa có kết quả GPU.** Các mục đợt2/dữ liệu bổ sung/xác nhận nhiều seed vẫn là đề xuất chưa triển khai. Sáu candidate DG v1 cùng kết quả đánh giá giữ nguyên. [Cách chạy v2](../docs/DG_V2_GUIDE.md) · [Kết quả v1](../results/dg_evaluation_seed42/README.md).

## Bằng chứng và giả thuyết

| Quan sát từ v1 | Giả thuyết cần kiểm tra, chưa phải nguyên nhân đã chứng minh |
| --- | --- |
| B2 aug ACDC52,47%, night28,75%; night thấp rõ rệt | Perturbation RGB toàn cục chưa bao phủ ngoại hình/độ tương phản/nhiễu ảnh đêm |
| B2 aug rider16,03%, bicycle25,54%, motorcycle27,12% | Mất cân bằng pixel, ít crop chứa đủ đối tượng và độ phân giải hữu hiệu thấp có thể cùng góp phần |
| B2 consistency không tăng mIoU gộp so aug | KL hiện tại có thể không bổ sung tín hiệu hữu ích, hoặc truyền dự đoán sai từ nhánh clean |
| DeepLab consistency tăng target dù giảm source | Không dùng riêng điểm Cityscapes sạch để suy luận độ bền vững; kiểm clean và source-corrupted dev |

Không suy từ IoU thấp rằng lớp đó chắc chắn hiếm. Phải đếm nguồn và xem confusion/precision/recall. CE chưa phải nguyên nhân đã chứng minh; tăng trọng số không khôi phục được thông tin đối tượng đã bị resize mất.

## Đợt 0: audit lỗi trước GPU

1. Thống kê từ Cityscapes fit: pixel/class, số ảnh có class, kích thước vùng và tỷ lệ lớp còn lại sau resize/crop. Không lấy ACDC làm nguồn tính weights hoặc sampler.
2. Xem các panel/nhãn/dự đoán/lỗi hiện có; bổ sung mẫu lỗi có quy tắc cố định cho từng thời tiết và lớp. ACDC chỉ dùng mô tả lỗi của vòng v1; công bố rằng hướng v2 đã chịu ảnh hưởng từ phản hồi này.
3. Tính precision/recall theo lớp từ cache hiện có. Tăng recall nhưng giảm precision mạnh có thể chỉ là model tô quá nhiều pixel vào lớp được ưu tiên.
4. Xác minh ảnh đêm tổng hợp còn thấy đường/đối tượng; kiểm mask biến đổi hình học đồng bộ và không để augmentation tạo ảnh gần đen hoàn toàn.

## Đợt 1: ma trận B2 có đối chứng

Tập trung B2 để chọn phương pháp, chưa nhân mọi ý tưởng sang cả hai model. Mỗi arm khởi tạo độc lập từ encoder ImageNet đã pin, 20 epoch, cùng seed42, split, số bước cập nhật và selection rule. Không học nối từ best B2 aug để gọi là cùng ablation.

| Arm phát triển | Aug RGB hiện tại | Weighted CE nhẹ | Aug ngoại hình mở rộng |
| --- | --- | --- | --- |
| baseline | Có | Không | Không |
| class_weights | Có | Có | Không |
| weather_aug | Có | Không | Có |
| weather_aug_class_weights | Có | Có | Có |

class_weights−baseline kiểm weighting; weather_aug−baseline kiểm augmentation; weather_aug_class_weights kiểm tương tác. Bốn arm phải chạy lại cùng split mới, không lấy điểm v1 full2975 để thay baseline nếu dữ liệu fit khác. Trong đợt này giữ crop/sampler/optimizer giống nhau để không lẫn tác động.

### Weighted CE nhẹ

Đếm tần suất pixel `f_c` từ **fit labels** sau khi chốt split; bỏ ignore255. Đề xuất khởi đầu, chưa tối ưu:

`w_c = clip(sqrt(median(f_positive) / (f_c + eps)), 0.75, 2.0)`.

Lớp vắng hoàn toàn phải được báo lỗi/thống kê riêng, không coi nhân trọng số sẽ tạo ra ví dụ học. Không đặt weights dựa trên `1/IoU_ACDC`. Lưu weights cùng hash fit manifest, số pixel và công thức để truy vết. Không tăng gấp10–20 lần cho lớp yếu.

Repo có `weighted_ce`, nhưng guard DG v1 hiện chỉ cho CE. Cần protocol/config v2 riêng và test bảo vệ nguồn dữ liệu, không nới lỏng âm thầm v1. Công thức căn bậc hai có chặn ở đây là lựa chọn thực nghiệm của project, không phải tái lập nguyên Class-Balanced Loss của Cui et al.

### Ngoại hình nguồn đa dạng hơn

Giữ ảnh sạch và augmentation nhẹ, bổ sung theo các nhánh có xác suất cố định; không chồng toàn bộ hiệu ứng lên mọi ảnh:

- Giảm phơi sáng/gamma theo nhiều mức; thêm biến thiên ánh sáng theo không gian và lệch màu nhẹ.
- Nhiễu phụ thuộc mức sáng, giảm màu/độ tương phản, blur nhẹ; một nhánh haze mức vừa từ nguồn. Haze không có depth chỉ là mô phỏng đơn giản, không nhận là sương vật lý chính xác.
- Giữ crop/flip/nhãn đồng bộ. Đặt giới hạn severity bằng kiểm trực quan và source-dev, không chọn bằng điểm ACDC.
- Một cấu hình khởi đầu để chốt trước chạy: 50% nhẹ hiện tại, 30% low-light, 20% haze/contrast; log mức sáng/clipping và ảnh preview. Đây là giả thuyết cần thử, không phải tỷ lệ đã được chứng minh tốt.

Nếu dùng sinh ảnh ISSA/StyleGAN hoặc diffusion, cần bước riêng để kiểm bảo toàn nội dung/nhãn và nguồn pretrained. Không gọi perturbation RGB mới là ISSA khi chưa cài phương pháp đó.

## Chia dữ liệu và chọn checkpoint

1. Chia Cityscapes train2975 thành fit/dev theo **thành phố hoặc nhóm ghi hình**, mục tiêu gần80/20; số ảnh thực tế phụ thuộc nhóm. Không tách ngẫu nhiên các ảnh rất gần nhau rồi nhận là kiểm tra độc lập.
2. Dev gồm bản sạch và các biến thể low-light/noise/haze cố định, khác seed và có tổ hợp severity chưa xuất hiện trong fit. Các biến thể của cùng ảnh luôn cùng split.
3. Chốt score trước chạy: `0.5 * clean_dev_mIoU + 0.5 * mean(corruption_dev_mIoU)`. Theo dõi thêm IoU/precision/recall lớp hiếm, hiệu quả từng corruption và chi phí. Đây là proxy nguồn, chưa bảo đảm chất lượng thực tế trên ACDC.
4. Tính weights/sampler từ fit, không dùng dev. Không chọn tỷ lệ split, augmentation severity hoặc checkpoint bằng ACDC.
5. Cityscapes val500 và ACDC val406 giữ để báo cáo sau khi chốt vòng mới. Chúng đã được xem trong v1, nên không nhận là test chưa từng thấy. Muốn kiểm chứng độc lập phải có benchmark/holdout khác chưa dùng để thiết kế/chọn cấu hình.

## Đợt 2: chỉ mở nhánh giải quyết lỗ hổng còn lại

Không triển khai tất cả lựa chọn dưới cùng lúc.

| Hướng | Khi nào ưu tiên | Đối chứng cần có |
| --- | --- | --- |
| Rare-class sampling + crop có ngữ cảnh | Dev vẫn bỏ sót lớp ít pixel | Arm tốt đã chọn so với cùng arm + sampling/crop; cùng số optimizer updates |
| Crop từ ảnh gốc độ phân giải cao | Đối tượng nhỏ biến mất sau resize toàn cảnh | Giữ canvas512×1024, lấy patch có chi tiết thật trước resize; không chỉ phóng ảnh đã mất thông tin |
| CE + Lovász-Softmax | mIoU lớp còn thấp dù learning ổn định | Một loss thay thế, cùng data/aug; không chồng ngay weighted CE, focal, Dice, OHEM và Lovász |
| Consistency có lọc độ tin cậy và ramp-up | Nhánh aug đã mạnh, teacher nguồn đủ tốt | Có/không KL; ví dụ ramp từ0 đến0,05 trong5epoch đầu, threshold chọn trên source-dev; kiểm lớp hiếm có bị bỏ hết vì confidence thấp |
| BatchNorm riêng cho DeepLab | Nhạy với hai view/real batch2 | Đối chứng cố định running stats BN encoder; không đóng băng mù quáng BN decoder mới từ đầu |
| Warmup/schedule | Đường LR/gradient cho thấy bất ổn | Hiện warmup20 updates trong tổng7438; thử warmup dài hơn thành arm riêng, không đồng thời tăng LR/epoch |

Rare-class sampler phải chọn ảnh/crop dựa vào nhãn fit, giữ đủ ngữ cảnh, giới hạn oversampling và log class coverage. Có tiện ích class-aware crop trong repo nhưng chưa chứng minh hiệu quả với DG. Crop và weighted CE có thể bù quá mức nếu bật mạnh cùng lúc.

Với consistency, clean prediction không phải đáp án đúng tuyệt đối. Confidence gating cũng có thể loại nhầm toàn bộ lớp khó; cần thống kê tỷ lệ pixel được nhận theo lớp. Không thêm teacher dùng ACDC hoặc pseudo-label target vào luồng DG.

## Dữ liệu bổ sung

- **Đầu tiên khai thác nguồn hiện có:** coverage của crop/lớp, ảnh gốc độ phân giải cao, style nguồn và chia nhóm đúng.
- Cityscapes train_extra có khoảng20.000 nhãn coarse. Có thể dùng pretraining nguồn rồi refine bằng fine labels; dữ liệu bổ sung phải thành đối chứng riêng. Nhãn thô không phù hợp để mặc định cải thiện ranh giới/đối tượng nhỏ. Tuyệt đối không lấy coarse labels của val/test vào train.
- Có thể mở multi-source DG với nguồn đường phố khác sau khi xác minh mapping19lớp, split/điều kiện dữ liệu và nguồn pretrained. Khi đó phải báo rõ đã đổi quyền dữ liệu so với Cityscapes-only, không gán toàn bộ lợi ích cho loss/module.
- Không lấy ảnh/nhãn ACDC train, reference hay style statistics target để cải thiện rồi tiếp tục gọi cùng protocol Cityscapes-only DG.

## Xác nhận cuối và tiêu chí quyết định

Sau phát triển nguồn, khóa phương pháp tốt nhất và một control. Train lại trên full Cityscapes train2975, đủ20epoch; xác nhận phương pháp trên B2 và DeepLab với cùng recipe dùng chung. Không bắt B2 dùng consistency chỉ để đồng nhất hình thức; nếu so khả năng chuyển phương pháp giữa kiến trúc, dùng cùng phương pháp và ghi kết quả âm nếu có.

Nếu ngân sách cho phép, chạy control và phương pháp cuối ít nhất3seed; pilot một seed không thay thế kiểm chứng ổn định. Bootstrap theo nhóm ảnh/ghi hình có thể đánh giá bất định của tập đánh giá, nhưng không thay thế nhiều seed train.

Báo mIoU chung, per-weather/per-class, precision/recall, clean-versus-corruption, train hours/VRAM và inference cost. Không lấy riêng lớp/thời tiết tăng để che mIoU gộp giảm. Mục tiêu là cải thiện đo được và giải thích được; không hứa52→70/80 trước thí nghiệm. Timing crop/aug/loss mới phải profile lại; không dùng thời gian v1 làm cam kết ngân sách.

## Nguồn tham khảo và phạm vi vận dụng

- [ISSA, WACV2023](https://openaccess.thecvf.com/content/WACV2023/html/Li_Intra-Source_Style_Augmentation_for_Improved_Domain_Generalization_WACV_2023_paper.html): đa dạng style nội nguồn, bảo toàn semantic layout; project v1 chưa tái lập StyleGAN/inversion của bài.
- [DAFormer, CVPR2022 — code chính thức](https://github.com/lhoyer/DAFormer): rare-class sampling được áp dụng ở nguồn. Toàn bộ bài thuộc UDA; chỉ tham khảo thành phần nguồn, không đưa self-training target vào DG.
- [Class-Balanced Loss, CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html): cơ sở về tái cân bằng; effective-number formula khác đề xuất trọng số nhẹ của project và không bảo đảm chuyển sang segmentation này.
- [Lovász-Softmax, CVPR2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html): surrogate cho IoU; một đối chứng loss đáng thử sau đợt đầu.
- [Cityscapes paper](https://arxiv.org/abs/1604.01685): nguồn fine/coarse; thêm train_extra là thay đổi dữ liệu, cần khai báo riêng.
