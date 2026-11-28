# Đề xuất kết hợp K1/K2 và B2 DG-v2

**Bản thảo tham khảo, chưa phải recipe đang chạy.** Sau bản thảo này, Toàn đề nghị phát triển ngân hàng style/SHADE trên nền bàn giao chung. Điểm xuất phát thực thi hiện tại nằm ở [DG_V2_HANDOFF.md](../docs/DG_V2_HANDOFF.md); các kết hợp K1/FiLM dưới đây vẫn chưa được triển khai hoặc chốt.

Ngày 28/11/2026. **Đề xuất chưa triển khai/merge; notebook05 hiện tại chưa chứa K1 hoặc FiLM của Toàn.** Đối chiếu nhánh `feat/k1-weather-augmentation` tại commit `ed623e6` với v2 local. K1/K2 có tính năng đáng thử, chưa được chứng minh tốt hơn trên target bằng đối chứng tương ứng.

## Định hướng chung

Giữ B2 làm backbone chính, dùng K1 làm một backend sinh thời tiết có kiểm soát và dùng weighted CE nhẹ để xử lý mất cân bằng. Chỉ thêm FiLM/condition head sau khi biết backend augmentation và weighting có hiệu quả trên source-dev. Giữ DG: không dùng ảnh, nhãn, reference hoặc thống kê ACDC để học.

Phần của Toàn đóng góp generator K1, ý tưởng/code FiLM self-conditioned, auxiliary condition supervision và kiểm initialization tương đương. Phần v2 local đóng góp city-disjoint fit/dev, fit-only class weights, selection clean+corruption, queue và checkpoint/resume. B0 vẫn là pilot/đối chứng riêng của Toàn, không trộn số B0 với B2 như cùng model hoặc cùng split.

## Dữ liệu và augmentation dùng chung

- Fit2.380/dev595 theo city như v2; weights chỉ từ nhãn fit, clip0,75–2,0. Không chọn trọng số bằng IoU ACDC.
- Cùng crop/flip, mask và source instance cho hai view clean/styled. K1 cần depth native nên sinh weather trước resize/crop. Khi so với backend RGB cũ trong vòng kết hợp, đặt cả hai backend ở cùng vị trí native trước cùng phép hình học; control này phải chạy mới, không coi như trùng hoàn toàn notebook05 đang áp style sau geometry.
- Mọi arm dùng cùng paired-batch probability0,5, CE trung bình clean/styled khi paired; weighting dùng cho cả hai view. Không đổi optimizer/batch/epochs giữa các arm.
- Khi paired, đề xuất sampling styled condition: clean40%, night30%, rain15%, snow15%; severity1–3. Đây là chính sách ứng viên từ nhận định ảnh đêm khó, chưa phải tỷ lệ tối ưu đã chứng minh; khóa trước GPU. Tỷ lệ trên là trong styled views, không phải tỷ lệ trên toàn bộ presentations. Severity có thể khác giữa các sample nhưng không là curriculum thay đổi theo epoch trong đợt đầu.
- Nếu có đủ disparity/camera cùng ảnh và kiểm đơn vị/geometry đạt: cấu hình riêng đủfog, ví dụ clean40%, night25%, rain10%, snow10%, fog15%, rồi dùng nhất quán cho mọi arm có K1. Chọn cấu hình có/khôngfog trước khi khóa thí nghiệm; không đổi giữa phiên hoặc giữa các đối chứng.
- Thiếu depth thì không gọi haze RGB là metric fog hoặc âm thầm far-fill. Nếu chủ động dùng far-fill, phải ghi rõ invalid policy/tỷ lệ invalid và đây chỉ là xấp xỉ.
- Không chồng thêm backend low-light v2 lên K1 ở cùng styled view; dễ làm ảnh quá nặng và mơ hồ condition ID.

Các mức night ở K1 được tính trong RGB tuyến tính; không sao chép trực tiếp exposure/gamma của backend v2 xử lý sRGB. Xem preview nhiều ảnh fit chọn bằng seed cố định, đủ severity/condition; kiểm chi tiết lớp nhỏ còn thấy và label geometry đúng. Profile throughput dataloader/CPU generator native để tránh GPU chờ dữ liệu.

## Đợt A: bốn đối chứng B2 ×20epoch

| Arm | Style backend | Loss | FiLM / auxiliary |
| --- | --- | --- | --- |
| J0 | RGB nguồn cũ, đặt ở cùng bước native | CE | Tắt |
| J1 | Như J0 | Weighted CE | Tắt |
| J2 | K1 | CE | Tắt |
| J3 | K1 | Weighted CE | Tắt |

J1−J0 kiểm weighting; J2−J0 kiểm thay backend; J3−J2 kiểm weighting với K1; so hai chênh lệch weighting để xem tương tác. Mọi arm khởi tạo độc lập từ cùng encoder ImageNet/head seed, cùng20epoch và số cập nhật. Không fine-tune từ best của arm trước.

Không đưa crop lớp hiếm, warmup500, canvas640×1280 hoặc40epoch vào ma trận này; sẽ không tách được ảnh hưởng K1/weighting. Các thay đổi recipe có thể là thí nghiệm sau, phải áp dụng cho cả control và proposed.

## Chọn phương pháp mà không chỉ học thuộc generator

Giữ dev sạch và bộ corruption dev cố định độc lập với K1: low-light theo spatial field/noise/haze của backend v2. Chốt mIoU selection bằng0,5 clean-dev +0,5 mean các corruption độc lập, giống định hướng v2. Bộ dev này chưa bao phủ đầy đủ mưa/tuyết; báo giới hạn và nếu mở thêm corruption dev thì phải khai báo trước khi train, không bổ sung sau khi nhìn thứ hạng.

Preview K1 ở severity/seed khác chỉ là QA generator, không phải bằng chứng OOD độc lập nếu vẫn cùng generator. ACDC val đã có phản hồi lịch sử; không gọi là test mới chưa từng xem. Khóa cấu hình/checkpoint trước đánh giá ACDC, không đổi epoch bằng target score.

## Đợt B: tách tác dụng của FiLM và auxiliary condition loss

Dùng cùng K1 và cùng lựa chọn CE/weighted CE đã chốt trên source-dev cho bốn biến thể: B2 gốc; B2+FiLM, loss điều kiện tắt; B2+condition-head-only; B2+FiLM+condition head. Hai biến thể có supervision điều kiện dùng hệ số khởi đầu0,1 như pilot của Toàn. Condition head có thể được giữ trong model FiLM-only nhưng không học khi hệ số bằng0.

So FiLM-only với control, condition-only với control, cả hai với từng thành phần để xem tương tác. K1 có thể cần FiLM mới có ích, nên đợt này vẫn đáng kiểm tra ngay cả khi riêng K1 chưa thắng RGB. Kết quả phương án cuối vẫn phải so với control RGB tốt nhất; không mặc định giữ mọi thành phần.

Nhãn điều kiện chỉ được dùng khi có generator sinh ra điều kiện đó. Nếu chuyển sang policy RGB không có ID K1 thì không dùng nhãn K1 giả cho auxiliary head. Khi dùng hai view, chuẩn hóa loss theo cùng số view: clean có label clean, styled có label phép biến đổi thực tế; không vô tình nhân đôi hệ số auxiliary.

Hiện Toàn chỉ có class `WeatherSegB0`. Phải port sang B2 bằng config channels/stages của B2 trong file riêng, không đổi tên B0 hoặc nạp B0 weights vào B2. FiLM final projection khởi tạo zero; cùng baseline weights phải cho logits ban đầu khớp trên CPU. Việc tạo tham số FiLM/head không được làm lệch RNG initialization/dropout của control.

FiLM tự dự đoán gamma/beta từ mean/variance feature. Condition head là nhiệm vụ phụ nhận nhãn phép sinh K1, không phải selector trực tiếp dùng nhãn ACDC để điều khiển FiLM. Inference chỉ cần ảnh; không yêu cầu người dùng khai báo thời tiết hoặc dùng GT condition. Accuracy synthetic-condition cao không thay thế mIoU thời tiết thật.

## Tổ chức GPU và xác nhận kết quả

Theo cập nhật ngày28/11, không lấy30giờ làm trần cho kế hoạch. Có thể dùng thêm quota, nhiều phiên hoặc nhiều GPU cho các run độc lập. Vẫn đo train/dev/checkpoint/ZIP để sắp xếp phiên và tránh bị ngắt; pilot weights bỏ đi, không dùng làm kết quả. Giữ20epoch cho mỗi arm đã khai báo, không tự tăng40 vì có thêm GPU.

Hoàn thành đợt A và đợt B theo thứ tự câu hỏi, không cắt đối chứng vì giới hạn30giờ. Sau khi chọn phương pháp, xác nhận control và phương pháp cuối bằng ít nhất3seed, ví dụ42/123/2026. Nếu muốn kết luận riêng về weighting hoặc FiLM thì lặp cả cặp ablation tương ứng qua nhiều seed. Báo mean/std của mIoU; khoảng bất định do resample ảnh không thay thế biến thiên giữa seed train.

Sau phát triển, train xác nhận trên full Cityscapes train2975 với cùng recipe giữa control và phương pháp. Khi đó595ảnh dev cũ đã vào train nên không còn là tập chọn best; dùng Cityscapes val500 và các corruption nguồn cố định theo quy tắc selection chốt trước, giống nhau cho mọi arm. Chỉ đánh giá ACDC sau khi khóa checkpoint. ACDC vẫn có feedback lịch sử, không trở thành test độc lập mới.

DeepLab là đối chứng kiến trúc bổ sung, không phải thành phần của B2 và không bắt buộc để triển khai hướng kết hợp. Ưu tiên B2 ablation/multi-seed. Nếu muốn chứng minh tính chuyển giao phương pháp, chạy control và data/loss đã chọn trên DeepLab; FiLM cho CNN cần thiết kế/kiểm riêng, không port nguyên class B2 bằng cách đổi tên. Kết quả thiếu xác nhận phải ghi là pilot, không hứa vượt paper.

## Tích hợp repo trước GPU

- Không merge toàn nhánh đè lên workspace có v2 chưa commit. Xem riêng thay đổi dataset, registry, protocol và trainer; tên notebook03–08 giữa hai nhánh đang mang nội dung khác nhau.
- Port generator theo module độc lập, thêm dataset adapter native paired-weather và protocol kết hợp riêng. Giữ v1 và snapshot v2 đã khóa để truy vết.
- Reuse một trainer/metrics/checkpoint; không tạo hai bản core training lệch nhau cho Toàn/Khang. FiLM B2 là file model riêng; K1 và weighted CE là thành phần dùng chung.
- Test target rejection, ảnh/mask geometry, điều kiện fog thiếudepth, fit-weight provenance, identity initialization, optimizer groups, AMP và interruption-resume. Test smoke CPU/fixture không thay thế GPU smoke thật.
- Sinh notebook/ZIP mới cho đợt kết hợp, tên khác notebook05 hiện tại; có log pipeline/model/source/data hash, status và output so sánh.

Nguồn nội bộ: [K1 của Toàn](https://github.com/MinhToan1001/adverse-weather-segmentation/blob/ed623e6/docs/K1_WEATHER.md), [K2/ablation](https://github.com/MinhToan1001/adverse-weather-segmentation/blob/ed623e6/docs/K2_TRAINING.md), [B2 DG-v2 local](../docs/DG_V2_GUIDE.md). Tất cả tác dụng dự kiến đều cần thí nghiệm đối chứng; chưa có kết quả kết hợp.
