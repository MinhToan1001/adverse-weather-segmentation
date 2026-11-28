# Project-CV · Phân đoạn đường phố trong thời tiết xấu

![Project-CV](docs/assets/project_banner.svg)

**Học từ Cityscapes → kiểm tra trên ACDC. SegFormer-B2 là model chính; DeepLabV3+ ResNet50 là đối chứng kiến trúc bổ sung.**

Ảnh đầu vào là một cảnh đường phố. Đầu ra là bản đồ 19 lớp cho từng pixel: đường, vỉa hè, người, xe, cây… Mục tiêu là giữ chất lượng khi gặp đêm, sương mù, mưa và tuyết, dù model chỉ học ảnh Cityscapes.

Cách làm này gọi là **Domain Generalization (DG)**: học ở điều kiện nguồn, kiểm tra ở điều kiện khác. ACDC không được dùng để train, lấy style hoặc chọn model trong luồng hiện hành.

## Đang làm: B2 DG v2 — hai notebook, mỗi notebook hai thí nghiệm

DG v2 là tên đợt phát triển pipeline. Các mã V0–V3 đã đổi thành tên mô tả; cả bốn thí nghiệm vẫn là **cùng model B2**, mỗi lượt20epoch, khởi tạo độc lập từ encoder ImageNet và head19lớp mới.

| Người/tài khoản chạy | Notebook import Kaggle | Hai thí nghiệm tự chạy |
| --- | --- | --- |
| Tài khoản1 | [05_dg_v2_segformer_b2.ipynb](notebooks/05_dg_v2_segformer_b2.ipynb) | `baseline` → `class_weights` |
| Tài khoản2 | [06_dg_v2_b2_weather.ipynb](notebooks/06_dg_v2_b2_weather.ipynb) | `weather_aug` → `weather_aug_class_weights` |

Hai bên nạp **cùng code ZIP mới + Cityscapes**, không nạp ACDC hoặc weights cũ. Dùng chung fit2380/dev595 theo thành phố, seed42 và quy tắc chọn checkpoint. Mỗi notebook có ZIP/checkpoint riêng, tiếp tục được khi hết phiên. `suite_complete` là **hai lượt trong nhóm đó đủ20/20**; cần cả hai ZIP để hoàn thành bốn đối chứng.

**Chưa có kết quả GPU v2.** [Hướng dẫn chạy từng bước](docs/DG_V2_GUIDE.md) · [Bàn giao cho Toàn: dữ liệu/config/augmentation/weights/selection](docs/DG_V2_HANDOFF.md). SHADE/ngân hàng style thuộc phần phát triển tiếp của Toàn, chưa có trong bốn lượt này.

## Project hiện làm được gì?

| Thành phần | Hiện trạng |
| --- | --- |
| Hai model có file code riêng | B2 và DeepLab đã có forward, backward, nạp/lưu weights; dùng chung pipeline |
| Đọc dữ liệu và xử lý ảnh–mask | Tạo danh sách cặp ảnh/nhãn, kiểm tra dữ liệu, chuyển nhãn, biến đổi hình học đồng bộ |
| Ba chiến thuật cho **mỗi model** | `baseline`, `aug`, `consistency`; đã có code và kiểm thử CPU |
| Train trên Kaggle | Hai notebook tự chuẩn bị dữ liệu → đo GPU → chạy ba chiến thuật, mỗi chiến thuật 20 epoch |
| Checkpoint và tiếp tục train | Lưu best/last, trạng thái optimizer và RNG; xuất ZIP sau mỗi epoch hoàn thành |
| Đánh giá và trực quan hóa | mIoU từng lớp/thời tiết, learning curves, ảnh nhãn/dự đoán/lỗi và độ tự tin |
| Demo Node.js | Upload ảnh, xem mask/overlay, tải kết quả; hiện dùng checkpoint B2 **lịch sử** trên CPU |
| Kết quả DG mới | **Đủ 6/6 nhánh × 20 epoch và 12 phép đánh giá Cityscapes/ACDC; đã nghiệm thu** |

Nghiệm thu cập nhật ngày 24/11/2026: **B2 baseline/aug/consistency đạt 73,98%/74,37%/74,44%; DeepLab đạt 71,82%/72,08%/71,58% Cityscapes val500**, mỗi nhánh đủ 20 epoch. Trên nguồn, consistency thấp hơn aug ở DeepLab và chỉ hơn rất ít ở B2. [B2](results/dg_segformer_b2_seed42/README.md) · [DeepLab](results/dg_deeplabv3plus_resnet50_seed42/README.md) · [Sáu checkpoint đã khóa](results/dg_cohort_seed42/README.md). Đánh giá ngày 26/11 đã hoàn tất: ACDC baseline/aug/consistency của **B2 đạt 49,02%/52,47%/52,46%; DeepLab đạt 34,26%/40,35%/42,20%**. Augmentation giúp cả hai model trong seed42; consistency chỉ tăng mIoU gộp ở DeepLab. [Bảng điểm và audit ACDC](results/dg_evaluation_seed42/README.md). [Bằng chứng và kết quả](results/README.md).

## Dữ liệu trong DG v1 đã hoàn tất

| Tập dữ liệu | Số ảnh | Vai trò hiện hành |
| --- | ---: | --- |
| Cityscapes train | 2.975 | Học model, tạo các phiên bản ảnh tăng cường |
| Cityscapes val | 500 | Theo dõi và chọn checkpoint tốt nhất |
| ACDC val | 406 | Chỉ đánh giá sau khi chốt cấu hình và checkpoint của sáu lượt |

**Không dùng ACDC train trong luồng DG.** Các thử nghiệm cũ từng học cả ảnh/nhãn ACDC thuộc hướng khác; giữ để truy vết và phục vụ demo, không đưa điểm đó vào bảng DG. ACDC cũng đã được xem trong quá trình phát triển trước đây, nên cần công bố giới hạn này khi báo cáo.

## Ba chiến thuật DG v1 đã được so sánh

| Tên trong code | Hiểu đơn giản | Mục đích |
| --- | --- | --- |
| `baseline` | Học ảnh nguồn với loss CE thông thường | Mốc so sánh của từng model |
| `aug` | Thêm ảnh đổi độ sáng/màu, nhiễu và mờ; giữ cùng nội dung/nhãn | Kiểm tra lợi ích tăng đa dạng ảnh nguồn |
| `consistency` | Như `aug`, thêm yêu cầu hai phiên bản ảnh dự đoán nhất quán | Kiểm tra riêng lợi ích của ràng buộc nhất quán |

CE là mức phạt khi model đoán sai lớp của pixel. Một epoch là một lượt học qua tập train. Checkpoint là bản lưu để dùng lại hoặc tiếp tục train. mIoU đo mức khớp giữa vùng dự đoán và nhãn thật, trung bình các lớp; càng cao càng tốt.

Cả B2 và DeepLab đều chạy đủ ba chiến thuật, **khởi tạo độc lập**, cùng seed và cùng recipe. Không học nối từ baseline sang aug/consistency. Tổng cộng sáu lượt × 20 epoch. [Thông số và công thức](docs/DG_METHOD.md).

## Thành viên mới bắt đầu từ đâu?

**Đang phát triển DG v2:** [notebook05/06](docs/DG_V2_GUIDE.md) chia hai cặp đối chứng B2 weighting × style nguồn, fit/dev theo thành phố. Input là **code ZIP DG-v2 mới + Cityscapes**, không ACDC. [Cách chạy và phạm vi](docs/DG_V2_GUIDE.md). Chưa có mIoU GPU v2; giữ nguyên kết quả v1 dưới đây.

**Training DG v1 đã hoàn thành; không cần chạy lại notebook train/resume v1.** Giữ hai ZIP suite hoàn chỉnh B2 và DeepLab `(1)` để truy vết. Notebook03 và Release epoch4 ghi lại lượt bàn giao cũ.

**Đánh giá trên Kaggle:** import [notebook04](notebooks/04_evaluate_dg_cohort.ipynb), thêm hai suite hoàn chỉnh + Cityscapes + ACDC. Không cần code ZIP; Run All tự kiểm sáu checkpoint rồi đánh giá cả hai tập. [Hướng dẫn từng bước](docs/KAGGLE_EVALUATION.md).

1. Đọc trang này để nắm bài toán, phạm vi dữ liệu và trạng thái thật.
2. Đọc [luồng code](docs/READING_GUIDE.md), rồi [quy ước dữ liệu](docs/DATA_CONTRACT.md).
3. Đọc [phương pháp DG](docs/DG_METHOD.md) để hiểu cách train và các phép so sánh.
4. Làm theo [hướng dẫn Kaggle/resume](docs/TRAINING_GUIDE.md) hoặc [chạy demo](web/README.md).
5. Trước khi sửa code, đọc [hướng dẫn đóng góp](docs/CONTRIBUTING.md). Việc còn lại ở [kế hoạch hoàn thành](research/PROJECT_COMPLETION_PLAN.md).

## Cấu trúc repo

```text
configs/         Recipe 20 epoch, preprocessing và augmentation dùng chung
src/data/        Dataset, nhãn, xử lý ảnh và tăng cường dữ liệu
src/models/      File B2, file DeepLab, registry chọn/nạp model
src/training/    Trainer, loss, optimizer, DG và checkpoint dùng chung
src/experiments/ Ghép config, chặn dùng sai dữ liệu, khóa kết quả
src/evaluation/  Metrics và xuất ảnh đánh giá
scripts/         Lệnh chuẩn bị, chạy suite, đánh giá và tạo notebook
notebooks/       05/06 chạy B2 v2; 01–04 giữ cho DG v1 đã hoàn tất
web/             Node.js API/giao diện + Python inference
tests/           Kiểm thử pipeline, checkpoint và protocol
docs/            Hướng dẫn đọc/chạy/sửa code
research/        Cơ sở bài báo và kế hoạch thực nghiệm
results/         Báo cáo nhỏ, bằng chứng kiểm tra và kết quả lịch sử
data/, outputs/  Dataset, weights và ZIP local; không đẩy lên Git
```

B0 chỉ còn loader trong `src/models/legacy/` để đọc checkpoint cũ. Các tiện ích replay/chia tập/loss khác còn giữ cho tương thích và kiểm thử; **không được bật trong recipe DG hiện hành**.

## Chuẩn bị và chạy

Python 3.11+, PyTorch từ 2.6 và TorchVision tương thích. Chọn bản PyTorch phù hợp môi trường CPU/CUDA trước khi cài dependencies. Node.js 22+ cần cho demo.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m scripts.build_dg_v2_notebook
```

Lệnh cuối tạo `outputs/kaggle/project-code-dg-v2.zip` và cập nhật hai notebook:

- [05_dg_v2_segformer_b2.ipynb](notebooks/05_dg_v2_segformer_b2.ipynb): baseline và class weights.
- [06_dg_v2_b2_weather.ipynb](notebooks/06_dg_v2_b2_weather.ipynb): augmentation mới và augmentation + class weights.

Trên Kaggle, thêm **code ZIP mới + Cityscapes train**, chọn GPU; mỗi tài khoản dùng một notebook trong bảng đầu trang. Mỗi notebook tự chạy hai lượt20epoch. `None` ở đường dẫn dataset nghĩa là tự tìm; `RESUME_INPUT=None` nghĩa là lượt mới. [Cách đặt đường dẫn, giữ ZIP và resume](docs/DG_V2_GUIDE.md).

Demo local: `npm start`, mở [localhost:3000](http://127.0.0.1:3000). Cần nạp checkpoint theo [web/README.md](web/README.md); clone source đơn thuần chưa có weights. Demo hiện chưa tích hợp weights DG mới hoặc DeepLab đã train.

```powershell
.venv/Scripts/python.exe -m pytest -q
npm test
```

## Còn phải hoàn thành

Sáu lượt train và 12 phép đánh giá đã được nghiệm thu; đã có bảng từng lớp/từng thời tiết và chi phí. Còn phân tích ảnh lỗi, tích hợp weights DG vào demo, hoàn thiện báo cáo/video. Không sửa recipe hoặc chọn lại checkpoint bằng điểm ACDC của cohort này.

[Cơ sở bài báo](research/README.md) · [Giấy phép và nguồn code](THIRD_PARTY_NOTICES.md) · [Kết quả hiện có](results/README.md)
