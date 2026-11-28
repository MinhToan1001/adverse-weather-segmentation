# DG v2 · bằng chứng triển khai local

Ngày 28/11/2026. [Biên bản kiểm chứng](verification.json) · [Split city thật](split.json) · [Class audit từ fit](class_audit.csv) · [Hướng dẫn chạy](../../docs/DG_V2_GUIDE.md).

**Đã triển khai, chưa train GPU.** Không có mIoU v2 hoặc kết luận cải thiện ACDC ở đây.

- 26 test riêng DG v2 đạt: source/target guards, fit weights, style, trainer interruption-resume, hai nhóm độc lập, bootstrap chạy lại không ghi chồng, resume ZIP/source giải nén và từ chối nhầm nhóm. Kết quả bộ test toàn repo lưu trong verification.json.
- Trainer thật chạy CPU hai epoch với tiny-model fixture có dropout, weighted CE và paired views. Chạy liền và resume cho weights/history giống từng bit. Đây là kiểm chứng orchestration/optimizer/RNG; không phải benchmark hoặc kiểm chứng GPU kernels của B2.
- Queue fixture kiểm cả bốn lượt chung và hai nhóm riêng40epoch request/nhóm; skip epoch committed khi resume, xuất ZIP khi Python lỗi. Đây là kiểm orchestration bằng fixture, không thực hiện80epoch GPU.
- Hai nhóm chuẩn bị toàn bộ Cityscapes train2975 thật, chuyển nhãn, chia14city fit2380 /4city dev595, tính weights từ fit và kiểm frozen inputs/config/source hashes. Official val/test và ACDC không tham gia.
- Review preview thật phát hiện thử nghiệm low-light ban đầu gần đen; đã giảm severity và tạo/kiểm preview của bản cuối. Hình dataset chỉ nằm trong outputs/ZIP riêng tư, không commit vào results.
- 68 file source/config/script của snapshot DG v1 giữ nguyên. V2 thêm module, script, config và notebook riêng; source ZIP mới khớp từng hash của bản triển khai đã kiểm.

Chưa đo timing/VRAM/GPU resume trên Kaggle của v2. Lưu lượng dev gồm bốn pass mỗi epoch nên không dùng ETA của v1 để cam kết quota. Notebook có kiểm source/data/config khi resume, checkpoint transactional best/last/previous, export sau mỗi epoch và cửa sổ phiên10giờ; hard-kill vẫn có thể bỏ qua finally.

Bàn giao thêm [manifest portable](../../configs/dg_v2/splits/fit.csv), [19 class weights](../../configs/dg_v2/class_weights_reference.json), config B2/augmentation và [quy tắc dùng chung với Toàn](../../docs/DG_V2_HANDOFF.md). CSV reference cố định LF cho Windows/Linux. Notebook05/06 mỗi notebook hai thí nghiệm, cùng source/split/seed và ZIP riêng. Không triển khai SHADE trong ma trận này.
