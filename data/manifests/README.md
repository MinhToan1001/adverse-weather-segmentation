# Danh sách ảnh–mask sinh tự động

Manifest là CSV gồm image, mask, domain, split. Suite DG tự tạo Cityscapes train2975/val500 trong workspace của nó; không cần đặt CSV vào thư mục này. Chuẩn bị/kiểm tra độc lập bằng `scripts.run_dg --prepare-only`; xem [data contract](../../docs/DATA_CONTRACT.md).

Đánh giá ACDC chỉ tạo manifest val406 sau khi khóa candidate, qua `scripts.make_manifest`. Manifest chứa đường dẫn local, được hash theo run và không commit. Thư mục này chỉ giữ hướng dẫn và placeholder.
