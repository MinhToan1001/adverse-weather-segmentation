# SegFormer — Cityscapes source baseline

Verified 2026-11-19 against the NeurIPS proceedings, paper Table 1(a), section 4.1 and the authors' implementation.

**Paper:** Enze Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021.

- [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html)
- [Paper, Table 1(a) and section 4.1](https://arxiv.org/html/2105.15203v3)
- [Official training/evaluation implementation and model downloads](https://github.com/NVlabs/SegFormer)
- [Published NVIDIA B2 Cityscapes checkpoint in Transformers format](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024)

## Published Cityscapes reference

| Model | Cityscapes val mIoU — single-scale | Cityscapes val mIoU — multi-scale |
| --- | ---: | ---: |
| SegFormer-B0 / MiT-B0 | 76.2% | 78.1% |
| SegFormer-B2 / MiT-B2 | 81.0% | 82.2% |

Single-scale evaluates one scale; multi-scale combines predictions at several scales. These are source-paper reference numbers, not measurements produced by this project, and are not ACDC scores.

The paper uses ImageNet-1K encoder pretraining followed by supervised segmentation training. Section 4.1 specifies 1024 × 1024 Cityscapes training crops and sliding-window evaluation. A 512 × 1024 fit-to-canvas research evaluator is a different recipe; do not claim it reproduces the paper's Cityscapes scores without checking the matching inference conditions and checkpoint.

## Vai trò trong project hiện hành

Dùng SegFormer-B2 làm kiến trúc chính, khởi tạo encoder ImageNet và head mới. Baseline học Cityscapes train2975, chọn best bằng val500,20epoch; ACDC chỉ đánh giá sau freeze. Recipe và inference của project khác paper, vì vậy không nhận là tái lập nguyên số Cityscapes của bài.

Hai nhánh aug/consistency kiểm tra độ bền vững khi chỉ biến đổi nguồn. Kiến trúc công bố là nền tảng; hiệu quả DG phải do project đo với đối chứng cùng protocol. Điểm B2 adapted63,02% lịch sử đã dùng nhãn ACDC, không thuộc bảng DG.

[Phương pháp và dữ liệu hiện hành](../../docs/DG_METHOD.md).
