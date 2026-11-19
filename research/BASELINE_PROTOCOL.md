> **SUPERSEDED / LỊCH SỬ:** Không phải giao thức hiện hành. Xem [DG recovery plan](DG_RECOVERY_PLAN.md). ACDC supervised không được dùng làm DG.

# Baseline decision — 2026-11-19

Status: **research direction and primary budget agreed with the user**. This is a protocol document, not an executable training configuration. The existing trainer, checkpoints and three model notebooks have not been changed to implement this protocol.

## Agreed decisions

1. Use a published **SegFormer-B2 Cityscapes segmentation checkpoint** as the common starting point for a supervised ACDC fine-tuning baseline and subsequent improvements.
2. Keep **20 passes over ACDC fit** in the primary comparison. Each primary arm sees the same ACDC examples the same number of times. Replay adds source-image processing; report that added cost instead of claiming matched compute.
3. Include an additional **update-matched target-only control** before attributing a gain specifically to replay. Its target exposure can differ from the primary 20-pass arms and must be stated separately.

No GPU training has been launched during this decision. Existing results remain valid records of the previous project recipe.

**Clarified study scope:** Cityscapes is the published source reference; ACDC is the user's downstream fine-tuning/evaluation extension. A paper does not need an ACDC score to provide the starting model for this study. Keep published Cityscapes results and project-measured ACDC adaptation results in separate tables. The [verified SegFormer review](paper_reviews/SegFormer_2021.md) records B0/B2 source-paper results and their evaluation conditions.

## Paper and artifact selection

| Reference | Role | Verified findings / limits |
| --- | --- | --- |
| [SegFormer, NeurIPS 2021 — official code](https://github.com/NVlabs/SegFormer) | Architecture and published source checkpoint | The authors publish Cityscapes configurations. [NVIDIA's B2 Cityscapes checkpoint on Hugging Face](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024) is a pretrained segmentation model, not only an ImageNet encoder. The model card is written by Hugging Face and describes 1024 × 1024 fine-tuning. Pin revision, weights hash and label mapping before using it. |
| [Kalb & Beyerer, CVPR 2023](https://arxiv.org/html/2303.14115v2) | Closely related analysis of supervised weather adaptation, augmentation and forgetting | Uses separate Cityscapes → Night/Rain/Snow/Fog tasks, including DeepLabV3+ ResNet50 and a B2 architecture study. The main described schedule is 200 source / 150 target epochs. [Author repository](https://github.com/tobiaskalb/feature-reuse-css) mainly supplies augmentation configurations and implementation pointers, not a complete ready-to-run training pipeline. Our pooled, 20-pass setting is not its exact reproduction. |
| [ACDC, ICCV 2021 — supervised training details](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Sakaridis_ACDC_The_Adverse_ICCV_2021_supplemental.pdf) | Dataset/benchmark reference | Reports supervised methods and per-condition/all-condition models. Its DeepLabv3+ all-condition model is an exception trained for 30 epochs. It is not the project's ResNet50 implementation or the newly selected B2 baseline. |

Refign/HRDA are UDA references, not directly matched baselines for supervised ACDC training. Their published scores can be cited in a separately labelled related-work table, not used as same-protocol competitors without reproducing the relevant evaluation and data-access conditions.

**Claim boundary:** this choice is a supervised ACDC study built on a published SegFormer checkpoint. It is not an exact reproduction of a paper's ACDC training experiment and must not be described as one.

## Experimental arms

| ID | Starting weights | Training | Purpose |
| --- | --- | --- | --- |
| S0 | Published B2 Cityscapes checkpoint | None | Measure the actual starting point on ACDC and Cityscapes with the project's evaluation recipe. |
| FT20 | Same S0 | ACDC fit only, CE, 20 passes, basic geometry augmentation | Main supervised fine-tuning baseline. No replay, class weights, new augmentation or new loss. |
| RP20 | Same S0 | Same ACDC exposure, CE, Cityscapes replay 1:1 | Isolate the practical accuracy/retention trade-off of adding replay, with added compute disclosed. |
| FT-U | Same S0 | Target-only CE, optimizer-update budget matched to RP20 | Check whether replay gains can instead be explained by extra optimization. Label its different ACDC exposure explicitly. |
| AUG20 | Same S0 | A selected primary arm + one predeclared augmentation change | Later ablation. Run after the FT20/RP20/FT-U comparison; no augmentation has been approved for implementation yet. |

Do not initialize FT20 from the already ACDC-adapted B2 checkpoint. Do not compare a newly trained candidate against the old 63.02% score as if they shared the same initialization. B0 and the previous B2 run stay historical architecture controls; their weights/results are preserved, not overwritten.

## Data and checkpoint selection proposal

- Retain the current ACDC train partition: **1,198 fit / 402 internal dev**, with identical ordered manifests for every arm. Official ACDC val 406 stays out of optimization and checkpoint selection in the new experiment.
- The old 188-image Cityscapes monitor comes from Cityscapes train. A published Cityscapes-trained checkpoint must not be treated as having held those images out. Retire that set as an unseen source-monitor criterion in the new experiment.
- Proposed selection: pick best by internal ACDC dev mIoU under a rule fixed before training. Evaluate Cityscapes val 500 for source retention after checkpoint freeze. Do not silently retain the old train-monitor-based 1-pp source gate.
- Report Cityscapes change relative to **S0 evaluated on the same 500 images**, alongside ACDC improvement. Historical inspection of official validation sets remains a development-feedback limitation.
- Official test submissions, if later performed, use the benchmark's withheld-label server. Distinguish internal dev, official validation and official test in every table.

## Remaining execution details

Before building a runnable notebook, pin/download the exact S0 revision and verify its tensor shapes, complete weight loading, 19-class trainId order, ignore 255 and input-normalization contract. No checkpoint has been downloaded or measured as part of this decision.

Use the previous adaptation settings as the initial implementation proposal (AdamW, encoder/head LR 1e-5, effective batch 8, weight decay 0.01, warm-up/poly scheduler), with no new loss or color augmentation. These are project choices, not the SegFormer paper's original training schedule. For FT-U, predeclare how its LR schedule follows its update budget; it must not resume a completed zero-LR schedule.

Keep source preparation, model implementation and evaluation separate from training-policy ablations. Record optimizer updates, ACDC/source exposures, AMP-skipped updates, elapsed time and peak memory; 20 passes do not imply equal updates or FLOPs across replay and target-only arms.

Proposed common research evaluation: preserve aspect ratio into the existing 512 × 1024 canvas, restore predictions to native image resolution, 19 trainIds, ignore 255, FP32 and no TTA. Apply it to S0 and every trained arm. This is the project's budget-conscious evaluation policy, not the checkpoint's published 1024 × 1024 setting. A faithful published-recipe reference must be evaluated and labelled separately; numbers from the two recipes must not be mixed.

Implementation and new GPU runs are pending. Current model code, training configurations, web demo and accepted B0/B2 artifacts remain unchanged.
