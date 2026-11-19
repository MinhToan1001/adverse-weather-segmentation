> Historical draft, superseded by the application scope. See [archive notice](README.md).

# Class-Aware Adaptation for Adverse-Weather Semantic Segmentation

### Shared Training and a Transformer–CNN Comparison

**Research draft — fresh GPU experiments pending**

[Code](../../README.md) · [Current method](../../docs/RESEARCH_FRAMEWORK.md) · [Execution](../../docs/TRAINING_GUIDE.md) · [Bibliography](references.bib)

## Abstract

We study whether class-aware training improves adverse-weather segmentation while preserving clear-weather knowledge. SegFormer-B0 is the lightweight baseline;SegFormer-B2 and a project DeepLabV3+ResNet50 implementation provide the development comparison. All start from ImageNet encoders/new19-class heads and share processing,20epoch source training and20epoch supervised ACDC replay. CE controls precede fit-weighted CE,Focal,CE+Dice and class-aware crop ablations. Transactional checkpoints and model-specific result archives support interrupted sessions. Software integration and recovery are tested on tiny real-image CPU probes;new GPU accuracy remains pending. Old project checkpoints/scores are excluded from the active comparison. Algorithmic novelty and superiority to prior work have not been established.

## 1. Introduction

Low per-class IoU can result from imbalance,lost detail after resizing,ambiguous appearance or inadequate representation. A more complex objective is a hypothesis rather than an improvement by definition. We ask which weak classes lose exposure,whether objective/crop interventions help,whether source performance is retained,and whether effects transfer across transformer and CNN architectures.

## 2. Related work

[SegFormer](https://arxiv.org/abs/2105.15203) motivates hierarchical transformer baselines. [DeepLabV3+](https://arxiv.org/abs/1802.02611) combines atrous context and decoder refinement;our R50 variant uses separable ASPP/decoder and is not a full paper recipe reproduction. [Kalb & Beyerer,CVPR2023](https://arxiv.org/abs/2303.14115) motivate studying forgetting and normalization in adverse-weather adaptation;our pooled supervised replay differs from their domain-incremental sequence.

[Focal](https://arxiv.org/abs/1708.02002) reduces contributions from easy examples;[V-Net](https://arxiv.org/abs/1606.04797) motivates region-overlap optimization. Our multiclass Dice reduction and bounded inverse-square-root class weights are specified in the method document,not claimed as reproductions of those full methods. HRDA/Refign use different target-label/protocol access;published UDA test scores are not placed into a superiority table with supervised validation scores.

## 3. Data

[Cityscapes](https://arxiv.org/abs/1604.01685) official-train2975 is grouped into2787fit/188monitor. [ACDC](https://openaccess.thecvf.com/content/ICCV2021/html/Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.html) official-train1600 is grouped by recording family into1198fit/402dev. Labels are19trainIds,void255 ignored. Recording tokens are a conservative grouping proxy,not proven geographic independence. Clear-reference pairs are recorded but not used by the current loss.

Frozen validation uses500Cityscapes and406ACDC images. Development feedback has previously involved these data;resetting weights does not create an independent test. Candidate selection uses monitor/internal-dev before official validation.

## 4. Shared method

All models use aspect-preserving512×1024 fit,RGB bilinear/mask nearest interpolation,ImageNet normalization and synchronized scale0.5–2/crop-pad/flip0.5. Evaluation removes padding and restores native logits. B0/B2/DeepLab have separate architecture files and a common logits/criterion/trainer/evaluator contract.

Parameters are3.72M/27.36M/26.70M. Counted forward conv/matmul work at batch1,512×1024 is35.51G/258.33G/150.71G FLOPs. DeepLab is within2.43% of B2's parameter count but has41.7% less counted compute;this is not exact compute matching or measured GPU latency.

Source:ImageNet encoder/new head,batch2/accum4,AdamW encoderLR6e-5/head×10,BN stats updating. Adaptation:its own new source best,source/target1:1,batch1/accum8,LR1e-5,BN stats fixed and affine trainable. Both use20epochs,no early stopping,weight decay0.01,warm-up20updates,poly power1 and GPUFP16AMP. Real batch2 is needed for DeepLab pooled BN;accumulation does not merge BN moments. SegFormer encoder LayerNorm differs from BN.

CE controls are followed by isolated loss/crop ablations. Fit-weighted CE uses bounded inverse-square-root priors without dev/val IoU. Focal uses gamma2. CE+0.5Dice averages overlap over GT-present minibatch classes. Optional crop selects an available class and attempts to retain64pixels;exposure successes/failures are logged. Exact formulas,void reductions and limitations are in [the method](../../docs/RESEARCH_FRAMEWORK.md).

Source best maximizes monitor mIoU. Adaptation best maximizes target-dev mIoU subject to source decrease≤1pp;an ineligible fallback is reported as such. Checkpoints include optimizer/scheduler/scaler/RNG and previous generations. Interrupted epochs repeat;model-specific atomic ZIPs preserve completed progress across saved sessions.

## 5. Experimental design and current results

Three independent notebooks first establish source/adaptCE controls,then B2 loss/crop ablations,selected transfer to DeepLab and seed43/44 confirmations. Report selected-best and common-epoch results,per-class/weather IoU,source retention,ECE,observed time and peak VRAM. Paired recording-group uncertainty is planned;it does not replace seed variation or independent testing.

| Fresh model | Source20 | Adapt20 | Cityval500 | ACDCval406 |
| --- | --- | --- | --- | --- |
| B0 | Pending | Pending | Pending | Pending |
| B2 | Pending | Pending | Pending | Pending |
| DeepLabV3+R50 | Pending | Pending | Pending | Pending |

[Software verification](../../results/repository-release/README.md) is separate from this table. Tiny CPU probes test serialization/geometry/gradients,not convergence or accuracy. No prior project score is reused.

## 6. Limitations and reproducibility

Shared recipes do not prove model-specific optimality. ImageNet recipes and compute differ;GPU memory/timing remain to be measured. Focal may emphasize noise,full-mask priors differ from crop exposure,and crop cannot reconstruct detail already lost by resizing. Historical feedback and small dev group counts constrain future conclusions. Fixed20epochs is a project design,not proof of convergence or an80%mIoU target.

Each model notebook uses the same generated template and source code. Run-bound config/source/manifests are hashed;resume rejects mismatched code/data/model plans. Raw data/checkpoints/ZIPs are excluded from Git. [Team workflow](../../docs/CONTRIBUTING.md) · [Software citation](../../CITATION.cff).
