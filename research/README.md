# Research context

[Active experiment matrix](EXPERIMENT_PLAN.md) · [Method/formulas](../docs/RESEARCH_FRAMEWORK.md) · [Primary bibliography](../docs/paper/references.bib)

**Next baseline agreed on 2026-11-19:** [published Cityscapes B2 → supervised ACDC FT20](BASELINE_PROTOCOL.md). The direction and 20-ACDC-pass budget are agreed; the detailed execution proposal is not yet implemented. Current notebooks still execute the previous shared source20/adapt20 pipeline.

[SegFormer 2021 source-paper review](paper_reviews/SegFormer_2021.md): published B0/B2 Cityscapes scores and the distinction between the Cityscapes reference and our ACDC fine-tuning extension.

Paper notes under paper_reviews/ are background literature,not instructions for the active pipeline. Current training is supervised Cityscapes→ACDC adaptation;published UDA scores have different label access/protocols. The fresh comparison is B0/B2/DeepLab with shared20epoch source/adapt policies. Old project scores are not reused.
