# Data contract

- RGB image and integer,one-channel mask share native dimensions.
- Cityscapes trainIds0–18;void255 excluded from loss/metrics. Raw labelIds are explicitly converted;never infer an encoding from pixel values.
- Cityscapes official-train2975 is grouped into2787fit/188monitor. ACDC official-train1600 is grouped by recording family into1198fit/402dev. Every weather is represented;declared groups do not overlap.
- Official-val Cityscapes500/ACDC406 is for frozen candidate checks. Grouping by recording tokens is a proxy,not proof of geographic independence;historical feedback prevents an independent-test claim.
- Training geometry is synchronized:scale0.5–2,crop/pad512×1024,flip0.5. RGB bilinear,mask nearest,pad255,ImageNet normalization. RGB augmentation never changes masks.
- Dev/monitor/evaluation have deterministic aspect-preserving fit;logits are unpadded and resized to native mask dimensions.
- Manifests are generated locally and hashed. Raw dataset,derived masks,weights and generated manifests are not committed.

Expected roots:Cityscape contains leftImg8bit/train and gtFine/train;ACDC contains rgb_anon_trainvaltest/rgb_anon and gt_trainval/gt. [Prepare/run instructions](TRAINING_GUIDE.md).
