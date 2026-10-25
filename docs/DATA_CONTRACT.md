# Data contract v1

CSV columns: `image,mask,domain,split`. `image` = RGB file; `mask` = **single channel trainId PNG**. Pixel IDs must be `0..18` or `255` ignore. RGB color masks, raw Cityscapes `labelIds` and color palettes are NOT usable as targets without an explicit verified mapping. For Cityscapes use `*_gtFine_labelTrainIds.png` when available, or correctly convert from official IDs; ACDC naming and trainId semantics must be verified against their official docs.

19 classes (Cityscapes trainId order): road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle.

**Splits:** Cityscapes train => train, Cityscapes val => val; never use hidden Cityscapes test labels. ACDC evaluation splits selected and recorded separately. No duplicate image across train/val/test. Store data locally under ignored `data/raw`/`data/processed`.

**Experimental setup for month 1:** train on clear weather Cityscapes then evaluate on ACDC adverse conditions; this is a *cross-domain generalization test*, not domain adaptation. If ACDC training data later used, label it as a separate protocol and do not compare as though identical.
