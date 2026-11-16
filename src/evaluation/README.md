# Shared evaluation

runner.py evaluates all three supported models with native output,pooled confusion,mIoU/per-class/weather scores,ECE,coverage and visual RGB/GT/prediction/confidence/error panels. semantic.py restores logits through the same unpad/resize contract. metrics.py ignores255;zero union is undefined,false-positive-only classes score0. calibration.py contains reliability diagnostics.

CLI:scripts.evaluate_research with checkpoint,manifest,scope and out. Scope enforces expected dataset counts/split. Models/config/manifests and confusion caches are hashed. Current inference is512×1024 canvas,float32,no TTA. Confidence metrics are diagnostics,not deployment safety evidence. Paired group-CI and independent-seed confirmation remain part of the experiment plan,not fabricated measurements.
