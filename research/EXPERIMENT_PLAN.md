# Predeclared development order — original shared-recipe study

The next study follows the [2026-11-19 baseline decision](BASELINE_PROTOCOL.md). B0/B2 below are completed historical controls; this original order is retained for provenance. A runnable published-checkpoint baseline and its new experiment matrix are pending implementation.

1. Train all three models independently:sourceCE20 then ACDC replayCE20. B0 is the small baseline;B2 and DeepLab form the main architecture comparison.
2. Diagnose per-class/weather errors,fit exposure,dev CE and source retention. Low IoU is not equivalent to class rarity.
3. On B2 from the same new source checkpoint,test weightedCE,Focal,CE+Dice and CE+classcrop separately;20epoch horizons and unchanged data/selection.
4. Choose by internal-dev/source gate,then test a selected combination and transfer the same intervention to DeepLab.
5. Repeat controls/candidates with seeds43/44. Freeze checkpoints before official-val. Report selected-best and common-epoch results,parameters,observed time and VRAM;do not claim exact compute matching.
6. Plan paired recording-group uncertainty and clearly separate seed variation. Historical dev/val feedback remains a limitation;no independent-test or SOTA claim yet.

These are existing techniques being evaluated in a controlled pipeline. A contribution needs demonstrated class gains/source retention/transfer,reliability evidence and clear limitations;adding losses alone is not novelty.
