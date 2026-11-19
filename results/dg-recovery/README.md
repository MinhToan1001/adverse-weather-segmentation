# DG recovery verification · 2026-11-19

[Protocol and method plan](../../research/DG_RECOVERY_PLAN.md) · [Machine-readable record](verification.json)

Restored the Cityscapes-only training/selection boundary. ACDC supervised training now requires an explicit legacy opt-in in trainer and launchers. DG configs reject target training/selection, wrong source split/counts and adapted-checkpoint initialization. The source→ACDC notebooks moved to archive; the active notebook only runs the new DG baseline.

Real local preflight converted/prepared **2.975 Cityscapes train / 500 val** pairs, validated the DG config and frozen hashes against current source. Six sampled converted masks matched image dimensions and valid trainIds. No ACDC images/masks or GPU training were used by this preflight.

The full regression suite passed **148 tests** before the final nested-ZIP resume adjustment. After that adjustment, **23 affected tests passed**, including the 10 DG-specific checks and actual generated notebook setup restoring Kaggle-autoexpanded source. Shared transactional checkpoint integrity tests are covered by the full regression; the new DG queue's pause/resume test uses an explicit fake trainer, not GPU convergence.

Historical B2 source config is Cityscapes-only; adapted config used ACDC labels. Their accepted best-weight hashes are recorded without copying weights. Historical source selection used Cityscapes monitor188 and is a reference, not a silently relabeled train2975/val500 DG baseline. Adapted63,02% is excluded from DG claims.

No new DG accuracy or proposed-method benefit is established. The baseline notebook has not completed a GPU run. The style+consistency proposal still requires implementation and controlled ablation. Prior ACDC feedback remains a reporting limitation. Raw data, local converted masks, ZIPs and full logs are ignored by Git.
