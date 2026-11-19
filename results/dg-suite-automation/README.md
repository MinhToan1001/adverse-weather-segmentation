# Automatic DG model suites · 2026-11-19

[Run guide](../../docs/TRAINING_GUIDE.md) · [Verification](verification.json)

Both B2 and DeepLabV3+ ResNet50 now automatically run **profile → baseline20 → aug20 → consistency20**. No ARM/PROFILE_ONLY editing is required. Each model has one notebook, shared converted source masks, one result ZIP and independent checkpoints for its three arms. Full candidates start independently from ImageNet; pilot weights are discarded.

**167 Python tests passed after repository cleanup**, with cached pretrained checks enabled, plus1Node API test. The original pre-cleanup suite had177tests; ten obsolete supervised-workflow cases were removed. Committed-epoch/archive checks were retained and fresh notebook setup tests migrated to the DG workflow. All170local Markdown links passed; both notebooks and the code-only ZIP were rebuilt and checked. Queue tests use explicit fake trainer/profiler callbacks and cover both models, all arms, pause during aug, resume without retraining baseline, one profiling pass, corrupted/missing sealed profile rejection, setup preservation when session time is short and error diagnostics export. Actual generated notebook setup supports Kaggle-autoexpanded nested source ZIPs. Frozen evaluation now requires all six candidates and accepts two completed suite roots.

Before this cleanup, real Cityscapes preparation verified train2975/val500 and three arm configs referencing one converted-mask tree with byte-identical copied manifests. This test used B2 configs on CPU; the same preparation function serves both models. Existing shared objective/numerical tests and the prior real CPU B2/DeepLab two-epoch QA remain separate from queue simulation.

No new full GPU run, timing claim or mIoU improvement is established. There are now six main runs, not the earlier five-run budget. Thirty hours of shared platform quota may be insufficient; the suites preserve progress and never silently omit aug or shorten20epochs. A new session still needs the saved ZIP as input; the notebook cannot create another Kaggle session or replenish quota itself.

Large converted masks, raw images, checkpoints, test logs and ZIPs remain Git-ignored under outputs/. Only compact verification evidence is committed. Superseded supervised launchers/plans and older software-test dumps have been removed; historical model reports and their numeric records remain unchanged. For old workflows, use Git history and each result ZIP's own source snapshot.
