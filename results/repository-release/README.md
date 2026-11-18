# Repository release verification

[Project](../../README.md) · [Three model runs](../../docs/TRAINING_GUIDE.md)

The active repository has three model notebooks,14CLI/build scripts and shared src/data,models,training,evaluation and experiments modules. Historical task notebooks/reports/scripts are local-only archive material. Current accuracy remains pending;old project checkpoints/scores are excluded.

| Verification | Evidence |
| --- | --- |
|Current regression suite:137passed,including optional cached-pretrained checks|[tests.txt](tests.txt),[JUnit](tests.xml)|
|Fresh checkout without datasets/weights:133passed,4optional pretrained checks skipped|[clean-checkout-tests.txt](clean-checkout-tests.txt)|
|Real-image CPU train/stop/resume of B0,B2,DeepLab;DeepLab continuous/resumed weights bitwise equal|[cpu_probe.json](cpu_probe.json)|
|Actual fresh/resume notebook setup,ZIP or extracted source,three models and strict plan separation|Current launcher tests;[handoff.json](handoff.json)|
|Checkpoint interruption,checksum corruption,rename gap,log rollback and previous-ZIP preservation|Current transactional tests|
|Former Streamlit startup (historical; replaced by the Node.js studio)|[app_smoke.json](app_smoke.json)|
|Architecture parameter/forward-operation counts|[complexity.json](complexity.json)|
|Provisional GPU budget from historical timing only|[runtime_budget.json](runtime_budget.json)|
|Legacy source archive verified;output relocation is reversible|[source_cleanup.json](source_cleanup.json),[output_cleanup.json](output_cleanup.json)|

CPU probes use4real-fit/2monitor images,64×96,two batches/epoch and random initialization under a20epoch scheduler. They test software,not20epoch convergence or GPU accuracy. GPU estimates are provisional;DeepLab has no direct GPU epoch measurement. Model-specific stage ETA updates from observed times.

Historical outputs (75 directories, about 16.7 GB) are retained in a local, Git-ignored archive. This reversible relocation does not free disk space or change raw datasets. The archive is not required to clone or run the project. New GPU accuracy remains pending.

The current web interface is [WeatherLens Studio](../../web/README.md); its [Node.js/browser verification](../web-studio/README.md) supersedes the historical Streamlit check above. This release snapshot predates the completed B0/B2 training reviews listed in the [results index](../README.md).
