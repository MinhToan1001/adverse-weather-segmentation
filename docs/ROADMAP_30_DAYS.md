# 30-day action plan (26 Oct–24 Nov 2026, ICT)

| Deadline | A (lead: training/research) | B (data/eval) | Exit check |
|---|---|---|---|
| 26 Oct | Create private GitHub repo, protect main, add B, initialize project/CI, open 8 issues | Install environment, review folder conventions, submit PR adding hardware note | PR/CI passes |
| 28 Oct | Run synthetic smoke, review logs, freeze class contract | Collect official download conditions and verify dataset folders/naming conventions | Both run tests + smoke |
| 31 Oct | Select backbone/optimizer, baseline protocol and compute budget | Write canonical 19-trainId converter and 20-sample visual audit; first manifests | G1 data review; no labels guessed |
| 3 Nov | Launch 1% subset overfit and 1 epoch baseline; debug loss | Audit class frequencies/splits, mask colors, mIoU and ignore=255 tests | Model reduces loss, metrics verified |
| 8 Nov | First reproducible full baseline run, checkpoint/config/log | Independent baseline validation + per-domain table template | G2 model + mIoU by domain |
| 12 Nov | Complete fair SegFormer B0 baseline and baseline ablation | Add ACDC validation; inspect failure examples rain/fog/night/snow | Domain shift analysis |
| 16 Nov | Propose exactly one modification, open design decision PR | Literature matrix: 5+ papers, gap analysis & comparison protocol | G3 approve novelty hypothesis |
| 20 Nov | Implement prototype + initial controlled run | Report confidence intervals, sample visualizations, runtime, failures | Basic ablation table |
| 24 Nov | Integrate results and final review, verify reproducibility | Draft method/data/metric/results sections | Report, clean git, known limits |

**Contingency:** If pretrained weights/dataset/GPU unavailable by Nov 3, use subset experiments and focus on auditable pipeline; do not invent metrics. Research extension may roll into next month.
