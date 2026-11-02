# Dataset QA report

Updated: 2026-11-01. Cityscapes: PASS for train/val preparation and source QA.
ACDC: extracted and PASS for file-level metadata QA; target pixels have not been opened.

| Check | Cityscapes | ACDC |
|---|---|---|
| Image/mask pairing | 2,975 train + 500 val | 1,600 train + 406 val pairs; 2,000 test images; val manifest created |
| Dimensions and decoding | All 3,475 RGB files decoded; 2048x1024, masks match | Deferred pixel QA |
| TrainId values | 0..18 / 255 only | Canonical trainId filenames verified; pixel values deferred |
| Mapping | All processed masks compared pixelwise to raw labelIds mapping | Not inspected |
| Leakage | No repeated paths or SHA-256-identical image files across combined source manifests | Combined source/target manifest path check PASS; target content hashing not performed |
| Visual audit | 20 samples, 10 train + 10 val; no obvious gross mispairing in reviewed sheets | Deferred |
| Actual configured loader | 20 source samples passed; train canvas 512x1024, native val labels | Constructor/role checks passed without pixel access; iteration deferred by protocol |
| Terms and version provenance | Supplied README/license SHA-256 recorded; archive version/hash unavailable | Supplied README/License.pdf SHA-256 recorded; official release/provider checksum unavailable |

Evidence: `results/cityscapes-qa-2026-11-01/audit.json`, `metadata.json`,
`loader_checks.json`, `selected.csv` and `tests-r02.txt` (81 passed).
The first test-log attempt failed because the output directory was missing;
the corrected run has a complete log and JUnit file.

ACDC metadata evidence: `results/datasets-metadata-2026-11-01/metadata.json`.
ACDC actual package roots are `data/raw/ACDC/rgb_anon_trainvaltest` and
`data/raw/ACDC/gt_trainval`; use these as --root and --mask-root respectively.
File counts, pair existence, nonzero sizes and split domains passed. This does
not certify archive checksums or target image/mask decoding.

Manifests: `data/manifests/cityscapes_train.csv`, `cityscapes_val.csv`, `acdc_val.csv`.
Converted masks: `data/processed/cityscapes/gtFine/{train,val}`. Raw data unchanged.
Visual sheets are local under `outputs/cityscapes-qa-2026-11-01/` and must stay
out of source control. Visual inspection is a sample check, not full semantic
annotation certification. Source file validity does not certify archive provenance.

Mapping reference checked:
https://raw.githubusercontent.com/mcordts/cityscapesScripts/master/cityscapesscripts/helpers/labels.py

Cityscapes is ready for source-only warm-up. Full training is intended for Colab
or Kaggle; no full baseline run has started and Gate B overall remains pending
provenance review and final sign-off. ACDC file-level metadata QA is now complete.

Final pre-freeze readiness and reproducibility: `results/in-progress-2026-11-01/dataset-readiness.json`. All three manifests regenerated identically. Task handoff: `docs/TASK_HANDOFF_2026-11-01.md`. Target pixel QA remains deferred by protocol, not certified by these metadata checks.
