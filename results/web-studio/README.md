# WeatherLens Node.js studio verification

**Historical verification of the initial B0/B2 UI.** The application now uses B2 before/after ACDC; see [current repo refresh verification](../repo-refresh/README.md). The original record below is retained for provenance.

[Run the studio](../../web/README.md) · [Verification record](verification.json)

The Node.js UI and HTTP server replace the former Streamlit app. A local Python worker reuses the accepted B0/B2 PyTorch checkpoints and shared native-resolution inference. Training/data/loss/model code is unchanged.

Validation includes the full Python regression suite, Node HTTP boundary checks and Chrome testing with real local checkpoints. The browser exercise covers sample selection, B0/B2 inference on the same image, adapted/source checkpoint switching, day/night inputs, real upload and invalid-file handling, result invalidation, opacity/view changes, all three PNG downloads and a 390px responsive layout. Downloaded label and color PNGs are checked for native dimensions and exact palette correspondence. These are software checks, not accuracy evaluations.

Desktop/mobile screenshots, downloaded masks, browser traces and full test logs stay local under `outputs/web-studio/`. They are not committed because they contain or derive from dataset images. No weights, result ZIPs or raw data are included in the web source.

Missing checkpoints are shown as unavailable; DeepLab awaits training. A fresh clone needs its own local weights to run inference. The server is local-only and does not provide hosted deployment or remote GPU execution.
