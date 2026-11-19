"""JSON-lines inference worker. Node owns HTTP; PyTorch stays in one CPU process."""
import base64
import binascii
from contextlib import redirect_stdout
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
import traceback
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 8_000_000
MODEL_NAMES = ("segformer_b2", "deeplabv3plus_resnet50")
VI_LABELS = ("Đường", "Vỉa hè", "Tòa nhà", "Tường", "Hàng rào", "Cột", "Đèn giao thông",
             "Biển báo", "Cây xanh", "Địa hình", "Bầu trời", "Người", "Người đi xe",
             "Ô tô", "Xe tải", "Xe buýt", "Tàu", "Xe máy", "Xe đạp")


def decode_image(encoded):
    if not isinstance(encoded, str) or len(encoded) > (MAX_BYTES * 4 // 3 + 8):
        raise ValueError("Ảnh tối đa 10 MB.")
    try:
        content = base64.b64decode(encoded, validate=True)
        if not content or len(content) > MAX_BYTES:
            raise ValueError("Ảnh trống hoặc vượt quá 10 MB.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                if image.format not in {"PNG", "JPEG", "WEBP", "BMP"}:
                    raise ValueError("Chọn ảnh PNG, JPG, WebP hoặc BMP.")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("Ảnh tối đa 8 triệu pixel. Hãy giảm kích thước rồi thử lại.")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("Không đọc được ảnh. Hãy chọn tệp ảnh hợp lệ.") from error


def image_url(image, format="PNG"):
    buffer = BytesIO()
    image.save(buffer, format=format, **({"quality": 94} if format == "JPEG" else {}))
    mime = "jpeg" if format == "JPEG" else "png"
    return f"data:image/{mime};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class Studio:
    def __init__(self):
        self.cached = None
        self.models = {}
        for name in MODEL_NAMES:
            report_file = ROOT / "results" / f"{name}_seed42" / "verification.json"
            report = json.loads(report_file.read_text(encoding="utf-8")) if report_file.is_file() else {}
            for stage in ("source", "adapt"):
                env_key = name.upper() + ("_SOURCE_CHECKPOINT" if stage == "source" else "_CHECKPOINT")
                default = ROOT / "outputs/reviews" / f"{name}_seed42" / stage / "best"
                path = Path(os.environ.get(env_key, str(default))).resolve()
                records = report.get("checkpoints", [])
                digest = next((r["sha256"] for r in records if r["stage"] == stage and r["generation"] == "best"), None)
                self.models[name, stage] = dict(path=path, digest=digest, report=report,
                    available=(path / "config.json").is_file() and (path / "model.safetensors").is_file())
        self.samples = self.discover_samples()

    @staticmethod
    def discover_samples():
        city = Path(os.environ.get("CITYSCAPES_ROOT", ROOT / "data/raw/cityscapes"))
        acdc = Path(os.environ.get("ACDC_ROOT", ROOT / "data/raw/ACDC"))
        samples = []
        choices = [("clear", "Trời quang", "Cityscapes", city / "leftImg8bit/val", "*_leftImg8bit.png")]
        choices += [(k, label, "ACDC", acdc / "rgb_anon_trainvaltest/rgb_anon" / k / "val", "*_rgb_anon.png")
                    for k, label in [("fog", "Sương mù"), ("rain", "Mưa"), ("snow", "Tuyết"), ("night", "Ban đêm")]]
        for key, label, dataset, folder, pattern in choices:
            path = next(iter(sorted(folder.rglob(pattern))), None) if folder.is_dir() else None
            if path:
                samples.append(dict(id=key, label=label, dataset=dataset, path=path))
        return samples

    def metadata(self):
        models = []
        for name, title, detail in [("segformer_b2", "SegFormer B2", "Mô hình chính"),
                                    ("deeplabv3plus_resnet50", "DeepLabV3+", "Đối chứng CNN")]:
            report = self.models[name, "adapt"]["report"]
            metrics = report.get("metrics", {})
            models.append(dict(id=name, name=title, detail=detail,
                stages={s: self.models[name, s]["available"] for s in ("source", "adapt")},
                parameters=report.get("configuration", {}).get("parameters"),
                dev_miou=metrics.get("acdc_dev_after"), source_dev_miou=metrics.get("acdc_dev_before")))
        return dict(models=models, samples=[{k: v for k, v in s.items() if k != "path"} for s in self.samples],
                    device="cpu", max_bytes=MAX_BYTES, max_pixels=MAX_PIXELS)

    def sample(self, sample_id):
        selected = next((s for s in self.samples if s["id"] == sample_id), None)
        if selected is None:
            raise ValueError("Không tìm thấy ảnh mẫu.")
        return dict(content=base64.b64encode(selected["path"].read_bytes()).decode("ascii"), mime="image/png")

    def predict(self, model_name, stage, encoded):
        record = self.models.get((model_name, stage))
        if record is None:
            raise ValueError("Model hoặc phiên bản chưa được hỗ trợ.")
        if not record["available"]:
            raise ValueError("Checkpoint chưa có trên máy. Xem hướng dẫn cài checkpoint của studio.")
        image = decode_image(encoded)
        started = time.perf_counter()
        with redirect_stdout(sys.stderr):
            import numpy as np
            import torch
            from src.inference import SegmentationPredictor
            from src.data.labels import CLASS_NAMES, PALETTE
            torch.set_num_threads(int(os.environ.get("INFERENCE_THREADS", "2")))
            key = (model_name, stage, (record["path"] / "model.safetensors").stat().st_mtime_ns)
            cold = self.cached is None or self.cached[0] != key
            if cold:
                self.cached = None  # One resident model keeps CPU memory bounded.
                predictor = SegmentationPredictor(record["path"], device="cpu", model_name=model_name,
                                                   expected_sha256=record["digest"])
                self.cached = key, predictor
            result = self.cached[1].predict(image)
            counts = np.bincount(np.asarray(result.train_ids).reshape(-1), minlength=19)
            classes = [dict(id=i, name=CLASS_NAMES[i], label=VI_LABELS[i], color=PALETTE[i].tolist(),
                            pixels=int(n), percent=round(100 * int(n) / (image.width * image.height), 4))
                       for i, n in enumerate(counts) if n]
        return dict(model=model_name, stage=stage, width=image.width, height=image.height,
                    device=result.device, cold_load=cold, forward_ms=round(result.forward_ms, 1),
                    inference_ms=round(result.latency_ms, 1),
                    total_ms=round(1000 * (time.perf_counter() - started), 1),
                    checkpoint_verified=bool(record["digest"]), classes=sorted(classes, key=lambda c: -c["pixels"]),
                    images=dict(original=image_url(result.image, "JPEG"), mask=image_url(result.color_mask),
                                labels=image_url(result.train_ids), overlay=image_url(result.overlay)))


def main():
    os.environ.update(USE_TF="0", USE_TORCH="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", HF_HUB_OFFLINE="1")
    studio = Studio()
    print(json.dumps(dict(event="ready")), flush=True)
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            action = request.get("action")
            if action == "metadata":
                value = studio.metadata()
            elif action == "sample":
                value = studio.sample(request.get("sample"))
            elif action == "predict":
                value = studio.predict(request.get("model"), request.get("stage", "adapt"), request.get("image"))
            else:
                raise ValueError("Thao tác không hợp lệ.")
            response = dict(id=request["id"], ok=True, result=value)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            message = str(error) if isinstance(error, ValueError) else "Chưa xử lý được ảnh. Kiểm tra checkpoint và thử lại."
            response = dict(id=request.get("id"), ok=False, error=message)
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
