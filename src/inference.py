"""Shared native-resolution inference for project B0,B2 and DeepLab checkpoints."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from src.data.labels import PALETTE
from src.data.preprocessing import ImagePreprocessor
import yaml


@dataclass
class Prediction:
    image: Image.Image
    train_ids: Image.Image
    color_mask: Image.Image
    overlay: Image.Image
    latency_ms: float
    forward_ms: float
    device: str


class SegmentationPredictor:
    def __init__(self, checkpoint, config=None, device="auto", expected_sha256=None, model_name=None):
        checkpoint = Path(checkpoint)
        weights = checkpoint / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"Chưa tìm thấy checkpoint tại {checkpoint}")
        if expected_sha256:
            with weights.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != expected_sha256:
                raise ValueError("Hash checkpoint khác bản đã chốt")
        if config is None:
            candidate=checkpoint.parent/'config.json'
            if candidate.is_file() and 'data' in json.loads(candidate.read_text()):config=candidate
        cfg=yaml.safe_load(Path(config).read_text()) if config else dict(model=dict(num_labels=19),data=dict(image_size=[512,1024]))
        if cfg["model"]["num_labels"] != 19:
            raise ValueError("Demo yêu cầu checkpoint 19 lớp Cityscapes")
        self.preprocessor = ImagePreprocessor(cfg["data"]["image_size"], cfg["data"].get("preprocessing"))
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device phải là auto, cpu hoặc cuda")
        selected = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if selected == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA chưa khả dụng trong môi trường hiện tại")
        self.device = torch.device(selected)
        from src.models.registry import load_checkpoint
        model=load_checkpoint(checkpoint,model_name or cfg['model'].get('name'))
        if model.config.num_labels != 19 or model.config.semantic_loss_ignore_index != 255:
            raise ValueError("Checkpoint phải có 19 lớp và ignore index 255")
        self.model = model.eval().to(self.device)
        self._lock = threading.Lock()

    def predict(self, image):
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                rgb = ImageOps.exif_transpose(source).convert("RGB")
        elif isinstance(image, Image.Image):
            rgb = ImageOps.exif_transpose(image).convert("RGB")
        else:
            raise ValueError("Ảnh đầu vào phải là PIL Image hoặc đường dẫn ảnh")
        if min(rgb.size) < 1:
            raise ValueError("Ảnh đầu vào phải có kích thước dương")

        with self._lock, torch.inference_mode():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            pixels = self.preprocessor.to_tensor(self.preprocessor.fit_image(rgb))[None].to(self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            forward_started = time.perf_counter()
            logits = self.model(pixel_values=pixels).logits
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            forward_ms = (time.perf_counter() - forward_started) * 1000
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Model trả logits không hữu hạn")
            logits = F.interpolate(logits, size=self.preprocessor.size, mode="bilinear", align_corners=False)
            top, left, height, width = self.preprocessor.fit_box(rgb.size)
            logits = logits[:, :, top:top + height, left:left + width]
            logits = F.interpolate(logits, size=(rgb.height, rgb.width), mode="bilinear", align_corners=False)
            labels = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            colored = PALETTE[labels]
            overlay = ((np.asarray(rgb, dtype=np.float32) + colored) / 2).astype(np.uint8)
            train_ids = Image.fromarray(labels)
            color_mask = Image.fromarray(colored)
            overlay_image = Image.fromarray(overlay)
            latency_ms = (time.perf_counter() - started) * 1000
        return Prediction(rgb, train_ids, color_mask, overlay_image, latency_ms, forward_ms, str(self.device))
