"""Offline tiny identity/gradient/save-load check, optionally CUDA + FP16 AMP."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile

import torch

from src.models.legacy.segformer_b0 import SegFormerB0
from src.models.proposed.weatherseg_b0 import WeatherSegB0
from src.models.registry import load_checkpoint, parameter_report
from src.data.augmentations.weather import WeatherGenerator
from src.data.preprocessing import ImagePreprocessor
from scripts.preview_weather import analytic_scene


def verify(device='cpu'):
    torch.manual_seed(42)
    base = SegFormerB0().eval().to(device)
    model = WeatherSegB0.from_b0(base.cpu()).to(device).eval()
    base = base.to(device)
    rgb, distance = analytic_scene((96,64))
    views = [WeatherGenerator().apply(rgb, condition, 2, seed=42, distance_m=distance)
             for condition in ('fog', 'night')]
    preprocessor = ImagePreprocessor((64,96))
    pixels = torch.stack([preprocessor.to_tensor(view.image) for view in views]).to(device)
    labels = torch.randint(0, 19, (2, 64, 96), device=device)
    labels[:, :4] = 255
    conditions = torch.tensor([view.condition_id for view in views], device=device)
    with torch.inference_mode():
        expected = base(pixels).logits
        actual = model(pixels)
        identity = torch.equal(expected, actual.logits)
    if not identity or actual.condition_logits.shape != (2, 5):
        raise AssertionError('K2 identity/condition shape check failed')
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cuda = str(device).startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=cuda, init_scale=1024.)
    context = torch.autocast('cuda', dtype=torch.float16) if cuda else nullcontext()
    with context:
        output = model(pixels, labels=labels, condition_labels=conditions)
    scaler.scale(output.loss).backward()
    gradients = [model.segformer.encoder.film[s].mlp[-1].weight.grad for s in range(4)]
    if any(g is None or not torch.isfinite(g).all() or not g.abs().sum() > 0 for g in gradients):
        raise AssertionError('FiLM gradients are absent/nonfinite/zero')
    if model.condition_head.weight.grad is None or not model.condition_head.weight.grad.abs().sum() > 0:
        raise AssertionError('Missing condition-head gradient')
    scaler.step(optimizer)
    scaler.update()
    model.eval()
    with torch.inference_mode():
        expected = model(pixels).logits
    with tempfile.TemporaryDirectory() as temporary:
        model.save_pretrained(temporary)
        restored = load_checkpoint(temporary, 'weatherseg_b0').to(device).eval()
        with torch.inference_mode():
            roundtrip = torch.equal(expected, restored(pixels).logits)
    if not roundtrip:
        raise AssertionError('K2 checkpoint roundtrip failed')
    return dict(status='passed', scope='tiny random-init software QA, not quality/convergence',
                device=device, amp=cuda, identity_exact=identity, checkpoint_roundtrip_exact=roundtrip,
                stage_film_gradients=True, condition_head_gradient=True,
                parameters=parameter_report(model), k1_conditions=[view.metadata['condition'] for view in views],
                input_scope='K1 procedural QA scene with known synthetic distance', torch=torch.__version__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = verify(args.device)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
