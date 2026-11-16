"""Same native evaluator, metrics, calibration and visual panels for all3 models."""
import argparse
import json
from pathlib import Path

from src.evaluation.runner import evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--scope', choices=('dev', 'monitor', 'official-city', 'official-acdc'), required=True)
    args = parser.parse_args()
    config = json.loads((Path(args.checkpoint) / 'config.json').read_text())
    family = 'deeplabv3plus' if config.get('model_type') == 'project_deeplabv3plus_resnet50' else 'segformer'
    counts = dict(dev=dict(fog=81, night=93, rain=154, snow=74), monitor=dict(cityscapes=188),
                  **{'official-city': dict(cityscapes=500),
                     'official-acdc': dict(fog=100, night=106, rain=100, snow=100)})
    print(json.dumps(evaluate(args.checkpoint, family, args.manifest, args.out,
        expected_counts=counts[args.scope], expected_split='val' if args.scope.startswith('official') else 'train'), indent=2))
