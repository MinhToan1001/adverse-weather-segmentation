"""Model-independent DG arm composition; all candidates start from ImageNet."""
from pathlib import Path
import yaml

from src.experiments.config import compose_experiment
from src.experiments.protocol import DG_PROTOCOL
from src.models.registry import MODEL_NAMES
from src.training.dg import ARMS, validate_settings

DEFAULTS = Path(__file__).resolve().parents[2] / 'configs/dg/paired_source.yaml'


def compose_dg(model, arm, data, output, seed=42, pair_probability=.5, consistency_weight=.1):
    if model not in MODEL_NAMES or arm not in ARMS:
        raise ValueError('Choose a supported DG model and arm')
    defaults = yaml.safe_load(DEFAULTS.read_text(encoding='utf-8'))
    settings = dict(arm=arm, pair_probability=pair_probability if arm != 'baseline' else 0.,
                    consistency_weight=consistency_weight if arm == 'consistency' else 0.,
                    source_style=defaults['source_style'] if arm != 'baseline' else None)
    validate_settings(settings)
    cfg = compose_experiment(model, 'source', data, output, seed=seed)
    cfg.update(experiment_protocol=DG_PROTOCOL, training_policy='dg_source_20e_v2',
               experiment=f'dg-{model}-{arm}-seed{seed}', dg=settings,
               selection=dict(metric='cityscapes_val_miou', target_data_access='none'))
    return cfg
