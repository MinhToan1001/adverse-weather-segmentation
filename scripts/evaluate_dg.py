"""Seal the cohort or evaluate its frozen best checkpoints; never select on ACDC."""
import argparse
from pathlib import Path

from src.experiments.dg_artifacts import seal_cohort,seal_suites, resolve_frozen


def evaluate_frozen(cohort,run_id,manifest,out,scope,device=None):
    candidate=resolve_frozen(cohort,run_id)  # Must pass before opening target data.
    if scope not in {'official-city','official-acdc'}:raise ValueError('Only official val splits are supported')
    from src.evaluation.runner import evaluate
    counts=dict(cityscapes=500) if scope=='official-city' else dict(fog=100,night=106,rain=100,snow=100)
    canvas=candidate['processing']['image_size']
    return evaluate(Path(candidate['workspace'])/'run/best',
        'segformer' if candidate['model']=='segformer_b2' else 'deeplabv3plus',
        manifest,out,canvas=canvas,device=device,expected_counts=counts,expected_split='val')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    freeze=commands.add_parser('freeze');inputs=freeze.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--workspaces',nargs='+');inputs.add_argument('--suites',nargs=2)
    freeze.add_argument('--out',required=True)
    evaluate=commands.add_parser('evaluate')
    for name in ['cohort','run-id','manifest','out']:evaluate.add_argument('--'+name,required=True)
    evaluate.add_argument('--scope',choices=['official-city','official-acdc'],required=True)
    evaluate.add_argument('--device',choices=['cpu','cuda'])
    a=p.parse_args()
    if a.command=='freeze':
        seal_suites(a.suites,a.out) if a.suites else seal_cohort(a.workspaces,a.out)
    else:evaluate_frozen(a.cohort,a.run_id,a.manifest,a.out,a.scope,a.device)
