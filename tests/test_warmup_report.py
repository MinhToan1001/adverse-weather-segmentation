import json
from pathlib import Path
import pytest
import yaml
from scripts.summarize_warmup import summarize
from src.data.preparation import write_manifest

def test_source_budget_projection_is_explicitly_unfrozen(tmp_path):
    for split,count in [('train',12),('val',4)]:
        rows=[dict(image=str(tmp_path/f'{split}_rgb_{i}.png'),mask=str(tmp_path/f'{split}_mask_{i}.png'),domain='synthetic',split=split) for i in range(count)]
        write_manifest(rows,tmp_path/f'{split}.csv')
    cfg={'data':{'train_manifest':str(tmp_path/'train.csv'),'val_manifest':str(tmp_path/'val.csv')},'train':{'epochs':3,'batch_size':2}}
    config=tmp_path/'full.yaml';config.write_text(yaml.safe_dump(cfg))
    history=[{'epoch':i,'train_loss':3-i*.1,'val':{'miou':.1}} for i in [1,2,3]]
    resources=[dict(epoch=i,device='cpu',gpu_name=None,train_seconds=3.,validation_seconds=1.,batches=3,val_images=2,max_memory_allocated_bytes=None) for i in [1,2,3]]
    (tmp_path/'history.jsonl').write_text('\n'.join(json.dumps(r) for r in history))
    (tmp_path/'resources.jsonl').write_text('\n'.join(json.dumps(r) for r in resources))
    result=summarize(tmp_path,config,tmp_path/'report')
    assert result['status']=='CPU_ONLY_NOT_A_GPU_BUDGET'
    assert result['projected_epoch_seconds'] is None
    assert result['budget_frozen'] is False
    assert (tmp_path/'report/loss.svg').is_file()
    assert (tmp_path/'report/loss.png').is_file()
    for row in resources:
        row.update(device='cuda',gpu_name='synthetic GPU log',max_memory_allocated_bytes=1024)
    (tmp_path/'resources.jsonl').write_text('\n'.join(json.dumps(r) for r in resources))
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    measured=summarize(tmp_path,config,tmp_path/'gpu-report-fixture')
    assert measured['projected_epoch_seconds']==8
    changed={**cfg,'model':{'checkpoint':'different'}}
    (tmp_path/'config.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='Model architecture'):
        summarize(tmp_path,config,tmp_path/'mismatch')
    rows=[dict(image=str(tmp_path/'target.png'),mask=str(tmp_path/'target_mask.png'),domain='fog',split='val')]
    write_manifest(rows,tmp_path/'val.csv')
    with pytest.raises(ValueError,match='source-only'):
        summarize(tmp_path,config,tmp_path/'rejected-report')
