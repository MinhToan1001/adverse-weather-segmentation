from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from scripts.audit_labels import audit
from src.data.preparation import write_manifest

def test_source_audit_counts_and_rejects_invalid_mask(tmp_path):
    image, mask=tmp_path/'image.png',tmp_path/'mask.png'
    Image.new('RGB',(4,3)).save(image)
    Image.fromarray(np.array([[0,1,255,0]]*3,dtype=np.uint8)).save(mask)
    path=tmp_path/'manifest.csv'
    write_manifest([dict(image=str(image),mask=str(mask),domain='cityscapes',split='train')],path)
    result=audit([path],True,True)
    assert result['status']=='PASS'
    assert result['splits']['cityscapes/train']['class_pixels']['255']==3
    Image.new('RGB',(4,3)).save(mask)
    with pytest.raises(ValueError,match='integer mask'):
        audit([path],True,True)

def test_target_file_check_never_decodes(tmp_path,monkeypatch):
    image,mask=tmp_path/'image.png',tmp_path/'mask.png'
    image.touch();mask.touch()
    path=tmp_path/'manifest.csv'
    write_manifest([dict(image=str(image),mask=str(mask),domain='fog',split='val')],path)
    def forbidden(*args,**kwargs):
        pytest.fail('Target pixels must stay unopened')
    monkeypatch.setattr(Image,'open',forbidden)
    assert audit([path],True,False)['status']=='PASS'
    with pytest.raises(ValueError,match='unopened'):
        audit([path],True,True)
