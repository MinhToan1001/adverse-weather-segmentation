import numpy as np
import pytest
from PIL import Image
from src.data.dataset import ManifestSegmentationDataset
from src.data.preparation import write_manifest

@pytest.mark.parametrize('prep', [{'std':[0,1,1]}, {'mean':[1,2]}, {'mask_interpolation':'bilinear'}, {'keep_aspect_ratio':False}])
def test_preprocessing_contract(tmp_path, prep):
    csv = tmp_path/'sample.csv'
    write_manifest([dict(image=str(tmp_path/'rgb.png'), mask=str(tmp_path/'mask.png'), domain='synthetic',split='val')], csv)
    with pytest.raises(ValueError):
        ManifestSegmentationDataset(csv, preprocessing=prep)

def test_configured_normalization(tmp_path):
    image, mask = tmp_path/'rgb.png', tmp_path/'mask.png'
    Image.fromarray(np.full((4,4,3), 255, dtype=np.uint8)).save(image)
    Image.fromarray(np.zeros((4,4), dtype=np.uint8)).save(mask)
    csv=tmp_path/'sample.csv'
    write_manifest([dict(image=str(image),mask=str(mask),domain='synthetic',split='val')], csv)
    sample=ManifestSegmentationDataset(csv,(4,4),preprocessing={'mean':[0,0,0],'std':[1,1,1]})[0]
    assert sample['pixel_values'].eq(1).all()
