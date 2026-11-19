"""Image and model boundary checks without requiring trained weights or GPU."""
import base64
from io import BytesIO
import json

import pytest
from PIL import Image

from web.worker import Studio, decode_image


def encoded(image, format='PNG'):
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.parametrize('value', [None, '', 'not-base64', base64.b64encode(b'fake PNG').decode()])
def test_rejects_invalid_uploads(value):
    with pytest.raises(ValueError):
        decode_image(value)


def test_image_limits_formats_and_exif_orientation(monkeypatch):
    import web.worker as worker
    monkeypatch.setattr(worker, 'MAX_PIXELS', 100)
    with pytest.raises(ValueError, match='triệu pixel'):
        decode_image(encoded(Image.new('RGB', (11, 10))))
    with pytest.raises(ValueError, match='PNG'):
        decode_image(encoded(Image.new('RGB', (2, 2)), 'TIFF'))
    rgba = decode_image(encoded(Image.new('RGBA', (3, 4), (20, 30, 40, 100))))
    assert rgba.mode == 'RGB' and rgba.size == (3, 4)
    image = Image.new('RGB', (3, 4)); exif = image.getexif(); exif[274] = 6
    buffer = BytesIO(); image.save(buffer, format='JPEG', exif=exif)
    assert decode_image(base64.b64encode(buffer.getvalue()).decode()).size == (4, 3)


def test_metadata_missing_checkpoints_and_no_path_access(tmp_path, monkeypatch):
    import web.worker as worker
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    for name in worker.MODEL_NAMES:
        monkeypatch.delenv(name.upper() + '_CHECKPOINT', raising=False)
        monkeypatch.delenv(name.upper() + '_SOURCE_CHECKPOINT', raising=False)
    studio = Studio()
    assert not any(m['stages']['adapt'] for m in studio.metadata()['models'])
    with pytest.raises(ValueError, match='hỗ trợ'):
        studio.predict('../arbitrary', 'adapt', '')
    with pytest.raises(ValueError, match='Checkpoint'):
        studio.predict('segformer_b2', 'adapt', '')
    with pytest.raises(ValueError, match='ảnh mẫu'):
        studio.sample('../../file')
    # A local checkpoint can be supplied, but raw filesystem paths are not sent to clients.
    ckpt = tmp_path / 'best'; ckpt.mkdir()
    (ckpt / 'config.json').write_text('{}'); (ckpt / 'model.safetensors').write_bytes(b'fixture')
    monkeypatch.setenv('SEGFORMER_B2_CHECKPOINT', str(ckpt))
    metadata = Studio().metadata()
    assert metadata['models'][0]['stages']['adapt'] is True
    assert str(tmp_path) not in json.dumps(metadata)


def test_retired_b0_is_not_exposed_by_studio():
    studio = Studio()
    assert "segformer_b0" not in [m["id"] for m in studio.metadata()["models"]]
    with pytest.raises(ValueError, match="hỗ trợ"):
        studio.predict("segformer_b0", "adapt", "")
