import pytest
import torch

from scripts.profile_dg import estimated_hours,profile


def test_forecast_weights_paired_and_single_cost_then_adds_validation():
    expected=20*(100*(.5*1+.5*2)+30)/3600
    assert estimated_hours(1,2,.5,100,30)==pytest.approx(expected)
    assert estimated_hours(1,2,0,100,30)==pytest.approx(20*130/3600)
    with pytest.raises(ValueError):estimated_hours(1,float('nan'),.5,100,30)


def test_cpu_profile_cannot_masquerade_as_gpu_eta(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(ValueError,match='CPU smoke timings are not GPU ETAs'):
        profile(tmp_path/'not_read',tmp_path/'out.json')
    assert not (tmp_path/'out.json').exists()
