import pytest

from src.data.replay import BalancedReplayDataset, replay_selection_rank, validate_replay_loader_options, validate_source_monitor


class Samples:
    def __init__(self, count, domain):
        self.count, self.domain = count, domain
    def __len__(self):
        return self.count
    def set_epoch(self, epoch):
        self.epoch = epoch
    def __getitem__(self, index):
        return self.domain, index


def test_replay_balances_every_update_and_final_partial_group():
    dataset = BalancedReplayDataset(Samples(6, "target"), Samples(15, "source"), 42)
    values = [dataset[i] for i in range(len(dataset))]
    assert len(values) == 12
    assert sorted(i for d, i in values if d == "target") == list(range(6))
    assert len({i for d, i in values if d == "source"}) == 6
    for start in (0, 8):
        block = values[start:start + 8]
        assert sum(d == "target" for d, _ in block) == sum(d == "source" for d, _ in block)
    dataset.set_epoch(1)
    later = [dataset[i] for i in range(len(dataset))]
    assert later != values
    dataset.set_epoch(0)
    assert [dataset[i] for i in range(len(dataset))] == values


def test_checkpoint_rank_enforces_source_constraint_and_explicit_fallback():
    assert replay_selection_rank(.4, -.5) > replay_selection_rank(.6, -2)
    assert replay_selection_rank(.5, -.9) > replay_selection_rank(.4, -.1)
    assert replay_selection_rank(.4, -2) > replay_selection_rank(.6, -3)
    with pytest.raises(ValueError, match="finite"):
        replay_selection_rank(float("nan"), 0)


def test_source_monitor_rejects_official_val_and_same_scene_other_frame():
    a = dict(image="leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png", domain="cityscapes", split="train")
    b = dict(a, image="leftImg8bit/train/aachen/aachen_000001_000019_leftImg8bit.png")
    validate_source_monitor([a], [b])
    with pytest.raises(ValueError, match="scene groups overlap"):
        validate_source_monitor([a], [dict(a, image=a["image"].replace("000019", "000020"))])
    with pytest.raises(ValueError, match="official train"):
        validate_source_monitor([a], [dict(b, split="val")])


@pytest.mark.parametrize('update',[dict(gradient_accumulation_steps=0),dict(gradient_accumulation_steps=3),dict(max_steps=3),dict(max_steps=-2),dict(batch_size=2)])
def test_invalid_options_cannot_break_domain_balance(update):
    options=dict(batch_size=1,gradient_accumulation_steps=8,max_steps=None)
    options.update(update)
    with pytest.raises(ValueError,match='positive even'):
        validate_replay_loader_options(options)
