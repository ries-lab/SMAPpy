"""netfit's training loop's parts: a schedule of samples, the stop rule's patience in samples, the
batch and its rate, batches drawn ahead in a thread the same as drawn in the loop."""
import pytest

torch = pytest.importorskip("torch")

from test_netfit_simulate import _calibration

from smappy.netfit import train
from smappy.netfit.render import SplineRenderer
from smappy.netfit.settings import CameraModel, Parts, SimulationSettings
from smappy.netfit.simulate import draw_batch


def test_the_schedule_of_samples_is_the_step_schedule_it_replaced_at_the_reference_batch():
    sch = dict(Parts().schedule.settings)
    w = torch.nn.Parameter(torch.zeros(1))
    old_opt, new_opt = torch.optim.SGD([w], lr=1.0), torch.optim.SGD([w], lr=1.0)
    old = torch.optim.lr_scheduler.StepLR(old_opt, step_size=sch["step_epochs"], gamma=sch["gamma"])
    new = train.schedule(new_opt, sch, 64)
    steps = sch["samples_per_epoch"] // 64
    for _ in range(25):                      # epochs
        for _ in range(steps):
            assert new.get_last_lr()[0] == pytest.approx(old.get_last_lr()[0])
            new.step()
        old.step()


def test_a_bigger_batch_decays_after_as_many_samples_and_takes_a_scaled_rate():
    sch = {**Parts().schedule.settings, "batch": 256}
    batch, lr = train.batch_and_rate(sch, "cpu")
    assert batch == 256 and lr == pytest.approx(sch["lr"] * 2)                   # sqrt(256 / 64)
    assert train.batch_and_rate({**sch, "lr_batch_scaling": "linear"}, "cpu")[1] == pytest.approx(sch["lr"] * 4)
    assert train.batch_and_rate({**sch, "batch": "auto"}, "cpu")[0] == 64       # no GPU: the reference
    s = train.schedule(torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0), sch, 256)
    decay = sch["step_epochs"] * sch["samples_per_epoch"]
    for _ in range(decay // 256 - 1):
        s.step()
    assert s.get_last_lr()[0] == 1.0
    s.step()
    assert s.get_last_lr()[0] == pytest.approx(sch["gamma"])


def test_the_stop_rules_patience_is_the_same_number_of_samples_whatever_the_epoch():
    assert train.stop_rule(32768)["window"] == train.STOP_RULE["window"]
    r = train.stop_rule(10000)
    assert r["window"] * 10000 == pytest.approx(train.STOP_RULE["window"] * 32768, rel=0.01)
    assert r["min_epochs"] * 10000 == pytest.approx(train.STOP_RULE["min_epochs"] * 32768, rel=0.01)


def test_batches_drawn_ahead_in_a_thread_are_the_batches_the_loop_would_draw():
    renderer = SplineRenderer(_calibration())
    sim = SimulationSettings(frame_px=16, density_per_40px=10.0)
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    feed = train._Prefetch(lambda: draw_batch(renderer, sim, CameraModel(em_gain=0.0), 4, g1), "cpu")
    try:
        for _ in range(4):
            a, b = feed.get(), draw_batch(renderer, sim, CameraModel(em_gain=0.0), 4, g2)
            assert torch.equal(a.frames, b.frames) and torch.equal(a.xyz, b.xyz)
    finally:
        feed.close()
