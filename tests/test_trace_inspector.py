"""*Trace inspector*: one trace's numbers, per region and on the view."""
import numpy as np
import pytest

from smappy import plugins
from smappy.locs import Localizations
from smappy.plugins.trace_inspector import trace_values
from smappy.session import Session

PLUGIN = "Analysis/Tracking/Trace inspector"


def table():
    rng = np.random.default_rng(0)
    # trace 1: 500 points, 3 nm noise and a slow drift of 10 nm; trace 2: 100 points
    t1 = np.arange(500) * 1e-3
    x1 = rng.normal(0, 3, 500) + 10 * t1 / t1[-1]
    y1 = rng.normal(0, 3, 500)
    t2 = np.arange(100) * 1e-3
    x2, y2 = 5000 + rng.normal(0, 2, 100), rng.normal(0, 2, 100)
    return Localizations({"x_nm": np.r_[x1, x2].astype(np.float32),
                          "y_nm": np.r_[y1, y2].astype(np.float32),
                          "time_s": np.r_[t1, t2], "frame": np.arange(600),
                          "tid": np.r_[np.full(500, 1), np.full(100, 2)],
                          "cfr": np.full(600, 0.3, np.float32)}, {})


def test_the_consecutive_precision_leaves_the_drift_out():
    v = trace_values(np.arange(500) * 1e-3,
                     {"x": np.random.default_rng(1).normal(0, 3, 500) + np.linspace(0, 30, 500)},
                     {})
    assert v["x_consecutive_nm"] == pytest.approx(3.0, rel=0.15)
    assert v["x_sd_nm"] > 2 * v["x_consecutive_nm"]


def test_on_the_view_it_takes_the_longest_trace_or_the_one_named():
    session = Session(table())
    session.show_grouped(0, False)
    cls = plugins.get(PLUGIN)
    r = session.run(cls(), cls.Settings())
    assert r.values["trace"] == 1 and r.values["n_locs"] == 500
    r = session.run(cls(), cls.Settings(trace=2))
    assert r.values["n_locs"] == 100 and r.values["cfr_median"] == pytest.approx(0.3)


def test_a_gallery_of_the_longest_traces():
    session = Session(table())
    session.show_grouped(0, False)
    cls = plugins.get(PLUGIN)
    r = session.run(cls(), cls.Settings(gallery=2))
    assert "2 longest" in r.text


def test_it_runs_per_region():
    assert plugins.get(PLUGIN).per_region
