"""*Quality filter* (`smappy.quality`, `plugins/quality_filter.py`): cfr
and efo thresholds found in data with a known single- and two-molecule
mixture, set as bounds, and not set where cfr is measured once a trace."""
import numpy as np
import pytest

from smappy import plugins
from smappy import quality as Q
from smappy.locs import Localizations
from smappy.session import Session

PLUGIN = "Analysis/MINFLUX/Quality filter"


def mixture(n=10000, cfr_measured=1.0, seed=0):
    rng = np.random.default_rng(seed)
    one = int(0.8 * n)
    cfr = np.r_[rng.normal(0.3, 0.08, int(0.9 * n)), rng.uniform(0.6, 1.2, n - int(0.9 * n))]
    cfr[rng.random(n) > cfr_measured] = np.nan
    efo = np.r_[rng.normal(100e3, 15e3, one), rng.normal(200e3, 25e3, n - one)]
    return Localizations({"x_nm": rng.uniform(0, 1e4, n).astype(np.float32),
                          "y_nm": rng.uniform(0, 1e4, n).astype(np.float32),
                          "frame": np.arange(n), "cfr": cfr.astype(np.float32),
                          "efo": efo.astype(np.float32)}, {})


def test_the_cfr_bound_is_the_median_plus_k_robust_sigmas():
    v = np.random.default_rng(1).normal(0.3, 0.1, 100000)
    assert Q.robust_upper(v, 2.0) == pytest.approx(0.5, abs=0.005)


def test_the_efo_bound_is_the_valley_before_the_two_molecule_peak():
    v = Q.efo_valley(np.asarray(mixture()["efo"]))
    assert v.valley and v.peak == pytest.approx(100e3, rel=0.05)
    assert 130e3 < v.bound < 165e3
    alone = Q.efo_valley(np.random.default_rng(2).normal(100e3, 15e3, 8000))
    assert not alone.valley and alone.bound > 125e3


def test_the_bounds_are_set_on_the_layers_and_a_second_run_finds_the_same():
    session = Session(mixture())
    session.show_grouped(0, False)
    cls = plugins.get(PLUGIN)
    first = session.run(cls(), cls.Settings()).data["bounds"]
    bounds = session.layer_configs()[0]["bounds"]
    assert bounds["cfr"][1] == pytest.approx(first["cfr"][1])
    again = session.run(cls(), cls.Settings()).data["bounds"]
    assert again["efo"][1] == pytest.approx(first["efo"][1])


def test_a_cfr_measured_once_a_trace_cuts_whole_traces():
    rng = np.random.default_rng(3)
    n_traces, per = 200, 20
    tid = np.repeat(np.arange(n_traces), per)
    good = rng.uniform(0.25, 0.35, n_traces)            # all well inside the bound
    good[:20] = rng.uniform(0.8, 1.0, 20)             # 20 traces with two molecules on
    cfr = np.full(len(tid), np.nan, np.float32)
    cfr[::per] = good                                  # measured once, before the rest
    locs = Localizations({"x_nm": rng.uniform(0, 1e4, len(tid)).astype(np.float32),
                          "y_nm": rng.uniform(0, 1e4, len(tid)).astype(np.float32),
                          "frame": np.arange(len(tid)), "tid": tid, "cfr": cfr}, {})
    session = Session(locs)
    session.show_grouped(0, False)
    cls = plugins.get(PLUGIN)
    result = session.run(cls(), cls.Settings(efo=False))
    assert "trace_cfr" in result.data["bounds"] and "cfr" not in result.data["bounds"]
    kept = np.asarray(session.context().selection.mask)
    ids = np.asarray(session.locs["tid"])
    # whole traces kept or dropped: the 180 good ones, none of the 20 bad
    assert set(np.unique(ids[kept])) == set(range(20, n_traces))
    assert kept.sum() == 180 * per
