"""*Trace statistics* (`smappy.traces`, `plugins/trace_statistics.py`):
tracks and still molecules with a known truth come back."""
import numpy as np
import pytest

from smappy import plugins
from smappy import traces as T
from smappy.locs import Localizations
from smappy.session import Session

PLUGIN = "Analysis/Tracking/Trace statistics"


def tracks(seed=0):
    """Track 1 moves 0.5 nm per ms at 30 degrees for 200 ms, precision 3 nm;
    track 2 is still, its first 15 localizations creeping in from 40 nm,
    precision 2 nm; track 3 moves along a quarter circle (a trend across)."""
    rng = np.random.default_rng(seed)
    t1 = np.arange(200) * 1e-3
    a = np.radians(30)
    s = 500 * t1
    x1, y1 = s * np.cos(a) + rng.normal(0, 3, 200), s * np.sin(a) + rng.normal(0, 3, 200)
    t2 = np.arange(100) * 1e-3
    creep = np.r_[np.linspace(40, 0, 15), np.zeros(85)]
    x2, y2 = 3000 + creep + rng.normal(0, 2, 100), 3000 + rng.normal(0, 2, 100)
    phi = np.linspace(0, np.pi / 2, 200)
    x3, y3 = 6000 + 300 * np.cos(phi) + rng.normal(0, 3, 200), 300 * np.sin(phi) + rng.normal(0, 3, 200)
    ids = np.r_[np.full(200, 1), np.full(100, 2), np.full(200, 3)]
    t = np.r_[t1, t2, t1]
    return ids, np.r_[x1, x2, x3], np.r_[y1, y2, y3], t


def sorted_runs(ids, x, y, t):
    r = T.runs(ids, t)
    return r, x[r.order], y[r.order], t[r.order]


def test_a_straight_track_gives_its_speed_direction_and_precision():
    r, x, y, t = sorted_runs(*tracks())
    st = T.statistics(r, x, y, t)
    assert st["velocity"][0] == pytest.approx(500, rel=0.1)        # nm/s
    assert st["angle_deg"][0] == pytest.approx(30, abs=2)
    assert st["precision_short_nm"][0] == pytest.approx(3, rel=0.15)
    assert st["trend_nm"][0] < 1.0
    # a curved track holds more across its axis than the scatter: a trend,
    # the arc's own spread across its chord
    phi = np.linspace(0, np.pi / 2, 200)
    arc = 300 * np.cos(phi - np.pi / 4)                    # distance from the centre, along the bisector
    assert st["trend_nm"][2] == pytest.approx(np.std(arc, ddof=1), rel=0.1)


def test_the_rotated_track_lies_along_x_and_runs_forward_in_time():
    r, x, y, t = sorted_runs(*tracks())
    st = T.statistics(r, x, y, t)
    xr, yr = T.rotate(r, x, y, st)
    one = r.code == 0
    assert np.corrcoef(t[one], xr[one])[0, 1] > 0.99
    assert np.std(yr[one]) == pytest.approx(3, rel=0.2)


def test_a_still_molecules_creeping_start_is_its_tail():
    r, x, y, t = sorted_runs(*tracks())
    tail = T.tails(r, x, y, t, "still", reference_from=20, reference_window=30, k=3.0)
    assert 8 <= tail.n[1] <= 15 and tail.reach_nm[1] == pytest.approx(40, abs=6)
    without = T.statistics(r, x, y, t, ~tail.in_tail)
    with_tail = T.statistics(r, x, y, t)
    assert without["std_long_nm"][1] < with_tail["std_long_nm"][1]


def test_a_cfr_violation_cuts_the_rest_of_the_trace():
    ids = np.repeat([1, 2], 5)
    r = T.runs(ids, np.arange(10.0))
    cut = T.cfr_cut(r, np.array([0.1, 0.2, 0.9, 0.1, 0.1, 0.1, np.nan, 0.2, 0.1, 0.3]), 0.8)
    assert cut.tolist() == [False, False, True, True, True] + [False] * 5


def test_the_plugin_keeps_a_region_table_and_writes_the_rotated_track_and_the_tail():
    ids, x, y, t = tracks()
    locs = Localizations({"x_nm": x.astype(np.float32), "y_nm": y.astype(np.float32),
                          "time_s": t, "frame": np.arange(len(x)), "tid": ids}, {})
    session = Session(locs)
    session.show_grouped(0, False)
    cls = plugins.get(PLUGIN)
    settings = cls.Settings()
    settings.tail.rule = "still"
    settings.write_rotated = settings.write_tail = True
    assert cls().changes_locs_for(settings)
    result = session.run(cls(), settings)
    table = session.id_regions["tid"]["Trace statistics"]
    assert set(table.ids) == {"1", "2", "3"}
    out = session.locs
    assert "x_rot_nm" in out and int(np.asarray(out["in_tail"]).sum()) == int(result.data["tails"].n.sum())
    assert "velocity" in result.text


def test_only_the_region_table_when_nothing_is_written():
    cls = plugins.get(PLUGIN)
    assert not cls().changes_locs_for(cls.Settings())
