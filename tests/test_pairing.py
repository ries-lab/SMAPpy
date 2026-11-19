"""`smappy.pairing` -- the bead measurement's pairing by time -- and *MINFLUX
dual colour* on two threads: a known pairing and offset come back."""
import numpy as np
import pytest

from smappy import pairing as P
from smappy import plugins
from smappy.locs import Localizations
from smappy.session import Session

OFFSET = (8.0, -5.0)


def two_threads(n_pairs=30, n_alone=5, seed=0):
    """Molecules seen in both threads, interleaved in time (thread 1 shifted
    by `OFFSET`), and a few seen in thread 0 alone; trace ids per thread."""
    rng = np.random.default_rng(seed)
    rows, t = [], 0.0
    for k in range(n_pairs):
        pos = rng.uniform(0, 5000, 2)
        for j in range(40):
            th = j % 2
            p = pos + (np.array(OFFSET) if th else 0) + rng.normal(0, 3, 2)
            rows.append((2 * k + 1 + th, th, t + j * 1e-3, p[0], p[1]))
        t += 0.1
    for k in range(n_alone):
        for j in range(20):
            rows.append((1000 + k, 0, t + j * 1e-3, 100.0, 100.0))
        t += 0.1
    a = np.array(rows)
    return Localizations({"tid": a[:, 0].astype(np.int64), "thi": a[:, 1].astype(np.int64),
                          "time_s": a[:, 2], "x_nm": a[:, 3].astype(np.float32),
                          "y_nm": a[:, 4].astype(np.float32),
                          "frame": np.arange(len(a))}, {})


def test_traces_measured_together_are_one_pair_and_a_lone_trace_its_own():
    locs = two_threads()
    pid = P.pair_traces(locs["tid"], locs["thi"], locs["time_s"])
    tid, thi = np.asarray(locs["tid"]), np.asarray(locs["thi"])
    assert len(np.unique(pid)) == 35
    # each pair holds one trace of each thread
    for p in np.unique(pid)[:30]:
        assert set(thi[pid == p]) == {0, 1} or len(set(tid[pid == p])) == 1


def test_the_offset_between_the_colours_comes_back():
    locs = two_threads()
    pid = P.pair_traces(locs["tid"], locs["thi"], locs["time_s"])
    off = P.pair_offsets(pid, locs["thi"], locs["time_s"], locs["x_nm"], locs["y_nm"])
    assert len(off.pair_id) == 30
    assert off.pooled() == pytest.approx(OFFSET, abs=0.5)


def test_a_trace_overlapping_two_takes_the_longer_overlap():
    # thread 0 trace 1 from 0 to 10 s; thread 1 trace 2 overlaps 1 s, trace 3 overlaps 5 s
    tid = np.array([1, 1, 2, 2, 3, 3])
    thi = np.array([0, 0, 1, 1, 1, 1])
    t = np.array([0.0, 10.0, -1.0, 1.0, 4.0, 9.0])
    pid = P.pair_traces(tid, thi, t)
    assert pid[0] == pid[4] and pid[2] != pid[0]


def test_the_plugin_writes_pair_id_and_keeps_the_offset_for_transform():
    session = Session(two_threads())
    session.show_grouped(0, False)
    cls = plugins.get("Analysis/MINFLUX/MINFLUX dual colour")
    result = session.run(cls(), cls.Settings())
    assert "pair_id" in session.locs
    t = session.transformations["thi 1 onto thi 0"]
    x, y, _ = t.apply(np.array([OFFSET[0]]), np.array([OFFSET[1]]))
    assert abs(x[0]) < 0.5 and abs(y[0]) < 0.5
    assert "30 in both colours" in result.text


def test_on_one_colour_the_plugin_says_so_and_changes_nothing():
    locs = two_threads()
    locs.columns["thi"] = np.zeros(len(locs), np.int64)
    cls = plugins.get("Analysis/MINFLUX/MINFLUX dual colour")
    session = Session(locs)
    result = session.run(cls(), cls.Settings())
    assert "one colour" in result.text and "pair_id" not in session.locs
