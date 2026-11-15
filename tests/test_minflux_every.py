"""Every iteration of a MINFLUX measurement, and one sequence per thread
(`io.minflux.table` with ``every`` and ``threads``).

Real data is used where it is set: ``SMAPPY_TEST_MSR`` a folder of `.msr`
files, ``SMAPPY_TEST_MINFLUX`` a folder of Imspector's exports (`.npy`,
`.json`); without them those tests skip.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from smappy.group import USE
from smappy.io import minflux
from smappy.locs import KEPT_BOUNDS

FLAT = [("vld", "?"), ("fnl", "?"), ("bot", "?"), ("eot", "?"), ("sta", "u1"),
        ("tim", "f8"), ("tid", "u4"), ("thi", "u1"), ("itr", "i4"), ("loc", "f8", (3,)),
        ("eco", "u4"), ("efo", "f4"), ("fbg", "f4"), ("cfr", "f2")]
NAN3 = (np.nan,) * 3
XY = (1e-6, 2e-6, 0.0)


def flat_records():
    """Two traces as m2410 writes them: trace 7 scouts (itr 0), catches
    (1, 2, cfr measured in 2) and localizes three times (3); trace 8 fails
    in scouting, without a position.  Times rise row by row."""
    rows = [  # vld fnl bot eot sta tim tid thi itr loc eco efo fbg cfr
        (True, False, True, False, 2, 0.0, 7, 0, 0, XY, 30, 3e4, 0, 0),
        (True, False, False, False, 2, 0.1, 7, 0, 1, XY, 60, 6e4, 0, 0),
        (True, False, False, False, 2, 0.2, 7, 0, 2, XY, 90, 9e4, 0, 0.3),
        (True, True, False, False, 1, 0.3, 7, 0, 3, XY, 100, 1e5, 0, 0),
        (False, False, True, True, 8, 0.35, 8, 0, 0, NAN3, 2, 2e3, 0, 0),
        (True, True, False, False, 1, 0.4, 7, 0, 3, XY, 200, 1e5, 0, 0),
        (True, True, False, False, 1, 0.5, 7, 0, 3, XY, 400, 1e5, 0, 0),
        (False, False, False, True, 8, 0.6, 7, 0, 3, NAN3, 3, 3e3, 0, 0),
    ]
    return np.array(rows, dtype=FLAT)


def seq(*geo, nm=None):
    return {"iterations": [{"Mode.pattern": "hexagon", "patGeoFactor": g} for g in geo],
            **({"excitation_nm": nm} if nm else {})}


SEQ = seq(0.79, 0.84, 0.42, 0.28)


def test_every_iteration_keeps_the_localizations_as_they_were_and_adds_the_rest():
    rec = flat_records()
    few, _ = minflux.table(rec, sequence=SEQ)
    every, counts = minflux.table(rec, sequence=SEQ, every=True)
    assert len(few) == 3 and len(every) == 8 and counts["localizations"] == 3
    final = np.asarray(every["final"]).astype(bool)
    for column in ("x_nm", "photons", "xy_err_nm", "cfr", "trace_index", "time_s"):
        assert np.allclose(np.asarray(few[column], float),
                           np.asarray(every[column], float)[final], equal_nan=True), column
    # the failed rows are there, without a position, and say why
    assert np.isnan(np.asarray(every["x_nm"])[~final]).sum() == 2
    assert sorted(np.asarray(every["sta"])[~final].tolist()) == [2, 2, 2, 8, 8]
    # the layers start on the localizations, and merging a trace merges them only
    assert every.metadata[KEPT_BOUNDS] == {"final": [1, 1]}
    assert np.array_equal(np.asarray(every[USE]).astype(bool), final)


def test_a_row_is_numbered_by_the_localization_it_leads_to():
    every, _ = minflux.table(flat_records(), sequence=SEQ, every=True)
    tid, index = np.asarray(every["tid"]), np.asarray(every["trace_index"])
    # trace 7: three catching rows lead to localization 1, then 1, 2, 3, and
    # the aborted row after the last one leads to a 4th that never came
    assert index[tid == 7].tolist() == [1, 1, 1, 1, 2, 3, 4]
    assert index[tid == 8].tolist() == [1]


def test_a_catching_rows_precision_is_its_own_iterations():
    every, _ = minflux.table(flat_records(), sequence=SEQ, every=True)
    it, err = np.asarray(every["iteration"]), np.asarray(every["xy_err_nm"])
    tid = np.asarray(every["tid"])
    row = (tid == 7) & (it == 1)
    assert err[row][0] == pytest.approx(minflux.crlb_xy(0.84 * 360, 60), rel=1e-5)


def test_each_thread_takes_its_own_pattern_size():
    rec = flat_records()
    rec["thi"][[5, 6]] = 1                     # two of trace 7's localizations
    a = seq(0.79, 0.84, 0.42, 0.28)
    b = seq(0.79, 0.84, 0.42, 0.28, nm=561.0)
    locs, _ = minflux.table(rec, sequence=a, threads=[a, b])
    thi = np.asarray(locs["thi"])
    L = 0.28 * 360
    assert np.asarray(locs["xy_err_nm"])[thi == 0][0] == pytest.approx(
        minflux.crlb_xy(L, 100), rel=1e-5)
    assert np.asarray(locs["xy_err_nm"])[thi == 1] == pytest.approx(
        minflux.crlb_xy(L * 561 / 642, np.array([200, 400])), rel=1e-5)


def test_a_threads_sequence_and_laser_are_read_from_the_metadata():
    def thread(geo, laser_nm):
        return {"lasers": [{"id": f"MINFLUX{laser_nm}", "purpose": 1,
                            "wavelength": laser_nm * 1e-9}],
                "sequences": [{"Itr": [{"Mode": {"pattern": "hexagon"}, "patGeoFactor": geo}]}]}
    attrs = {"measurement": {"threads": [thread(0.28, 640), thread(0.42, 561)]}}
    seqs = minflux.sequences(attrs)
    assert len(seqs) == 2 and seqs[1]["excitation_nm"] == pytest.approx(561)
    assert minflux.pattern_sizes(seqs[1])["xy"][1] == pytest.approx(0.42 * 360 * 561 / 642)
    assert len(minflux.sequences({})) == 1


def test_an_m2205_record_is_unnested_into_its_iterations():
    n = 3
    sub = np.zeros((n, 2), dtype=[("loc", "f8", (3,)), ("eco", "f8"), ("efo", "f8"),
                                  ("fbg", "f8"), ("cfr", "f8")])
    sub["loc"][:] = 1e-6
    sub["eco"][:, 0], sub["eco"][:, 1] = 50, [100, 200, 400]
    sub["cfr"][:, 0] = [0.1, 0.2, 0.3]
    rec = np.zeros(n, dtype=[("itr", sub.dtype, (2,)), ("tim", "f8"), ("tid", "i8"),
                             ("vld", "?")])
    rec["itr"], rec["tim"], rec["tid"], rec["vld"] = sub, np.arange(n), 1, [True, True, False]
    few, _ = minflux.table(rec, sequence=seq(0.42, 0.28))
    every, _ = minflux.table(rec, sequence=seq(0.42, 0.28), every=True)
    assert len(few) == 2 and len(every) == 6
    final = np.asarray(every["final"]).astype(bool)
    assert final.sum() == 2 and np.asarray(every["iteration"]).tolist() == [0, 1] * 3
    assert np.allclose(np.asarray(every["photons"])[final], [100, 200])
    assert np.allclose(np.asarray(every["cfr"])[final], [0.1, 0.2])   # from iteration 0


EXPORTS = os.environ.get("SMAPPY_TEST_MINFLUX")
MSRS = os.environ.get("SMAPPY_TEST_MSR")


@pytest.mark.skipif(not EXPORTS, reason="SMAPPY_TEST_MINFLUX=dir of MINFLUX exports")
def test_real_exports_read_alike_and_every_iteration_keeps_their_localizations():
    paths = sorted(p for p in Path(EXPORTS).iterdir() if p.suffix in (".npy", ".json"))
    assert paths
    tables = {}
    read = 0
    for path in paths:
        try:
            rec = minflux.records(path)
        except ValueError:                      # a sequence file beside the exports
            continue
        read += 1
        few, _ = minflux.table(rec)
        every, _ = minflux.table(rec, every=True)
        final = np.asarray(every["final"]).astype(bool)
        assert final.sum() == len(few) and len(every) >= len(few), path.name
        assert np.allclose(np.asarray(few["x_nm"], float),
                           np.asarray(every["x_nm"], float)[final], equal_nan=True)
        tables.setdefault(path.stem, []).append(few)
    assert read
    for stem, same in tables.items():         # one measurement in several formats
        for other in same[1:]:
            assert np.allclose(np.asarray(same[0]["x_nm"]), np.asarray(other["x_nm"]),
                               atol=1e-3), stem


@pytest.mark.skipif(not MSRS, reason="SMAPPY_TEST_MSR=dir of real .msr files")
def test_real_measurements_read_with_every_iteration():
    from smappy.io import msr
    paths = sorted(Path(MSRS).glob("*.msr"))
    assert paths
    for path in paths:
        m = msr.contents(path).measurements[0]
        few, _ = msr.read_measurement(path, m)
        every, meta = msr.read_measurement(path, m, every=True)
        final = np.asarray(every["final"]).astype(bool)
        assert final.sum() == len(few) and meta["every_iteration"]
        assert np.array_equal(np.asarray(few["tid"]), np.asarray(every["tid"])[final])
        assert np.allclose(np.asarray(few["xy_err_nm"]),
                           np.asarray(every["xy_err_nm"])[final], equal_nan=True)
