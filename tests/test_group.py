"""Linking and combining, on tables whose groups are known by construction."""
import numpy as np
import pytest

from smappy.group import (COMBINE_MODES, GroupSettings, combine, connect, group,
                          sorted_order)
from smappy.locs import Localizations


def _runs(spec, dx_jitter=0.0, seed=0):
    """A table from (x, y, first_frame, length) runs; one emitter per run."""
    rng = np.random.default_rng(seed)
    x, y, frame, truth = [], [], [], []
    for i, (px, py, f0, n) in enumerate(spec):
        for k in range(n):
            x.append(px + rng.normal(0, dx_jitter))
            y.append(py + rng.normal(0, dx_jitter))
            frame.append(f0 + k)
            truth.append(i)
    order = rng.permutation(len(x))          # input order must not matter
    return (np.array(x)[order], np.array(y)[order],
            np.array(frame)[order], np.array(truth)[order])


def _same_partition(a, b):
    """Whether two labellings define the same grouping, ignoring the labels."""
    return len(set(zip(np.asarray(a).tolist(), np.asarray(b).tolist()))) == \
        len(np.unique(a)) == len(np.unique(b))


def test_consecutive_frames_are_linked():
    x, y, frame, truth = _runs([(100, 100, 0, 5), (300, 300, 0, 3),
                                (100, 100, 20, 4)], dx_jitter=2.0)
    ids = connect(x, y, frame, dx=20.0, dt=0)
    assert _same_partition(ids, truth)
    assert ids.min() == 1


def test_a_gap_larger_than_dt_starts_a_new_group():
    # one emitter, on for 3 frames, dark for 2, on for 3
    x = np.full(6, 100.0)
    y = np.full(6, 100.0)
    frame = np.array([0, 1, 2, 5, 6, 7])
    assert len(np.unique(connect(x, y, frame, dx=20.0, dt=1))) == 2
    assert len(np.unique(connect(x, y, frame, dx=20.0, dt=2))) == 1


def test_the_search_region_is_a_box():
    """SMAP tests |dx| and |dy| separately, so the corner of the box links."""
    frame = np.array([0, 1])
    inside = connect(np.array([0.0, 9.0]), np.array([0.0, 9.0]), frame, 10.0, 0)
    outside = connect(np.array([0.0, 11.0]), np.array([0.0, 0.0]), frame, 10.0, 0)
    assert len(np.unique(inside)) == 1      # 12.7 away, but inside the box
    assert len(np.unique(outside)) == 2


def test_linking_does_not_cross_blocks():
    x = np.full(4, 100.0)
    y = np.full(4, 100.0)
    frame = np.array([0, 1, 2, 3])
    assert len(np.unique(connect(x, y, frame, 20.0, 1))) == 1
    blocks = np.array([0, 0, 1, 1])
    ids = connect(x, y, frame, 20.0, 1, blocks=blocks)
    assert len(np.unique(ids)) == 2 and sorted(np.unique(ids)) == [1, 2]


def test_every_localization_gets_exactly_one_group():
    """The bug fixed in the C port: SMAP left the first and last unassigned."""
    x, y, frame, _ = _runs([(i * 50.0, 0.0, i % 7, 1 + i % 4) for i in range(60)],
                           dx_jitter=1.0, seed=3)
    ids = connect(x, y, frame, dx=20.0, dt=1)
    assert (ids > 0).all()
    assert np.array_equal(np.unique(ids), np.arange(1, ids.max() + 1))


def _table(n=12):
    rng = np.random.default_rng(1)
    return Localizations({
        "x_nm": np.repeat([100.0, 500.0, 900.0], 4).astype(np.float32),
        "y_nm": np.zeros(n, np.float32),
        "loc_precision_nm": rng.uniform(5, 20, n).astype(np.float32),
        "photons": rng.uniform(1000, 5000, n).astype(np.float32),
        "photons_err": rng.uniform(50, 200, n).astype(np.float32),
        "logl_rel": rng.normal(-1, 0.3, n).astype(np.float32),
        "frame": np.arange(n, dtype=np.int64) % 4,
    }, {"units": "nm"})


def test_combine_follows_the_rules_per_column():
    locs = _table()
    gi = np.repeat([1, 2, 3], 4)
    grouped = combine(locs, gi)
    assert len(grouped) == 3
    assert np.array_equal(grouped["n_in_group"], [4, 4, 4])

    w = 1.0 / np.asarray(locs["loc_precision_nm"], np.float64) ** 2
    for g in range(3):
        s = slice(4 * g, 4 * g + 4)
        assert grouped["x_nm"][g] == pytest.approx(
            np.average(locs["x_nm"][s], weights=w[s]), rel=1e-5)
        assert grouped["photons"][g] == pytest.approx(
            np.sum(locs["photons"][s]), rel=1e-5)
        assert grouped["loc_precision_nm"][g] == pytest.approx(
            1 / np.sqrt(np.sum(1 / locs["loc_precision_nm"][s] ** 2)), rel=1e-5)
        # the error of a *sum* adds in quadrature, not by the precision rule
        assert grouped["photons_err"][g] == pytest.approx(
            np.sqrt(np.sum(locs["photons_err"][s] ** 2)), rel=1e-5)
        assert grouped["logl_rel"][g] == pytest.approx(max(locs["logl_rel"][s]))
        assert grouped["frame"][g] == min(locs["frame"][s])


def test_grouping_improves_precision_and_conserves_photons():
    locs = _table()
    gi = np.repeat([1, 2, 3], 4)
    grouped = combine(locs, gi)
    assert np.asarray(grouped["photons"]).sum() == pytest.approx(
        np.asarray(locs["photons"], np.float64).sum(), rel=1e-5)
    for g in range(3):
        best = locs["loc_precision_nm"][4 * g:4 * g + 4].min()
        assert grouped["loc_precision_nm"][g] < best


def test_the_meaningless_columns_are_dropped_not_guessed():
    """A group's raw logl is a sum over its members; no reduction is right."""
    locs = _table()
    locs.columns["logl"] = np.full(len(locs), -100.0, np.float32)
    grouped = combine(locs, np.repeat([1, 2, 3], 4))
    assert "logl" not in grouped and "logl_rel" in grouped
    assert COMBINE_MODES["logl_rel"] == "max"


def test_group_end_to_end_and_metadata():
    locs = _table()
    grouped, gi = group(locs, GroupSettings(dx=50.0, dt=1))
    assert len(grouped) == len(np.unique(gi))
    assert np.asarray(grouped["n_in_group"]).sum() == len(locs)
    assert grouped.metadata["grouped"] is True
    assert grouped.metadata["units"] == "nm"


def test_summed_errors_stay_consistent_with_shot_noise():
    """std(N) = sqrt(N) must survive grouping: std(N1+N2) = sqrt(N1+N2).

    This is what picks quadrature over SMAP's precision rule for the error of a
    summed quantity -- the identity holds exactly, not approximately.
    """
    photons = np.array([1000.0, 2000.0, 4000.0, 300.0], np.float32)
    locs = Localizations({
        "x_nm": np.zeros(4, np.float32), "y_nm": np.zeros(4, np.float32),
        "photons": photons, "photons_err": np.sqrt(photons).astype(np.float32),
        "loc_precision_nm": np.full(4, 10.0, np.float32),
        "frame": np.arange(4, dtype=np.int64),
    }, {"units": "nm"})
    grouped = combine(locs, np.ones(4, np.int64))
    assert grouped["photons"][0] == pytest.approx(photons.sum(), rel=1e-5)
    assert grouped["photons_err"][0] == pytest.approx(np.sqrt(photons.sum()),
                                                      rel=1e-5)
    # SMAP's rule would give this instead, six times too small
    smap = 1 / np.sqrt(np.sum(1 / np.sqrt(photons.astype(np.float64)) ** 2))
    assert smap < 0.2 * grouped["photons_err"][0]


def test_each_coordinate_is_weighted_by_its_own_error():
    """Under astigmatism x_err and y_err diverge, so the pooled weight is wrong."""
    locs = Localizations({
        "x_nm": np.array([-100.0, 0.0, 40.0], np.float32),
        "y_nm": np.array([40.0, 0.0, -100.0], np.float32),
        "x_err_nm": np.array([50.0, 5.0, 50.0], np.float32),
        "y_err_nm": np.array([5.0, 50.0, 5.0], np.float32),
        "loc_precision_nm": np.array([35.4, 35.4, 35.4], np.float32),
        "frame": np.arange(3, dtype=np.int64),
    }, {"units": "nm"})
    grouped = combine(locs, np.ones(3, np.int64))

    for name, err in (("x_nm", "x_err_nm"), ("y_nm", "y_err_nm")):
        values = np.asarray(locs[name], np.float64)
        own = np.average(values, weights=1 / np.asarray(locs[err], np.float64) ** 2)
        pooled = values.mean()          # the pooled precision is equal here
        assert grouped[name][0] == pytest.approx(own, abs=1e-3)
        assert abs(own - pooled) > 1.0  # the two really do differ


def test_z_is_weighted_by_its_own_error_when_the_table_has_one():
    z = np.array([-100.0, 0.0, 40.0], np.float32)   # asymmetric, so the
    # two weightings cannot coincide by symmetry
    z_err = np.array([50.0, 5.0, 50.0], np.float32)      # the middle one wins
    locs = Localizations({
        "x_nm": np.zeros(3, np.float32), "y_nm": np.zeros(3, np.float32),
        "z_nm": z, "z_err_nm": z_err,
        "loc_precision_nm": np.array([5.0, 50.0, 5.0], np.float32),
        "frame": np.arange(3, dtype=np.int64),
    }, {"units": "nm"})
    grouped = combine(locs, np.ones(3, np.int64))

    by_z_err = np.average(z.astype(np.float64), weights=1 / z_err.astype(np.float64) ** 2)
    by_lateral = np.average(z.astype(np.float64),
                            weights=1 / locs["loc_precision_nm"].astype(np.float64) ** 2)
    assert grouped["z_nm"][0] == pytest.approx(by_z_err, abs=1e-3)
    assert abs(by_z_err - by_lateral) > 1.0        # the two really do differ here
    assert grouped["z_err_nm"][0] == pytest.approx(
        1 / np.sqrt(np.sum(1 / z_err.astype(np.float64) ** 2)), rel=1e-5)


def test_linking_is_lateral_and_does_not_look_at_z():
    """Two emitters above each other link into one blink, on purpose.

    The z window that would have kept them apart is gone: emitters this close
    in xy overlap in the raw frames and were never fitted apart anyway, so the
    window mostly split real blinks whose fitted z scatters between frames.
    """
    from smappy.group import GroupSettings, group
    from smappy.locs import Localizations
    n = 40
    locs = Localizations({"x_nm": np.full(n, 100.0, np.float32), "y_nm": np.full(n, 100.0, np.float32),
                          "z_nm": np.where(np.arange(n) % 2 == 0, 0.0, 400.0).astype(np.float32),
                          "frame": np.arange(n, dtype=np.int64),
                          "loc_precision_nm": np.full(n, 10, np.float32)}, {})
    flat, _ = group(locs, GroupSettings(dx=50, dt=1))
    assert len(flat) == 1
    with pytest.raises(TypeError):
        GroupSettings(dx=50, dt=1, dz=100)


# --------------------------------------------------------------- the sort ---
# `sorted_order` replaced the `np.lexsort` in front of the linking walk (70 s of
# `connect`'s 71.7 s on a 57 M localization file).  It is only allowed to exist
# if it returns *the same permutation*, ties and all, so that is what is pinned.

def _orders_match(x, frame, keys=()):
    from smappy.group import sorted_order
    x = np.asarray(x, np.float64)
    frame = np.asarray(frame, np.int64)
    mine = sorted_order(x, frame, keys)
    lex = np.lexsort((x, frame) + tuple(reversed(keys)))
    return np.array_equal(mine, lex)


@pytest.mark.parametrize("name, n, frames", [
    ("empty", 0, 1), ("one", 1, 1), ("one frame", 1000, 1),
    ("one localization per frame", 1000, 1000), ("dense", 5000, 40),
    ("sparse", 30, 1_000_000), ("two localizations", 2, 1),
])
def test_sorted_order_is_the_lexsort_it_replaces(name, n, frames):
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    x = rng.random(n)
    frame = rng.integers(0, frames, n).astype(np.int64)
    assert _orders_match(x, frame)
    assert _orders_match(x, frame, (rng.integers(0, 3, n).astype(np.int64),))
    assert _orders_match(x, frame, (rng.integers(0, 2, n).astype(np.int64),
                                    rng.integers(0, 3, n).astype(np.int64)))


def test_sorted_order_keeps_the_order_of_ties():
    """Equal (frame, x) must come out in input order: both sorts are stable,
    and the linker claims the first match, so a swap here changes groups."""
    x = np.repeat([1.0, 2.0], 8)
    frame = np.zeros(16, np.int64)
    assert _orders_match(x, frame)
    assert _orders_match(x, frame, (np.arange(16) % 2,))


def test_sorted_order_handles_negative_and_far_apart_frames():
    rng = np.random.default_rng(7)
    x = rng.random(2000)
    assert _orders_match(x, rng.integers(-500, 500, 2000).astype(np.int64))
    assert _orders_match(x, (rng.integers(0, 4, 2000) * 10 ** 7).astype(np.int64))


# ----------------------------------------------------- chunked linking ------
# `smappy._group_chunked` links slices of the frame axis in parallel and repairs
# the seams.  It is an approximation and is not wired into `group()`; these pin
# the two things that make it usable at all -- that one chunk is the sequential
# walk exactly, and that more chunks stay within a hair of it.

def _blinking(n_emitters=3000, frames=400, mean_on=3.0, seed=0):
    rng = np.random.default_rng(seed)
    ex, ey = rng.uniform(0, 5000, n_emitters), rng.uniform(0, 5000, n_emitters)
    xs, ys, fs = [], [], []
    start = rng.integers(0, frames, n_emitters)
    length = 1 + rng.geometric(1.0 / mean_on, n_emitters)
    for e in range(n_emitters):
        k = int(min(length[e], frames - start[e]))
        if k <= 0:
            continue
        xs.append(rng.normal(ex[e], 12.0, k))
        ys.append(rng.normal(ey[e], 12.0, k))
        fs.append(np.arange(start[e], start[e] + k))
    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate(fs).astype(np.int64))


def _same_partition(a, b):
    """Fraction of localizations whose group holds the same localizations."""
    pair = a.astype(np.int64) * (int(b.max()) + 1) + b
    _, inv, count = np.unique(pair, return_inverse=True, return_counts=True)
    return float(((count[inv] == np.bincount(a)[a])
                  & (count[inv] == np.bincount(b)[b])).sum()) / len(a)


def test_one_chunk_is_the_sequential_walk():
    from smappy._group_chunked import connect_chunked
    x, y, f = _blinking()
    assert np.array_equal(connect_chunked(x, y, f, 50.0, 1, n_chunks=1),
                          connect(x, y, f, 50.0, 1))


def test_chunked_linking_stays_within_a_hair_of_the_control():
    from smappy._group_chunked import connect_chunked
    x, y, f = _blinking()
    control = connect(x, y, f, 50.0, 1)
    for n_chunks in (2, 4, 8):
        stats = {}
        ids = connect_chunked(x, y, f, 50.0, 1, n_chunks=n_chunks, stats=stats)
        assert ids.min() == 1 and len(np.unique(ids)) == ids.max()   # no gaps
        assert _same_partition(control, ids) > 0.999
        assert stats["rejoined"] > 0            # seams were found and repaired


def test_chunked_linking_keeps_blocks_apart():
    """Localizations of different channels may not be linked, however the
    frame axis is cut."""
    from smappy._group_chunked import connect_chunked
    x, y, f = _blinking(n_emitters=800, frames=200)
    channel = (np.arange(len(x)) % 2).astype(np.int64)
    ids = connect_chunked(x, y, f, 50.0, 1, channel[:, None], n_chunks=4)
    for g in np.unique(ids):
        assert len(np.unique(channel[ids == g])) == 1


def test_the_sort_assumes_nothing_about_the_order_it_is_given():
    """The radix partitions on a key every localization carries, not on where
    it sits, so a table straight out of a plugin that reordered everything
    sorts the same as a tidy one -- and `connect` answers in input order."""
    rng = np.random.default_rng(11)
    n = 50_000
    x = rng.random(n) * 40_000
    frame = rng.integers(0, 90_000, n).astype(np.int64)      # spans two chunks
    channel = rng.integers(0, 3, n).astype(np.int64)
    filenumber = rng.integers(0, 2, n).astype(np.int64)
    keys = (filenumber, channel)
    assert np.array_equal(sorted_order(x, frame, keys),
                          np.lexsort((x, frame, channel, filenumber)))

    shuffle = rng.permutation(n)
    y = rng.random(n) * 40_000
    blocks = np.stack(keys, axis=1)
    here = connect(x, y, frame, 50.0, 1, blocks)
    there = connect(x[shuffle], y[shuffle], frame[shuffle], 50.0, 1, blocks[shuffle])
    # ids are handed back against the rows they came in as, so the two
    # labellings have to be the same partition of the same localizations
    assert _same_partition(here[shuffle], there)== 1.0


def test_frames_that_do_not_start_at_zero_or_run_in_order():
    rng = np.random.default_rng(12)
    x = rng.random(20_000) * 1000
    for frame in (rng.integers(500_000, 700_000, 20_000),      # far from zero
                  rng.integers(-40_000, -30_000, 20_000),      # negative
                  rng.choice([3, 999_999], 20_000)):           # two, far apart
        frame = frame.astype(np.int64)
        assert np.array_equal(sorted_order(x, frame), np.lexsort((x, frame)))


# ------------------------------------------------ threaded, and repeatable ---
# `combine` runs its columns across threads and `group()` links across chunks.
# Neither may make the answer depend on what the machine happened to do.

def test_group_ids_do_not_depend_on_the_thread_count():
    """`link_chunks` is a setting, not a thread count: the cut points come
    from the data, so the same file groups the same way on any machine."""
    from smappy._group_chunked import connect_chunked
    x, y, f = _blinking(n_emitters=4000, frames=600, seed=2)
    first = connect_chunked(x, y, f, 50.0, 1, n_chunks=8, workers=1)
    for workers in (2, 4, 16):
        assert np.array_equal(connect_chunked(x, y, f, 50.0, 1, n_chunks=8,
                                              workers=workers), first)


def test_grouping_uses_the_chunked_linker_and_can_be_told_not_to():
    x, y, f = _blinking(n_emitters=4000, frames=600, seed=3)
    locs = Localizations({"x_nm": x.astype(np.float32), "y_nm": y.astype(np.float32),
                          "frame": f, "photons": np.ones(len(x), np.float32),
                          "loc_precision_nm": np.full(len(x), 10.0, np.float32)},
                         {"units": "nm"})
    assert GroupSettings().link_chunks == 8
    chunked, _ = group(locs, GroupSettings())
    exact, index = group(locs, GroupSettings(link_chunks=1))
    assert np.array_equal(index, connect(x, y, f, 50.0, 1))   # 1 chunk is the walk
    assert abs(len(chunked) - len(exact)) / len(exact) < 1e-3


def test_combine_reduces_each_column_by_its_own_rule():
    rng = np.random.default_rng(5)
    x, y, f = _blinking(n_emitters=2000, frames=300, seed=4)
    n = len(x)
    locs = Localizations({
        "x_nm": x.astype(np.float32), "y_nm": y.astype(np.float32), "frame": f,
        "photons": rng.uniform(100, 900, n).astype(np.float32),
        "photons_err": rng.uniform(5, 40, n).astype(np.float32),
        "loc_precision_nm": rng.uniform(5, 25, n).astype(np.float32),
        "x_err_nm": rng.uniform(5, 25, n).astype(np.float32),
        "y_err_nm": rng.uniform(5, 25, n).astype(np.float32),
        "logl_rel": rng.uniform(-2, 0, n).astype(np.float32),
    }, {"units": "nm"})
    gi = connect(x, y, f, 50.0, 1)
    once, twice = combine(locs, gi), combine(locs, gi)
    for name in once.keys():
        assert np.array_equal(np.asarray(once[name]), np.asarray(twice[name]),
                              equal_nan=True), name

    # and the values are still the per-column rules, checked against the slow way
    size = int(gi.max())
    frame_min = np.full(size, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(frame_min, gi - 1, np.asarray(f))
    assert np.array_equal(np.asarray(once["frame"]), frame_min)
    photons = np.zeros(size)
    np.add.at(photons, gi - 1, np.asarray(locs["photons"], np.float64))
    assert np.allclose(np.asarray(once["photons"]), photons.astype(np.float32), rtol=1e-5)


def test_radix_argsort_is_the_stable_argsort_it_replaces():
    from smappy.group import radix_argsort
    rng = np.random.default_rng(6)
    for values in (rng.integers(0, 5, 1000), rng.integers(0, 10 ** 7, 50_000),
                   np.zeros(100, np.int64), np.arange(999, -1, -1)):
        values = np.asarray(values, np.int64)
        assert np.array_equal(radix_argsort(values),
                              np.argsort(values, kind="stable"))


# ------------------------------------------------- the compiled combiner -----
# `combine` reduces each column in C++ when the extension is built and in numpy
# when it is not.  The two have to agree, mode for mode, or the compiled one is
# quietly changing what a grouped table means.

def _numpy_combine(locs, gi, monkeypatch):
    """`combine` with the compiled reducer hidden, i.e. the numpy reference."""
    import smappy.group as g
    monkeypatch.setattr(g, "_group", None)
    return g.combine(locs, gi)


def test_the_compiled_combiner_agrees_with_numpy_in_every_mode(monkeypatch):
    from smappy import group as g
    pytest.importorskip("smappy._group")
    rng = np.random.default_rng(9)
    x, y, f = _blinking(n_emitters=3000, frames=400, seed=9)
    n = len(x)
    locs = Localizations({                      # one column per combine mode
        "x_nm": x.astype(np.float32),           # mean, weighted by x_err_nm
        "y_nm": y.astype(np.float32),           # mean, weighted by y_err_nm
        "z_nm": rng.normal(0, 300, n).astype(np.float32),
        "frame": f,                             # min, and an int64 column
        "photons": rng.uniform(100, 900, n).astype(np.float32),        # sum
        "background": rng.uniform(1, 50, n).astype(np.float32),        # sum
        "photons_err": rng.uniform(5, 40, n).astype(np.float32),       # quad
        "loc_precision_nm": rng.uniform(5, 25, n).astype(np.float32),  # precision
        "x_err_nm": rng.uniform(5, 25, n).astype(np.float32),
        "y_err_nm": rng.uniform(5, 25, n).astype(np.float32),
        "z_err_nm": rng.uniform(20, 90, n).astype(np.float32),
        "logl_rel": rng.uniform(-2, 0, n).astype(np.float32),          # max
        "iterations": rng.integers(1, 30, n).astype(np.int32),         # max, int32
        "sigma_nm": rng.uniform(90, 180, n).astype(np.float32),        # mean
    }, {"units": "nm"})
    gi = connect(x, y, f, 50.0, 1)

    modes = {name: g._mode(name) for name in locs.keys()}
    assert set(modes.values()) == {"mean", "sum", "quad", "precision", "min", "max"}

    compiled = g.combine(locs, gi)
    reference = _numpy_combine(locs, gi, monkeypatch)
    assert set(compiled.keys()) == set(reference.keys())
    for name in compiled.keys():
        a, b = np.asarray(compiled[name]), np.asarray(reference[name])
        assert a.dtype == b.dtype, name
        assert np.allclose(a, b, rtol=1e-5, atol=1e-5, equal_nan=True), name


def test_the_compiled_combiner_gives_the_same_answer_on_any_thread_count():
    from smappy import _group
    pytest.importorskip("smappy._group")
    rng = np.random.default_rng(10)
    n, n_groups = 40_000, 9_000
    gi = np.sort(rng.integers(1, n_groups + 1, n))
    order = np.argsort(gi, kind="stable")
    starts = np.concatenate(([0], np.flatnonzero(gi[order][1:] != gi[order][:-1]) + 1,
                             [n]))
    values = rng.random(n).astype(np.float32)
    weights = rng.random(n) + 0.5
    for mode in range(6):
        one = _group.combine(values, weights, order, starts, mode, 1)
        for threads in (2, 4, 8):
            assert np.array_equal(
                _group.combine(values, weights, order, starts, mode, threads), one)


def test_a_group_of_one_survives_every_mode():
    """Groups of a single localization are most of a real table; min, max and
    the sums all have to give that localization's own value back."""
    from smappy import _group
    pytest.importorskip("smappy._group")
    values = np.array([3.0, 5.0, 7.0], np.float32)
    weights = np.ones(3)
    order = np.array([0, 1, 2], np.int64)
    starts = np.array([0, 1, 2, 3], np.int64)
    for mode, expect in [(0, [3, 5, 7]), (1, [3, 5, 7]), (4, [3, 5, 7]), (5, [3, 5, 7])]:
        assert np.allclose(_group.combine(values, weights, order, starts, mode, 1),
                           expect)
