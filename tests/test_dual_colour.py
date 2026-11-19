"""*MINFLUX dual colour* (`smappy.dual_colour`, `plugins/dual_colour.py`)
and the shipped kinesin chain: two-colour molecules with a known offset,
lead and colour drift come back; molecules that only cross are not joined;
a bead measurement open beside the tracking gives the offset."""
import numpy as np
import pytest

from smappy import dual_colour as DC
from smappy import plugins
from smappy.io.formats import FileInfo
from smappy.locs import Localizations
from smappy.session import Session

C = np.array([-1.5, 11.5])          # colour 1 - colour 0, as the kinesin beads had it
LEAD = -6.0                          # colour 1 behind colour 0 along the motion (nm)
NOISE = 4.0


def kinesins(n_walk=30, n_still=10, seed=0, t0=0.0, c=C, lead=LEAD):
    """Molecules seen in both colours, the localizations alternating between
    the colours every 3 ms: walkers at 400 nm/s in random directions, still
    ones; colour 1 offset by ``c`` and, along the motion, by ``lead``.
    Returns the columns and the truth per molecule."""
    rng = np.random.default_rng(seed)
    rows, t = [], t0
    tid = {0: 100, 1: 5000}
    for k in range(n_walk + n_still):
        walk = k < n_walk
        phi = rng.uniform(0, 2 * np.pi)
        u = np.array([np.cos(phi), np.sin(phi)])
        start = rng.uniform(0, 20000, 2)
        tid[0] += 1
        tid[1] += 1
        for j in range(120):
            ts = t + 3e-3 * j
            colour = j % 2
            p = start + (u * 400.0 * (ts - t) if walk else 0.0)
            if colour:
                p = p + c + (lead * u if walk else 0.0)
            p = p + rng.normal(0, NOISE, 2)
            rows.append((tid[colour], colour, ts, p[0], p[1]))
        t += 0.5
    a = np.array(rows)
    order = np.argsort(a[:, 2], kind="stable")
    a = a[order]
    n = len(a)
    return {"tid": a[:, 0].astype(np.int64), "channel": a[:, 1].astype(np.int16),
            "thi": a[:, 1].astype(np.int32), "time_s": a[:, 2],
            "x_nm": a[:, 3].astype(np.float32), "y_nm": a[:, 4].astype(np.float32),
            "frame": np.arange(n), "efo": rng.normal(100e3, 5e3, n).astype(np.float32),
            "cfr": rng.normal(0.3, 0.05, n).astype(np.float32),
            "xy_err_nm": np.full(n, 3.0, np.float32), "sigma_nm": np.full(n, 150.0, np.float32)}


def bead(seed=1, c=C, t0=0.0):
    """One bead, followed for 3000 localizations per colour."""
    rng = np.random.default_rng(seed)
    n = 6000
    colour = np.arange(n) % 2
    t = t0 + np.arange(n) * 2.6e-3
    p = np.array([5000.0, 5000.0]) + np.where(colour[:, None] == 1, c, 0.0) \
        + rng.normal(0, 2.0, (n, 2))
    return {"tid": np.where(colour == 1, 309, 308).astype(np.int64),
            "channel": colour.astype(np.int16), "thi": (1 - colour).astype(np.int32),
            "time_s": t, "x_nm": p[:, 0].astype(np.float32), "y_nm": p[:, 1].astype(np.float32),
            "frame": np.arange(n), "efo": np.full(n, 100e3, np.float32),
            "cfr": np.full(n, 0.3, np.float32), "xy_err_nm": np.full(n, 2.0, np.float32),
            "sigma_nm": np.full(n, 150.0, np.float32)}


def info(name, label, date, did):
    meta = {"label": label, "did": did, "acquisition_date": date,
            "channels": [{"channel": 0, "excitation_nm": 638.0, "name": "638 nm"},
                         {"channel": 1, "excitation_nm": 560.0, "name": "560 nm"}],
            "mbm": {"used": []}}
    return FileInfo(name, f"/data/{name}", "Abberior MSR",
                    metadata={"minflux": {"measurements": [meta]}})


def session_of(*files):
    s = Session()
    for k, (cols, inf) in enumerate(files):
        s.add_file(Localizations(cols, {"minflux": inf.metadata["minflux"]}), inf,
                   append=k > 0)
    s.show_grouped(0, False)
    return s


# -------------------------------------------------------------------- pairing

def test_molecules_in_both_colours_are_joined_and_their_offset_measured():
    k = kinesins()
    links = DC.link_traces(k["tid"], k["channel"], k["time_s"], k["x_nm"], k["y_nm"],
                           offset=lambda t: np.tile(C, (len(t), 1)))
    assert links.both.sum() == 40 and len(links.ids) == 40
    # still molecules: their offset is the registration
    still = links.displacement_nm < 30
    assert still.sum() == 10
    assert np.median(links.dx[still]) == pytest.approx(C[0], abs=1.5)
    assert np.median(links.dy[still]) == pytest.approx(C[1], abs=1.5)
    assert np.all(links.within > 0.9)


def test_two_trajectories_that_only_cross_are_not_one_molecule():
    t = np.arange(200) * 3e-3
    colour = np.arange(200) % 2
    # colour 0 walks along x, colour 1 along y; they meet half way
    x = np.where(colour == 0, -300 + 1000 * t, 0.0)
    y = np.where(colour == 0, 0.0, -300 + 1000 * t)
    tid = np.where(colour == 0, 1, 2)
    links = DC.link_traces(tid, colour, t, x, y)
    assert len(links.ids) == 2 and not links.both.any()
    # a median distance alone would have let them through: they are close
    # for a while either side of the crossing
    together = DC.link_traces(tid, colour, t, x, y, min_fraction=0.0)
    assert together.both.all()


def test_a_trace_that_stops_and_starts_again_joins_its_partner_once():
    t = np.arange(300) * 3e-3
    colour = np.arange(300) % 2
    x = 400.0 * t + np.where(colour == 1, 10.0, 0.0)
    y = np.zeros(300)
    # colour 1 lost in the middle: two traces, one after the other
    tid = np.where(colour == 0, 1, np.where(t < 0.4, 2, 3))
    gap = (colour == 1) & (t > 0.35) & (t < 0.45)
    tid = np.where(gap, -1, tid)
    links = DC.link_traces(tid, colour, t, x, y)
    assert len(links.ids) == 1 and links.n_traces[0] == 3


def test_the_fit_tells_the_registration_from_the_lead_along_the_track():
    k = kinesins(n_walk=40, n_still=10, seed=2)
    links = DC.link_traces(k["tid"], k["channel"], k["time_s"], k["x_nm"], k["y_nm"],
                           max_distance=100.0)
    both = links.both
    # a few chance pairs: offsets far off
    dx, dy = links.dx[both].copy(), links.dy[both].copy()
    dx[:3] += 60.0
    fit = DC.fit_track_offset(dx, dy, links.ux[both], links.uy[both],
                              links.displacement_nm[both] > 30)
    assert fit.c == pytest.approx(C, abs=1.5)
    assert fit.a == pytest.approx(LEAD, abs=1.5)
    assert not fit.used[:3].any() and fit.n_moving == 40 and fit.n_still == 10


def test_a_bead_measurement_gives_the_offset():
    b = bead()
    got = DC.bead_offset(b["tid"], b["channel"], b["time_s"], b["x_nm"], b["y_nm"])
    assert got.offset == pytest.approx(C, abs=0.2)
    assert got.n_pairs == 1 and got.n_matched > 2900


def test_the_mbm_colour_drift_comes_back_whatever_each_beads_bias():
    rng = np.random.default_rng(3)
    t = np.arange(0.0, 400.0, 18.0)
    drift = np.column_stack([0.01 * t, -0.005 * t])            # colour 1 vs colour 0
    rows, entries, tid = [], [], 0
    for b in range(6):
        bias = rng.normal(0, 20, 2)
        for colour in (0, 1):
            tid += 1
            p = np.array([1000.0 * b, 0.0]) + (bias + drift if colour else 0.0) \
                + rng.normal(0, 1.0, (len(t), 2))
            rows += [(tid, colour, ti, *pi) for ti, pi in zip(t + 0.1 * colour, p)]
            entries.append({"tid": tid, "gri": 61 + 62 * b, "name": f"R{b}",
                            "channel": colour, "measurement": 0})
    a = np.array(rows)
    beads = Localizations({"tid": a[:, 0].astype(np.int64), "channel": a[:, 1].astype(np.int16),
                           "time_s": a[:, 2], "x_nm": a[:, 3], "y_nm": a[:, 4]}, {})
    got = DC.mbm_colour_drift(beads, entries, knot=200.0)
    assert len(got.names) == 6
    grid = np.array([0.0, 200.0, 396.0])
    want = np.column_stack([0.01 * grid, -0.005 * grid])
    assert np.abs(got(grid) - want).max() < 0.8


def test_beads_before_and_after_are_interpolated_with_the_drift_as_the_shape():
    class Linear:
        def __call__(self, t):
            t = np.clip(np.asarray(t, float), 0.0, 100.0)
            return np.column_stack([0.05 * t, np.zeros(len(t))])
    curve = DC.offset_curve((np.array([0.0, 10.0]), -50.0), (np.array([8.0, 14.0]), 150.0),
                            Linear())
    # the drift explains 5 nm in x of the 8; 3 nm in x and 4 in y are spread
    assert curve(np.array([-50.0]))[0] == pytest.approx([0.0, 10.0])
    assert curve(np.array([150.0]))[0] == pytest.approx([8.0, 14.0])
    assert "rest (3.0, 4.0)" in curve.note


# --------------------------------------------------------------------- plugin

def test_the_bead_measurement_open_beside_the_tracking_sets_the_offset_and_moves_colour_1():
    s = session_of((bead(), info("beads.msr", "12_alignment", "2026-11-13T16:20:34+01:00", "b")),
                   (kinesins(), info("tracks.msr", "13_tracking", "2026-11-13T16:26:46+01:00",
                                     "t")))
    for layer in s.layers:
        layer.set_files([1])
    cls = plugins.get("Analysis/MINFLUX/MINFLUX dual colour")
    assert cls.name == "MINFLUX dual colour"
    result = s.run(cls(), cls.Settings())
    assert "beads in beads.msr" in result.text and "40 in both colours" in result.text
    # colour 1 moved onto colour 0: a still molecule's colours on top of each other
    links = result.data["targets"][0][5]
    still = links.both & (links.displacement_nm < 30)
    assert np.abs(np.median(links.dx[still])) < 1.5 and np.abs(np.median(links.dy[still])) < 1.5
    assert "560 nm onto 638 nm" in s.transformations
    # run again: from the positions as read, not from the first run's
    x1 = np.asarray(s.locs["x_nm"]).copy()
    s.run(cls(), cls.Settings())
    assert np.allclose(np.asarray(s.locs["x_nm"]), x1, atol=1e-3)
    # the bead measurement is not paired: it is the reference
    beads = np.asarray(s.locs["filenumber"]) == 0
    assert (np.asarray(s.locs["pair_id"])[beads] == -1).all()


def test_without_beads_the_offset_comes_from_the_molecules():
    s = session_of((kinesins(seed=4), info("tracks.msr", "13_tracking",
                                           "2026-11-13T16:26:46+01:00", "t")))
    cls = plugins.get("Analysis/MINFLUX/MINFLUX dual colour")
    result = s.run(cls(), cls.Settings())
    curve, how, fit = result.data["targets"][0][1:4]
    assert "fitted over" in how
    assert curve.c == pytest.approx(C, abs=1.5)
    assert fit.a == pytest.approx(LEAD, abs=2.0)


def test_a_bead_measurement_read_with_the_tracking_is_told_by_its_long_traces():
    b, k = bead(t0=0.0), kinesins(t0=400.0)
    cols = {c: np.concatenate([b[c], k[c]]) for c in b}
    cols["measurement"] = np.concatenate([np.zeros(len(b["tid"])),
                                          np.ones(len(k["tid"]))]).astype(np.int16)
    cols["frame"] = np.arange(len(cols["tid"]))
    metas = [info("x", "12_alignment", "2026-11-13T16:20:34+01:00", "b")
             .metadata["minflux"]["measurements"][0],
             info("x", "13_tracking", "2026-11-13T16:26:46+01:00", "t")
             .metadata["minflux"]["measurements"][0]]
    metas[1]["time_offset_s"] = 400.0
    inf = FileInfo("both.msr", "/data/both.msr", "Abberior MSR",
                   metadata={"minflux": {"measurements": metas}})
    s = session_of((cols, inf))
    cls = plugins.get("Analysis/MINFLUX/MINFLUX dual colour")
    result = s.run(cls(), cls.Settings())
    assert "beads in both.msr: 12_alignment" in result.text
    measurement = np.asarray(s.locs["measurement"])
    assert (np.asarray(s.locs["pair_id"])[measurement == 0] == -1).all()


# ---------------------------------------------------------------------- chain

CHAIN = "Analysis/MINFLUX/Workflow/MINFLUX kinesin"


def test_the_kinesin_chain_is_shipped_and_asks_four_settings():
    found = plugins.get(CHAIN)
    assert len(found.interface()) == 4


def test_the_kinesin_chain_finds_the_walkers_and_their_steps():
    s = session_of((bead(), info("beads.msr", "12_alignment", "2026-11-13T16:20:34+01:00", "b")),
                   (kinesins(), info("tracks.msr", "13_tracking", "2026-11-13T16:26:46+01:00",
                                     "t")))
    for layer in s.layers:
        layer.set_files([1])
    chain = plugins.get(CHAIN)
    result = s.run(chain(), chain.Settings())
    walkers = [r for r in s.rois.rois.values() if r.use]
    assert len(walkers) == 30                          # the still ones left out
    assert "30 traces of pair_id" in result.text
