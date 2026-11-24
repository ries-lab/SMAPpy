"""In situ learning in rounds, and the coverslip edge that pins the focus (M5, step 5).

The rounds on whole simulated movies -- a focus started 250 nm off brought back by the edge -- take minutes and
are a benchmark, not a test: benchmarks/uipsf/synthetic/insitu_movie.py."""
import numpy as np
import pytest
import torch

from uipsf.workflows.insitu_rounds import _focus_per_height, _moved, below_coverslip, coverslip_edge, edge_over_field


def heights(n=1500, on_coverslip=0.2, shift=120.0, sd=30.0, scatter=1.2, outliers=0.01, seed=0):
    """Localised heights of molecules uniform 0–1400 nm deep, ``on_coverslip`` of them at 0, all shifted by
    ``shift``; each with a CRLB about ``sd`` that understates its scatter ``scatter`` times, and a few
    ``outliers`` hundreds of nanometres too low."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 1400, n)
    t[rng.random(n) < on_coverslip] = 0.0
    s = sd * (0.6 + 0.8 * rng.random(n))
    h = t + shift + rng.normal(0, s * scatter)
    out = rng.random(n) < outliers
    h[out] -= rng.uniform(100, 400, out.sum())
    return h, s


def test_the_edge_is_where_the_molecules_on_the_coverslip_are_whatever_the_outliers():
    found = [coverslip_edge(*heights(seed=k)) for k in range(5)]
    assert all(e.spike > 0.2 for e in found)
    shifts = np.array([e.shift_nm for e in found])
    assert abs(shifts.mean() - 120.0) < 3 and shifts.std() < 5
    assert np.median([e.sd_nm for e in found]) < 5


def test_without_molecules_on_the_coverslip_the_flat_edge_is_found_and_a_spike_rarely_invented():
    found = [coverslip_edge(*heights(on_coverslip=0.0, seed=k)) for k in range(5)]
    assert sum(e.spike > 0 for e in found) <= 1
    assert abs(np.mean([e.shift_nm for e in found]) - 120.0) < 10


def test_molecules_on_the_coverslip_stay_and_those_placed_far_below_it_are_out():
    rng = np.random.default_rng(3)
    n = 1500
    t = rng.uniform(0, 1400, n)
    on = rng.random(n) < 0.2
    t[on] = 0.0
    sd = 30.0 * (0.6 + 0.8 * rng.random(n))
    h = t + 120.0 + rng.normal(0, 1.2 * sd)
    bad = rng.random(n) < 0.03                 # misplaced by a model that does not fit them
    h[bad] -= rng.uniform(250, 500, bad.sum())
    edge = coverslip_edge(h, sd)
    assert edge.kappa >= 1.0
    out = below_coverslip(h - edge.shift_nm, sd, edge)
    assert out[on & ~bad].mean() < 0.01
    assert out[bad & (t < 100)].mean() > 0.9
    assert not below_coverslip(np.array([-10.0]), np.array([30.0]), edge)[0]


def test_too_few_heights_give_no_edge():
    assert coverslip_edge(np.arange(10.0), np.ones(10)) is None


def test_after_two_corrections_the_focus_per_height_is_their_secant():
    assert _focus_per_height([(700.0, 30.0)], 1.6) == 1.6
    assert _focus_per_height([(700.0, 30.0), (650.0, 5.0)], 1.6) == pytest.approx(2.0)
    assert _focus_per_height([(700.0, 30.0), (705.0, 5.0)], 1.6) == 1.6        # the foci too close


def test_moved_is_the_rms_change_in_standard_errors_of_the_change():
    sd = {"pupil.phase": torch.full((4,), 0.01), "geometry.focus": torch.tensor(5.0)}
    old = {"pupil.phase": torch.zeros(4), "geometry.focus": torch.tensor(700.0)}
    new = {"pupil.phase": torch.tensor([0.01, -0.01, 0.0, 0.0]) * np.sqrt(2), "geometry.focus": torch.tensor(700.0)}
    assert _moved(new, old, sd, sd) == pytest.approx(np.sqrt(2 / 5))


def test_the_coverslip_edge_measured_over_the_field_finds_a_tilted_focus():
    """A focus tilted against the coverslip by 3 nm/µm over a 100 µm field: every region's lowest molecules
    sit that much higher or lower, and the plane through the regions' edges gives the tilt back."""
    rng = np.random.default_rng(3)
    n = 6000
    xy = rng.uniform(0, 1000, (n, 2))                         # chip pixels, 100 nm each
    h, s = heights(n=n, shift=0.0, seed=4)
    tilt = 3.0                                                # nm per µm, along x
    h = h + tilt * (xy[:, 0] - 500) * 0.1
    edges = edge_over_field(h, s, xy, (100.0, 100.0), tiles=3)
    assert edges is not None and len(edges.shift_nm) == 9
    assert abs(edges.plane[1] - tilt) < 0.5 and abs(edges.plane[2]) < 0.5
    assert edges.span_nm == pytest.approx(tilt * 66.7, rel=0.25)        # the regions' centres span ~2/3
    assert "tilted" in edges.summary()
    flat = edge_over_field(*heights(n=n, shift=0.0, seed=5), xy, (100.0, 100.0))
    assert abs(flat.plane[1]) < 0.5 and flat.span_nm < 50.0
    assert edge_over_field(h[:100], s[:100], xy[:100], (100.0, 100.0)) is None      # too few
