"""The precision a MINFLUX table is given: the Cramér-Rao bound at the
pattern's centre (`io.minflux.crlb_xy`, `crlb_z`), with L from the sequence
and the photons of the last lateral iteration.

The bound is checked twice against a simulated measurement: the Fisher
information computed numerically for patterns of several shapes, and the
spread of maximum-likelihood positions from Poisson counts.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from smappy.io import minflux


def _pattern(k, L, centre=False):
    a = 2 * np.pi * np.arange(k) / k
    pts = np.c_[L / 2 * np.cos(a), L / 2 * np.sin(a)]
    return np.vstack([pts, [0.0, 0.0]]) if centre else pts


def _rates(r, pts, b):
    """The quadratic zero, I = |r - r_i|^2 + b, per exposure."""
    return ((r[None, :] - pts) ** 2).sum(1) + b


def _fisher_sigma(pts, b, n_total, h=1e-4):
    """sqrt of the x element of the inverse Fisher information of the
    multinomial counts, at the centre, by finite differences."""
    def p(r):
        lam = _rates(r, pts, b)
        return lam / lam.sum()
    p0 = p(np.zeros(2))
    grad = np.stack([(p(np.array([h, 0])) - p(np.array([-h, 0]))) / (2 * h),
                     (p(np.array([0, h])) - p(np.array([0, -h]))) / (2 * h)], 1)
    seen = p0 > 0               # a background-free centre: no counts and no slope, no information
    fisher = n_total * (grad[seen].T / p0[seen]) @ grad[seen]
    return np.sqrt(np.linalg.inv(fisher)[0, 0])


@pytest.mark.parametrize("k, centre", [(3, False), (3, True), (4, True), (6, False), (6, True)])
@pytest.mark.parametrize("sbr", [np.inf, 4.0, 1.0])
def test_the_lateral_bound_is_the_fisher_information_of_any_ring_pattern(k, centre, sbr):
    L, n_offsets = 100.0, 200.0
    pts = _pattern(k, L, centre)
    signal = (L / 2) ** 2                       # the rate at an offset, emitter at the centre
    b = 0.0 if np.isinf(sbr) else signal / sbr
    rates = _rates(np.zeros(2), pts, b)
    n_total = n_offsets * rates.sum() / rates[:k].sum()  # the centre's counts on top
    s = signal / (signal + b)                    # the molecule's share of the offsets' photons
    expected = minflux.crlb_xy(L, n_offsets * s, n_offsets * (1 - s))
    assert np.isclose(_fisher_sigma(pts, b, n_total), expected, rtol=1e-4)
    if np.isinf(sbr):
        assert np.isclose(expected, L / (2 * np.sqrt(2 * n_offsets)))


def test_the_axial_bound_is_the_fisher_information_of_a_line_of_two():
    L, n, b = 80.0, 150.0, 400.0
    xs = np.array([-L / 2, L / 2])
    def p(x):
        lam = (x - xs) ** 2 + b
        return lam / lam.sum()
    h = 1e-4
    grad = (p(h) - p(-h)) / (2 * h)
    sigma = 1 / np.sqrt(n * (grad ** 2 / p(0.0)).sum())
    s = (L / 2) ** 2 / ((L / 2) ** 2 + b)
    assert np.isclose(sigma, minflux.crlb_z(L, n * s, n * (1 - s)), rtol=1e-4)


def test_maximum_likelihood_positions_of_a_hexagon_spread_as_the_bound_says():
    from scipy.optimize import minimize
    rng = np.random.default_rng(3)
    L, n_mean, sbr = 100.0, 400.0, 5.0
    pts = _pattern(6, L)
    signal = (L / 2) ** 2
    b = signal / sbr
    lam = _rates(np.zeros(2), pts, b)
    lam *= n_mean / lam.sum()
    fits = []
    for _ in range(400):
        n = rng.poisson(lam)
        def nll(r):
            q = _rates(r, pts, b)
            return -(n * np.log(q / q.sum())).sum()
        fits.append(minimize(nll, np.zeros(2), method="Nelder-Mead",
                             options={"xatol": 1e-3, "fatol": 1e-9}).x)
    spread = np.std(np.asarray(fits), axis=0)
    bound = minflux.crlb_xy(L, n_mean * sbr / (1 + sbr), n_mean / (1 + sbr))
    assert np.allclose(spread, bound, rtol=0.12), (spread, bound)


def test_the_photons_are_split_by_the_background_rate_and_are_all_signal_without():
    signal, background = minflux.photon_split([100, 100, 100], [1e5, 1e5, 1e5],
                                              [0, 2.5e4, np.nan])
    assert np.allclose(signal, [100, 75, 100]) and np.allclose(background, [0, 25, 0])
    signal, background = minflux.photon_split(100, 1e4, 5e4)   # more background than photons
    assert np.isclose(background / 100, minflux.MAX_BACKGROUND_FRACTION)
    assert minflux.photon_split(100, 0.0, 0.0) == (100, 0)


def test_the_bound_with_background_is_marin_and_ries_eq_33_in_1d_and_its_2d_form():
    # Eq. 33 (1D, -L/2, 0, L/2): Nb over all three exposures, two of them offsets
    L, ns, nb = 60.0, 150.0, 90.0
    eq33 = L / (4 * np.sqrt(ns)) * np.sqrt(1 + 2 / 3 * nb / ns)
    assert np.isclose(minflux.crlb_z(L, ns, nb * 2 / 3), eq33)
    assert np.isclose(minflux.crlb_xy(L, ns, 0.0), L / (2 * np.sqrt(2 * ns)))
    assert minflux.crlb_xy(L, ns, nb) > minflux.crlb_xy(L, ns)


def _seq(*steps):
    return {"iterations": [{"Mode.pattern": p, "patGeoFactor": g} for p, g in steps]}


def test_the_pattern_sizes_are_the_last_lateral_and_axial_iterations():
    two_d = _seq(("hexagon", 0.79), ("hexagon", 0.84), ("hexagon", 0.42), ("hexagon", 0.28))
    assert minflux.pattern_sizes(two_d) == {"xy": (3, pytest.approx(100.8))}
    three_d = _seq(("hexagon", 0.8), ("zline", [0.8, 4]), ("square", 0.42), ("zline2", 0.42),
                   ("square", 0.21), ("zline2", 0.21))
    assert minflux.pattern_sizes(three_d) == {"xy": (4, pytest.approx(75.6)),
                                              "z": (5, pytest.approx(75.6))}
    assert minflux.pattern_sizes(None) == {}


def _records_3d():
    """Two localizations of one trace, each a lateral iteration 2 (its own
    photons) then the final axial iteration 3."""
    dt = [("vld", "?"), ("fnl", "?"), ("itr", "i4"), ("tim", "f8"), ("tid", "i8"),
          ("loc", "f8", (3,)), ("eco", "u4"), ("efo", "f4"), ("fbg", "f4"), ("cfr", "f4")]
    rows = [(True, False, 2, 0.0, 1, (np.nan,) * 3, 800, 4e5, 1e5, 0.0),
            (True, True, 3, 0.1, 1, (1e-6, 1e-6, 1e-8), 200, 2e5, 0.0, 0.0),
            (True, False, 2, 0.2, 1, (np.nan,) * 3, 200, 4e5, 0.0, 0.0),
            (True, True, 3, 0.3, 1, (1e-6, 1e-6, 1e-8), 50, 2e5, 0.0, 0.0)]
    return np.array(rows, dtype=dt)


def test_a_3d_localization_takes_its_own_lateral_iterations_photons():
    seq = _seq(("square", 0.42), ("zline2", 0.42), ("square", 0.21), ("zline2", 0.21))
    locs, _ = minflux.table(_records_3d(), dimensionality=3, z_scale=0.7, sequence=seq)
    L = 0.21 * minflux.PATTERN_UNIT_NM
    assert np.allclose(locs["photons"], [200, 50])            # the final's, as before
    assert np.allclose(locs["xy_err_nm"], [minflux.crlb_xy(L, 600, 200),
                                           minflux.crlb_xy(L, 200)], rtol=1e-5)
    assert np.allclose(locs["z_err_nm"], minflux.crlb_z(L * 0.7, [200, 50]), rtol=1e-5)


def test_m2205_takes_the_lateral_photons_from_their_column_of_the_nested_record():
    n = 3
    sub = np.zeros((n, 2), dtype=[("loc", "f8", (3,)), ("eco", "f8"), ("efo", "f8"),
                                  ("fbg", "f8"), ("cfr", "f8")])
    sub["loc"][:, -1] = 1e-6
    sub["eco"][:, 0], sub["eco"][:, 1] = [100, 200, 400], 50
    rec = np.zeros(n, dtype=[("itr", sub.dtype, (2,)), ("tim", "f8"), ("tid", "i8"), ("vld", "?")])
    rec["itr"], rec["tim"], rec["vld"] = sub, np.arange(n), True
    locs, _ = minflux.table(rec, dimensionality=3, sequence=_seq(("square", 0.28), ("zline", 0.28)))
    assert np.allclose(locs["xy_err_nm"], minflux.crlb_xy(100.8, [100, 200, 400]), rtol=1e-5)


def test_without_a_sequence_the_default_pattern_size_is_used_and_z_has_no_bound():
    locs, _ = minflux.table(_records_3d(), dimensionality=3)
    assert np.allclose(locs["xy_err_nm"], minflux.crlb_xy(minflux.DEFAULT_PATTERN_NM, [200, 50]))
    assert "z_err_nm" not in locs


@pytest.mark.skipif(not os.environ.get("SMAPPY_TEST_MSR"),
                    reason="SMAPPY_TEST_MSR=dir of real .msr files")
def test_real_measurements_get_a_bound_of_a_few_nanometres():
    from smappy.io import msr
    paths = sorted(Path(os.environ["SMAPPY_TEST_MSR"]).glob("*.msr"))
    assert paths
    for path in paths:
        for m in msr.contents(path).measurements:
            locs, meta = msr.read_measurement(path, m)
            sizes = minflux.pattern_sizes(meta["sequence"])
            assert "xy" in sizes, (path.name, m.label)
            median = float(np.median(locs["xy_err_nm"]))
            assert 0.2 < median < 20, (path.name, m.label, median)
