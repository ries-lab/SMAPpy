"""*Stability*'s random scatter split into the noise that averages away
and a floor (`stability.noise_floor`)."""
import numpy as np
import pytest

from smappy import stability as st


def scatter_with(eta, floor, precision=3.0, drift_from=None):
    """A Scatter whose Allan deviation is (eta p)^2/m + floor^2, rising as
    drift after ``drift_from`` points."""
    m = np.asarray(st.averaging_points(1000), np.int64)
    var = (eta * precision) ** 2 / m + floor ** 2
    if drift_from is not None:
        var = var + np.where(m > drift_from, (0.02 * (m - drift_from)) ** 2, 0.0)
    count = np.full(len(m), 1000, np.int64)
    sq = 2 * count * var                        # allan_nm = sqrt(sq / (2 count))
    return st.Scatter("x", precision, 1e-3, m, sq, count, np.zeros(10))


def test_white_noise_at_the_precision_has_no_floor():
    eta, floor = st.noise_floor(scatter_with(1.0, 0.0))
    assert eta == pytest.approx(1.0, rel=1e-3) and floor == pytest.approx(0.0, abs=1e-3)


def test_a_floor_in_quadrature_comes_back_and_the_drift_after_the_minimum_is_left_out():
    eta, floor = st.noise_floor(scatter_with(1.3, 0.8, drift_from=100))
    assert eta == pytest.approx(1.3, rel=0.05) and floor == pytest.approx(0.8, rel=0.1)


def test_real_white_noise_gives_eta_one():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 3, 40000)
    m = np.asarray(st.averaging_points(1000), np.int64)
    sq, count = st.allan_sums(x, m)
    eta, floor = st.noise_floor(st.Scatter("x", 3.0, 1e-3, m, sq, count, np.diff(x)))
    assert eta == pytest.approx(1.0, rel=0.05) and floor < 0.3
