"""Where an astigmatic spline is narrowest in x and in y, and how far apart (`calibrate.focal_planes`)."""
import numpy as np
import pytest

from smappy.calibrate.core import spline_coefficients
from smappy.calibrate.focal_planes import (
    focal_planes,
    gaussian_sigma,
    of_calibration,
    summary_text,
)
from smappy.io.calibration import SplineCalibration

NZ, BOX = 61, 17


def astigmatic(zx_nm: float, zy_nm: float, dz_nm: float = 25.0, depth_nm: float = 500.0) -> SplineCalibration:
    """Gaussians narrowest in x at ``zx_nm`` and in y at ``zy_nm`` (the spline's z), each widening as
    sqrt(1 + (dz / d)^2) about its own focus."""
    z0 = NZ // 2
    z = (-(np.arange(NZ) - z0) * dz_nm)[:, None, None]          # SplineCalibration.z_index_to_nm
    y, x = np.mgrid[:BOX, :BOX] - (BOX - 1) / 2
    sx = 1.2 * np.sqrt(1 + ((z - zx_nm) / depth_nm) ** 2)
    sy = 1.2 * np.sqrt(1 + ((z - zy_nm) / depth_nm) ** 2)
    psf = np.exp(-x ** 2 / (2 * sx ** 2) - y ** 2 / (2 * sy ** 2)) / (2 * np.pi * sx * sy)
    return SplineCalibration(spline_coefficients(psf.astype(np.float32)), dz_nm, z0, x0=(BOX - 1) / 2)


def test_a_gaussian_fit_with_an_offset_finds_the_width_under_a_pedestal():
    i = np.arange(15.0)
    assert gaussian_sigma(3 * np.exp(-0.5 * ((i - 7.3) / 1.7) ** 2) + 0.4) == pytest.approx(1.7, rel=1e-3)


@pytest.mark.parametrize("zx, zy", [(-230.0, 190.0), (140.0, -310.0), (35.0, 35.0)])
def test_the_focal_lines_are_found_between_the_splines_planes(zx, zy):
    found = focal_planes(astigmatic(zx, zy))
    assert found.x_nm == pytest.approx(zx, abs=3.0)
    assert found.y_nm == pytest.approx(zy, abs=3.0)
    assert found.separation_nm == pytest.approx(abs(zy - zx), abs=5.0)
    assert found.sigma_x_px == pytest.approx(1.2, rel=0.05)


def test_a_focus_beyond_the_spline_is_no_focal_plane():
    found = focal_planes(astigmatic(-100.0, 2000.0))
    assert found.x_nm == pytest.approx(-100.0, abs=3.0)
    assert found.y_nm is None and found.separation_nm is None
    assert "not found" in found.text()


def test_the_text_names_both_lines_and_their_distance():
    text = summary_text(of_calibration(astigmatic(-200.0, 200.0)))
    assert text.startswith("focal separation 400 nm") and "x narrowest at z -200 nm" in text
