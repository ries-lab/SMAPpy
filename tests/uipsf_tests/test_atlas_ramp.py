"""The residual atlas's background-ramp check on maps that cannot fix a plane (fast; the atlas's fits are in
test_atlas.py's slow tier)."""
import numpy as np
import pytest

from uipsf.diagnostics.atlas import ramp_z2


def test_a_ramp_is_measured_on_the_maps_that_fix_a_plane_and_the_others_are_passed_over():
    Y = X = 9
    gy, gx = np.mgrid[:Y, :X]
    ramp = 0.5 * (gx - 4.0)                     # a slope in x, none in y
    full = np.full((Y, X), 4)
    row = np.zeros((Y, X), int)
    row[4] = 4                                  # pixels on one row only: no slope in y to fit
    maps = np.stack([ramp, ramp, ramp])
    counts = np.stack([full, np.zeros((Y, X), int), row])   # whole; masked whole, as a partner's region; a row
    z2 = ramp_z2(maps, counts)
    assert len(z2) == 2                          # the first map's x and y; the other two passed over
    assert z2[0] == pytest.approx(0.25 * 4 * (Y * ((gx[0] - 4.0) ** 2).sum()))   # slope²/var, var = 1/Σw·x²
    assert z2[1] == pytest.approx(0.0, abs=1e-12)
    assert ramp_z2(maps[1:], counts[1:]) == []
