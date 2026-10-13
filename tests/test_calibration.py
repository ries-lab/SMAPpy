"""Axis-convention tests for the spline calibration reader.

These need a real SMAP _3Dcal.mat; set SMAPPY_TEST_CAL to point at one.
"""
import os
import numpy as np
import pytest

from smappy.io.calibration import (
    _smap_flipped, _to_scalar, compute_delta3d, evaluate_spline, load_spline_calibration,
)

CAL = os.environ.get("SMAPPY_TEST_CAL")
needs_cal = pytest.mark.skipif(not CAL, reason="set SMAPPY_TEST_CAL")


def test_compute_delta3d_ordering():
    d = compute_delta3d(2.0, 3.0, 5.0)
    for pz in range(4):
        for py in range(4):
            for px in range(4):
                assert d[16 * pz + 4 * py + px] == pytest.approx(5.0**pz * 3.0**py * 2.0**px)
    assert compute_delta3d(0.0, 0.0, 0.0)[0] == 1.0
    assert compute_delta3d(0.0, 0.0, 0.0)[1:].sum() == 0.0


@needs_cal
def test_coefficients_match_bead_stack():
    """coeff[0] is the model value at a knot, so it must match the PSF stack."""
    cal = load_spline_calibration(CAL)
    assert cal.psf is not None
    nz, ny, nx = cal.coeff.shape[1:]
    # converted from mirrored beads, an odd grid gains a cell in x: compare
    # the square part, so that the transposed control has the same shape
    n = min(ny, nx)
    c0 = cal.coeff[0, :, :n, :n]
    ref = cal.psf[1:1 + nz, 1:1 + n, 1:1 + n]

    def corr(b):
        gb = np.isfinite(b)
        return np.corrcoef(c0[gb], b[gb])[0, 1]

    direct = corr(ref)
    assert direct > 0.99
    # wrong axis conventions must be clearly worse
    assert direct > corr(np.ascontiguousarray(ref.transpose(0, 2, 1))) + 0.1
    assert direct > corr(np.ascontiguousarray(ref[:, :, ::-1])) + 0.1
    assert direct > corr(np.ascontiguousarray(ref[::-1])) + 0.1


@needs_cal
def test_evaluate_spline_reproduces_coefficients_at_knot():
    cal = load_spline_calibration(CAL)
    _, nz, ny, nx = cal.coeff.shape
    r, z = 13, int(round(cal.z0))
    model = evaluate_spline(cal, r / 2 - 0.5, r / 2 - 0.5, float(z), r)
    ox = int(np.floor((nx + 1 - r) / 2))
    oy = int(np.floor((ny + 1 - r) / 2))
    assert np.allclose(model, cal.coeff[0, z, oy:oy + r, ox:ox + r])


@needs_cal
def test_smaps_mirror_flag_and_its_record_of_the_flip_agree():
    """OWNER: the reading of ``emmirror`` in `_smap_flipped` comes from SMAP's
    code and was never checked on a real file.  It takes ``SXY.EMon`` as
    SMAP's record that ``readbeadimages`` flipped the beads (its ``isem``),
    and the popup ``emmirror`` (1 from metadata, 2 no mirror, 3 mirrored) as
    what was asked.  On a file from SMAP's bead calibrator the two agree: 0
    or 2 never flipped, 3 always did.  If this fails, or a converted file
    fails `test_coefficients_match_bead_stack` above, the reading is wrong."""
    cal = load_spline_calibration(CAL)
    emmirror = _to_scalar(cal.parameters.get("emmirror"))
    print(f"emmirror={emmirror} EMon={cal.em_on}: "
          f"{cal.parameters.get('smappy_conversion', 'not converted')}")
    if emmirror is None or cal.em_on is None:
        pytest.skip("this file records no emmirror or no EMon")
    if int(emmirror) in (0, 2):
        assert cal.em_on is False
    if int(emmirror) == 3:
        assert cal.em_on is True
    assert _smap_flipped(emmirror, cal.em_on, CAL) is cal.em_on
    assert ("smappy_conversion" in cal.parameters) is cal.em_on
