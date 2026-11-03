"""The excitation beam (`model.excitation`): its focal field against Leutenegger's library, and its physics."""
import json
import math
import os

import numpy as np
import pytest
import torch

from uipsf.model import MaskSpec, ZernikePupil, make_mask
from uipsf.model.excitation import BeamSpec, ExcitationBeam
from uipsf.optics import Optics
from uipsf.optics.grid import PupilGrid
from uipsf.optics.vectorial import _vector_fields
from uipsf.params import Params

ORACLE = np.load(os.path.join(os.path.dirname(__file__), "fixtures", "oracle_focus.npz"))
OIL = Optics(NA=1.4, wavelength_nm=640.0, n_imm=1.518, n_med=1.518, pixel_size_nm=(25.0, 25.0))


def make_beam(optics=OIL, P=64, mask=None, **beam):
    grid = PupilGrid.build(optics, P, device="cpu", dtype=torch.float64)
    m = make_mask(mask, grid, aperture_na=optics.NA) if mask is not None else None
    b = ExcitationBeam(grid, ZernikePupil(grid, n_max=4, magnitude_n_max=0, mask=m), BeamSpec(**beam))
    return b, Params(b.specs(), dtype=torch.float64).physical()


def oracle_case(name, weighting, P=128):
    """(the library's E, uipsf's E), each [3, X, Y, Z] as the library lays it out, for one case of the fixture.
    ``weighting`` "library" gives uipsf's fields the library's radial weighting, 1/cos θ_imm and its lens
    (`scripts/uipsf/make_oracles_focus.py`), "uipsf" leaves its own.  The library's E_z is multiplied by n_m
    (`test_the_library_underweights_E_z_by_the_immersions_index`)."""
    s = json.loads(str(ORACLE[f"{name}.settings"]))
    optics = Optics(NA=s["NA"], wavelength_nm=s["lo"] * 1e9, n_imm=s["nm"], n_med=s["ns"], pixel_size_nm=(25, 25))
    ms = s["maskshift"]
    # the library's pupil x is uipsf's −u; its vortex shifts x by maskshift[1], its top-hat by maskshift[0]
    mask = (MaskSpec(kind="vortex", centre=(ms[1], -ms[0])) if "phaseramp2" in s["Ei"]
            else MaskSpec(kind="top-hat", centre=(ms[0], -ms[1])) if "pishift" in s["Ei"] else None)
    fill = s["wa"] / (s["Da"] / 2) if "gauss" in s["Ei"] else None
    beam, p = make_beam(optics, P, mask, polarisation="x" if "linear" in s["Ei"] else "circular", fill=fill)
    if weighting == "library":
        g = beam.grid
        sin_imm = (g.rho * s["NA"] / s["nm"]).clamp(max=0.99).numpy()
        tp, ts = (torch.as_tensor(np.interp(sin_imm, ORACLE["lens.sin"], ORACLE[f"lens.{k}"])) for k in ("tp", "ts"))
        apod = g.aperture * g.apodisation / torch.sqrt(g.cos_imm)
        sin_med = (g.rho * s["NA"] / s["ns"]).to(g.kz_med.dtype)
        beam.hx, beam.hy = (apod * _vector_fields(g, g.fresnel_p * tp, g.fresnel_s * ts, g.cos_med, sin_med, a)
                            for a in "xy")
    z, n = ORACLE[f"{name}.z"], len(ORACLE[f"{name}.x"])
    if s["ns"] != s["nm"]:
        E = beam.field(p, (n, n), (25, 25), np.zeros_like(z), depth_nm=z)
    else:
        E = beam.field(p, (n, n), (25, 25), z)
    L = ORACLE[f"{name}.E"].copy()
    L[2] *= s["nm"]
    return L, E.numpy().transpose(1, 3, 2, 0)


def shape_difference(L, U):
    """(intensity, field): the largest difference of the intensities, each scaled to its peak in focus, and of
    the complex fields after one complex factor, each as a fraction of the peak."""
    IL, IU = (np.abs(L) ** 2).sum(0), (np.abs(U) ** 2).sum(0)
    k = IL.shape[-1] // 2
    dI = np.abs(IL / IL[..., k].max() - IU / IU[..., k].max()).max()
    a = (np.conj(U) * L).sum() / (np.abs(U) ** 2).sum()
    return dI, np.abs(L - a * U).max() / np.abs(L).max()


@pytest.mark.parametrize("name, field_tol", [("gauss_x", 0.005), ("gauss_circular", 0.005), ("vortex", 0.005),
                                             ("vortex_shifted", 0.005), ("tophat", 0.012)])
def test_the_focal_field_is_leuteneggers_when_weighted_as_his(name, field_tol):
    # what is left at 128 pupil samples is sampling: a tenth of it at 256 (the top-hat's sharp step, a third)
    dI, dE = shape_difference(*oracle_case(name, "library"))
    assert dE < field_tol


def test_the_focal_field_in_water_two_microns_deep_is_leuteneggers_to_sampling():
    # NA 1.35 into water: the supercritical band is evanescent and the depth's phase steep; 4 % at 128 samples,
    # 2.4 % at 256
    dI, dE = shape_difference(*oracle_case("depth_water", "library"))
    assert dE < 0.06 and dI < 0.06


def test_the_library_underweights_E_z_by_the_immersions_index():
    """The library writes E_z = t_p sin θ_imm / n_s where the p wave's z component is t_p sin θ_s =
    t_p n_m sin θ_imm / n_s: its |E_z|² is n_m² too small, 2.3 in oil (MINFLUXexcitation's MATLAB, SimuFLUX's
    Python).  uipsf's agrees with it once its E_z is multiplied by n_m (the oracle cases above)."""
    L, U = oracle_case("gauss_x", "library")
    L[2] /= 1.518                                     # undo the correction
    k = L.shape[-1] // 2
    ratio = (np.abs(U[2, ..., k]).max() / np.abs(U[0, ..., k]).max()) / (
        np.abs(L[2, ..., k]).max() / np.abs(L[0, ..., k]).max())
    assert ratio == pytest.approx(1.518, rel=0.01)


def test_uipsfs_aplanatic_weighting_differs_from_the_librarys_by_a_percent_and_fills_a_top_hats_zero_more():
    """The library weights the pupil by 1/cos θ_imm and an air-lens-air transmission, uipsf by the aplanatic
    √cos θ_imm (detection's, by reciprocity).  For Gaussian beams that is about a per cent of the peak; at a
    top-hat's zero, a balance of the inner and the outer pupil, it is 0.12 of the peak against 0.06.  The
    objective's own transmission is learnt (a minimal pupil's γ, a Zernike magnitude), so this is the start."""
    dI, _ = shape_difference(*oracle_case("gauss_circular", "uipsf"))
    assert 0.004 < dI < 0.02
    L, U = oracle_case("tophat", "uipsf")
    IL, IU = (np.abs(L) ** 2).sum(0), (np.abs(U) ** 2).sum(0)
    mid = IL.shape[0] // 2
    zero_L, zero_U = IL[mid, mid, 2] / IL[..., 2].max(), IU[mid, mid, 2] / IU[..., 2].max()
    assert zero_L == pytest.approx(0.058, abs=0.01) and zero_U == pytest.approx(0.124, abs=0.01)


def test_a_flat_x_polarised_beam_integrates_to_one_over_the_focal_plane():
    # over 18 µm: a hard-edged aperture's Airy rings carry 4.5 % of the power beyond 1.6 µm, 0.5 % beyond 9
    beam, p = make_beam(P=128, polarisation="x", fill=None)
    I = beam.intensity(p, (301, 301), (60.0, 60.0), [0.0])
    assert float(I.sum()) * 60.0 ** 2 == pytest.approx(1.0, abs=0.01)


def test_a_vortex_keeps_the_beams_power_so_a_donut_and_a_gaussian_keep_their_relative_heights():
    gauss, pg = make_beam(P=128)
    donut, pd = make_beam(P=128, mask=MaskSpec(kind="vortex"))
    Ig = gauss.intensity(pg, (151, 151), (40.0, 40.0), [0.0])
    Id = donut.intensity(pd, (151, 151), (40.0, 40.0), [0.0])
    assert float(Id.sum()) == pytest.approx(float(Ig.sum()), rel=0.02)
    assert float(Id.max() / Ig.max()) == pytest.approx(0.285, abs=0.01)    # the crest against the peak


def test_a_vortex_has_its_zero_only_with_the_circular_polarisation_of_the_opposite_handedness():
    """exp(iφ) with (1, i)/√2: the transverse parts cancel on the axis and E_z's does too, the zero is dark;
    with (1, −i)/√2 E_z fills it -- why MINFLUX's λ/4 must be set right (scanned-psfs.md §3)."""
    right, pr = make_beam(mask=MaskSpec(kind="vortex"), handedness=1)
    wrong, pw = make_beam(mask=MaskSpec(kind="vortex"), handedness=-1)
    Ir = right.intensity(pr, (21, 21), (20.0, 20.0), [0.0])[0]
    Iw = wrong.intensity(pw, (21, 21), (20.0, 20.0), [0.0])[0]
    assert float(Ir[10, 10] / Ir.max()) < 1e-4
    assert float(Iw[10, 10] / Iw.max()) > 0.1


def test_a_radially_polarised_beam_is_all_E_z_on_the_axis():
    beam, p = make_beam(polarisation="radial", fill=None)
    E = beam.field(p, (21, 21), (20.0, 20.0), [0.0])[0]
    assert float(E[:2, 10, 10].abs().max()) < 1e-3 * float(E[2, 10, 10].abs())
    assert int((E[2].abs() ** 2).argmax()) == 10 * 21 + 10


def test_linear_polarisation_at_ninety_degrees_is_y():
    lin, pl = make_beam(polarisation="linear", angle=math.pi / 2)
    y, py = make_beam(polarisation="y")
    El = lin.field(pl, (11, 11), (30.0, 30.0), [0.0, 200.0])
    Ey = y.field(py, (11, 11), (30.0, 30.0), [0.0, 200.0])
    assert torch.allclose(El, Ey, atol=1e-9)


def test_a_dye_is_the_sum_of_the_three_fixed_dipoles():
    beam, p = make_beam(mask=MaskSpec(kind="top-hat"))
    args = ((15, 15), (30.0, 30.0), [-200.0, 0.0])
    free = beam.intensity(p, *args)
    fixed = sum(beam.intensity(p, *args, dipole=d) for d in torch.eye(3, dtype=torch.float64))
    assert torch.allclose(free, fixed, rtol=1e-10, atol=1e-14)


def test_an_aberration_free_beam_in_a_matched_medium_is_symmetric_about_the_focus():
    # linearly polarised: exactly.  A circular beam's −z is the other handedness's +z, its mirror image, which a
    # square pupil grid reproduces to 4e-4 of the peak
    beam, p = make_beam(polarisation="x")
    I = beam.intensity(p, (15, 15), (40.0, 40.0), [-300.0, 300.0])
    assert torch.allclose(I[0], I[1], rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize("name", ["beam.centre", "beam.fill", "beam.ellipticity", "beam.angle"])
def test_the_beams_parameters_have_derivatives(name):
    beam, p = make_beam(P=32, polarisation="elliptical", ellipticity=0.3, angle=0.2, fill=1.0, centre=(0.05, -0.1))

    def f(v):
        q = dict(p, **{name: v})
        return beam.intensity(q, (7, 7), (60.0, 60.0), [0.0, 250.0])

    assert torch.autograd.gradcheck(f, (p[name].clone().requires_grad_(True),), eps=1e-6, atol=1e-6)


def test_beam_parameters_are_held_unless_freed_and_refused_where_they_mean_nothing():
    beam, _ = make_beam(free=("centre", "ellipticity"))
    held = {s.name: s.fixed for s in beam.specs() if s.name.startswith("beam.")}
    assert held == {"beam.angle": True, "beam.ellipticity": False, "beam.fill": True, "beam.centre": False}
    with pytest.raises(ValueError):
        BeamSpec(polarisation="radial", free=("angle",))
    with pytest.raises(ValueError):
        BeamSpec(fill=None, free=("centre",))
    with pytest.raises(ValueError):
        BeamSpec(free=("tilt",))
