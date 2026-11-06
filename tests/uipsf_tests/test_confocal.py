"""A confocal raster scan (`model.confocal`): the excitation times the detection through a pinhole -- its limits, the
shared optical path, the pinhole's offset, and fitting."""
import pytest
import torch

from uipsf.model import (
    BeamSpec,
    Bleaching,
    ConfocalPSF,
    ExcitationBeam,
    ForwardModel,
    MaskSpec,
    ScannedBeamPSF,
    VectorPSF,
    ZernikePupil,
    make_mask,
)
from uipsf.optics import Optics
from uipsf.optics.grid import PupilGrid
from uipsf.params import Params

STEP = 25.0
OIL = Optics(NA=1.4, wavelength_nm=640.0, n_imm=1.518, n_med=1.518, pixel_size_nm=(STEP, STEP))
EMISSION = 680.0


def make(pinhole_au=1.0, roi=(25, 25), P=48, mask=None, aberrations=None, polarisation="circular", **kw):
    grid = PupilGrid.build(OIL, P, device="cpu", dtype=torch.float64)
    m = make_mask(mask, grid, aperture_na=OIL.NA) if mask is not None else None
    beam = ExcitationBeam(grid, ZernikePupil(grid, n_max=4, magnitude_n_max=0, mask=m),
                          BeamSpec(polarisation=polarisation))
    psf = ConfocalPSF(beam, roi, emission_nm=EMISSION, pinhole_au=pinhole_au, **kw)
    p = Params(psf.specs(), dtype=torch.float64).physical()
    for j, c in (aberrations or {}).items():
        p["pupil.phase"][psf.pupil.phase_modes.index(j)] = c
    return psf, p


def at(*xyz):
    return torch.tensor([[list(v) for v in xyz]], dtype=torch.float64)


def unblurred(p):
    return dict(p, **{"psf.sigma": torch.zeros(2, dtype=torch.float64)})


def test_a_pinhole_wide_open_gives_the_open_detectors_scan():
    psf, p = make(pinhole_au=20.0, roi=(21, 21))
    conf = psf(p, at((20.0, -10.0, 0.0)))
    open_ = ScannedBeamPSF(psf.beam, (21, 21))(unblurred(p), at((20.0, -10.0, 0.0)))
    assert float((conf - open_).abs().max() / open_.max()) < 0.015


def test_a_point_pinhole_gives_the_product_of_excitation_and_emission():
    """R → 0: D = π R² I_em(r), the classic confocal PSF I_exc·I_em, up to the constant."""
    psf, p = make(pinhole_au=0.02, aberrations={6: 0.3, 7: 0.2})
    xyz = at((30.0, 10.0, 80.0))
    conf = psf(p, xyz)[0, 0]
    exc = ScannedBeamPSF(psf.beam, psf.roi_shape)(unblurred(p), xyz)[0, 0]
    em = psf.detection.resized(psf.roi_shape)(unblurred(p), xyz)[0, 0]
    product = exc * em
    assert torch.allclose(conf / conf.max(), product / product.max(), atol=2e-3)


def test_a_one_airy_unit_pinhole_passes_three_quarters_of_what_an_emitter_on_the_axis_sends():
    # a scalar Airy disc holds 84 % inside its first dark ring; at NA 1.4 the vectorial PSF's is wider
    psf, p = make(pinhole_au=1.0, roi=(5, 5))
    assert float(psf.pinhole(p, at((0.0, 0.0, 0.0)))[0, 0, 2, 2]) == pytest.approx(0.76, abs=0.02)


def test_the_pinhole_integral_is_the_emission_psf_summed_over_the_disc():
    """D(r_s) = ∫_disc I_em(r_s + r_ph + ρ − r_e) d²ρ by quadrature on a 10 nm lattice inside the disc, against
    the FFT with the jinc (`ConfocalPSF.pinhole`)."""
    psf, p = make(pinhole_au=0.6, aberrations={5: 0.3, 8: 0.2}, free=("offset",))
    p["pinhole.offset"] = torch.tensor([40.0, -20.0], dtype=torch.float64)
    R = psf.radius_nm
    s = torch.arange(-R, R + 1e-9, 10.0, dtype=torch.float64)
    x, y = torch.meshgrid(s, s, indexing="ij")
    inside = x ** 2 + y ** 2 <= R ** 2
    rho = torch.stack([x[inside], y[inside]], -1)
    e = torch.tensor([30.0, 10.0, 50.0], dtype=torch.float64)
    # the emitter seen from a pinhole point ρ off its centre: I_em at r_s − (r_e − r_ph − ρ)
    xyz = torch.cat([e[:2] - p["pinhole.offset"] - rho, e[2:].expand(len(rho), 1)], -1)[:, None]
    roi = psf.roi_shape
    em = psf.detection.resized(roi)(unblurred(p), xyz)[:, 0]          # per scan position: density × step²
    brute = em.sum(0) * 10.0 ** 2 / STEP ** 2
    D = psf.pinhole(p, e[None, None])[0, 0]
    assert float((D - brute).abs().max() / D.max()) < 0.01


def test_the_detection_sees_the_objectives_path_at_its_own_wavelength_and_not_the_mask():
    psf, p = make(mask=MaskSpec(kind="vortex"), aberrations={5: 0.4, 8: -0.3})
    grid = PupilGrid.build(OIL.__class__(**{**OIL.__dict__, "wavelength_nm": EMISSION}), 48, device="cpu",
                           dtype=torch.float64)
    pupil = ZernikePupil(grid, n_max=4, magnitude_n_max=0)
    alone = VectorPSF(grid, pupil, psf.detection.roi_shape, fit_sigma=False)
    q = unblurred(Params(alone.specs(), dtype=torch.float64).physical())
    q["pupil.phase"] = p["pupil.phase"] * 640.0 / EMISSION
    xyz = at((15.0, -25.0, 60.0))
    assert torch.allclose(psf.detection(unblurred(p), xyz), alone(q, xyz), rtol=1e-10, atol=1e-15)


def test_a_pinhole_off_the_axis_pulls_the_scan_the_other_way_by_about_half():
    """D peaks where the pinhole sees the emitter, r_s = r_e − r_ph, the excitation at r_s = r_e: with nearly
    equal widths the product lies about halfway, the other way from the offset."""
    psf, p = make(pinhole_au=0.2, roi=(41, 41), free=("offset",))
    p["pinhole.offset"] = torch.tensor([100.0, 0.0], dtype=torch.float64)
    mu = psf(p, at((0.0, 0.0, 0.0)))[0, 0]
    x = (torch.arange(41, dtype=torch.float64) - 20) * STEP
    assert float((mu.sum(0) * x).sum() / mu.sum()) == pytest.approx(-47.0, abs=4.0)


def test_a_bead_is_the_product_averaged_over_the_spheres_projection():
    R = 50.0
    psf, p = make(bead_radius_nm=R)
    point, _ = make()
    s = torch.arange(-R + 5, R, 10.0, dtype=torch.float64)
    x, y, z = torch.meshgrid(s, s, s, indexing="ij")
    inside = x ** 2 + y ** 2 + z ** 2 <= R ** 2
    xyz = torch.stack([x[inside], y[inside], torch.zeros_like(x[inside])], -1)[:, None]
    brute = point(p, xyz)[:, 0].mean(0)
    bead = psf(p, at((0.0, 0.0, 0.0)))[0, 0]
    assert float((bead - brute).abs().max() / bead.max()) < 0.01


def test_gold_and_unknown_pinhole_parameters_are_refused():
    with pytest.raises(ValueError):
        make(sample="gold")
    with pytest.raises(ValueError):
        make(free=("diameter",))


@pytest.mark.slow
def test_a_confocal_scan_is_fitted_through_the_whole_model():
    """The solver's pupil tangents assume one pupil field: a confocal model says it is not linearisable and gets
    JVPs through the product.  One plane of circular light of known handedness determines the even modes'
    signs (scanned-psfs.md §7a), and the pinhole's offset comes back."""
    from uipsf import Batch
    from uipsf.simulate import parameters, simulate, template
    from uipsf.workflows.beads import fit_beads
    truth = {5: 0.15, 6: 0.25, 7: 0.15, 8: -0.1}
    roi = (21, 21)
    psf, _ = make(roi=roi, P=32, free=("offset",))
    fm = ForwardModel(psf, intensity=Bleaching(order=0))
    assert not fm.linearisable
    n = 6
    g = torch.Generator().manual_seed(5)
    xyz = (torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5) * torch.tensor([60.0, 60.0, 80.0])
    b0 = template(n, Batch.z_from_objective(torch.tensor([0.0])), roi, dtype=torch.float64, device="cpu")

    def start(ab, offset):
        phase = torch.tensor([ab.get(j, 0.0) for j in psf.pupil.phase_modes], dtype=torch.float64)
        return parameters(fm, b0, {"pupil.phase": phase, "pinhole.offset": torch.tensor(offset, dtype=torch.float64),
                                   "emitter.xyz": xyz, "emitter.photons": torch.full((n,), 1e5),
                                   "emitter.background": torch.full((n,), 1.0)}, dtype=torch.float64, device="cpu")
    data = simulate(fm, start(truth, (60.0, -40.0)), b0, seed=6)
    res = fit_beads(fm, data, start=start({}, (0.0, 0.0)), first_order=None, max_dz_nm=None)
    q = res.params.physical()
    phase = dict(zip(psf.pupil.phase_modes, q["pupil.phase"].tolist(), strict=True))
    for j, c in truth.items():
        assert phase[j] == pytest.approx(c, abs=0.04)
    assert q["pinhole.offset"].tolist() == pytest.approx([60.0, -40.0], abs=15.0)
