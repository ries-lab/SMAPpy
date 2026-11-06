"""A beam scanned over an emitter behind an open detector (`model.scanned`): the scan image, what the sample makes
of the beam's components, a bead's size, the twin in one plane, and fitting."""
import math

import pytest
import torch

from uipsf.model import (
    BeamSpec,
    Bleaching,
    ExcitationBeam,
    ForwardModel,
    MaskSpec,
    ScannedBeamPSF,
    ZernikePupil,
    make_mask,
    sample_weights,
)
from uipsf.model.scanned import collection
from uipsf.optics import Optics
from uipsf.optics.grid import PupilGrid
from uipsf.params import Params

STEP = 25.0
OIL = Optics(NA=1.4, wavelength_nm=640.0, n_imm=1.518, n_med=1.518, pixel_size_nm=(STEP, STEP))


def make_scan(mask=None, roi=(25, 25), P=48, aberrations=None, sample="dye", bead_radius_nm=0.0, **beam):
    grid = PupilGrid.build(OIL, P, device="cpu", dtype=torch.float64)
    m = make_mask(mask, grid, aperture_na=OIL.NA) if mask is not None else None
    b = ExcitationBeam(grid, ZernikePupil(grid, n_max=4, magnitude_n_max=0, mask=m), BeamSpec(**beam))
    psf = ScannedBeamPSF(b, roi, sample=sample, bead_radius_nm=bead_radius_nm)
    p = Params(psf.specs(), dtype=torch.float64).physical()
    for j, c in (aberrations or {}).items():
        p["pupil.phase"][psf.pupil.phase_modes.index(j)] = c
    return psf, p


def at(*xyz):
    return torch.tensor([[list(v) for v in xyz]], dtype=torch.float64)


def test_the_scan_is_the_beams_intensity_point_reflected_about_the_emitter():
    # coma and a decentred vortex make the beam odd, so a scan that were not mirrored would show it the other way
    psf, p = make_scan(MaskSpec(kind="vortex", centre=(0.08, -0.03)), aberrations={7: 0.4, 9: 0.2})
    scan = psf(p, at((60.0, -40.0, 50.0)))[0, 0]
    # the beam's intensity at r_e − r_s: evaluated on the scan's grid moved by −r_e, then flipped through its centre
    I = psf.beam.intensity(p, psf.roi_shape, (STEP, STEP), [50.0], offset_nm=(-60.0, 40.0))[0].flip(-2, -1)
    assert torch.allclose(scan, I * STEP ** 2, rtol=1e-10, atol=1e-14)


def test_a_flat_beam_scanned_over_the_whole_focal_plane_gives_the_emitter_one_photon():
    psf, p = make_scan(roi=(201, 201), P=96, fill=None, polarisation="x")
    psf.dft = type(psf.dft)(psf.grid, psf.roi_shape, (60.0, 60.0))
    psf.spacing = (60.0, 60.0)
    psf.norm = 60.0 ** 2 / (psf.energy * psf.dft.scale)
    assert float(psf(p, at((0.0, 0.0, 0.0))).sum()) == pytest.approx(1.0, abs=0.02)


def test_the_sample_weights_put_E_z_below_the_in_plane_components_for_gold_and_beads():
    grid = PupilGrid.build(OIL, 64, device="cpu", dtype=torch.float64)
    assert torch.equal(sample_weights("dye"), torch.ones(3, dtype=torch.float64))
    gold, bead = sample_weights("gold", grid), sample_weights("bead", grid)
    assert gold[1] == pytest.approx(1.0) and bead[1] == pytest.approx(1.0)
    assert float(gold[2]) == pytest.approx(0.65, abs=0.01) and float(bead[2]) == pytest.approx(0.85, abs=0.01)
    with pytest.raises(ValueError):
        sample_weights("quantum dot", grid)


def test_a_fluorescent_beads_weights_are_its_fixed_random_dyes_averaged_over_the_sphere():
    """Each dye μ absorbs |μ·E|² and the open detector collects Σ_k μ_k² η_k of what it emits; the average over
    orientations is Σ_d w_d |E_d|² / 15 with w_d = Σ η + 2 η_d (`sample_weights`)."""
    grid = PupilGrid.build(OIL, 64, device="cpu", dtype=torch.float64)
    eta = collection(grid).to(torch.float64)
    n = 20000                                                      # a Fibonacci sphere
    i = torch.arange(n, dtype=torch.float64) + 0.5
    cos_t, phi = 1 - 2 * i / n, math.pi * (1 + 5 ** 0.5) * i
    sin_t = torch.sqrt(1 - cos_t ** 2)
    mu = torch.stack([sin_t * torch.cos(phi), sin_t * torch.sin(phi), cos_t], -1)
    E = torch.tensor([0.7 + 0.2j, -0.3 + 0.5j, 0.4 - 0.1j], dtype=torch.complex128)
    signal = ((mu.to(E.dtype) @ E).abs() ** 2 * (mu ** 2 @ eta)).mean()
    w = eta.sum() + 2 * eta
    assert float(signal) == pytest.approx(float((w * E.abs() ** 2).sum() / 15), rel=1e-4)


def test_a_vortexs_zero_is_filled_by_the_wrong_handedness_through_E_z_less_for_gold_than_for_a_dye():
    def zero(sample, handedness):
        psf, p = make_scan(MaskSpec(kind="vortex"), roi=(5, 5), sample=sample, handedness=handedness)
        I = psf(p, at((0.0, 0.0, 0.0)))[0, 0]
        ring, _ = make_scan(MaskSpec(kind="vortex"), roi=(41, 41), sample=sample, handedness=1)
        return float(I[2, 2] / ring(p, at((0.0, 0.0, 0.0))).max())
    assert zero("dye", 1) < 1e-6
    dye, bead, gold = zero("dye", -1), zero("bead", -1), zero("gold", -1)
    assert dye > bead > gold > 0.3


def test_a_bead_in_one_plane_is_the_point_scan_averaged_over_the_spheres_projection():
    R = 50.0
    psf, p = make_scan(MaskSpec(kind="vortex"), bead_radius_nm=R)
    point, _ = make_scan(MaskSpec(kind="vortex"))
    bead = psf(p, at((0.0, 0.0, 0.0)))[0, 0]
    # the sphere sampled on a 5 nm cube, every sample at the bead's height: the projection
    s = torch.arange(-R + 2.5, R, 5.0, dtype=torch.float64)
    x, y, z = torch.meshgrid(s, s, s, indexing="ij")
    inside = x ** 2 + y ** 2 + z ** 2 <= R ** 2
    xyz = torch.stack([x[inside], y[inside], torch.zeros_like(x[inside])], -1)[:, None]
    brute = point(p, xyz)[:, 0].mean(0)
    assert float((bead - brute).abs().max() / bead.max()) < 0.01
    assert float(bead[12, 12] / bead.max()) > 0.03            # a 100 nm bead fills a donut's zero
    stack = psf(p, at((0.0, 0.0, -100.0), (0.0, 0.0, 0.0), (0.0, 0.0, 100.0)), z_step_nm=100.0)
    assert stack.shape == (1, 3, 25, 25)                      # in a stack the sphere is 3-D, cropped the same


@pytest.mark.parametrize("pattern, polarisation", [(None, "circular"), ("top-hat", "circular"),
                                                   ("half-moon", "x")])
def test_one_plane_cannot_tell_a_beam_from_its_twin_unless_the_handedness_is_held(pattern, polarisation):
    """The twin s*(−k) -- even modes flipped, the mask's centre at −c, the circular handedness flipped -- gives
    the same scan in focus.  A model with the λ/4 held cannot flip the handedness, and for circular light that
    alone tells the twin apart; linear light is its own conjugate (scanned-psfs.md §7a)."""
    ab = {5: 0.2, 6: 0.3, 7: 0.15, 8: -0.1, 11: 0.12}
    twin_ab = {j: (-c if j in (5, 6, 11) else c) for j, c in ab.items()}
    mask = None if pattern is None else MaskSpec(kind=pattern, centre=(0.05, 0.03))
    twin_mask = None if pattern is None else MaskSpec(kind=pattern, centre=(-0.05, -0.03))
    kw = dict(roi=(25, 25), polarisation=polarisation)
    psf, p = make_scan(mask, aberrations=ab, **kw)
    twin, q = make_scan(twin_mask, aberrations=twin_ab, handedness=-1, **kw)
    held, r = make_scan(twin_mask, aberrations=twin_ab, **kw)
    mu, mu_twin, mu_held = (3e4 * f(v, at((0.0, 0.0, 0.0))) + 1 for f, v in ((psf, p), (twin, q), (held, r)))

    def kl(a, b):
        return float((a * torch.log(a / b) - a + b).sum())
    assert kl(mu, mu_twin) < 1.5
    if polarisation == "circular":
        assert kl(mu, mu_held) > 5
    else:
        assert kl(mu, mu_held) < 1e-6


def test_the_mirrored_pupils_fast_tangents_are_the_derivatives_of_the_scan():
    psf, p = make_scan(MaskSpec(kind="vortex", centre=(0.05, 0.0)), aberrations={7: 0.3, 6: 0.2})
    xyz = at((30.0, -20.0, 40.0), (30.0, -20.0, -60.0))
    t = torch.zeros_like(p["pupil.phase"])
    t[psf.pupil.phase_modes.index(9)] = 1.0

    def scan(c):
        return psf(dict(p, **{"pupil.phase": c}), xyz)

    def pupil(c):
        return psf.pupil_field(dict(p, **{"pupil.phase": c}))
    _, dA = torch.func.jvp(pupil, (p["pupil.phase"],), (t,))
    _, d_scan = torch.func.jvp(scan, (p["pupil.phase"],), (t,))
    fast = psf.pupil_linearisation(p, xyz)(dA[None])[0]
    assert torch.allclose(fast, d_scan, rtol=1e-8, atol=1e-12 * float(d_scan.abs().max()))


def test_one_scanned_plane_of_a_vortex_gives_back_the_even_modes_signs_from_the_twin():
    """A vortex breaks the twin (its charge cannot flip), so a fit of one plane started with the even modes flipped,
    the centre mirrored and the heights mirrored ends where one started at the truth, with the even modes' signs
    right.  Where that is: the centre trades with coma in one plane (scanned-psfs.md §7), so they are only near."""
    from uipsf import Batch
    from uipsf.simulate import parameters, simulate, template
    from uipsf.workflows.beads import fit_beads
    truth = {5: 0.15, 6: 0.25, 7: 0.15, 8: -0.1, 11: 0.1}
    roi = (33, 33)
    psf, _ = make_scan(MaskSpec(kind="vortex", free=("centre",)), roi=roi, P=64)
    fm = ForwardModel(psf, intensity=Bleaching(order=0))
    n = 8
    g = torch.Generator().manual_seed(3)
    xyz = (torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5) * torch.tensor([80.0, 80.0, 100.0])
    b0 = template(n, Batch.z_from_objective(torch.tensor([0.0])), roi, dtype=torch.float64, device="cpu")

    def start(ab, centre, z_sign):
        phase = torch.tensor([ab.get(j, 0.0) for j in psf.pupil.phase_modes], dtype=torch.float64)
        return parameters(fm, b0, {"pupil.phase": phase, "mask.centre": torch.tensor(centre, dtype=torch.float64),
                                   "emitter.xyz": xyz * torch.tensor([1.0, 1.0, z_sign]),
                                   "emitter.photons": torch.full((n,), 3e4),
                                   "emitter.background": torch.full((n,), 1.0)}, dtype=torch.float64, device="cpu")
    data = simulate(fm, start(truth, (0.05, 0.03), 1.0), b0, seed=4)
    twin = {j: (-c if j in (5, 6, 11) else c) for j, c in truth.items()}
    from_truth = fit_beads(fm, data, start=start(truth, (0.05, 0.03), 1.0), first_order=None, max_dz_nm=None)
    from_twin = fit_beads(fm, data, start=start(twin, (-0.05, -0.03), -1.0), first_order=None, max_dz_nm=None)
    assert from_twin.objective == pytest.approx(from_truth.objective, abs=0.5)
    q = from_twin.params.physical()
    phase = dict(zip(psf.pupil.phase_modes, q["pupil.phase"].tolist(), strict=True))
    for j in (5, 6):
        assert phase[j] == pytest.approx(truth[j], abs=0.03)
    assert phase[11] > 0.05
    assert phase[7] == pytest.approx(truth[7], abs=0.06)
    assert q["mask.centre"].tolist() == pytest.approx([0.05, 0.03], abs=0.03)
