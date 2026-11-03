"""Phase masks in the pupil (`uipsf.model.mask`): their designs, their placement as held or freed parameters, and
a calibration recovering a misplaced mask."""
import math
from dataclasses import dataclass, field

import pytest
import torch

import uipsf
from uipsf.export.model_file import build, describe
from uipsf.model import MaskSpec, VectorPSF, ZernikePupil, make_mask, start_from
from uipsf.params import Params
from uipsf.simulate import parameters, simulate, template
from uipsf.workflows import BeadSettings, PSFModelSettings, bead_model, calibrate_beads

OPTICS = uipsf.Optics(NA=1.45, wavelength_nm=670, n_imm=1.518, n_med=1.518, pixel_size_nm=(100.0, 100.0))


@dataclass
class MaskedModel(PSFModelSettings):
    """The PSF model's settings with a mask: what the builder reads (`model.build`, ``phase_mask``)."""
    phase_mask: MaskSpec = field(default_factory=MaskSpec)


def psf_with(spec, roi=(15, 15), dtype=torch.float64):
    grid = uipsf.PupilGrid.build(OPTICS, 64, device="cpu", dtype=dtype)
    pupil = ZernikePupil(grid, n_max=4, magnitude_modes=[], mask=make_mask(spec, grid))
    return VectorPSF(grid, pupil, roi, fit_sigma=False)


def image(psf, p, z=0.0):
    xyz = torch.zeros(1, 1, 3, dtype=psf.grid.dtype)
    xyz[..., 2] = z
    return psf(p, xyz)[0, 0]


def physical(psf, **values):
    p = Params(psf.specs(), device="cpu", dtype=psf.grid.dtype).physical()
    return p | {k: torch.as_tensor(v, dtype=psf.grid.dtype) for k, v in values.items()}


def test_a_vortex_transmits_its_charge_as_a_phase_winding_around_its_centre():
    psf = psf_with(MaskSpec(kind="vortex", charge=2))
    t = psf.pupil.transmission(physical(psf))
    g = psf.grid
    ring = (g.rho > 0.5) & (g.rho < 0.6)
    # exp(2iφ): the transmission over e^{2iφ} is constant on a ring away from the singularity, its magnitude
    # one but for the winding's curvature within a pupil sample, which its mean over the sub-samples sees
    ratio = t[ring] * torch.exp(-2j * g.phi[ring].to(t.dtype))
    assert torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-3)
    assert float(ratio.angle().abs().max()) < 1e-5
    assert float(t.abs()[ring].min()) == pytest.approx(1.0, abs=1e-3)


def test_a_top_hat_makes_a_zero_at_focus_that_a_decentred_one_fills():
    psf = psf_with(MaskSpec(kind="top-hat"), roi=(15, 15))
    centred = image(psf, physical(psf))
    off = image(psf, physical(psf, **{"mask.centre": [0.1, 0.0]}))
    assert float(centred[7, 7] / centred.max()) < 0.05
    assert float(off[7, 7] / off.max()) > 2 * float(centred[7, 7] / centred.max())


def test_a_mask_leaves_the_psf_normalised_to_one_photon():
    plain = psf_with(MaskSpec(kind="none"), roi=(41, 41))
    vortex = psf_with(MaskSpec(kind="vortex"), roi=(41, 41))
    assert float(image(vortex, physical(vortex)).sum()) == pytest.approx(float(image(plain, physical(plain)).sum()),
                                                                         rel=0.02)


@pytest.mark.parametrize("kind", ["vortex", "top-hat", "half-moon", "double helix"])
def test_the_psf_has_exact_derivatives_with_respect_to_the_masks_centre(kind):
    psf = psf_with(MaskSpec(kind=kind, free=("centre",)), roi=(9, 9))
    p0 = physical(psf)

    def f(c):
        return image(psf, p0 | {"mask.centre": c})
    assert torch.autograd.gradcheck(f, (torch.tensor([0.031, -0.017], dtype=torch.float64, requires_grad=True),),
                                    eps=1e-5, atol=1e-6, rtol=1e-3)


def test_a_masks_parameters_are_held_unless_freed():
    psf = psf_with(MaskSpec(kind="double helix", free=("rotation",)))
    specs = {s.name: s for s in psf.specs()}
    assert not specs["mask.rotation"].fixed
    assert specs["mask.centre"].fixed and specs["mask.scale"].fixed and specs["mask.alpha"].fixed


def test_a_mask_offers_only_the_placements_it_can_show():
    assert {s.name for s in psf_with(MaskSpec(kind="vortex")).specs() if s.name.startswith("mask.")} == {
        "mask.centre", "mask.alpha"}
    with pytest.raises(ValueError, match="vortex mask cannot learn scale"):
        psf_with(MaskSpec(kind="vortex", free=("scale",)))


def test_a_rotated_zernike_mask_is_its_design_turned():
    """Astigmatism cos 2φ turned by 45° is astigmatism sin 2φ: a design rotates as a block."""
    turned = psf_with(MaskSpec(kind="zernike", zernike={6: 1.0}, rotation=math.pi / 4))
    other = psf_with(MaskSpec(kind="zernike", zernike={5: 1.0}))
    a, b = (m.pupil.transmission(physical(m)) for m in (turned, other))
    inside = turned.grid.rho < 0.95
    assert torch.allclose(a[inside], b[inside], atol=1e-3)


def test_a_model_with_a_mask_describes_and_builds_again():
    model = bead_model(OPTICS, BeadSettings(psf=MaskedModel(zernike_order=4, phase_mask=MaskSpec(
        kind="top-hat", free=("centre", "radius"), radius=0.7))), (13, 13), device="cpu")
    again = build(describe(model), device="cpu")
    mask = again.psf.pupil.mask
    assert mask.kind == "top-hat" and mask.spec.radius == 0.7 and set(mask.spec.free) == {"centre", "radius"}
    assert [s.name for s in again.specs(template(1, torch.zeros(3), (13, 13)))] == [
        s.name for s in model.specs(template(1, torch.zeros(3), (13, 13)))]


def test_a_calibration_starts_its_mask_where_a_previous_model_put_it():
    psf = psf_with(MaskSpec(kind="vortex", free=("centre",)))
    description = psf.pupil.describe(physical(psf, **{"mask.centre": [0.05, -0.02]}))
    start = start_from(psf.pupil, description, OPTICS.wavelength_nm)
    assert torch.allclose(start["mask.centre"], torch.tensor([0.05, -0.02], dtype=start["mask.centre"].dtype))


def _twin_divergence(kind):
    """The Poisson KL divergence, nats per bead at 10⁴ photons, between the in-focus image of a pupil with even
    and odd aberrations and a mask 0.05/0.03 off, and its twin's: the even modes flipped, the centre at −c."""
    spec = None if kind == "none" else MaskSpec(kind=kind)
    psf = psf_with(spec, roi=(21, 21))
    modes = psf.pupil.phase_modes

    def p(sign, centre):
        phase = torch.zeros(len(modes), dtype=torch.float64)
        for j, c in {5: 0.2, 6: 0.3, 7: 0.15, 8: -0.1, 11: 0.12}.items():
            phase[modes.index(j)] = c * (sign if j in (5, 6, 11) else 1)
        return physical(psf, **{"pupil.phase": phase}, **({} if spec is None else {"mask.centre": centre}))
    mu, alt = (1e4 * image(psf, p(s, c)) + 10 for s, c in ((1, [0.05, 0.03]), (-1, [-0.05, -0.03])))
    return float((mu * torch.log(mu / alt) - mu + alt).sum())


@pytest.mark.parametrize("kind", ["none", "half-moon"])
def test_in_focus_a_pupil_and_its_twin_look_alike_without_a_mask_or_behind_a_twin_symmetric_one(kind):
    assert _twin_divergence(kind) < 0.05


@pytest.mark.parametrize("kind", ["vortex", "double helix"])
def test_a_vortex_or_a_double_helix_tells_a_pupil_from_its_twin_in_one_plane(kind):
    assert _twin_divergence(kind) > 100


@pytest.mark.slow
def test_a_calibration_finds_a_decentred_vortex_and_the_aberrations_beside_it():
    """Beads imaged through a vortex in emission (the Vortex PSF), the plate 0.06 and −0.04 of the pupil radius
    off and astigmatism and coma beside it: freed, the centre comes back, and the Zernikes are the system's."""
    centre = torch.tensor([0.06, -0.04])
    phase = torch.tensor([0.3, -0.2, 0.15, 0.1])                   # Noll 5–8
    settings = BeadSettings(psf=MaskedModel(zernike_order=4, magnitude="fixed", blur="fixed", blur_nm=0.0,
                                            phase_mask=MaskSpec(kind="vortex", free=("centre",))))
    roi, K, n = (15, 15), 21, 8
    model = bead_model(OPTICS, settings, roi, z_max_nm=800, device="cpu")
    g = torch.Generator().manual_seed(1)
    values = {"pupil.phase": torch.cat([phase, torch.zeros(len(model.psf.pupil.phase_modes) - 4)]),
              "mask.centre": centre, "psf.sigma": torch.tensor([0.0]),
              "emitter.xyz": (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([100.0, 100.0, 150.0]),
              "emitter.photons": 30000 * (1 + torch.rand(n, generator=g)), "emitter.background": torch.full((n,), 10.0)}
    b0 = template(n, uipsf.Batch.z_from_objective(torch.linspace(-800, 800, K)), roi, device="cpu")
    beads = simulate(model, parameters(model, b0, values, device="cpu"), b0, seed=2)
    shared = calibrate_beads(beads, OPTICS, settings, device="cpu").shared()
    assert torch.allclose(shared["mask.centre"].cpu(), centre, atol=0.01)
    assert torch.allclose(shared["pupil.phase"][:4].cpu(), phase, atol=0.03)
