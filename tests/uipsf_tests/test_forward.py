import dataclasses

import torch

import uipsf
from uipsf.model import BeadStackGeometry, Bleaching, Constant, PerObservation, PerSliceFactor, Planar, rough_start
from uipsf.model.start import BACKGROUND_FLOOR
from uipsf.simulate import parameters, template
from uipsf_tests.conftest import small_beads

f64 = torch.float64


def test_the_expected_photons_are_photons_times_the_psf_plus_the_background(optics):
    model, truth, batch = small_beads(optics, n=2)
    p = truth.physical()
    mu = model(p, batch)
    xyz = model.geometry(p, batch)
    I = model.psf(p, xyz)
    expected = p["emitter.photons"][:, None, None, None] * I + p["emitter.background"][:, None, None, None]
    assert mu.shape == batch.data.shape
    assert torch.allclose(mu[:, 0], expected, rtol=1e-12)


def test_a_bead_moves_through_the_stack_in_smappys_z_sign():
    # Objective raised by 100 nm: the bead is 100 nm further below the focal plane.
    z = uipsf.Batch.z_from_objective(torch.tensor([-100.0, 0.0, 100.0], dtype=f64))
    batch = template(1, z, (5, 5), dtype=f64)
    geo = BeadStackGeometry()
    p = uipsf.Params(geo.specs(batch, {"xyz": torch.tensor([[1.0, 2.0, 30.0]])}), 1, dtype=f64).physical()
    xyz = geo(p, batch)
    assert xyz[0, :, 2].tolist() == [130.0, 30.0, -70.0]
    assert xyz[0, :, 0].tolist() == [1.0, 1.0, 1.0]


def test_the_stage_terms_are_declared_fixed_until_asked_for():
    batch = template(2, torch.zeros(4), (5, 5))
    start = {"xyz": torch.zeros(2, 3)}
    fixed = {s.name: s.fixed for s in BeadStackGeometry().specs(batch, start)}
    assert fixed == {"emitter.xyz": False, "geometry.drift": True, "geometry.z_scale": True}
    free = {s.name: s.fixed for s in BeadStackGeometry(drift=True, z_scale=True).specs(batch, start)}
    assert not any(free.values())


def test_the_drift_and_z_scale_move_every_bead_alike():
    batch = template(2, torch.tensor([100.0, 0.0, -100.0], dtype=f64), (5, 5), dtype=f64)
    geo = BeadStackGeometry(drift=True, z_scale=True)
    p = uipsf.Params(geo.specs(batch, {"xyz": torch.tensor([[0.0, 0, 0], [10.0, 0, 5]])}), 2, dtype=f64,
                     values={"geometry.z_scale": 1.1}).physical()
    p["geometry.drift"] = torch.tensor([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]], dtype=f64)
    xyz = geo(p, batch)
    assert torch.allclose(xyz[1] - xyz[0], torch.tensor([10.0, 0, 5], dtype=f64).expand(3, 3))
    assert torch.allclose(xyz[0, :, 2], torch.tensor([110.0, 0.0, -107.0], dtype=f64))


def test_bleaching_is_exponential_about_the_middle_of_the_stack():
    batch = template(1, torch.zeros(5), (3, 3), dtype=f64)
    model = Bleaching(order=1)
    p = uipsf.Params(model.specs(batch, {"photons": torch.tensor([100.0])}), 1, dtype=f64,
                     values={"emitter.bleaching": torch.tensor([[-0.5]])}).physical()
    N = model(p, batch)[0]
    assert torch.allclose(N, 100 * torch.exp(-0.5 * torch.linspace(-1, 1, 5, dtype=f64)))
    assert abs(N[2].item() - 100) < 1e-9


def test_a_per_slice_factor_scales_every_bead_in_that_slice_and_has_a_prior():
    batch = template(2, torch.zeros(3), (3, 3), dtype=f64)
    model = PerSliceFactor(order=0)
    specs = model.specs(batch, {"photons": torch.tensor([10.0, 20.0])})
    p = uipsf.Params(specs, 2, dtype=f64, values={"intensity.slice_factor": torch.tensor([1.0, 0.5, 2.0])})
    N = model(p.physical(), batch)
    assert torch.allclose(N, torch.tensor([[10.0, 5.0, 20.0], [20.0, 10.0, 40.0]], dtype=f64))
    assert p.prior_penalty() > 0


def test_a_free_count_per_observation_is_a_block_of_the_emitter():
    batch = template(2, torch.zeros(4), (3, 3), dtype=f64)
    model = PerObservation()
    p = uipsf.Params(model.specs(batch, {"photons": torch.tensor([10.0, 20.0])}), 2, dtype=f64)
    assert p.emitter_matrix().shape == (2, 4)
    assert torch.allclose(model(p.physical(), batch)[1], torch.full((4,), 20.0, dtype=f64))


def test_a_planar_background_is_the_level_at_the_centre_plus_the_slopes():
    batch = template(1, torch.zeros(1), (3, 5), dtype=f64)
    bg = Planar((100.0, 50.0))
    p = uipsf.Params(bg.specs(batch, {"background": torch.tensor([10.0])}), 1, dtype=f64,
                     values={"emitter.background_slope": torch.tensor([[0.01, -0.02]], dtype=f64)}).physical()
    b = bg(p, batch)[0, 0, 0]
    assert abs(b[1, 2].item() - 10) < 1e-9
    assert abs((b[1, 3] - b[1, 2]).item() - 1.0) < 1e-9          # 100 nm · 0.01 per nm
    assert abs((b[2, 2] - b[1, 2]).item() + 1.0) < 1e-9          # 50 nm · −0.02
    assert torch.allclose(Constant()(p, batch)[0, 0, 0, 0], torch.tensor(10.0, dtype=f64))


def test_each_emitter_sees_only_its_own_parameters_through_the_whole_model(optics):
    # The contract the solver's blocks rely on (plan §5.4), for every emitter parameter at once.
    model, truth, batch = small_beads(optics, n=3)
    ue = truth.emitter_matrix()
    tangent = torch.zeros_like(ue)
    tangent[1] = 1.0
    _, dmu = torch.func.jvp(lambda v: model(truth.replace(emitter=v).physical(), batch), (ue,), (tangent,))
    assert dmu[1].abs().max() > 0
    assert dmu[0].abs().max() == 0 and dmu[2].abs().max() == 0


def test_the_rough_start_finds_the_bead_near_its_true_place(optics):
    # A ROI of the usual size: in a small one the defocused spot fills the border.
    model, truth, batch = small_beads(optics, n=4, photons=20000.0, P=32, roi=(17, 17), K=15)
    start = rough_start(batch, model.pixel_size_nm)
    tru = truth.physical()
    assert (start["xyz"][:, :2] - tru["emitter.xyz"][:, :2]).abs().max() < 60       # nm
    assert (start["xyz"][:, 2] - tru["emitter.xyz"][:, 2]).abs().max() < 300        # within two slices
    # 20 000–40 000 photons over a background of 10: the spot's wings still reach the border.
    assert (start["background"] / tru["emitter.background"]).max() < 2.5
    assert (start["photons"] / tru["emitter.photons"]).min() > 0.5


def test_the_rough_start_keeps_a_dim_background_off_the_softplus_floor(optics):
    # Roadmap item 4: Zenodo 2ch's dim channel, offset-subtracted in steps of the camera gain, has border
    # medians of 0 or below in some slice for most beads.  Started at 10⁻³, where softplus' slope is 10⁻³,
    # such a background froze stage 1; at the floor of one photon it is a background like any other.
    model, truth, batch = small_beads(optics, n=4)
    g = torch.Generator().manual_seed(0)
    dark = torch.poisson(torch.full_like(batch.data, 2.0), generator=g) - 4.0    # an offset 4 photons too high
    data = torch.round(dark / 4.7) * 4.7                                          # in steps of the gain
    start = rough_start(dataclasses.replace(batch, data=data), model.pixel_size_nm)
    assert (start["background"] == BACKGROUND_FLOOR).all()


def test_a_simulation_is_the_model_plus_poisson_noise(optics):
    model, truth, batch = small_beads(optics, n=20, photons=500.0)
    mu = model(truth.physical(), batch)
    r = (batch.data - mu) / mu.sqrt()
    assert abs(r.mean().item()) < 0.02 and abs(r.std().item() - 1) < 0.02
    assert parameters(model, batch, {}).n_emitters == 20
