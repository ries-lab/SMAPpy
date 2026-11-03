"""Single molecules on the coverslip through repeated z sweeps: the geometry (`CoverslipStackGeometry`)."""
import numpy as np
import pytest
import torch

import uipsf
from uipsf.export.model_file import build, describe
from uipsf.model import Bleaching, CoverslipStackGeometry
from uipsf.params import Params

f64 = torch.float64


def _batch(z_nm, sweep, field_xy):
    E = len(z_nm)
    return uipsf.Batch(data=torch.zeros(E, 1, 1, 9, 9, dtype=f64), z_nm=torch.tensor(z_nm, dtype=f64)[:, None],
                       stack_id=torch.tensor(sweep), field_xy=torch.tensor(field_xy, dtype=f64))


def _params(geometry, batch, values):
    E = batch.n_emitters
    start = {"xyz": values.get("emitter.xyz", torch.zeros(E, 3, dtype=f64))}
    return Params(geometry.specs(batch, start), E, dtype=f64, values=values).physical()


def test_the_defocus_is_the_stage_z_less_the_sweeps_focus_and_the_focal_plane_at_the_molecule():
    g = CoverslipStackGeometry(2, [100.0, -50.0], centre_px=(50, 50), pixel_nm=(100.0, 100.0))
    batch = _batch([0.0, 200.0, 200.0], [0, 1, 1], [[50, 50], [50, 50], [60, 40]])
    p = _params(g, batch, {"geometry.plane": torch.tensor([3.0, -2.0], dtype=f64),
                           "geometry.height": torch.tensor(12.0, dtype=f64)})
    out = g(p, batch)                                           # [E, 1, 4]
    # the third molecule is 1 µm right of and 1 µm above the centre: 3·1 + (−2)·(−1) = 5 nm more focus
    assert torch.allclose(out[:, 0, 2], torch.tensor([-100.0, 250.0, 245.0], dtype=f64))
    assert torch.allclose(out[:, 0, 3], torch.full((3,), 12.0, dtype=f64))


def test_the_z_check_frees_each_molecules_offset_from_the_plane():
    g = CoverslipStackGeometry(1, [0.0], centre_px=(0, 0), pixel_nm=(100.0, 100.0))
    batch = _batch([0.0], [0], [[0, 0]])
    xyz = torch.tensor([[0.0, 0.0, 30.0]], dtype=f64)
    learning = {s.name: s for s in g.specs(batch, {"xyz": xyz})}
    checking = {s.name: s for s in g.checking().specs(batch, {"xyz": xyz})}
    assert learning["emitter.xyz"].prior is not None and checking["emitter.xyz"].prior is None
    out = g.checking()(_params(g.checking(), batch, {"emitter.xyz": xyz}), batch)
    assert float(out[0, 0, 2]) == 30.0


def test_a_held_height_is_not_learnt_and_a_learnt_one_starts_at_the_glass():
    batch = _batch([0.0], [0], [[0, 0]])
    held = {s.name: s for s in CoverslipStackGeometry(1, [0.0], (0, 0), (100, 100), height_nm=20.0)
            .specs(batch, {"xyz": torch.zeros(1, 3)})}
    free = {s.name: s for s in CoverslipStackGeometry(1, [0.0], (0, 0), (100, 100))
            .specs(batch, {"xyz": torch.zeros(1, 3)})}
    assert held["geometry.height"].fixed and held["geometry.height"].init == 20.0
    assert not free["geometry.height"].fixed and free["geometry.height"].init < 1.0


def test_the_geometry_is_written_to_the_model_file_and_read_back():
    optics = uipsf.Optics(NA=1.5, wavelength_nm=676.0, n_imm=1.518, n_med=1.334, pixel_size_nm=(105.0, 106.0))
    grid = uipsf.PupilGrid.build(optics, 16, device="cpu", dtype=f64)
    psf = uipsf.VectorPSF(grid, uipsf.ZernikePupil(grid, n_max=4), (9, 9))
    g = CoverslipStackGeometry(3, [10.0, 20.0, 30.0], (64.5, 63.5), (105.0, 106.0), height_nm=8.0, fit_plane=False)
    back = build(describe(uipsf.ForwardModel(psf, geometry=g, intensity=Bleaching(order=0)))).geometry
    assert isinstance(back, CoverslipStackGeometry)
    assert back.sweeps == 3 and back.focus_nm.tolist() == [10.0, 20.0, 30.0]
    assert back.centre_px == (64.5, 63.5) and back.height_nm == 8.0 and not back.fit_plane


# ------------------------------------------------------------------------------- learning, in rounds
OPTICS = uipsf.Optics(NA=1.45, wavelength_nm=680.0, n_imm=1.518, n_med=1.334, pixel_size_nm=(110.0, 110.0))
ROI = 13
TRUE_PHASE_NM = {5: -25.0, 6: 60.0, 11: 40.0}


def stack_molecules(sweeps=3, per_frame=6, step_nm=100.0, span_nm=900.0, height_nm=10.0, gamma=0.5,
                    foci=(40.0, -30.0, 10.0), plane=(6.0, -4.0), seed=0, dtype=torch.float32):
    """(model, truth, batch, defocus [E]): molecules on the coverslip at ``height_nm``, each seen once, in
    ``sweeps`` z sweeps of ``step_nm`` over ±``span_nm`` with each sweep's own focus and a tilted focal plane;
    the pupil astigmatic and spherical (`TRUE_PHASE_NM`), γ ``gamma``."""
    from smappy.calibrate.pupil import PSFModelSettings

    from uipsf.optics.zernike import noll_to_nm  # noqa: F401  (Noll indices of the pupil's modes)
    from uipsf.simulate import parameters, simulate
    from uipsf.workflows.calibrate import molecule_stack_model
    spec = PSFModelSettings(zernike_order=4, magnitude="minimal", blur="round", blur_nm=20.0)
    stage = np.arange(-span_nm, span_nm + 1, step_nm)
    rng = np.random.default_rng(seed)
    z, sweep, field = [], [], []
    for s in range(sweeps):
        for zk in stage:
            z += [zk] * per_frame
            sweep += [s] * per_frame
            field += rng.uniform(10, 118, (per_frame, 2)).tolist()
    E = len(z)
    model = molecule_stack_model(OPTICS, spec, (ROI, ROI), sweeps, np.zeros(sweeps), (64.0, 64.0), device="cpu",
                                 dtype=dtype)
    batch = uipsf.Batch(data=torch.zeros(E, 1, 1, ROI, ROI, dtype=dtype), z_nm=torch.tensor(z, dtype=dtype)[:, None],
                        stack_id=torch.tensor(sweep), field_xy=torch.tensor(field, dtype=dtype))
    pupil = model.psf.pupil
    phase = torch.zeros(len(pupil.phase_modes), dtype=dtype)
    for k, j in enumerate(pupil.phase_modes):
        phase[k] = 2 * np.pi * TRUE_PHASE_NM.get(int(j), 0.0) / OPTICS.wavelength_nm
    xy = torch.tensor(rng.uniform(-40, 40, (E, 2)), dtype=dtype)
    values = {"pupil.phase": phase, "pupil.gamma": torch.tensor(gamma, dtype=dtype),
              "psf.sigma": torch.tensor([20.0], dtype=dtype),
              "geometry.focus": torch.tensor(foci[:sweeps], dtype=dtype),
              "geometry.plane": torch.tensor(plane, dtype=dtype),
              "geometry.height": torch.tensor(height_nm, dtype=dtype),
              "emitter.xyz": torch.cat([xy, torch.zeros(E, 1, dtype=dtype)], 1),
              "emitter.photons": torch.tensor(rng.uniform(3000, 6000, E), dtype=dtype),
              "emitter.background": torch.full((E,), 15.0, dtype=dtype)}
    truth = parameters(model, batch, values, dtype=dtype)
    data = simulate(model, truth, batch, seed=seed + 1)
    start = {"xyz": torch.cat([xy + torch.tensor(rng.normal(0, 15, (E, 2)), dtype=dtype),
                               torch.zeros(E, 1, dtype=dtype)], 1),
             "photons": values["emitter.photons"] * 0.9, "background": values["emitter.background"] * 1.1}
    from dataclasses import replace
    return model, truth, replace(data, start=start), np.asarray(z) - np.repeat(np.asarray(foci[:sweeps]), E // sweeps)


@pytest.mark.slow
def test_the_rounds_recover_the_pupil_the_height_the_foci_and_the_plane_and_find_no_z_bend():
    from uipsf.workflows.molecule_stack import StackRoundSettings, learn_molecule_stack
    model, truth, batch, defocus = stack_molecules(sweeps=2)
    run = learn_molecule_stack(model, batch, defocus, settings=StackRoundSettings(max_rounds=2))
    p = {k: v.detach().double() for k, v in run.final.calibration.shared().items()}
    t = {k: v.detach().double() for k, v in truth.physical().items()}
    nm = (p["pupil.phase"] - t["pupil.phase"]) * OPTICS.wavelength_nm / (2 * np.pi)
    assert float(nm.abs().max()) < 3.0
    assert abs(float(p["pupil.gamma"] - t["pupil.gamma"])) < 0.1
    # the height is told only by the supercritical light's decay, and trades with the common focus: a few nm
    assert abs(float(p["geometry.height"] - t["geometry.height"])) < 5.0
    assert float((p["geometry.focus"] - t["geometry.focus"]).abs().max()) < 10.0
    assert float((p["geometry.plane"] - t["geometry.plane"]).abs().max()) < 3.0
    check = run.final.check
    # ~12 molecules a bin: the bend is the noise of their median, a third of a CRLB
    assert check.bend_rms(600) < 0.5 * np.median(check.crlb_nm) and 0.7 < check.scatter < 1.4
    assert check.bad.mean() < 0.05
