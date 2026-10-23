"""A Zernike pupil with some modes held at given values (in situ holds the objective's symmetric magnitude
and the high orders from a bead model of the microscope)."""
import torch

import uipsf
from uipsf.export import build, describe, load_model, save_model
from uipsf.workflows import learn_beads
from uipsf.workflows.insitu import twin, twin_model
from uipsf_tests.conftest import small_beads

f64 = torch.float64
OPTICS = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.518, n_med=1.33, pixel_size_nm=(110, 100))
PHASE = {5: 0.05, 6: 0.6, 7: -0.1, 11: 0.2, 12: -0.04, 22: 0.03}
MAGNITUDE = {4: -0.1, 6: 0.02, 11: -0.07}


def pupils():
    grid = uipsf.PupilGrid.build(OPTICS, 32, device="cpu", dtype=f64)
    learnt = uipsf.ZernikePupil(grid, n_max=6)
    held = uipsf.ZernikePupil(grid, n_max=6, held_phase={j: v for j, v in PHASE.items() if j > 10},
                              held_magnitude={j: v for j, v in MAGNITUDE.items() if j in (4, 11)})
    return learnt, held


def values(pupil, phase, magnitude):
    return {"pupil.phase": torch.tensor([phase.get(j, 0.0) for j in pupil.phase_modes], dtype=f64),
            "pupil.magnitude": torch.tensor([magnitude.get(j, 0.0) for j in pupil.magnitude_modes], dtype=f64)}


def test_held_modes_make_the_same_pupil_as_learnt_ones_and_are_no_parameters():
    learnt, held = pupils()
    a = learnt(values(learnt, PHASE, MAGNITUDE))
    b = held(values(held, PHASE, MAGNITUDE))
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])
    assert not set(held.phase_modes) & {11, 12, 22} and 6 in held.phase_modes
    assert not set(held.magnitude_modes) & {4, 11} and 6 in held.magnitude_modes
    sizes = {s.name: s.shape for s in held.specs()}
    assert sizes["pupil.phase"] == (len(learnt.phase_modes) - 3,)        # 11, 12 and 22, all of order ≤ 6


def test_a_first_stage_keeps_the_held_modes_whatever_their_order():
    _, held = pupils()
    low = held.truncated(4)
    assert all(j <= 15 for j in low.phase_modes)
    p = values(low, PHASE, MAGNITUDE)
    full = held(values(held, {j: v for j, v in PHASE.items() if j in low.phase_modes}, {}))
    assert torch.allclose(low({**p, "pupil.magnitude": torch.zeros(len(low.magnitude_modes), dtype=f64)})[1],
                          full[1])


def test_the_coefficients_are_the_whole_pupil_learnt_and_held():
    _, held = pupils()
    c = held.coefficients(values(held, PHASE, MAGNITUDE), "phase")
    assert c[11] == PHASE[11] and c[22] == PHASE[22] and c[6] == PHASE[6] and list(c) == sorted(c)


def test_the_twin_of_a_partly_held_pupil_is_the_twin_of_the_whole():
    # the twin mirrors held and learnt modes alike, so holding some changes nothing about the test
    model, _, _ = small_beads(OPTICS, n=2, dtype=f64)
    grid = model.psf.grid
    whole = uipsf.ZernikePupil(grid, n_max=6)
    part = uipsf.ZernikePupil(grid, n_max=6, held_phase={11: PHASE[11], 12: PHASE[12]},
                              held_magnitude={4: MAGNITUDE[4], 6: MAGNITUDE[6]})
    model.psf.pupil = part
    shared = values(part, PHASE, MAGNITUDE)
    mirrored = twin_model(model).psf.pupil(twin(model, shared))
    model.psf.pupil = whole
    expected = whole(twin(model, values(whole, PHASE, MAGNITUDE)))
    assert torch.allclose(mirrored[0], expected[0]) and torch.allclose(mirrored[1], expected[1])
    assert part.held_phase[11] == PHASE[11]                            # the original is left as it was


def test_a_model_file_keeps_the_held_modes(tmp_path):
    model, _, batch = small_beads(OPTICS, n=6, dtype=torch.float32)
    grid = model.psf.grid
    model.psf.pupil = uipsf.ZernikePupil(grid, n_max=4, held_magnitude={4: -0.1}, held_phase={22: 0.05})
    cal = learn_beads(model, batch, first_order=None)
    assert build(describe(cal.model)).psf.pupil.held_magnitude == {4: -0.1}
    path = tmp_path / "held.uipsf.h5"
    save_model(cal, path)
    m = load_model(path)
    assert m.model.psf.pupil.held_phase == {22: 0.05}
    xyz = torch.tensor([[0.0, 0.0, 0.0], [30.0, -20.0, 300.0]])
    assert torch.allclose(m.psf(xyz), cal.psf(xyz), atol=1e-7)
    assert (22, 0.05, 0.0) in m.zernikes("phase") and (4, -0.1, 0.0) in m.zernikes("magnitude")
