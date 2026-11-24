"""4Pi in situ as a method of the PSF calibration window (`psf_methods.FourPiInSitu`; docs/4pi.md §27): a
bead model file, a simulated 4Pi movie in quadrant files, the molecules fitted, chosen, cut and learnt from,
and the held-out molecules saying the learnt model is the better one."""
import math

import numpy as np
import pytest
import torch
from smappy import fourpi
from smappy.calibrate.fourpi_beads import build_model, model_from_calibration, recipe_record, shared_record
from smappy.calibrate.pupil import PSFModelSettings, RunSettings
from smappy.channels import side_by_side
from smappy.metadata import CameraMetadata

from uipsf.workflows.fourpi import calibration

R, SIZE = 11, 48
CAMERA = CameraMetadata(conversion=0.5, offset=100.0, pixelsize_um=0.129)


def _recipe():
    return {"NA": 1.35, "wavelength_nm": 600.0, "n_imm": 1.406, "n_med": None, "pixel_size_nm": 129.0,
            "roi_shape": (R, R), "n_channels": 4, "n_phases": 3, "steps_rad": [2.0, 0.0, -2.0],
            "spec": PSFModelSettings(zernike_order=4, magnitude="fixed", blur="fixed", blur_nm=40.0),
            "bead_radius_nm": 0.0, "photons_per_observation": False, "kind": "scalar", "plates": "aligned",
            "magnitude_modes": None, "same_magnitude": True}


def _shared(model, astig_up, astig_lo, coherence, dz):
    modes = model.pupils[0].phase_modes
    up, lo = torch.zeros(len(modes)), torch.zeros(len(modes))
    up[modes.index(6)], lo[modes.index(6)] = astig_up, astig_lo
    up[modes.index(11)] = 0.1
    return {"pupil.phase": up, "lo.pupil.phase": lo, "pupil.magnitude": torch.zeros(0),
            "psf.sigma": torch.tensor([40.0]),
            "fourpi.offset": torch.tensor([0.0, 0.0, dz]), "fourpi.steps": torch.tensor([-2.0, -4.0]),
            "fourpi.delta": torch.tensor([math.pi / 2, math.pi, 3 * math.pi / 2]),
            "fourpi.share": torch.tensor([0.25, 0.25, 0.25]), "fourpi.balance": torch.full((4,), 0.5),
            "fourpi.coherence": torch.full((4,), coherence),
            **{f"ch{c}.transform": torch.tensor([1.0, 0, 0, 0, 1.0, 0, 0, 0]) for c in (1, 2, 3)}}


def _bead_model_file(path, model, shared):
    cal = calibration((model, shared), z_range_nm=500, dz_nm=25)
    cal.parameters.update(model=recipe_record(_recipe()), shared=shared_record(shared))
    fourpi.save_fourpi_calibration(path, cal)
    return cal


def test_a_bead_model_file_keeps_the_model_to_start_from_again(tmp_path):
    model = build_model(_recipe())
    shared = _shared(model, 0.4, -0.4, 0.7, 40.0)
    _bead_model_file(tmp_path / "beads_4pi.h5", model, shared)
    again, values = model_from_calibration(fourpi.load_fourpi_calibration(tmp_path / "beads_4pi.h5"))
    assert again.kind == "scalar" and again.n_phases == 3 and again.psf_up.roi_shape == (R, R)
    assert torch.allclose(values["pupil.phase"], shared["pupil.phase"])
    assert model_from_calibration(fourpi.model_calibration()) is None


@pytest.mark.long
def test_the_window_method_learns_a_4pi_movie_and_its_held_out_molecules_prefer_it(tmp_path):
    """The bead model a day off (astigmatism, coherence, Δz); the movie drawn with the sample's model, the
    cavity drifting.  The method fits the movie with the bead model, learns, and the held-out molecules are
    explained better by the in-situ model; it is saved for Spline 4Pi with the movie's layout."""
    from smappy.calibrate.fourpi_insitu import FourPiInSituSettings
    from smappy.calibrate.psf_methods import FOURPI
    from smappy.calibrate.psf_run import calibrate
    from tests.test_fourpi_channels import _write_quadrant_series
    model = build_model(_recipe())
    truth = _shared(model, 0.5, -0.3, 0.9, 80.0)
    bead = _shared(model, 0.35, -0.45, 0.7, 30.0)
    truth_cal = calibration((model, truth), z_range_nm=500, dz_nm=25)
    _bead_model_file(tmp_path / "beads_4pi.h5", model, bead)
    layout = side_by_side(4, SIZE, SIZE)
    rng = np.random.default_rng(0)
    n_frames = 300
    frame = np.repeat(np.arange(n_frames), 2)
    x, y = rng.uniform(8, SIZE - 8, (2, len(frame)))
    x[1::2] = np.where(np.abs(x[1::2] - x[::2]) < 14, (x[::2] + 20) % (SIZE - 16) + 8, x[1::2])
    z = rng.uniform(-350, 350, len(frame))
    drift = fourpi.cavity_drift(n_frames, rate_rad=0.01, start_rad=0.3)
    frames = fourpi.render_frames(truth_cal, layout, frame, x, y, z, 5000, phase0=drift[frame],
                                  n_frames=n_frames, background=10)
    adu = (frames / CAMERA.adu_to_photons + CAMERA.offset).astype(np.uint16)
    paths = _write_quadrant_series(tmp_path, [layout.cut(adu)[:, c] for c in range(4)], files=1)
    method = FOURPI["insitu"]
    settings = FourPiInSituSettings(bead_model=str(tmp_path / "beads_4pi.h5"), fit_frames=n_frames, molecules=250,
                                    roi_size=R, rounds=1, held_out=0.25, min_photons=1.0)
    run = calibrate(method, False, [str(p) for p in paths],
                    {"fourpi_insitu": settings, "camera": CAMERA, "run": RunSettings(save_model=False)})
    done = run.result["insitu"]
    assert len(done.molecules) > 200 and len(done.held) == round(0.25 * len(done.molecules))
    assert done.gain_nats > 1.0
    assert float(done.fit.shared["fourpi.coherence"].mean()) > 0.8
    assert "better than the bead model" in method.summary(run)
    path = method.save_as(run, tmp_path / "insitu_4pi.h5")
    saved = fourpi.load_fourpi_calibration(path)
    assert saved.layout is not None and saved.parameters["method"] == "uiPSF v2 4Pi in-situ model"
    assert model_from_calibration(saved) is not None                   # in situ can start from it again
    assert [p.title for p in method.pages(run)] == ["Pupils", "Bead model's pupils", "Model", "Molecules",
                                                    "Parameters"]
