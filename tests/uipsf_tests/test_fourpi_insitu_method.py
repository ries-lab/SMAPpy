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


def test_spline_4pi_fits_with_what_in_situ_learnt_or_with_the_model_where_its_checks_speak_against_it(
        tmp_path, monkeypatch):
    # the fitter's decisions alone, the learning stubbed: a Run learns and fits with the model learnt, the
    # registration then the model's; a Preview with nothing kept fits as calibrated; what a Run kept is found
    # again; a learning in situ's checks speak against keeps the model and says why
    from dataclasses import replace

    from smappy.plugins import Context, PreflightQuestion
    from smappy.plugins import fit_4pi
    from smappy.plugins.fit import SourceSettings
    from smappy.plugins.fit_4pi import FourPiFit, FourPiFitSettings, FourPiModelSettings
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(FourPiFit, "unknown_camera", lambda self, settings: None)
    (tmp_path / "movie.h5").write_bytes(b"")
    (tmp_path / "beads_4pi.h5").write_bytes(b"")
    settings = FourPiFitSettings(source=SourceSettings(path=str(tmp_path / "movie.h5")),
                                 model=FourPiModelSettings(calibration=str(tmp_path / "beads_4pi.h5"),
                                                           insitu="channels"))
    settings.model.registration.source = "beads"
    plugin = FourPiFit()
    learnt = []
    warned = []

    def learn(ctx, s, camera, path, key, fallback=False):
        learnt.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")
        record = {"key": __import__("json").loads(__import__("json").dumps(key, default=str)),
                  "fallback": fallback, "warnings": ["held-out no better"] if fallback else [],
                  "summary": "learnt maps from 300 molecules (100 held out)"}
        path.with_suffix(".json").write_text(__import__("json").dumps(record))
        return record
    monkeypatch.setattr(fit_4pi, "learn_insitu", learn)
    ctx = Context()
    ctx.warn = warned.append
    asked = plugin.preflight(ctx, settings)
    assert isinstance(asked, PreflightQuestion) and "20000 frames" in asked.text
    assert asked.choices[0].settings.model.insitu == "off"
    # a Preview learns nothing: the model as calibrated
    preview = plugin.with_focus(ctx, settings, None, None, None)
    assert not learnt and preview.model.insitu == "off" and preview.model.calibration == settings.model.calibration
    plugin._may_learn = True
    used = plugin.with_focus(ctx, settings, None, None, None)
    assert len(learnt) == 1 and used.model.calibration == str(learnt[0]) and used.model.insitu == "off"
    assert used.model.registration.source == "model"                  # the maps learnt are the model's
    assert plugin.with_focus(ctx, settings, None, None, None).model.calibration == str(learnt[0])
    assert len(learnt) == 1 and plugin.preflight(ctx, settings) is None      # kept: not learnt again
    # learnt for other settings: learnt again, and this time its checks speak against it
    other = replace(settings, model=replace(settings.model, insitu_molecules=500))
    monkeypatch.setattr(fit_4pi, "learn_insitu", lambda *a: learn(*a, fallback=True))
    back = plugin.with_focus(ctx, other, None, None, None)
    assert back.model.calibration == settings.model.calibration and back.model.registration.source == "beads"
    assert warned == ["in situ: held-out no better"]


@pytest.mark.long
def test_spline_4pi_in_situ_learns_a_movie_and_fits_it_with_the_model_learnt(tmp_path, monkeypatch):
    """The bead model a day off, as above; Spline 4Pi with *in situ* fits the first frames with it, learns,
    keeps the model learnt -- the held-out molecules prefer it -- and fits the movie with it, the fitted
    file carrying it."""
    from smappy.io import embedded
    from smappy.plugins import Context
    from smappy.plugins.fit import CameraSettings, OutputSettings, SourceSettings
    from smappy.plugins.fit_4pi import FourPiFit, FourPiFitSettings, FourPiModelSettings
    from tests.test_fourpi_channels import _write_quadrant_series
    import functools

    from smappy.calibrate import fourpi_insitu
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    # one round on the model's own box, as the window's test learns: minutes fewer
    monkeypatch.setattr(fourpi_insitu, "FourPiInSituSettings",
                        functools.partial(fourpi_insitu.FourPiInSituSettings, roi_size=R, rounds=1,
                                          held_out=0.25, min_photons=1.0))
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
    settings = FourPiFitSettings(
        source=SourceSettings(path=str(paths[0])),
        camera=CameraSettings(conversion=0.5, offset=100.0, pixelsize_um=0.129),
        model=FourPiModelSettings(calibration=str(tmp_path / "beads_4pi.h5"), insitu="channels",
                                  insitu_frames=n_frames, insitu_molecules=250),
        output=OutputSettings(path=str(tmp_path / "out_locs.hdf5")))
    result = FourPiFit().run(Context(), settings)
    used = result.settings.model
    assert used.insitu == "off" and "fourpi_insitu" in used.calibration, result.text
    assert used.registration.source == "model"
    kept = fourpi.load_fourpi_calibration(tmp_path / "out_locs.hdf5")
    assert kept.parameters["method"] == "uiPSF v2 4Pi in-situ model"
    assert embedded.carried_format(tmp_path / "out_locs.hdf5") == "smappy-4pi-calibration"
