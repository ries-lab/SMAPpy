"""In situ on a chip split in two (Spline 3D 2C): a two-channel model learnt from the movie's own molecules.

The movies are uipsf's (`uipsf.movie`, every molecule drawn in both halves where the map puts it), the steps
SMAPpy's (`smappy.calibrate.insitu_dual`, `depth.insitu_dual`, `plugins.fit.DualSplineFit`).  The start is a
dual pupil calibration whose astigmatism is half the movie's in both channels: what a calibration made on
another day, or through another coverslip, looks like to the data.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import torch
from smappy.calibrate.dual import ChannelTransform, split_geometry
from smappy.calibrate.insitu import InSituSettings
from smappy.metadata import CameraMetadata

import uipsf
from uipsf.model import Bleaching, Channel, InSituGeometry
from uipsf.movie import MovieSettings, simulate_movie

f64 = torch.float64
OPTICS = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))
FOCUS = 700.0
SHAPE = (64, 128)                                   # (Y, X): the left half main, the right half secondary
H = np.array([[1.0, 0, -64 + 0.3], [0, 1, -0.2], [0, 0, 1]])        # secondary -> main chip pixels
GEOMETRY = split_geometry("right-left", "left", SHAPE)
DZ, SHARE = -300.0, 0.45                             # biplane: the secondary's plane 300 nm higher
CAMERA = CameraMetadata(conversion=2.0, offset=100.0, pixelsize_um=(0.127, 0.116), read_noise_e=1.6)
SELECTION = InSituSettings(block_frames=150, molecules=63)
PHASE = (torch.tensor([0.1, 0.7, -0.2, 0.15, 0.1]), torch.tensor([0.05, 0.6, 0.1, -0.1, 0.15]))


def truth(P=24, roi=(15, 15)):
    """The movie's two-channel model: each channel its own astigmatism, coma and spherical aberration."""
    channels = []
    for c in range(2):
        grid = uipsf.PupilGrid.build(OPTICS, P, device="cpu", dtype=f64)
        psf = uipsf.VectorPSF(grid, uipsf.ZernikePupil(grid, n_max=4), roi, sigma_nm=(30.0, 30.0))
        channels.append(Channel(psf, transform=H if c else None))
    model = uipsf.ForwardModel(channels, geometry=InSituGeometry(FOCUS), intensity=Bleaching(order=0))
    shared = {"geometry.focus": torch.tensor(FOCUS, dtype=f64), "ch1.transform": torch.tensor(H.ravel()[:8]),
              "ch1.share": torch.tensor(SHARE, dtype=f64), "ch1.dz": torch.tensor(DZ, dtype=f64)}
    for c, pre in enumerate(("", "ch1.")):
        phase = torch.zeros(len(channels[c].psf.pupil.phase_modes), dtype=f64)
        phase[:5] = PHASE[c]
        shared[pre + "pupil.phase"] = phase
        shared[pre + "pupil.magnitude"] = torch.zeros(len(channels[c].psf.pupil.magnitude_modes), dtype=f64)
        shared[pre + "psf.sigma"] = torch.tensor([35.0, 35.0], dtype=f64)
    return model, shared


def movie(dyes=None, seed=1):
    from smappy.simulate import BlinkingSettings
    model, shared = truth()
    s = MovieSettings(size_px=SHAPE, n_frames=150, fluorophores=150, heights_nm=(0.0, 1400.0), on_coverslip=0.25,
                      seed=seed, region=(0, 0, 64, 64), dyes=dyes,
                      blinking=BlinkingSettings(photons=6000.0, photons_std=3000.0, on_time=1.5, blinks=3.0))
    return simulate_movie(model, shared, s)


def write(m, folder) -> Path:
    path = folder / "split.tif"
    tifffile.imwrite(path, np.round(m.frames / CAMERA.adu_to_photons + CAMERA.offset).astype(np.uint16))
    return path


@pytest.fixture(scope="module")
def biplane(tmp_path_factory):
    m = movie()
    return write(m, tmp_path_factory.mktemp("insitu_dual")), m


def test_a_split_movie_draws_each_molecule_in_both_halves_where_the_map_puts_it_with_its_share():
    # one molecule, no noise to speak of: its light in the secondary half lies where H⁻¹ puts it, the share of it
    from uipsf.movie import MovieTruth, render
    model, shared = truth()
    a = np.array
    x, y = 30.0, 20.0
    t = MovieTruth(a([0]), a([0]), a([0]), a([x * 127.0]), a([y * 116.0]), a([300.0]), a([10000.0]))
    image = np.zeros((1, *SHAPE))
    render(model, shared, t, image, 21)
    left, right = image[0, :, :64], image[0, :, 64:]
    assert right.sum() / (left.sum() + right.sum()) == pytest.approx(SHARE, abs=0.02)
    yy, xx = np.mgrid[:SHAPE[0], :64]
    assert (xx * left).sum() / left.sum() == pytest.approx(x, abs=0.5)
    # the secondary's centre of mass at H⁻¹(x, y): 64 - 0.3 to the right, 0.2 down
    assert (xx * right).sum() / right.sum() + 64 == pytest.approx(x + 64 - 0.3, abs=0.5)
    assert (yy * right).sum() / right.sum() == pytest.approx(y + 0.2, abs=0.5)
    # two dyes: each molecule its own share
    t2 = replace(t, share=a([0.8]))
    image[:] = 0
    render(model, shared, t2, image, 21)
    assert image[0, :, 64:].sum() / image[0].sum() == pytest.approx(0.8, abs=0.02)


def test_the_molecules_of_both_halves_are_paired_each_region_about_its_own_place_on_the_chip(biplane):
    from smappy.calibrate.insitu_dual import dual_movie_rois
    path, m = biplane
    rois = dual_movie_rois([path], ChannelTransform(H, GEOMETRY, {}), SELECTION, CAMERA)
    used = rois.ids("used")
    assert len(used) >= 40, rois.counts()
    batch, _ = rois.batch()
    assert tuple(batch.data.shape[:2]) == (len(used), 2) and batch.roi_xy.shape == (len(used), 2, 2)
    # each secondary region where the map puts the main one: both cut about whole pixels near their own spot
    main = rois.roi_xy[used, 0]
    expected = np.c_[main[:, 0] + 64 - 0.3, main[:, 1] + 0.2]
    assert np.abs(rois.roi_xy[used, 1] - expected).max() <= 2.0
    assert np.median(np.abs(rois.roi_xy[used, 1] - expected)) <= 0.8
    # the secondary carries its share of each molecule's light
    light = rois.channel_regions[used].sum((2, 3)) - np.median(rois.channel_regions[used], axis=(2, 3)) * 25 ** 2
    assert np.median(light[:, 1] / light.sum(1)) == pytest.approx(SHARE, abs=0.05)


@pytest.mark.slow
def test_a_share_per_molecule_tells_two_dyes_apart_where_one_share_cannot(tmp_path):
    # two dyes on a ratiometric split: each molecule localised on the movie's own pupils, with its own share
    from smappy.calibrate.insitu_dual import dual_movie_rois

    from uipsf.workflows.insitu import localise_from_heights, start_heights
    from uipsf.workflows.scoring import localise
    m = movie(dyes=(0.2, 0.8), seed=2)
    rois = dual_movie_rois([write(m, tmp_path)], ChannelTransform(H, GEOMETRY, {}), SELECTION, CAMERA)
    model, shared = truth(P=16, roi=(25, 25))
    model.channels[1] = Channel(model.channels[1].psf, transform=H, share_per_emitter=True)
    batch, noise = rois.batch()
    batch = batch.to(dtype=f64)
    fixed = {k: v for k, v in shared.items() if k != "ch1.share"}
    # from a grid of heights, the best kept, as the rounds localise (`localise_from_heights`)
    start, _ = localise_from_heights(model, fixed, batch, noise, start_heights(model, FOCUS)[::2])
    fit, _ = localise(model, fixed, replace(batch, start=start), noise)
    found = fit.params.physical()["emitter.ch1.share"].detach().numpy()
    # each unit's dye: the truth's nearest molecule in its frame
    t = m.truth
    used = rois.ids("used")
    dye = []
    for k in used:
        f = int(rois.units.frames[k][0])
        rows = np.flatnonzero(t.frame == f)
        d = np.hypot(t.x_nm[rows] / 127 - rois.units.x_px[k], t.y_nm[rows] / 116 - rois.units.y_px[k])
        dye.append(t.share[rows[np.argmin(d)]])
    dye = np.asarray(dye)
    assert set(np.unique(dye)) == {0.2, 0.8}
    for share in (0.2, 0.8):
        assert np.median(found[dye == share]) == pytest.approx(share, abs=0.05)
    assert np.mean((found > 0.5) == (dye > 0.5)) > 0.9


def test_two_channel_in_situ_keeps_what_it_learnt_only_where_the_molecules_held_out_agree(monkeypatch):
    # the decision alone, as one channel's: the focus learnt first, the aberrations from the same molecules,
    # the better on the molecules held out kept -- here twice, once each way
    from smappy.calibrate import depth, insitu_run
    runs = []

    def one(dual, start, paths, camera, n_med, focus_nm, learn, detection, rounds, registration,
            share_per_molecule, progress, cancel, rois=None, hold_focus=False, order=None):
        rois = rois or SimpleNamespace(ids=lambda state: np.arange(7))
        out = depth.AutoDepth(650.0 if learn == "blur" else 700.0, 10.0, 0.1, 50, method="in situ")
        out.result = SimpleNamespace(learn=learn, rois=rois)
        runs.append((learn, focus_nm, rois, registration, share_per_molecule))
        return out
    monkeypatch.setattr(depth, "_insitu_dual", one)
    monkeypatch.setattr(depth, "model_of", lambda cal, channels=1: object())
    dual = SimpleNamespace(main=None)
    for better, kept in ((5.0, "phase"), (20.0, "blur")):
        runs.clear()
        monkeypatch.setattr(insitu_run, "heldout_deviance",
                            lambda result, b=better: 10.0 if result.learn == "blur" else b)
        found = depth.insitu_dual(dual, ["movie.tif"], CAMERA, learn="phase", registration=True,
                                  share_per_molecule=True)
        assert [r[0] for r in runs] == ["blur", "phase"] and runs[1][1] == 650.0
        assert runs[1][2] is runs[0][2]                       # the same molecules, the same held out
        assert all(r[3] and r[4] for r in runs)               # what is asked for reaches both runs
        assert found.heldout == {"blur": 10.0, "phase": better}
        if kept == "phase":
            assert found.focus_nm == 700.0 and found.fallback is None
        else:
            assert found.focus_nm == 650.0 and found.fallback == "phase"
            assert any("no better" in m for m in found.messages)
    with pytest.raises(ValueError, match="pupil"):
        monkeypatch.setattr(depth, "model_of", lambda cal, channels=1: None)
        depth.insitu_dual(dual, ["movie.tif"], CAMERA)


def test_spline_3d_2c_learns_in_situ_at_run_never_at_preview_and_asks_first(monkeypatch, tmp_path):
    # *learn* in the sample part, *learn registration* shown with it; the preflight asks, offering a fit
    # without learning; a Preview fits from the edge and learns nothing; a Run learns, keeps it, and fits with
    # what it kept -- the registration the model's where it was learnt
    from smappy.calibrate import depth
    from smappy.calibrate.depth import SampleSettings
    from smappy.plugins import Context, PreflightQuestion
    from smappy.plugins.fit import DualModelSettings, DualSplineFit, DualSplineFitSettings, SourceSettings
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(DualSplineFit, "unknown_camera", lambda self, settings: None)
    movie_path = tmp_path / "movie.tif"
    movie_path.write_bytes(b"")
    calibration = tmp_path / "pair_3dcal.h5"
    calibration.write_bytes(b"")
    settings = DualSplineFitSettings(source=SourceSettings(path=str(movie_path)),
                                     model=DualModelSettings(calibration=str(calibration),
                                                             sample=SampleSettings(focus="edge", learn="phase"),
                                                             learn_registration=True))
    plugin = DualSplineFit()
    asked = plugin.preflight(Context(), settings)
    assert isinstance(asked, PreflightQuestion) and "minutes" in asked.text
    assert asked.choices[0].settings.model.sample.learn == "nothing"
    linked = replace(settings, model=replace(settings.model, link_photons=True))
    assert plugin._insitu_key(settings) != plugin._insitu_key(linked)

    original = DualSplineFit.with_focus
    calls = []

    def with_focus(self, ctx, s, source, camera, finder):
        if s.model.sample.mode == "auto":
            calls.append("auto")
            return replace(s, model=replace(s.model, sample=replace(s.model.sample, focus="typed", focus_nm=600.0)))
        return original(self, ctx, s, source, camera, finder)
    monkeypatch.setattr(DualSplineFit, "with_focus", with_focus)
    monkeypatch.setattr(DualSplineFit, "registration", lambda self, s: SimpleNamespace(main="beads"))
    monkeypatch.setattr(depth, "has_model", lambda cal: True)
    learnt = []

    def insitu_dual(dual, paths, camera, n_med, focus_nm, learn, registration, share_per_molecule, progress,
                    cancel, hold_focus=False, order=None):
        learnt.append((focus_nm, learn, registration, share_per_molecule))
        out = depth.AutoDepth(710.0, 10.0, 0.1, 50, method="in situ", blur_nm=40.0)
        out.messages.append("a warning")
        return out
    monkeypatch.setattr(depth, "insitu_dual", insitu_dual)
    monkeypatch.setattr(depth, "keep", lambda key, found: depth.kept_path(key))
    ctx = Context()
    previewed = plugin.with_focus(ctx, settings, None, CAMERA, None)
    assert not learnt and calls == ["auto"] and previewed.model.sample.focus == "typed"
    plugin._may_learn = True
    used = plugin.with_focus(ctx, settings, None, CAMERA, None)
    # started where the edge put it; each molecule its own share, the photons not being linked
    assert learnt == [(600.0, "phase", True, True)]
    assert used.model.calibration == str(depth.kept_path(plugin._insitu_key(settings)))
    assert used.model.sample.focus == "calibrated" and used.model.sample.focus_nm == 710.0
    assert used.model.registration.map == "calibration"
    assert any("a warning" in w for w in ctx.warnings)
    # learnt once: the next run of these settings finds it
    plugin._may_learn = False
    learnt.clear()
    again = plugin.with_focus(ctx, settings, None, CAMERA, None)
    assert not learnt and again.model.calibration == used.model.calibration
    assert plugin.preflight(Context(), settings) is None


def test_the_form_offers_learn_registration_only_while_learning(monkeypatch):
    from smappy.calibrate.depth import SampleSettings
    from smappy.plugins import FormState
    from smappy.plugins.fit import DualModelSettings, DualSplineFit, DualSplineFitSettings
    plugin = DualSplineFit()
    for learn, shown in (("phase", True), ("nothing", False)):
        state = FormState()
        settings = DualSplineFitSettings(model=DualModelSettings(sample=SampleSettings(learn=learn)))
        monkeypatch.setattr("smappy.calibrate.depth.sample_state",
                            lambda main, sample, prefix, found: ({"model.sample.focus": True}, {}, {}))
        plugin._sample_form(settings, state)
        assert ("edge", "coverslip edge") in state.choices["model.sample.focus"]
        assert state.shown["model.learn_registration"] is shown


@pytest.mark.long
def test_spline_3d_2c_learns_both_pupils_a_start_lacks_in_situ_keeps_them_and_fits_closer_to_the_truth(
        biplane, monkeypatch, tmp_path):
    # a dual calibration with half the movie's astigmatism in both channels: a Run with *in situ* learns the
    # aberrations from the paired molecules -- near the movie's, with the channels' offset in z and their share --
    # explains the molecules held out better than the start with the focus learnt, keeps it, and fits with it:
    # z scatters less about the truth than with the start drawn at the focus on auto
    from smappy.calibrate import depth
    from smappy.calibrate import pupil as pupil_module
    from smappy.calibrate.depth import SampleSettings
    from smappy.calibrate.dual import save_dual_color_calibration
    from smappy.plugins import Context
    from smappy.plugins.fit import (
        CameraSettings,
        DualModelSettings,
        DualSplineFit,
        DualSplineFitSettings,
        OutputSettings,
        SourceSettings,
        SplineTransformSettings,
    )

    import uipsf.workflows
    import uipsf.workflows.insitu as insitu_module
    from uipsf.export import LearnedPSF, describe
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(uipsf.workflows, "insitu_model", small_model)
    monkeypatch.setattr(insitu_module, "HEIGHT_STEP_NM", 300.0)
    path, m = biplane
    model, shared = truth()
    start = {k: v.clone() for k, v in shared.items()}
    start["pupil.phase"][1] *= 0.5
    start["ch1.pupil.phase"][1] *= 0.5
    learned = LearnedPSF(model, start, {}, describe(model), geometry=dict(GEOMETRY),
                         options={"psf": {"zernike_order": 4, "blur_nm": 30.0, "pixel_blur": False}})
    dual = pupil_module.dual_calibration(learned, 20.0, 700.0, 17)
    for channel in (dual.main, dual.secondary):
        channel.parameters = dict(channel.parameters, model_file="pair.uipsf.h5")
    calibration = tmp_path / "pair_3dcal.h5"
    save_dual_color_calibration(calibration, dual)
    (tmp_path / "pair.uipsf.h5").write_bytes(b"")
    monkeypatch.setattr(depth, "load", lambda p, channels=1: learned)
    plugin = DualSplineFit()
    scatter = {}
    for learn in ("nothing", "phase"):
        settings = DualSplineFitSettings(source=SourceSettings(path=str(path)),
                                         camera=CameraSettings(conversion=2.0, offset=100.0, pixelsize_um=0.127,
                                                               pixelsize_y_um=0.116),
                                         # no channel move: the map the molecules are paired by and
                                         # held at is then the calibration's, exactly
                                         model=DualModelSettings(calibration=str(calibration), link_photons=True,
                                                                 sample=SampleSettings(learn=learn),
                                                                 registration=SplineTransformSettings(move="none")),
                                         output=OutputSettings(save=False))
        result = plugin.run(Context(), settings)
        scatter[learn] = _z_scatter(result.locs, m.truth)
    used = result.settings.model
    key = plugin._insitu_key(settings)
    assert used.sample.focus == "calibrated" and used.calibration == str(depth.kept_path(key))
    found = depth.found(key)
    assert found.fallback is None, found.messages
    assert found.heldout["phase"] < found.heldout["blur"]
    got = found.drawn.shared
    for pre, truth_phase in zip(("", "ch1."), PHASE, strict=True):
        assert float(got[pre + "pupil.phase"][1]) == pytest.approx(float(truth_phase[1]), abs=0.1)  # from half
    assert float(got["ch1.dz"]) == pytest.approx(DZ, abs=40)
    assert float(got["ch1.share"]) == pytest.approx(SHARE, abs=0.03)
    assert found.focus_nm == pytest.approx(FOCUS, abs=150)
    assert np.allclose(found.spline.transformation, H / H[2, 2], atol=1e-6)    # held: not asked to learn it
    assert scatter["phase"] < 0.9 * scatter["nothing"], scatter
    # kept as a dual calibration with its model inside: a new session reads it back instead of learning again
    depth._FOUND.clear()
    again = depth.kept(key, dual=True)
    assert again is not None and again.focus_nm == pytest.approx(found.focus_nm)
    assert again.heldout == found.heldout and depth.has_model(again.spline.main)


def _z_scatter(locs, truth) -> float:
    """The median absolute deviation of z less the truth's height, over the localizations within 100 nm of a
    molecule on in their frame."""
    x, y, z, f = (np.asarray(locs[k]) for k in ("x_nm", "y_nm", "z_nm", "frame"))
    dz = []
    for k in range(len(z)):
        rows = np.flatnonzero(truth.frame == f[k])
        if len(rows):
            d = np.hypot(truth.x_nm[rows] - x[k], truth.y_nm[rows] - y[k])
            if d.min() < 100:
                dz.append(z[k] - truth.h_nm[rows[np.argmin(d)]])
    dz = np.asarray(dz)
    assert len(dz) > 300
    return float(np.median(np.abs(dz - np.median(dz))))


def small_model(optics, spec, roi_shape, focus_nm, z_max_nm=2000.0, device=None, dtype=torch.float32, held=None,
                fit_focus=True, fit_tilt=False, field_centre_px=None, transforms=None, channel_options=None):
    """`insitu_model` of two channels with pupils of 16 samples on the CPU instead of 64: seconds a round."""
    from uipsf.model import make_psf, pupil_grid
    spec = replace(spec, saf_damping=False)
    optics = list(optics)
    held = list(held) if isinstance(held, (list, tuple)) else [held] * len(optics)
    options = [{}] + list(channel_options or [{}])
    channels = []
    for c, o in enumerate(optics):
        psf = make_psf(o, spec, roi_shape, held=held[c], grid=pupil_grid(o, spec, size=16, device="cpu", dtype=dtype))
        channels.append(Channel(psf, transform=(transforms or [None, None])[c], **options[c]))
    return uipsf.ForwardModel(channels, geometry=InSituGeometry(focus_nm, fit_focus=fit_focus, fit_tilt=fit_tilt),
                              intensity=Bleaching(order=0))


def test_a_dual_pupil_calibration_keeps_the_beads_em_gain_on_disk(tmp_path):
    """Without it a movie taken with the other EM gain -- mirrored in x by the EM register -- went unwarned:
    the Zenodo two-colour beads (normal port) against their movie (EM port)."""
    import smappy.calibrate.pupil as pupil_module
    from smappy.calibrate.dual import load_dual_color_calibration, save_dual_color_calibration

    from uipsf.export import LearnedPSF, describe
    model, shared = truth()
    learned = LearnedPSF(model, shared, {}, describe(model), geometry=dict(GEOMETRY),
                         options={"psf": {"zernike_order": 4, "blur_nm": 30.0, "pixel_blur": False}})
    for camera, em in ((CameraMetadata(em_on=True), True), (CameraMetadata(em_on=False), False), (None, None)):
        path = tmp_path / f"pair_{em}_3dcal.h5"
        save_dual_color_calibration(path, pupil_module.dual_calibration(learned, 20.0, 700.0, 17, camera=camera))
        read = load_dual_color_calibration(path)
        assert read.main.em_on is em and read.secondary.em_on is em
