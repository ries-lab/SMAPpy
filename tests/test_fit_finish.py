"""What the two-channel fits do once the last frame is in.

Drift correction and colour assignment need the whole acquisition -- a drift
curve is measured across it, the colour histogram's modes are a property of
the sample -- so they run once at the end rather than per block, and the file
that is saved is the finished table.  Here: that both steps really run, that
the saved file is the finished one, and that a step which fails costs its own
step and not the fit.
"""
import numpy as np
import pytest
from test_assign_colors import simulate as simulate_colours
from test_drift import SETTINGS as COMET_SETTINGS
from test_drift import simulate as simulate_drift
from test_dual_gauss_fit import SHAPE, SPLIT, WIDTHS, camera, draw, transform

from smappy.locs import Localizations
from smappy.plugins.fit import FinishSettings, finish_localizations


def drifted_two_colours(n_frames=50, per_frame=40, seed=3):
    """Two dyes on a fixed structure, the whole thing moved by a known drift.

    The two simulations are independent -- a molecule's colour has nothing to
    do with where the stage went -- so the colour table's photons are grafted
    onto the drift table's positions rather than simulated together.
    """
    locs, drift, true_xyz = simulate_drift(n_frames=n_frames, per_frame=per_frame,
                                           seed=seed)
    colours, species = simulate_colours(n=len(locs), seed=seed)
    for name in ("photons", "photons_ch0", "photons_ch1",
                 "photons_err_ch0", "photons_err_ch1"):
        locs.columns[name] = colours[name]
    return locs, drift, true_xyz, species


def test_the_finishing_step_subtracts_the_drift_and_then_assigns_the_colours():
    locs, drift, true_xyz, species = drifted_two_colours()
    settings = FinishSettings(assign_colors=True, drift="comet",
                              comet=COMET_SETTINGS)

    said = []
    finished = finish_localizations(locs, settings, said.append)

    assert finished.changed
    # the drift is gone: every localization sits on the structure it came from
    moved = np.c_[finished.locs["x_nm"], finished.locs["y_nm"], finished.locs["z_nm"]]
    residual = moved - true_xyz
    assert np.abs(residual - residual.mean(0)).mean() < 2.0
    # and the colours are the ones that were simulated
    assert (finished.locs["channel"] == species).mean() > 0.99

    # both runs are in the log, with the settings they used, and both figures
    # came back with the result
    assert [entry["what"] for entry in finished.history] == [
        "Analysis/Drift/COMET", "Analysis/After fit/Assign colours"]
    assert finished.history[0]["settings"]["max_drift_nm"] == COMET_SETTINGS.max_drift_nm
    assert len(finished.notes) == 2
    assert any("Assign colours" in name for name in finished.plots)


def test_the_rcc_estimator_is_the_other_choice():
    locs, drift, true_xyz, _ = drifted_two_colours(n_frames=60, per_frame=120)
    finished = finish_localizations(
        locs, FinishSettings(assign_colors=False, drift="rcc"))

    assert finished.changed and len(finished.history) == 1
    assert finished.history[0]["what"] == "Analysis/Drift/RCC"
    assert "channel" not in finished.locs
    moved = np.c_[finished.locs["x_nm"], finished.locs["y_nm"]]
    residual = moved - true_xyz[:, :2]
    assert np.abs(residual - residual.mean(0)).mean() < 15.0


def test_nothing_is_asked_for_and_nothing_happens():
    locs, _, _, _ = drifted_two_colours(n_frames=5, per_frame=5)
    finished = finish_localizations(
        locs, FinishSettings(assign_colors=False, drift="none"))
    assert finished.locs is locs and not finished.changed and not finished.history


def test_a_finishing_step_that_fails_costs_only_itself():
    """A fit that ran for twenty minutes must not be lost to a missing file."""
    locs, _ = widths_table(100)
    finished = finish_localizations(locs, FinishSettings(astigmatic_z=True))

    assert finished.locs is locs and not finished.changed
    assert len(finished.notes) == 1 and "failed" in finished.notes[0]


def test_a_step_the_table_cannot_take_is_passed_over_not_failed():
    """One part for every fitter: colours after a one-colour fit, z without
    the widths, are nothing to run rather than a failure to report."""
    locs = Localizations({"frame": np.arange(100, dtype=np.int64),
                          "x_nm": np.zeros(100), "y_nm": np.zeros(100),
                          "photons": np.full(100, 500.0)}, {"units": "nm"})
    finished = finish_localizations(
        locs, FinishSettings(assign_colors=True, astigmatic_z=True))
    assert finished.locs is locs and not finished.changed and not finished.notes


def widths_table(n=400, seed=0):
    """An elliptical Gaussian fit's table: each localization's widths at a known
    z, on the typed calibration's curves (the after-fit part's default)."""
    from smappy.plugins.astigmatic_z import TypedCalibration
    cal = TypedCalibration().calibration()
    rng = np.random.default_rng(seed)
    z = rng.uniform(-500.0, 500.0, n)
    sx, sy = cal.widths(z)
    return Localizations({"frame": np.arange(n, dtype=np.int64),
                          "x_nm": rng.uniform(0, 5000, n), "y_nm": rng.uniform(0, 5000, n),
                          "photons": np.full(n, 1000.0),
                          "sigma_x_nm": np.asarray(sx), "sigma_y_nm": np.asarray(sy)},
                         {"units": "nm"}), z


def typed_z(**more):
    from smappy.plugins.astigmatic_z import AstigmaticZSettings
    return FinishSettings(astigmatic_z=True, assign_colors=False,
                          astigmatic=AstigmaticZSettings(calibration="typed"), **more)


def test_astigmatic_z_turns_the_widths_into_the_z_they_were_drawn_at():
    locs, z = widths_table()
    finished = finish_localizations(locs, typed_z())
    assert finished.changed
    assert np.abs(finished.locs["z_nm"] - z).max() < 1.0
    assert [entry["what"] for entry in finished.history] == ["Analysis/After fit/Astigmatic z"]


def test_z_comes_before_the_drift_so_that_the_drift_can_correct_it():
    settings = FinishSettings(astigmatic_z=True, drift="comet", assign_colors=True)
    assert [plugin.path for plugin, _ in settings.steps()] == [
        "Analysis/After fit/Astigmatic z", "Analysis/Drift/COMET",
        "Analysis/After fit/Assign colours"]


def test_a_chain_runs_last_on_the_finished_table(tmp_path):
    from smappy.chain import ChainSpec, Step, write
    from smappy.plugins.after_fit import AfterFitSettings
    spec = ChainSpec(steps=[Step(plugin="Analysis/After fit/Astigmatic z",
                                 values={"calibration": "typed"})])
    path = write(spec, tmp_path / "z.chain.yaml")
    locs, z = widths_table()
    settings = AfterFitSettings(assign_colors=False, chain=str(path))
    finished = finish_localizations(locs, settings)
    assert finished.changed and np.abs(finished.locs["z_nm"] - z).max() < 1.0
    assert [entry["what"] for entry in finished.history][-1].startswith("Analysis/Chains/")


def test_a_chain_that_fits_is_refused_and_the_rest_still_runs(tmp_path):
    from smappy.chain import ChainSpec, Step, write
    spec = ChainSpec(steps=[Step(plugin="Localize/Gaussian 2D")])
    path = write(spec, tmp_path / "fits.chain.yaml")
    locs, z = widths_table()
    finished = finish_localizations(locs, typed_z(chain=str(path)))
    assert "nothing to fit" in finished.notes[0]
    assert np.abs(finished.locs["z_nm"] - z).max() < 1.0


def test_each_fitter_shows_only_the_steps_its_table_can_take():
    """Colours after two channels, z after an elliptical Gaussian; each
    step's settings only while it is on, a drift method's only when chosen."""
    from smappy.plugins.fit import (DualGaussianFit, DualSplineFit, GaussianFit,
                                    GaussianFitSettings, SplineFit)

    def shown(cls, settings=None):
        state = cls().describe(None, settings or cls.Settings())
        return {k[len("finish."):]: v for k, v in state.shown.items()
                if k.startswith("finish.")}

    for cls in (GaussianFit, SplineFit):
        assert shown(cls)["assign_colors"] is False and shown(cls)["colors"] is False
    for cls in (DualGaussianFit, DualSplineFit):
        assert shown(cls)["assign_colors"] is True and shown(cls)["colors"] is True
        assert shown(cls)["astigmatic_z"] is False
    plain = shown(GaussianFit)
    assert plain["astigmatic_z"] is False and plain["rcc"] is False and plain["comet"] is False
    elliptical = GaussianFitSettings()
    elliptical.model.elliptical = True
    assert shown(GaussianFit, elliptical)["astigmatic_z"] is True
    assert shown(GaussianFit, elliptical)["astigmatic"] is False      # until ticked
    elliptical.finish.astigmatic_z = True
    elliptical.finish.drift = "rcc"
    on = shown(GaussianFit, elliptical)
    assert on["astigmatic"] and on["rcc"] and not on["comet"]
    assert GaussianFit().describe(None, elliptical).titles["finish"] == \
        "after the fit: astigmatic z, RCC"


def test_the_fitters_button_runs_after_the_fit_on_the_open_table_as_its_own_step():
    """Not ticked during the fit: the button inside *after the fit* runs the
    steps on the session's table, one undo step, logged as After the fit."""
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.plugins.fit import GaussianFit
    from smappy.session import Session

    locs, z = widths_table()
    session = Session(locs)
    panel = PluginPanel(GaussianFit, session)
    panel.form.set_values({"model.elliptical": True, "finish.astigmatic_z": True,
                           "finish.astigmatic.calibration": "typed"})
    button = panel.part_buttons["finish"]
    assert button.isEnabled()
    loop = QEventLoop()
    panel.ended.connect(lambda *_: loop.quit())
    button.click()
    QTimer.singleShot(20000, loop.quit)
    loop.exec()
    app.processEvents()

    assert np.abs(session.locs["z_nm"] - z).max() < 1.0
    assert session.history[-1]["what"] == "Analysis/After fit/After the fit"
    session.undo()
    assert "z_nm" not in session.locs


def test_colours_are_assigned_by_default_and_drift_is_not():
    """The two defaults, which are the point of the feature: colour assignment
    is seconds and is what the fit was for, a drift estimate is minutes."""
    from smappy.plugins.fit import DualGaussianFitSettings, DualSplineFitSettings

    for cls in (DualGaussianFitSettings, DualSplineFitSettings):
        finish = cls().finish
        assert finish.assign_colors is True
        assert finish.drift == "none"


# ------------------------------------------------------------- the whole fit
def two_colour_movie(n_frames, rng, t, ratios=(0.25, 0.75), photons=6000.,
                     background=15., per=4):
    """A split-frame movie whose emitters are one of two species.

    ``ratio`` is the fraction of the photons the secondary half gets, so the
    two species land at r = +0.5 and -0.5 -- far enough apart that what is
    tested is the wiring and not the colour assignment, which has its own
    tests.
    """
    forward = np.linalg.inv(t.transformation)
    frames = []
    for _ in range(n_frames):
        frame = np.full(SHAPE, float(background))
        positions = np.c_[rng.uniform(6, SHAPE[1] - 6, per),
                          rng.uniform(6, SPLIT - 6, per)]
        for (x, y), ratio in zip(positions, rng.choice(ratios, per)):
            partner = (forward @ np.array([x, y, 1.0]))[:2]
            draw(frame, x, y, photons * (1 - ratio), WIDTHS[0])
            draw(frame, partner[0], partner[1], photons * ratio, WIDTHS[1])
        frames.append(rng.poisson(np.clip(frame, 0, None)).astype(np.float32))
    return np.stack(frames)


def test_the_2c_fit_saves_the_table_with_its_colours_on_it(tmp_path):
    """End to end: the file the fit leaves behind is the finished one, not the
    raw fit the writer streamed while the frames were coming in."""
    import tifffile

    from smappy.io.hdf5 import load_localizations
    from smappy.pipeline import FitSettings
    from smappy.plugins import Context
    from smappy.plugins.fit import (CameraSettings, ChannelTransformSettings,
                                    DetectionSettings, DualGaussianFit,
                                    DualGaussianFitSettings,
                                    DualGaussianModelSettings, OutputSettings,
                                    SourceSettings)
    from smappy.calibrate.transform import save_channel_transform

    rng = np.random.default_rng(11)
    t = transform()
    stack = tmp_path / "split.tif"
    tifffile.imwrite(stack, (two_colour_movie(120, rng, t) + 100).astype(np.uint16))
    given = tmp_path / "given_2ct.h5"
    save_channel_transform(given, t)

    out = tmp_path / "out.hdf5"
    settings = DualGaussianFitSettings(
        source=SourceSettings(path=str(stack), chunk=40),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=25.0),
        model=DualGaussianModelSettings(sigma=1.2),
        transform=ChannelTransformSettings(path=str(given)),
        fit=FitSettings(roisize=13),
        output=OutputSettings(path=str(out)))

    said = []
    result = DualGaussianFit().run(Context(progress=said.append), settings)

    assert result.locs is not None and "channel" in result.locs
    assert set(np.unique(result.locs["channel"])) <= {0, 1, 2}
    assert (result.locs["channel"] > 0).mean() > 0.9
    assert "Assign colours" in result.text

    saved = load_localizations(out)
    assert len(saved) == len(result.locs)
    assert "channel" in saved and "color_ratio" in saved
    # and the file says what was done to it
    history = saved.metadata["history"]
    assert [entry["what"] for entry in history] == \
        ["Analysis/After fit/Assign colours"]
