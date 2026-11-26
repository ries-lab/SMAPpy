"""*Pixel equalisation*: correct, revert, and the cases it refuses."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.pixel_equalisation import (
    SHIFTS,
    PixelEqualisation,
    PixelEqualisationSettings,
)
from smappy.training.reference import pixel_modulation

PIXEL = (110.0, 120.0)


def _pixelated(n=200_000, files=1, seed=0):
    """A fit's table whose places within the pixel are pulled toward its centre, more at large |z|."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 60, (n, 2))
    z = rng.uniform(-600, 600, n)
    edge = np.round(t) - 0.5
    u = t - edge
    f = edge + u + (0.03 + 0.1 * np.abs(z)[:, None] / 600) * np.sin(2 * np.pi * u) / (2 * np.pi)
    err = rng.uniform(5, 20, (n, 2))
    columns = {"x_nm": (f[:, 0] * PIXEL[0]).astype(np.float32), "y_nm": (f[:, 1] * PIXEL[1]).astype(np.float32),
               "z_nm": z.astype(np.float32), "x_err_nm": err[:, 0].astype(np.float32),
               "y_err_nm": err[:, 1].astype(np.float32), "frame": rng.integers(0, 1000, n),
               "filenumber": np.arange(n) % files}
    return Localizations(columns, {"pixelsize_nm": float(np.mean(PIXEL)), "pixelsize_nm_xy": list(PIXEL)})


def _modulation(locs):
    return pixel_modulation(np.r_[np.asarray(locs["x_nm"], float) / PIXEL[0],
                                  np.asarray(locs["y_nm"], float) / PIXEL[1]])


def test_correcting_flattens_the_places_and_running_again_reverts_to_the_table_as_it_was():
    locs = _pixelated()
    plugin, settings = PixelEqualisation(), PixelEqualisationSettings()
    assert plugin.describe(Context(locs=locs), settings).run_label == "Correct"
    corrected = plugin.run(Context(locs=locs), settings).locs
    assert _modulation(locs) > 0.05 and _modulation(corrected) < 0.01
    assert all(c in corrected for c in SHIFTS) and "pixel_equalisation" in corrected.metadata
    assert plugin.describe(Context(locs=corrected), settings).run_label == "Revert"
    back = plugin.run(Context(locs=corrected), settings).locs
    assert not any(c in back for c in SHIFTS) and "pixel_equalisation" not in back.metadata
    for c in ("x_nm", "y_nm"):
        np.testing.assert_allclose(back[c], locs[c], atol=0.02)


def test_a_revert_after_a_translation_takes_off_the_same_shifts():
    locs = _pixelated()
    plugin, settings = PixelEqualisation(), PixelEqualisationSettings()
    corrected = plugin.run(Context(locs=locs), settings).locs
    columns = dict(corrected.columns)
    columns["x_nm"] = np.asarray(columns["x_nm"]) + np.float32(37.0)      # as a drift correction moves it
    back = plugin.run(Context(locs=Localizations(columns, dict(corrected.metadata))), settings).locs
    np.testing.assert_allclose(back["x_nm"], np.asarray(locs["x_nm"]) + 37.0, atol=0.02)


def test_each_file_is_equalised_on_its_own_maps():
    locs = _pixelated(files=2)
    out = PixelEqualisation().run(Context(locs=locs), PixelEqualisationSettings()).locs
    assert len(out.metadata["pixel_equalisation"]["maps"]) == 2


def test_a_corrected_or_drift_corrected_table_is_not_corrected_again_and_a_plain_one_not_reverted():
    plugin = PixelEqualisation()
    locs = _pixelated(20_000)
    with pytest.raises(ValueError, match="not been equalised"):
        plugin.run(Context(locs=locs), PixelEqualisationSettings(action="revert"))
    corrected = plugin.run(Context(locs=locs), PixelEqualisationSettings()).locs
    with pytest.raises(ValueError, match="already equalised"):
        plugin.run(Context(locs=corrected), PixelEqualisationSettings(action="correct"))
    drifted = Localizations(dict(locs.columns), dict(locs.metadata, drift_correction={"method": "RCC"}))
    with pytest.raises(ValueError, match="drift-corrected"):
        plugin.run(Context(locs=drifted), PixelEqualisationSettings())
    assert plugin.describe(Context(locs=drifted), PixelEqualisationSettings()).notice is not None


def test_the_panels_run_button_says_correct_or_revert_as_the_table_changes():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from smappy.gui.plugin_panel import PluginPanel
    app = QApplication.instance() or QApplication([])
    from smappy.session import Session
    locs = _pixelated(20_000)
    session = Session(locs)
    panel = PluginPanel(PixelEqualisation, session)
    assert panel.run_button.text() == "Correct"
    session.set_locs(PixelEqualisation().run(Context(locs=locs), PixelEqualisationSettings()).locs)
    app.processEvents()
    assert panel.run_button.text() == "Revert"
    session.set_locs(locs)
    app.processEvents()
    assert panel.run_button.text() == "Correct"


def test_after_the_fit_equalises_a_networks_table_first_and_passes_over_any_other():
    from dataclasses import replace
    from smappy.plugins.after_fit import AfterFitSettings, finish_localizations
    locs = _pixelated(50_000)
    wanted = replace(AfterFitSettings(), pixel_equalisation=True, assign_colors=False)
    # a table no network fitted has no grid to take out: the step is not run
    plain = finish_localizations(locs, wanted)
    assert not plain.changed and not any(c in plain.locs for c in SHIFTS)
    decoded = Localizations(dict(locs.columns), dict(locs.metadata, decode={"model": "m.safetensors"}))
    finished = finish_localizations(decoded, wanted)
    assert finished.changed and all(c in finished.locs for c in SHIFTS)
    assert _modulation(finished.locs) < _modulation(locs) / 3
    assert finished.history[0]["what"] == "Analysis/After fit/Pixel equalisation"
    # first, before the drift: the rest of the steps see equalised positions
    assert wanted.steps()[0][0].path == "Analysis/After fit/Pixel equalisation"
    assert wanted.steps()[0][1].action == "correct"


def test_only_the_network_fitter_offers_pixel_equalisation_after_the_fit():
    from smappy.plugins.fit import GaussianFit, SplineFit
    from smappy.plugins.fit_decode import DecodeFit

    def shown(cls, settings=None):
        state = cls().describe(None, settings or cls.Settings())
        return state, {k[len("finish."):]: v for k, v in state.shown.items() if k.startswith("finish.")}

    for cls in (GaussianFit, SplineFit):
        assert shown(cls)[1]["pixel_equalisation"] is False
    settings = DecodeFit.Settings()
    state, on = shown(DecodeFit, settings)
    assert on["pixel_equalisation"] is True and on["equalisation"] is False      # until ticked
    settings.finish.pixel_equalisation = True
    state, on = shown(DecodeFit, settings)
    assert on["equalisation"] is True and on["equalisation.action"] is False
    assert state.titles["finish"].startswith("after the fit: pixels")


def test_the_halves_of_a_split_chip_get_maps_of_their_own():
    from smappy.plugins.pixel_equalisation import _groups
    locs = _pixelated(1000)
    columns = dict(locs.columns, camera_channel=np.arange(1000) % 2)
    assert sorted(_groups(Localizations(columns, {}))) == [(0, 0), (0, 1)]
