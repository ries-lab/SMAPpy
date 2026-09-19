"""The preview: one frame, drawn, with nothing saved."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt    # noqa: E402

from smappy.plugins import Plugin                          # noqa: E402
from smappy.plugins.fit import GaussianFit, SplineFit      # noqa: E402
from smappy.session import Session                         # noqa: E402


@pytest.fixture
def stack(tmp_path):
    """A few frames with a handful of bright, well-separated emitters."""
    import tifffile
    rng = np.random.default_rng(7)
    frames = np.full((4, 48, 60), 100.0)
    yy, xx = np.mgrid[:48, :60]
    for y, x in ((10, 12), (10, 40), (34, 20), (34, 48)):
        frames += 900*np.exp(-((yy-y)**2+(xx-x)**2)/(2*1.3**2))
    frames = rng.poisson(frames).astype(np.uint16)
    path = tmp_path / "run_MMStack.ome.tif"
    tifffile.imwrite(path, frames)
    return path


def settings_for(plugin, path):
    s = plugin.Settings()
    s.source.path = str(path)
    s.camera.conversion = 1.0
    s.camera.offset = 100.0
    s.camera.pixelsize_um = 0.1
    s.camera.em_on = False
    return s


def test_only_the_plugins_that_can_preview_say_so():
    assert GaussianFit.has_preview() and SplineFit.has_preview()

    class Plain(Plugin):
        pass

    assert not Plain.has_preview()


def test_a_preview_detects_and_fits_one_frame_and_changes_nothing(stack):
    plugin = GaussianFit()
    session = Session()
    result = plugin.preview(session.context(), settings_for(plugin, stack), frame=2)
    assert result.data["frame"] == 2
    assert result.data["candidates"] == 4
    assert result.data["error"] == ""
    assert len(result.data["locs"]) == 4
    assert "4 fitted" in result.text
    assert result.locs is None and not result.files     # nothing to apply
    assert len(session.locs) == 0
    figure = plt.figure()
    result.plot.draw_into(figure)                     # three panels, declared
    assert result.plot.panels == 3            # the three, plus a colour bar
    assert len(figure.axes) == 4
    plt.close(figure)


def test_a_frame_past_the_end_is_clamped_rather_than_an_error(stack):
    plugin = GaussianFit()
    result = plugin.preview(Session().context(), settings_for(plugin, stack),
                            frame=10_000)
    assert result.data["frame"] == 3


def test_without_a_calibration_the_detections_are_still_drawn(stack):
    plugin = SplineFit()
    result = plugin.preview(Session().context(), settings_for(plugin, stack), frame=0)
    assert result.data["candidates"] == 4
    assert result.data["locs"] is None
    assert "calibration" in result.data["error"]
    assert "not fitted" in result.text
    figure = plt.figure()
    result.plot.draw_into(figure)                     # draws, with the warning
    assert figure._suptitle.get_text().startswith("detection only")
    plt.close(figure)


def test_the_camera_hints_say_what_auto_resolves_to(stack):
    plugin = GaussianFit()
    settings = plugin.Settings()
    settings.source.path = str(stack)
    hints = plugin.hints(settings)
    assert set(hints) == {f"camera.{n}" for n in plugin.CAMERA_FIELDS}
    # a value the user did set comes back as theirs, not as the file's
    settings.camera.offset = 42.0
    assert plugin.hints(settings)["camera.offset"] == 42.0


def test_hints_are_silent_when_there_is_nothing_to_resolve():
    plugin = GaussianFit()
    assert plugin.hints(plugin.Settings()) is None
    settings = plugin.Settings()
    settings.source.path = "/nowhere/at/all.tif"
    assert plugin.hints(settings) is None
