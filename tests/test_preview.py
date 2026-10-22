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
        changes_locs = False
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
    hints = plugin.describe(None, settings).hints
    # each field's, and the microscope control's own account under the part's name
    assert set(hints) - {"camera"} == {f"camera.{n}" for n in plugin.CAMERA_FIELDS}
    # a value the user did set comes back as theirs, not as the file's
    settings.camera.offset = 42.0
    assert plugin.describe(None, settings).hints["camera.offset"] == 42.0


def test_hints_are_silent_when_there_is_nothing_to_resolve():
    plugin = GaussianFit()
    assert not plugin.describe(None, plugin.Settings()).hints
    settings = plugin.Settings()
    settings.source.path = "/nowhere/at/all.tif"
    assert not plugin.describe(None, settings).hints


def _maxima_of_frame(plugin, settings, path, index):
    """The frame's strict 3x3 maxima in the filtered image, computed here and
    not by the code under test, and the candidates detection kept."""
    import tifffile
    from smappy.detect import find_candidates, local_maxima
    camera = plugin._camera(settings)
    photons = ((tifffile.imread(path)[index].astype(float) - camera.offset)
               / camera.conversion)
    finder = plugin.finder(settings, camera)
    filtered = finder.filter(photons[None].astype(np.float32))
    values = local_maxima(filtered, use_extension=False)[3]
    kept = find_candidates(filtered, finder.cutoff).value
    return values, kept


@pytest.mark.parametrize("mode, value", [("dynamic", 1.7), ("absolute", 20.0)])
def test_the_reported_cutoff_is_the_one_detection_applied(stack, mode, value):
    """Once a percentile of every pixel was reported: 322 where detection held
    the frame's maxima to 8.2."""
    from smappy.detect import DynamicCutoff
    plugin = GaussianFit()
    settings = settings_for(plugin, stack)
    settings.detection.cutoff_mode, settings.detection.cutoff = mode, value
    result = plugin.preview(Session().context(), settings, frame=1)
    cutoff = result.data["cutoff"]

    values, kept = _maxima_of_frame(plugin, settings, stack, 1)
    expected = DynamicCutoff(value)(values) if mode == "dynamic" else value
    assert cutoff == pytest.approx(expected, rel=1e-6)
    if mode == "absolute":
        assert cutoff == value                  # the fixed number, as typed
    assert cutoff < 50                          # not the brightest pixels'
    # the threshold that split kept from dropped
    assert np.all(kept > cutoff) and not np.any(np.setdiff1d(values, kept) > cutoff)
    assert len(kept) == result.data["candidates"] == 4
    assert f"cutoff {cutoff:.3g} on the filtered image" in result.text

    figure = plt.figure()
    result.plot.draw_into(figure)
    bar = figure.axes[-1]
    lines = [line.get_ydata()[0] for line in bar.lines]
    assert lines == [pytest.approx(cutoff)]
    plt.close(figure)
