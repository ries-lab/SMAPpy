"""Register layers: the shift of one layer against another, kept for Transform.

A layer and a shifted copy give the shift back (x, y, and z when asked); an
image layer against a picture of the same structure; and the transformation
the form offers, applied by Transform, lays the layers on each other.
"""
import numpy as np
import pytest

from smappy import plugins
from smappy.images import ImageData
from smappy.io.formats import FileInfo
from smappy.layer_registration import register
from smappy.locs import Localizations
from smappy.session import Session

PATH = "Analysis/Register/Register layers"


def structure(n=60000, seed=0, shift=(0.0, 0.0, 0.0)):
    """Localizations around 300 random points in 3D, 10 nm apart from them."""
    rng = np.random.default_rng(seed)
    sites = np.column_stack([rng.uniform(0, 6000, 300), rng.uniform(0, 6000, 300),
                             rng.uniform(-300, 300, 300)])
    points = sites[rng.integers(0, 300, n)] + rng.normal(0, 10, (n, 3)) + shift
    return points


def as_table(points, frame0=0):
    return Localizations({"x_nm": points[:, 0], "y_nm": points[:, 1],
                          "z_nm": points[:, 2],
                          "frame": frame0 + np.arange(len(points)) // 100}, {})


def two_files(shift):
    """One structure seen twice: file 2 shifted by ``shift``, a layer each."""
    session = Session()
    session.add_file(as_table(structure(seed=1)), FileInfo("one", "", "test"))
    second = structure(seed=1)[::-1] + shift          # other localizations order
    session.add_file(as_table(second), FileInfo("two", "", "test"), append=True)
    session.layers[0].set_files([0])
    session.add_layer().set_files([1])
    return session


def test_a_layer_and_a_shifted_copy_give_the_shift():
    a = structure(seed=2)
    b = structure(seed=2)[::-1] + (137.0, -52.0, 0.0)
    found = register({"x": a[:, 0], "y": a[:, 1]}, {"x": b[:, 0], "y": b[:, 1]})
    assert found.dx_nm == pytest.approx(137.0, abs=1.5)
    assert found.dy_nm == pytest.approx(-52.0, abs=1.5)
    assert found.dz_nm is None


def test_with_z_the_axial_shift_comes_back_too():
    session = two_files((40.0, -75.0, 60.0))
    settings = plugins.get(PATH).Settings(use_z=True)
    result = session.run(plugins.get(PATH)(), settings)
    dx, dy, dz = result.data["shift_nm"]
    assert (dx, dy, dz) == pytest.approx((40.0, -75.0, 60.0), abs=3.0)


def test_a_shift_beyond_max_shift_is_not_found():
    a = structure(seed=3)
    b = a + (900.0, 0.0, 0.0)
    near = register({"x": a[:, 0], "y": a[:, 1]}, {"x": b[:, 0], "y": b[:, 1]},
                    max_shift_nm=400.0)
    assert abs(near.dx_nm - 900.0) > 100
    anywhere = register({"x": a[:, 0], "y": a[:, 1]}, {"x": b[:, 0], "y": b[:, 1]},
                        max_shift_nm=None)
    assert anywhere.dx_nm == pytest.approx(900.0, abs=2.0)


def test_an_image_layer_against_a_rendering_of_the_same_structure():
    """A widefield-like picture -- blurred, on a pedestal, coarse pixels --
    placed 230 nm right and 120 nm up of the localizations."""
    from scipy.ndimage import gaussian_filter
    points = structure(seed=4)
    pixel = 50.0
    counts, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=(120, 120),
                                  range=((0, 6000), (0, 6000)))
    picture = gaussian_filter(counts, 3.0) + 20.0
    image = ImageData(picture.astype(np.float32), pixel, x0=230.0, y0=-120.0, name="wf")
    session = Session(as_table(points))
    session.add_image(image)
    result = session.run(plugins.get(PATH)(), plugins.get(PATH).Settings(
        reference=1, target=2, bin_nm=25.0))
    dx, dy, _ = result.data["shift_nm"]
    assert dx == pytest.approx(230.0, abs=8.0)
    assert dy == pytest.approx(-120.0, abs=8.0)


def test_the_offered_transformation_applied_by_transform_aligns_the_layers():
    session = two_files((-310.0, 145.0, 0.0))
    plugin = plugins.get(PATH)()
    settings = plugin.Settings()
    session.run(plugin, settings)
    name = "layer 2 onto layer 1"
    assert name in session.transformations
    notice = plugin.describe(session.context(), settings).notice
    assert notice.window == "transform" and notice.action == "Apply with Transform"
    assert (notice.settings.source, notice.settings.named) == ("named", name)
    assert notice.settings.file == 1                  # the target layer's file
    session.run(plugins.get("Analysis/Process/Transform")(), notice.settings)
    after = session.run(plugin, plugin.Settings(name="check"))
    assert np.hypot(*after.data["shift_nm"][:2]) < 2.0
    # the reference file did not move
    first = session.locs[np.asarray(session.locs["filenumber"]) == 0]
    np.testing.assert_allclose(np.sort(first["x_nm"]),
                               np.sort(structure(seed=1)[:, 0]), atol=1e-3)


def test_the_same_layer_twice_and_z_of_an_image_are_refused():
    session = Session(as_table(structure(seed=5)))
    session.add_image(ImageData(np.ones((10, 10), np.float32), 100.0))
    plugin = plugins.get(PATH)()
    with pytest.raises(ValueError, match="same layer"):
        session.run(plugin, plugin.Settings(reference=1, target=1))
    with pytest.raises(ValueError, match="no z"):
        session.run(plugin, plugin.Settings(reference=1, target=2, use_z=True))


def test_after_a_run_the_panel_offers_to_apply_it_and_opens_transform(tmp_path,
                                                                       monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path))
    from smappy.gui.plugin_panel import PluginWindow
    session = two_files((50.0, 20.0, 0.0))
    opened = []
    session.window_openers["transform"] = opened.append
    window = PluginWindow(plugins.get(PATH), session)
    panel = window.panel
    assert panel.notice_button.isHidden()             # nothing measured yet
    session.run(plugins.get(PATH)(), panel.form.value())
    assert not panel.notice_button.isHidden()
    panel.notice_button.click()
    assert opened[0].named == "layer 2 onto layer 1" and opened[0].file == 1
    window.close()
    app.processEvents()
