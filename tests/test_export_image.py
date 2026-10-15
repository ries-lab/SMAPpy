"""Export Image: one renderer of the view, for the plugin and the windows' Save."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.picture import (ViewFrame, auto_bar, pixel_size_of, read_description,
                            render_picture)
from smappy.plugins import Context, settings_from
from smappy.plugins.file import ExportImageSettings, export_image
from smappy.session import Session

Image = pytest.importorskip("PIL.Image")


def table(n=6000, seed=0) -> Localizations:
    rng = np.random.default_rng(seed)
    return Localizations({
        "x_nm": rng.uniform(0, 4000, n), "y_nm": rng.uniform(0, 3000, n),
        "z_nm": rng.normal(0, 200, n), "xy_err_nm": rng.uniform(5, 15, n),
        "frame": np.arange(n),
    }, {"units": "nm"})


def two_layers() -> Session:
    """Two layers of one table: the second coloured by z."""
    session = Session(table())
    session.add_layer()
    session.layers[1].state.settings.color_field = "z_nm"
    session.layers[1].state.settings.color_range = (-300.0, 300.0)
    session.layers[1].state.display.lut = "jet"
    return session


def frame_on(session, pixel=20.0, panels=("overlay",), stacked=False):
    session.view_frames["2D"] = lambda: ViewFrame((0.0, 4000.0), (0.0, 3000.0), pixel,
                                                  tuple(panels), stacked)


def pixels(path):
    with Image.open(path) as image:
        return np.asarray(image)


BARE = dict(scale_bar=False, colour_bars=False, labels=False)


def test_the_composite_is_the_layers_added_up_as_the_window_renders_them(tmp_path):
    from smappy.layer_state import render_panels, snapshot_layers
    from smappy.render import FieldOfView
    session = two_layers()
    frame_on(session)
    path, = export_image(session.context(), ExportImageSettings(
        path=str(tmp_path / "c.png"), what="composite", **BARE))
    fov = FieldOfView.from_range((0, 4000), (0, 3000), 20.0)
    (overlay,), _ = render_panels(snapshot_layers(session.layers), fov)
    assert pixels(path).shape == (150, 200, 3)
    assert np.array_equal(pixels(path), np.round(overlay * 255).astype(np.uint8))


def test_each_layer_comes_out_in_its_colour_and_in_grey(tmp_path):
    from smappy.layer_state import render_panels, snapshot_layers
    from smappy.render import FieldOfView, normalize
    session = two_layers()
    frame_on(session)
    fov = FieldOfView.from_range((0, 4000), (0, 3000), 20.0)
    planes = {}
    images, _ = render_panels(snapshot_layers(session.layers), fov, [0, 1], planes=planes)
    colour = export_image(session.context(), ExportImageSettings(
        path=str(tmp_path / "c.png"), what="layers", **BARE))
    assert [p.name for p in colour] == ["c_layer1.png", "c_layer2.png"]
    for path, image in zip(colour, images):
        assert np.array_equal(pixels(path), np.round(image * 255).astype(np.uint8))
    # the second is coloured by z, the first is not: they differ in hue
    a, b = pixels(colour[0]).astype(int), pixels(colour[1]).astype(int)
    assert (np.abs(a - b).sum(axis=2) > 60).mean() > 0.01

    grey = export_image(session.context(), ExportImageSettings(
        path=str(tmp_path / "g.tif"), what="layers grey", **BARE))
    for i, path in enumerate(grey):
        written = pixels(path)
        assert written.ndim == 2 and written.dtype == np.uint8
        expected, _ = normalize(planes[i].weight, None, session.layers[i].state.display.contrast)
        assert np.array_equal(written, np.round(expected * 255).astype(np.uint8))
    # one table, one intensity: the colour by z goes, the grey stays the same
    assert np.array_equal(pixels(grey[0]), pixels(grey[1]))


def test_the_intensity_is_the_summed_weights_as_floats_in_a_tiff(tmp_path):
    import tifffile
    from smappy.layer_state import render_panels, snapshot_layers
    from smappy.render import FieldOfView
    session = two_layers()
    frame_on(session)
    path, = export_image(session.context(), ExportImageSettings(
        path=str(tmp_path / "i.png"), what="intensity"))
    assert path.suffix == ".tif"                    # PNG has no float pixels
    fov = FieldOfView.from_range((0, 4000), (0, 3000), 20.0)
    _, weight = render_panels(snapshot_layers(session.layers), fov)
    data = tifffile.imread(path)
    assert data.dtype == np.float32 and np.allclose(data, weight)


@pytest.mark.parametrize("pixel, length", [(5.0, 1000.0), (8.0, 2000.0)])
def test_the_scale_bar_is_its_length_over_the_pixel_size_in_pixels(pixel, length):
    from smappy.layer_state import LayerSnapshot
    from smappy.render import FieldOfView, RenderSettings
    locs = table(10)
    nothing = LayerSnapshot(name="empty", locs=locs, mask=np.zeros(len(locs), bool),
                            settings=RenderSettings())
    fov = FieldOfView.from_range((0, 6000), (0, 3000), pixel)
    picture, = render_picture((nothing,), fov, what="composite", scale_bar_length=length,
                              colour_bars=False, labels=False)
    n = int(round(length / pixel))
    assert picture.scale_bar_px == n
    # on a black picture the bar is the only wide run of white: its rows
    # hold exactly its length
    white = (picture.image == 255).all(axis=2)
    assert white.sum(axis=1).max() == n
    # automatic: the window's, a round fifth of the width
    picture, = render_picture((nothing,), fov, what="composite", colour_bars=False,
                              labels=False)
    assert picture.scale_bar == auto_bar(6000) == 2000.0
    assert (picture.image == 255).all(axis=2).sum(axis=1).max() == round(2000 / pixel)


@pytest.mark.parametrize("suffix", [".tif", ".png", ".jpg"])
def test_the_description_round_trips_the_settings(tmp_path, suffix):
    session = two_layers()
    session.layers[0].set_bound("xy_err_nm", None, 12.0)
    frame_on(session, pixel=12.5)
    settings = ExportImageSettings(path=str(tmp_path / f"d{suffix}"), what="composite",
                                   region="all", pixelsize_nm=12.5, scale_bar_nm=500.0,
                                   colour_bars=False)
    path, = export_image(session.context(), settings)
    found = read_description(path)
    assert settings_from(ExportImageSettings, found["export"]) == settings
    assert found["pixelsize"] == 12.5 and found["scale_bar"] == 500.0
    assert found["grouping"]["dx"] == session.group_settings.dx
    first, second = found["layers"]
    assert first["filter"]["xy_err_nm"] == [None, 12.0]
    assert first["render"]["mode"] == session.layers[0].state.settings.mode
    assert second["render"]["color_field"] == "z_nm"
    assert second["display"]["lut"] == "jet"
    if suffix != ".jpg":                       # JPEG's density cannot hold nm
        assert pixel_size_of(path) == pytest.approx(12.5, rel=1e-6)


def test_the_area_is_the_view_the_roi_or_everything(tmp_path):
    from smappy.regions import Region
    session = Session(table())
    frame_on(session, pixel=40.0)
    out = lambda **k: pixels(export_image(session.context(), ExportImageSettings(
        path=str(tmp_path / "a.png"), what="composite", **BARE, **k))[0])
    assert out().shape[:2] == (75, 100)                       # the view, as displayed
    assert out(pixelsize_nm=20.0).shape[:2] == (150, 200)     # the view, finer
    session.set_roi(Region.rect(1000, 500, 2000, 1500))
    assert out(region="roi", pixelsize_nm=10.0).shape[:2] == (100, 100)
    (x0, x1), (y0, y1) = session.full_view()
    assert out(region="all", pixelsize_nm=40.0).shape[:2] == \
        (round((y1 - y0) / 40), round((x1 - x0) / 40))


def test_the_view_has_its_panels_side_by_side_named(tmp_path):
    session = two_layers()
    frame_on(session, panels=(0, 1, "overlay"))
    path, = export_image(session.context(), ExportImageSettings(path=str(tmp_path / "v.png")))
    image = pixels(path)
    assert image.shape == (150, 3 * 200 + 2 * 2, 3)       # a 2-pixel line between
    assert (image[:, 200:202] == 128).all()
    found = read_description(path)
    assert found["export"]["what"] == "view"


def test_a_script_exports_its_table_without_a_session(tmp_path):
    locs = table()
    path, = export_image(Context(locs=locs), ExportImageSettings(
        path=str(tmp_path / "s.png"), pixelsize_nm=20.0, **BARE))
    assert pixels(path).any()


def test_the_export_draws_the_layers_grouped_table(tmp_path):
    from smappy import plugins
    from smappy.simulate import simulate
    session = Session(simulate(n_frames=800, seed=3))
    layer = session.layers[0]
    layer.show_grouped(True)
    plugin = plugins.get("File/Export/Image")()
    result = plugin.run(session.context(), ExportImageSettings(path=str(tmp_path / "g.png")))
    shown = read_description(result.written)["layers"][0]["n_shown"]
    assert 0 < shown < len(session.locs)          # one row per blink
    layer.set_bound("photons", 1e12, None)
    layer.show_grouped(False)
    with pytest.raises(ValueError, match="nothing selected"):
        plugin.run(session.context(), ExportImageSettings(path=str(tmp_path / "e.png")))


# ------------------------------------------------------------- the windows
@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def a_window(app, session):
    from PySide6.QtCore import QRectF
    from smappy.gui.render_view import RenderToolBar, RenderView
    view = RenderView(session)
    bar = RenderToolBar(view)
    view.resize(420, 320)
    view.show()
    app.processEvents()
    view.view.setRange(QRectF(500, 500, 2000, 1500), padding=0)
    app.processEvents()
    return view, bar


def test_the_composite_equals_the_windows_render(app, tmp_path):
    session = two_layers()
    view, _ = a_window(app, session)
    path, = export_image(session.context(), ExportImageSettings(
        path=str(tmp_path / "w.png"), what="composite", **BARE))
    shown = view.render_now()
    assert pixels(path).shape == shown.shape
    assert np.array_equal(pixels(path), np.round(shown * 255).astype(np.uint8))
    view.close()


def test_the_toolbar_and_the_plugin_give_the_same_file(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from smappy import plugins
    session = two_layers()
    view, bar = a_window(app, session)
    view.set_layout_mode("side+overlay")
    app.processEvents()
    quick = tmp_path / "quick.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(quick), "")))
    bar._png()
    plugin = tmp_path / "plugin.png"
    plugins.get("File/Export/Image")().run(session.context(),
                                           ExportImageSettings(path=str(plugin)))
    assert np.array_equal(pixels(quick), pixels(plugin))
    a, b = read_description(quick), read_description(plugin)
    a["export"].pop("path"), b["export"].pop("path")
    assert a == b
    # three panels, as the window lays them out, and Copy takes the same
    from smappy.render import FieldOfView
    frame = session.view_frames["2D"]()
    assert frame.panels == (0, 1, "overlay")
    fov = FieldOfView.from_range(frame.xrange, frame.yrange, frame.pixelsize)
    h, w = pixels(quick).shape[:2]
    assert (h, w) == ((3 * fov.ny + 4, fov.nx) if frame.stacked else (fov.ny, 3 * fov.nx + 4))
    image = view.qimage()
    assert (image.width(), image.height()) == (w, h)
    view.close()


def test_save_image_opens_the_plugin_filled_from_the_window(app):
    session = two_layers()
    view, bar = a_window(app, session)
    asked = []
    session.window_openers["export image"] = asked.append
    from PySide6.QtGui import QAction
    entries = [a.text() for a in bar.findChildren(QAction)]
    assert "PNG as displayed..." in entries and "Save image..." in entries
    bar._save_image()
    assert asked and asked[0].view == "2D" and asked[0].pixelsize_nm is None
    view.close()


def test_the_3d_view_exports_its_projection(app, tmp_path):
    from smappy.gui.view3d import View3DWindow
    from smappy.view3d import render_3d
    session = Session(table())
    window = View3DWindow(session)
    window.resize(500, 400)
    window.show()
    app.processEvents()
    asked = []
    session.window_openers["export image"] = asked.append
    window._save_image()
    settings = asked[0]
    assert settings.view == "3D"
    settings.path = str(tmp_path / "p.png")
    settings.what = "composite"
    settings.scale_bar = settings.colour_bars = settings.labels = False
    path, = export_image(session.context(), settings)
    frame = session.view_frames["3D"]()
    from smappy.render import FieldOfView
    fov = FieldOfView.from_range(frame.xrange, frame.yrange, frame.pixelsize)
    expected = render_3d(session.layers, frame.projection, frame.slab, fov)
    assert np.array_equal(pixels(path), np.round(np.clip(expected, 0, 1) * 255).astype(np.uint8))
    assert read_description(path)["projection"]["zoom"] == pytest.approx(frame.pixelsize)
    window.close()
