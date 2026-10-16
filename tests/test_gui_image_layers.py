"""Image layers in the Render tab: the + menu, the source list, the frames."""
import numpy as np
import pytest

pytest.importorskip("PySide6")

from smappy.images import ImageData                                   # noqa: E402
from smappy.io.formats import FileInfo                                # noqa: E402
from smappy.locs import Localizations                                 # noqa: E402
from smappy.rawframes import RawFrameKeeper                           # noqa: E402
from smappy.metadata import CameraMetadata                            # noqa: E402
from smappy.session import Session                                    # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def session_with_raw_frames():
    rng = np.random.default_rng(0)
    session = Session()
    session.add_file(Localizations({"x_nm": rng.uniform(0, 3000, 500),
                                    "y_nm": rng.uniform(0, 3000, 500),
                                    "frame": rng.integers(0, 40, 500)}, {}),
                     FileInfo("a.hdf5", "/a.hdf5", "smappy"))
    keeper = RawFrameKeeper(3, 0, 40)
    keeper.see(0, np.arange(40)[:, None, None] + np.full((40, 30, 30), 100.0))
    raw = keeper.image(CameraMetadata(conversion=1.0, offset=100.0, pixelsize_um=0.1),
                       name="a.tif: raw frames")
    raw.metadata["filenumber"] = 0
    session.images.append(raw)
    return session, raw


def test_the_plus_menu_adds_an_image_layer_on_the_kept_average(app):
    from smappy.gui.render_tab import RenderTab
    session, raw = session_with_raw_frames()
    tab = RenderTab(session)
    image_action = [a for a in tab.strip.add.menu().actions() if a.text() == "image layer"][0]
    image_action.trigger()               # used to do nothing: see LayerStrip
    layer = session.layers[-1]
    assert layer.is_image and layer.image is raw and layer.frame == 0
    assert tab.strip.current == len(session.layers) - 1
    assert tab.image_section.isVisibleTo(tab)
    assert tab.image_frame_label.text() == "average of 40 frames"
    # the average of frames 0..39, each (frame + 100) counts: 19.5 photons
    assert layer.image.plane_at(layer.frame)[0, 0] == pytest.approx(19.5)


def test_the_frame_slider_steps_through_the_kept_frames_by_their_numbers(app):
    from smappy.gui.render_tab import RenderTab
    session, raw = session_with_raw_frames()
    tab = RenderTab(session)
    tab.strip.add.menu().actions()[1].trigger()
    assert raw.frames.tolist() == [-1, 0, 20, 39]
    tab.image_frame.setValue(3)
    assert session.layers[-1].frame == 3
    assert tab.image_frame_label.text() == "frame 39"
    assert raw.frame == 0                    # the layer's, not the image's


def test_the_source_list_offers_every_image_and_opens_a_new_one(app, tmp_path,
                                                                   monkeypatch):
    import tifffile
    from PySide6.QtWidgets import QFileDialog
    from smappy.gui.render_tab import RenderTab
    session, raw = session_with_raw_frames()
    widefield = ImageData(np.ones((5, 5), np.float32), 100.0, name="wf.tif")
    session.images.append(widefield)
    tab = RenderTab(session)
    tab.strip.add.menu().actions()[1].trigger()
    box = tab.image_source
    assert [box.itemText(i) for i in range(box.count())] == \
        ["a.tif: raw frames", "wf.tif", "open file..."]
    assert box.currentText() == "a.tif: raw frames"

    box.setCurrentIndex(1)
    box.activated.emit(1)
    assert session.layers[-1].image is widefield
    assert tab.image_pixelsize.value() == pytest.approx(100.0)

    path = tmp_path / "other.tif"
    tifffile.imwrite(path, np.zeros((6, 6), np.uint16), imagej=True,
                     resolution=(10.0, 10.0), metadata={"unit": "um"})
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    box.setCurrentIndex(2)
    box.activated.emit(2)
    assert session.layers[-1].image.name == "other.tif"
    assert session.layers[-1].image.pixelsize == pytest.approx(100.0)
    assert box.itemText(box.count() - 1) == "open file..."
    assert "other.tif" in [box.itemText(i) for i in range(box.count())]


def test_an_image_layer_asked_for_before_any_image_waits_and_takes_the_first(app):
    """During a live fit there are no frames yet: the layer opens drawing
    nothing, and shows the fit's frames when they arrive."""
    from smappy.gui.render_tab import RenderTab
    rng = np.random.default_rng(1)
    session = Session(Localizations({"x_nm": rng.uniform(0, 3000, 50),
                                     "y_nm": rng.uniform(0, 3000, 50),
                                     "frame": np.arange(50)}, {}))
    tab = RenderTab(session)
    [a for a in tab.strip.add.menu().actions() if a.text() == "image layer"][0].trigger()
    layer = session.layers[-1]
    assert layer.is_image and layer.image.is_empty
    assert session.images == [] and session.saved_images() == []
    assert layer.bounds() is None
    assert tab.image_source.currentText().startswith("none yet")
    _, raw = session_with_raw_frames()
    session.images.append(raw)
    session.changed("images")
    assert layer.image is raw and layer.name == raw.name
    assert tab.image_source.currentText() == "a.tif: raw frames"


def _zstack_session():
    session = Session(Localizations({"x_nm": np.zeros(3), "y_nm": np.zeros(3),
                                     "z_nm": np.array([-100.0, 0.0, 100.0])}, {}))
    stack = ImageData(np.arange(5)[:, None, None] * np.ones((5, 8, 8), np.float32), 100.0)
    stack.axis, stack.z_step_nm, stack.z0_nm = "z", 50.0, -100.0
    layer = session.add_image(stack)
    return session, layer


def test_the_image_rows_set_what_a_stack_shows_and_how_3d_draws_it(app):
    from smappy.gui.render_tab import RenderTab
    session, layer = _zstack_session()
    tab = RenderTab(session)
    tab.strip.select(len(session.layers) - 1)
    assert tab.image_axis.currentData() == "z" and tab.image_step.value() == 50.0
    assert tab.image_3d.isEnabled() and tab.image_frame_label.text() == "z -100 nm"
    tab.image_show.setCurrentIndex(tab.image_show.findData("max"))
    tab.image_from.setValue(2)
    tab.image_to.setValue(4)
    assert layer.image_show == "max" and layer.image_planes == (1, 3)
    assert np.allclose(layer.shown_plane(), 3.0)              # planes 1..3: max is 3
    assert not tab.image_frame.isEnabled()                   # a projection has no slice
    tab.image_3d.setCurrentIndex(tab.image_3d.findData("volume"))
    assert layer.image_3d == "volume"
    # called time frames, it is no z-stack: the 3D view has no volume of it
    tab.image_axis.setCurrentIndex(tab.image_axis.findData("t"))
    assert layer.image.axis == "t" and not tab.image_3d.isEnabled()


def test_a_stack_the_file_does_not_describe_is_asked_about_and_split(app, tmp_path, monkeypatch):
    import tifffile
    from PySide6.QtWidgets import QDialog
    from smappy.gui import dialogs
    path = tmp_path / "pages.tif"
    tifffile.imwrite(path, np.zeros((8, 6, 6), np.float32), photometric="minisblack",
                     resolution=(10, 10), resolutionunit="CENTIMETER")

    def answer(self):
        if isinstance(self, dialogs.PixelSizeDialog):
            return QDialog.Accepted
        self.channels.setValue(2)
        self.axis.setCurrentIndex(self.axis.findData("z"))
        self.step.setValue(25.0)
        return QDialog.Accepted

    monkeypatch.setattr(QDialog, "exec", answer)
    images = dialogs.read_image_file(path)
    assert [im.n_frames for im in images] == [4, 4]
    assert all(im.is_zstack and im.z_step_nm == 25.0 for im in images)
    # three channels do not divide eight planes: OK is refused, not guessed at
    dialog = dialogs.StackAxesDialog("pages.tif", 8, (6, 6))
    dialog.channels.setValue(3)
    dialog._accept()
    assert dialog.result() != QDialog.Accepted and "do not divide" in dialog.error.text()
