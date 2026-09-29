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
    image_action = [a for a in tab.strip.add.menu().actions() if a.text() == "image..."][0]
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


def test_without_kept_frames_the_plus_menu_asks_for_a_file(app, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from smappy.gui.render_tab import RenderTab
    session, _ = session_with_raw_frames()
    session.images = []
    asked = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: asked.append(1) or ("", "")))
    tab = RenderTab(session)
    n = len(session.layers)
    tab.strip.add.menu().actions()[1].trigger()
    assert asked == [1] and len(session.layers) == n     # cancelled: no layer
