"""A picture with no window: what a program with its own event loop needs."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.render import DisplaySettings, RenderSettings, save_image

Image = pytest.importorskip("PIL.Image")


def table(n=200) -> Localizations:
    rng = np.random.default_rng(0)
    return Localizations({
        "x_nm": rng.uniform(0, 1000, n), "y_nm": rng.uniform(0, 500, n),
        "loc_precision_nm": rng.uniform(5, 25, n),
        "z_nm": rng.uniform(-300, 300, n),
        "frame": np.arange(n),
    }, {"units": "nm"})


def test_it_writes_an_image_of_the_field(tmp_path):
    path = save_image(table(), tmp_path / "locs.png", pixelsize=10.0)
    with Image.open(path) as image:
        assert image.mode == "RGB"
        width, height = image.size
    # 1000 x 500 nm at 10 nm/pixel, give or take the margin the field adds
    assert 90 <= width <= 130 and 40 <= height <= 80


def test_it_reads_a_saved_file_too(tmp_path):
    from smappy.io.hdf5 import save_localizations

    h5 = save_localizations(tmp_path / "locs.h5", table())
    path = save_image(h5, tmp_path / "from_file.png", pixelsize=20.0)
    assert path.exists() and path.stat().st_size > 0


def test_the_render_and_the_display_are_both_honoured(tmp_path):
    hot = save_image(table(), tmp_path / "hot.png", pixelsize=10.0,
                     settings=RenderSettings(mode="hist"),
                     display=DisplaySettings(lut="hot"))
    inverted = save_image(table(), tmp_path / "inv.png", pixelsize=10.0,
                          settings=RenderSettings(mode="hist"),
                          display=DisplaySettings(lut="hot", invert=True))
    assert hot.read_bytes() != inverted.read_bytes()


def test_a_filter_narrows_what_is_drawn(tmp_path):
    from smappy.filter import LocFilter

    locs = table()
    keep = LocFilter(locs, loc_precision_nm=(None, 10))
    path = save_image(locs, tmp_path / "filtered.png", pixelsize=10.0, select=keep)
    with Image.open(path) as image:
        assert np.asarray(image).any()      # something was drawn


def test_it_needs_no_window(monkeypatch, tmp_path):
    """It must not import matplotlib, let alone open a figure."""
    import sys
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    save_image(table(), tmp_path / "headless.png", pixelsize=10.0)
