"""The application icon ships, and is what `scripts/make_icon.py` draws."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from smappy.gui.icon import ICONS, app_icon, install_icon   # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_icon.py"


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _make_icon():
    spec = importlib.util.spec_from_file_location("make_icon", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_shipped_svg_is_what_the_script_draws():
    assert (ICONS / "smappy.svg").read_text() == _make_icon().svg()


def test_every_size_the_script_renders_is_shipped():
    for n in _make_icon().SIZES:
        assert (ICONS / f"smappy_{n}.png").is_file(), n


def test_the_application_gets_an_icon_at_window_sizes(app):
    install_icon(app)
    icon = app.windowIcon()
    assert not icon.isNull()
    sizes = {s.width() for s in app_icon().availableSizes()}
    assert {16, 32, 256} <= sizes
    # beside the crosshair, the PSF's brightest pixel: near white in `hot`
    image = icon.pixmap(256, 256).toImage()
    centre = image.pixelColor(122, 96)
    assert centre.red() > 240 and centre.green() > 240
