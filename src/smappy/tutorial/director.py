"""Drive the GUI offscreen and keep a screenshot, and what to point at, per step.

The windows are the program's own -- `ControlWindow` and `RenderWindow` over a
`Session` -- on Qt's offscreen platform, laid out on a virtual desktop of a
fixed size so that every run of a storyboard produces the same pictures.  The
desktop is drawn at twice its size (``QT_SCALE_FACTOR=2``): the player zooms
into a 380-px control window to make its labels legible, and a zoom into a
1x screenshot is a blur.

Nothing here knows what a tutorial says.  A storyboard does things to the
widgets -- ``button.click()``, ``edit.setText(...)``, the session's own calls
-- and asks for a `shot`; the director waits until the render thread and any
plugin run have finished, composites every visible window onto the desktop
with a title bar, and records the rectangles of the widgets it was told about
in desktop coordinates.  Those rectangles go to the player, which draws the
spotlight and the pointer itself, so they can move between steps.

The configuration is a directory of its own (``SMAPPY_CONFIG_DIR``, and a
QSettings path beside it): a tutorial shows the shipped workspace, not the
favourites of whoever runs it, and writes nothing into theirs.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

SIZE = (1600, 900)          # the virtual desktop, in logical pixels (16:9)
SCALE = 2                   # device pixels per logical pixel in the screenshots
TITLE = 26                  # the height of a drawn title bar
MARGIN = 14

Rect = Tuple[float, float, float, float]            # x, y, w, h on the desktop


@dataclass
class Step:
    """One slide: what is on screen, what is said, where to look."""
    say: str
    image: Optional[str] = None          # the screenshot; None for a card
    spot: List[Rect] = field(default_factory=list)  # lit, the rest dimmed
    point: Optional[Tuple[float, float]] = None     # where the pointer goes
    click: bool = False                  # and whether it clicks there
    zoom: Optional[Rect] = None          # the part of the desktop to show
    card: Optional[dict] = None          # title / body / figure, instead of a picture
    chapter: Optional[str] = None        # starts a chapter of the progress bar
    duration: float = 0.0                # seconds; filled in by the player


def _environment(config: Path, size: Tuple[int, int]) -> None:
    """The offscreen platform with a screen of our size, before Qt starts."""
    screen = config / "screen.json"
    screen.write_text(json.dumps({"screens": [{
        "name": "tutorial", "x": 0, "y": 0,
        "width": size[0] * SCALE, "height": size[1] * SCALE,
        "logicalDpi": 96, "logicalBaseDpi": 96}]}))
    os.environ["QT_QPA_PLATFORM"] = f"offscreen:configfile={screen}"
    os.environ["QT_SCALE_FACTOR"] = str(SCALE)
    os.environ["SMAPPY_CONFIG_DIR"] = str(config / "smappy")


class Director:
    """The program, a desktop to put it on, and the steps shot so far.

    Must be made before anything else creates a QApplication: the platform
    and the scale are fixed when Qt starts.
    """

    def __init__(self, out: Union[str, Path], size: Tuple[int, int] = SIZE):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.size = size
        self._config = Path(tempfile.mkdtemp(prefix="smappy-tutorial-"))
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is not None:
            raise RuntimeError("a tutorial needs its own QApplication: make the "
                               "Director before anything else starts Qt")
        _environment(self._config, size)
        from PySide6.QtCore import QSettings
        QSettings.setDefaultFormat(QSettings.IniFormat)
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope,
                          str(self._config / "qsettings"))
        self.app = QApplication([sys.argv[0]])
        self.app.setApplicationName("smappy-tutorial")
        from ..gui.app import ControlWindow, RenderWindow
        from ..gui.widgets import CONTROL_WIDTH, apply_style
        from ..session import Session
        apply_style(self.app)
        self.session = Session()
        self.render = RenderWindow(self.session)
        self.control = ControlWindow(self.session, self.render)
        height = size[1] - 2 * MARGIN - TITLE
        self.place(self.control, MARGIN, MARGIN, CONTROL_WIDTH, height)
        left = 2 * MARGIN + CONTROL_WIDTH
        self.place(self.render, left, MARGIN, size[0] - left - MARGIN, height)
        self._order: list = [self.control, self.render]     # bottom to top
        self.steps: List[Step] = []
        self._chapter: Optional[str] = None
        self.settle()

    # ------------------------------------------------------------ the windows
    def place(self, window, x: float, y: float, w: Optional[float] = None,
              h: Optional[float] = None) -> None:
        """Put a window's title bar at (x, y) and show it, on top."""
        if w is not None and h is not None:
            window.resize(int(w), int(h))
        window.move(int(x), int(y + TITLE))
        window.show()
        self.raise_(window)

    def raise_(self, window) -> None:
        if window in getattr(self, "_order", []):
            self._order.remove(window)
        if hasattr(self, "_order"):
            self._order.append(window)

    def _visible(self) -> list:
        """The windows to draw, bottom to top: ours in order, then any new ones
        -- a plugin's result window, a menu -- in the order they turned up."""
        from PySide6.QtWidgets import QApplication
        for w in QApplication.topLevelWidgets():
            if w.isVisible() and w not in self._order:
                self._order.append(w)
        self._order = [w for w in self._order if _alive(w) and w.isVisible()]
        return list(self._order)

    # ------------------------------------------------------------- the clock
    def pump(self, seconds: float = 0.05) -> None:
        """Run the event loop for a while, as `app.exec` would.

        `processEvents` alone never runs a `deleteLater`: those wait for control
        to return to the event loop, which it never does here.  Without the
        explicit flush, every rebuilt row of buttons stays in its layout beside
        its replacement -- a filter with its quick buttons three times over.
        """
        from PySide6.QtCore import QCoreApplication, QEvent
        end = time.monotonic() + seconds
        while True:
            self.app.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            if time.monotonic() >= end:
                return
            time.sleep(0.01)

    def settle(self, timeout: float = 120.0) -> None:
        """Wait for the picture and every plugin run to finish.

        A render is asked for 40 ms after the view moves and done on its own
        thread; a plugin runs on another.  A screenshot taken in between is a
        black picture or a half-run panel, so "done" is all of them idle
        twice in a row, a little apart.
        """
        end = time.monotonic() + timeout
        quiet = 0
        while quiet < 2:
            self.pump(0.08)
            quiet = quiet + 1 if self._idle() else 0
            if time.monotonic() > end:
                raise TimeoutError("the GUI did not settle")

    def settle_picture(self, timeout: float = 30.0) -> None:
        """Wait for the picture only, not for a plugin still running: a shot of
        a fit half done, with its localizations drawn so far."""
        end = time.monotonic() + timeout
        quiet = 0
        while quiet < 2 and time.monotonic() < end:
            self.pump(0.04)
            quiet = quiet + 1 if self._picture_idle() else 0

    def _picture_idle(self) -> bool:
        view = self.render.view
        return not (view._timer.isActive() or view._busy or view._pending)

    def _idle(self) -> bool:
        if not self._picture_idle():
            return False
        for tab in self.control.plugin_tabs:
            for panel in tab.panels():
                thread = getattr(panel, "_thread", None)
                if thread is not None and thread.isRunning():
                    return False
        return True

    # ------------------------------------------------------------- geometry
    def rect(self, widget, pad: float = 3) -> Rect:
        """Where a widget is on the desktop, a little larger than itself.

        Only the part of it that is on screen: a field in a scrolled form can
        be wider than the column showing it, and a spotlight on the rest lit
        up the render window beside it.
        """
        from PySide6.QtCore import QPoint, QRect
        top = widget.window()
        seen = widget.visibleRegion().boundingRect() if widget.isVisible() else QRect()
        if seen.isEmpty():
            seen = QRect(0, 0, widget.width(), widget.height())
        corner = (widget.mapTo(top, seen.topLeft()) if widget is not top
                  else seen.topLeft())
        g = top.geometry()
        return (g.x() + corner.x() - pad, g.y() + corner.y() - pad,
                seen.width() + 2 * pad, seen.height() + 2 * pad)

    def union(self, *items, pad: float = 3) -> Rect:
        rects = [self._as_rect(i, pad) for i in items]
        x0 = min(r[0] for r in rects)
        y0 = min(r[1] for r in rects)
        x1 = max(r[0] + r[2] for r in rects)
        y1 = max(r[1] + r[3] for r in rects)
        return (x0, y0, x1 - x0, y1 - y0)

    def window_rect(self, window) -> Rect:
        """A window with its drawn title bar."""
        g = window.geometry()
        return (g.x(), g.y() - TITLE, g.width(), g.height() + TITLE)

    def centre(self, item) -> Tuple[float, float]:
        if isinstance(item, tuple) and len(item) == 2:
            return item
        x, y, w, h = self._as_rect(item, 0)
        return (x + w / 2, y + h / 2)

    def at_data(self, x: float, y: float) -> Tuple[float, float]:
        """Where a point of the picture, in data units, is on the desktop."""
        from PySide6.QtCore import QPointF
        view = self.render.view
        scene = view.view.mapViewToScene(QPointF(x, y))
        local = view.graphics.mapFromScene(scene)
        gx, gy, _, _ = self.rect(view.graphics, pad=0)
        return (gx + local.x(), gy + local.y())

    def data_rect(self, x0, y0, x1, y1, pad: float = 0) -> Rect:
        ax, ay = self.at_data(x0, y0)
        bx, by = self.at_data(x1, y1)
        return (min(ax, bx) - pad, min(ay, by) - pad,
                abs(bx - ax) + 2 * pad, abs(by - ay) + 2 * pad)

    def _as_rect(self, item, pad: float) -> Rect:
        if isinstance(item, tuple) and len(item) == 4:
            return item
        return self.rect(item, pad)

    def around(self, item, width: float, height: Optional[float] = None) -> Rect:
        """A zoom of ``width`` (16:9 unless told) centred on an item, kept on
        the desktop so the player never shows beyond its edge."""
        height = height or width * self.size[1] / self.size[0]
        cx, cy = self.centre(item)
        x = min(max(cx - width / 2, 0), self.size[0] - width)
        y = min(max(cy - height / 2, 0), self.size[1] - height)
        return (x, y, width, height)

    # --------------------------------------------------------------- shots
    def chapter(self, name: str) -> None:
        """The next step starts a chapter of the progress bar."""
        self._chapter = name

    def shot(self, say: str, spot: Sequence = (), point=None, click: bool = False,
             zoom=None, pad: float = 3, wait: bool = True) -> Step:
        """Wait for the GUI, keep a picture of it and the step that goes with it.

        ``wait=False`` waits for the picture but not for a running plugin.
        """
        self.settle() if wait else self.settle_picture()
        n = len(self.steps) + 1
        name = f"{n:03d}.webp"
        self.composite().save(str(self.out / name), "webp", 92)
        step = Step(say=say, image=name,
                    spot=[self._as_rect(s, pad) for s in spot],
                    point=self.centre(point) if point is not None else None,
                    click=click,
                    zoom=self._as_rect(zoom, 0) if zoom is not None else None,
                    chapter=self._take_chapter())
        self.steps.append(step)
        return step

    def card(self, title: str, body: str, figure: str = "", say: str = "") -> Step:
        """A slide of its own: a concept, drawn rather than shown."""
        step = Step(say=say or title, card={"title": title, "body": body,
                                              "figure": figure},
                    chapter=self._take_chapter())
        self.steps.append(step)
        return step

    def _take_chapter(self) -> Optional[str]:
        chapter, self._chapter = self._chapter, None
        return chapter

    def composite(self):
        """Every visible window on the desktop, at the screenshots' scale."""
        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtGui import (QColor, QFont, QImage, QLinearGradient, QPainter,
                                   QPainterPath, QPen)
        w, h = self.size
        image = QImage(w * SCALE, h * SCALE, QImage.Format_RGB32)
        image.setDevicePixelRatio(SCALE)
        p = QPainter(image)
        p.setRenderHint(QPainter.Antialiasing)
        ground = QLinearGradient(0, 0, w, h)
        ground.setColorAt(0, QColor("#3b4a5a"))
        ground.setColorAt(1, QColor("#1f2833"))
        p.fillRect(QRectF(0, 0, w, h), ground)
        for window in self._visible():
            g = window.geometry()
            popup = window.windowType() in (Qt.Popup, Qt.ToolTip)
            top = g.y() if popup else g.y() - TITLE
            frame = QRectF(g.x(), top, g.width(), g.bottom() + 1 - top)
            for i, alpha in enumerate((40, 25, 12)):          # a soft shadow
                p.fillRect(frame.adjusted(-i - 1, -i + 1, i + 3, i + 5),
                           QColor(0, 0, 0, alpha))
            if not popup:
                bar = QRectF(g.x(), top, g.width(), TITLE)
                path = QPainterPath()
                path.addRoundedRect(bar.adjusted(0, 0, 0, 6), 6, 6)
                p.fillPath(path, QColor("#e4e6ea"))
                for k, colour in enumerate(("#ff5f57", "#febc2e", "#28c840")):
                    p.setBrush(QColor(colour))
                    p.setPen(Qt.NoPen)
                    p.drawEllipse(QRectF(g.x() + 10 + 18 * k, top + 8, 11, 11))
                p.setPen(QPen(QColor("#333")))
                font = QFont()
                font.setPixelSize(13)
                p.setFont(font)
                p.drawText(bar, Qt.AlignCenter, window.windowTitle() or "SMAPpy")
            p.drawPixmap(g.x(), g.y(), window.grab())
        p.end()
        return image

    def data_dir(self) -> Path:
        """Where a storyboard puts the files it makes -- an acquisition to fit.

        Short and plain, because it is on screen: a file field shows the path,
        and a temporary directory's name is noise to whoever watches.
        """
        path = Path(tempfile.gettempdir()) / "SMAPpy demo"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ----------------------------------------------------------- the result
    def manifest(self) -> List[dict]:
        return [asdict(s) for s in self.steps]

    def close(self) -> None:
        from ..gui.render_view import stop_render_threads
        for w in list(self._visible()):
            w.hide()
        stop_render_threads()
        self.pump(0.05)


def _alive(widget) -> bool:
    try:
        widget.isVisible()
        return True
    except RuntimeError:          # deleted on the C++ side
        return False
