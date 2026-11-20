"""matplotlib's Qt canvas, safe to delete while a redraw is queued.

`FigureCanvasQT.draw_idle` queues its redraw as `QTimer.singleShot(0,
self._draw_idle)`, with no context object, so the call is made even when the
canvas has been deleted in the meantime -- a result window closed and freed
right after it was shown -- and `_draw_idle` then raises "Internal C++ object
already deleted".  An exception from a singleShot callable is left pending
inside `processEvents` (PySide6 6.8 to 6.10), and the next Python override Qt
calls in that same pass -- a pyqtgraph item's `boundingRect` -- hangs on 6.8
and 6.9 and segfaults on 6.10.  That was `test_live_session` crashing after
`test_figure_window`, in about half the full runs on 6.10.3.  So the queued
redraw of a deleted canvas does nothing, which is what it would have drawn.
"""
from __future__ import annotations

_canvas = None


def canvas_class():
    """`FigureCanvasQTAgg`, with a queued redraw that outlives it harmlessly.

    Imported late, as matplotlib's Qt backend is not free to import.
    """
    global _canvas
    if _canvas is None:
        import shiboken6
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        class Canvas(FigureCanvasQTAgg):
            def _draw_idle(self):
                if shiboken6.isValid(self):
                    super()._draw_idle()

        _canvas = Canvas
    return _canvas


_toolbar = None


def toolbar_class():
    """matplotlib's `NavigationToolbar2QT`, whose house button goes to an axis's *full view* when the
    drawing gave it one: ``ax.full_view = (xlim, ylim)``, which a Qt-free drawing sets as a plain attribute.

    A figure that opens zoomed onto what matters -- the CRLB around focus, its blow-up at the spline's ends
    clipped -- would otherwise have its house button return to that same zoom, the first view the toolbar
    saw, and the full range would be reachable only by zooming out by hand.
    """
    global _toolbar
    if _toolbar is None:
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

        class Toolbar(NavigationToolbar2QT):
            def home(self, *args):
                axes = [a for a in self.canvas.figure.axes if getattr(a, "full_view", None)]
                if not axes:
                    return super().home(*args)
                self.push_current()                  # *back* returns to the zoom
                for ax in axes:
                    xlim, ylim = ax.full_view
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                self.push_current()
                self.set_history_buttons()
                self.canvas.draw_idle()

        _toolbar = Toolbar
    return _toolbar

