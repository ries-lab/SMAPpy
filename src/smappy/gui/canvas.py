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
