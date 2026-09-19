"""Result figures, drawn where the user already put the window.

A plugin that is run again, and an evaluator that is run on the next ROI, both
want the figure they drew last time rather than another copy of it on top: the
window has been moved and resized to be looked at, and the whole point of
stepping through a list of sites is that the picture changes in place while
everything around it stays still.

So a figure is identified by a key -- the plugin and the name of its plot --
and the window that key was last drawn in is cleared and reused.  A window the
user closed is opened again, which is how a figure is dismissed and got back.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Tuple


def draw_figure(cache: Dict[str, object], key: str, title: str,
                draw: Callable) -> Optional[object]:
    """Draw `draw(ax)` into the window `key` owns, and show it.

    `cache` is the caller's map of key to figure; it is updated here.  The
    figure is cleared first, so a `plot(ax)` that repopulates the whole figure
    -- a three-panel preview, say -- leaves nothing of the last one behind.
    """
    import matplotlib
    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt
    figure = cache.get(key)
    if figure is not None and plt.fignum_exists(figure.number):
        figure.clear()
        ax = figure.subplots()
    else:
        figure, ax = plt.subplots()
        cache[key] = figure
    draw(ax)
    figure.canvas.manager.set_window_title(title)
    figure.canvas.draw_idle()
    figure.show()
    return figure


def draw_figures(cache: Dict[str, object], figures: Iterable[Tuple[str, Callable]],
                 title: str, prefix: str = "") -> None:
    """Every figure of one result, one window each, named `title: plot`.

    `prefix` distinguishes the windows of two things drawing the same plots --
    two evaluators in one pipeline -- without appearing in the title, which
    says what is being looked at now.
    """
    for name, draw in figures:
        draw_figure(cache, f"{prefix}{name}", f"{title}: {name}" if name else title,
                    draw)
