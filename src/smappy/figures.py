"""A result's figures without a window: laid out on a page, or written to files.

The GUI draws a `Plot` into a Qt canvas when its tab is looked at
(`gui.figures`); a batch run has no window and wants the same figures on
disk.  Both go through `Plot.draw_into`, so what is saved is exactly what the
window shows, and the page of small multiples is the same function in both --
which is why it lives here rather than beside the Qt code that imports
PySide6 at the top.

The figures are `matplotlib.figure.Figure` objects made directly, never
through pyplot: pyplot keeps every figure in a global registry, and a batch
that draws six figures per file over two hundred files would hold all twelve
hundred of them.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence

from .plugins import Plot

# What a figure is drawn at when the plot does not say: the size a window
# opens at, near enough, so a saved figure reads like the one on screen.
DEFAULT_SIZE = (8.0, 6.0)


def summary_plot(plots: Sequence[Plot], name: str = "all") -> Plot:
    """Every figure on one page, each in a subfigure of it.

    A page is worth having for the glance and for what gets exported, and it
    is worth nothing if it costs the figures being drawn twice, so it is a
    tab like any other and is drawn when it is looked at.

    Each plot is handed a `SubFigure`, which takes `subplots` exactly as a
    figure does -- which is why a plot declares its panels instead of
    helping itself to `ax.figure`: a plot that seizes the figure would take
    the whole page with it.
    """
    plots = list(plots)

    def draw(figure) -> None:
        columns = 1 if len(plots) == 1 else 2
        rows = -(-len(plots) // columns)
        cells = figure.subfigures(rows, columns, squeeze=False).ravel()
        for plot, cell in zip(plots, cells):
            if plot.name:
                cell.suptitle(plot.name, fontsize=9)
            plot.draw_into(cell)
        for cell in cells[len(plots):]:
            cell.set_visible(False)

    return Plot(draw=draw, name=name, panels=len(plots) + 1)


def render(plot: Plot):
    """A `Plot` drawn into a figure of its own, at the size it asks for."""
    from matplotlib.figure import Figure
    figure = Figure(figsize=plot.size or DEFAULT_SIZE, layout="constrained")
    plot.draw_into(figure)
    return figure


def file_name(text: str) -> str:
    """A figure's name as a file name: ``"drift: x(t)"`` -> ``drift_x_t``."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_.")
    return cleaned or "figure"


def save(plots: Sequence[Plot], folder, prefix: str = "", fmt: str = "png",
         dpi: int = 120) -> List[Path]:
    """Write each plot to ``folder/<prefix>_<name>.<fmt>``; the paths written.

    A plot that fails to draw costs that file and nothing else: the numbers
    of the run are already in hand, and a figure is the one part of a result
    that may be given up.  Two plots whose names clean to the same file name
    are numbered rather than overwriting one another.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    taken = set()
    for plot in plots:
        stem = file_name("_".join(p for p in (prefix, plot.name) if p))
        candidate, n = stem, 2
        while candidate in taken:
            candidate, n = f"{stem}_{n}", n + 1
        taken.add(candidate)
        path = folder / f"{candidate}.{fmt}"
        try:
            figure = render(plot)
            figure.savefig(path, dpi=dpi)
        except Exception:
            continue
        written.append(path)
    return written
