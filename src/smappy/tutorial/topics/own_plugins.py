"""Your own plugins: an analysis no plugin does, made into one.

Built on the plugins tutorial: a plugin is a form and a Run button, and this
one shows where a plugin of one's own comes from -- Plugins > New plugin...,
which copies a working example into the user's folder (`smappy.user_plugins`)
-- and how it becomes the analysis wanted, by describing it to an AI
assistant started in that folder.

The assistant is not run here: the storyboard writes the file an assistant
writes for that description (`_ASKED`, a plugin as AGENTS.md describes one),
and the program reads it the way it reads any change -- the Plugins menu
opens the new code.  The dialogs *New plugin...* opens are answered, so the
click runs the window's own code.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager

from .common import menu, menu_item, open_section, place_beside, show_tab

TITLE = "Your own plugins: an analysis described, then run"
DESCRIPTION = ("When no plugin measures what you need: a new plugin from the "
               "example, changed by an AI assistant into what you describe.")

NAME = "Neighbours"
PATH = "User/Neighbours"

# what an assistant writes, asked for "the number of localizations within a
# radius of each one, as a histogram, and its median per ROI"
_ASKED = '''"""Neighbours: how many localizations lie within a radius of each one."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smappy.plugins import Context, Plugin, Result, param


def neighbour_counts(x, y, radius):
    """For each point, how many other points lie within ``radius``."""
    from scipy.spatial import cKDTree
    points = np.column_stack([x, y])
    tree = cKDTree(points)
    return np.array([len(n) - 1 for n in tree.query_ball_point(points, radius)])


@dataclass
class NeighboursSettings:
    radius_nm: float = param(50.0, label="radius", unit="nm", min=1.0,
                             help="count the localizations closer than this")


class Neighbours(Plugin):
    """How many localizations lie within a radius of each one."""
    name = "Neighbours"
    Settings = NeighboursSettings
    changes_locs = False
    per_region = True

    def run(self, ctx: Context, settings: NeighboursSettings) -> Result:
        locs = ctx.selection.apply(ctx.locs)
        counts = neighbour_counts(locs["x_nm"], locs["y_nm"], settings.radius_nm)
        median = float(np.median(counts)) if len(counts) else float("nan")

        def draw(ax):
            ax.hist(counts, bins=np.arange(counts.max() + 2) - 0.5 if len(counts) else 10)
            ax.axvline(median, color="k", lw=1)
            ax.set_xlabel(f"localizations within {settings.radius_nm:g} nm")
            ax.set_ylabel("localizations")

        return Result(text=f"median {median:g} neighbours within {settings.radius_nm:g} nm",
                      values={"neighbours_median": median}, plot=draw, settings=settings)
'''

_FOLDER = """
<svg viewBox="0 0 560 170" role="img" aria-label="your plugin folder: the new plugin, the example, and the instructions for an assistant">
  <g font-size="13" fill="currentColor" font-family="ui-monospace, monospace">
    <text x="20" y="25" font-weight="600">your plugin folder/</text>
    <text x="40" y="52">neighbours.py</text><text x="220" y="52" opacity=".7" font-family="inherit">the new plugin: the example, renamed</text>
    <text x="40" y="78">AGENTS.md</text><text x="220" y="78" opacity=".7" font-family="inherit">how a plugin is written, for an assistant</text>
    <text x="40" y="104">CLAUDE.md</text><text x="220" y="104" opacity=".7" font-family="inherit">reads AGENTS.md; your own notes</text>
    <text x="40" y="130">README.md</text><text x="220" y="130" opacity=".7" font-family="inherit">the same, for you</text>
    <text x="40" y="156">_template.py</text><text x="220" y="156" opacity=".7" font-family="inherit">the example, kept as it is</text>
  </g>
</svg>
"""

_ASKING = """
<svg viewBox="0 0 560 150" role="img" aria-label="a terminal in the plugin folder: the assistant is started and told what to measure">
  <rect x="10" y="8" width="540" height="134" rx="8" fill="none" stroke="currentColor" opacity=".35"/>
  <g font-size="13" fill="currentColor" font-family="ui-monospace, monospace">
    <text x="28" y="38" opacity=".7">$ cd "your plugin folder"</text>
    <text x="28" y="62" opacity=".7">$ claude</text>
    <text x="28" y="92">&gt; Change neighbours.py: for each localization, count</text>
    <text x="28" y="112">  the others within a radius, show them as a histogram,</text>
    <text x="28" y="132">  and give the median per ROI.</text>
  </g>
</svg>
"""


@contextmanager
def _answering_new_plugin(name: str):
    """Answer what Plugins > New plugin... asks and shows: the name, the
    message that follows, and the file manager it opens, which the offscreen
    platform has no one to look at."""
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QInputDialog, QMessageBox
    kept = [(QInputDialog, "getText", QInputDialog.getText),
            (QMessageBox, "information", QMessageBox.information),
            (QDesktopServices, "openUrl", QDesktopServices.openUrl)]
    try:
        QInputDialog.getText = staticmethod(lambda *a, **k: (name, True))
        QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
        QDesktopServices.openUrl = staticmethod(lambda url: True)
        yield
    finally:
        for cls, attr, original in kept:
            setattr(cls, attr, original)


def _place(d, window) -> None:
    """A plugin window the size of its form, at the left of the picture."""
    x, y, _, _ = d.window_rect(d.render)
    d.place(window, x + 30, y + 60, 400, 240)
    d.settle()


def _open_user_menu(d):
    """Plugins, and its User submenu beside it, as a person opens them."""
    from PySide6.QtCore import QPoint
    plugins_menu = menu(d, "Plugins")
    action = next(a for a in plugins_menu.actions() if a.text() == "User")
    r = plugins_menu.actionGeometry(action)
    submenu = action.menu()
    plugins_menu.setActiveAction(action)
    submenu.popup(plugins_menu.mapToGlobal(QPoint(r.right(), r.top())))
    d.settle()
    return plugins_menu, submenu


def _run(d, window):
    """Run a plugin window's panel and put its figure beside it."""
    panel = window.panel
    panel.run_button.click()
    d.settle()
    figure = panel._window
    place_beside(d, figure, 0.55)
    return panel, figure


def make(d) -> None:
    from ... import config
    control = d.control

    # the tour's simulated table, made the way it shows (File -> Simulate)
    file_tab = show_tab(d, "File")
    _, simulate = open_section(d, file_tab, "Simulate structure")
    simulate.run_button.click()
    d.settle()

    d.chapter("Your own analysis")
    d.card(TITLE,
           "<p>Sometimes no plugin measures what you need. Then make one: a "
           "plugin of your own works like every other, with its settings, Run "
           "and figures.</p>"
           "<p>You do not have to program it. Describe it to an AI assistant, "
           "and it writes the plugin.</p>",
           say="When no plugin measures what you need, make your own. An AI "
               "assistant can write it from your description.")

    d.chapter("A new plugin")
    plugins_menu = menu(d, "Plugins")
    item = menu_item(d, plugins_menu, "New plugin")
    d.shot("Plugins, New plugin asks for a name, and puts a working example "
           "under that name in your plugin folder.",
           spot=[item], point=item, click=True, zoom=d.around(plugins_menu, 700))
    plugins_menu.close()
    with _answering_new_plugin(NAME):
        next(a for a in control.plugins_menu.actions()
             if a.text() == "New plugin...").trigger()
    d.settle()
    file = config.user_plugin_dir() / "neighbours.py"
    assert file.is_file(), f"New plugin wrote no {file}"
    d.card("Your plugin folder",
           "<p>Every plugin in this folder is in SMAPpy, under User. Nothing "
           "has to be installed, and SMAPpy need not be restarted.</p>"
           "<p>Beside your plugins are the instructions an AI assistant reads, "
           "so it knows how a SMAPpy plugin is written.</p>",
           figure=_FOLDER,
           say="Your plugins live in this folder, beside the instructions an "
               "assistant reads.")

    d.chapter("It runs at once")
    plugins_menu, submenu = _open_user_menu(d)
    item = menu_item(d, submenu, NAME)
    d.shot("It is in the Plugins menu, under User. The plus in a tab pins it, "
           "as any other plugin.",
           spot=[item], point=item, click=True)
    submenu.close()
    plugins_menu.close()
    window = control.open_plugin_window(PATH)
    assert window is not None, f"{PATH} did not open"
    _place(d, window)
    d.shot("The example measures how far each localization is from its "
           "nearest neighbour: its settings, and Run.",
           spot=[d.window_rect(window)], point=window.panel.run_button, click=True)
    panel, figure = _run(d, window)
    d.shot("It is small on purpose: a starting point to change into what you "
           "need.",
           spot=[d.window_rect(figure)])
    figure.hide()
    window.hide()

    d.chapter("Describe what you need")
    d.card("Ask an assistant",
           "<p>Open a terminal in your plugin folder, start an AI assistant "
           "there, and say what to measure, on which localizations, and what "
           "to show.</p>"
           "<p>With a chat assistant, paste AGENTS.md and your description, "
           "and save the answer in the folder.</p>",
           figure=_ASKING,
           say="Start an assistant in your plugin folder, and describe what "
               "to measure and what to show.")
    # what the assistant writes; a newer time, so the change is seen even on
    # a file system that keeps whole seconds
    file.write_text(_ASKED, encoding="utf-8")
    later = time.time() + 2
    os.utime(file, (later, later))

    d.chapter("Run it again")
    window = control.open_plugin_window(PATH)
    assert window is not None and "radius_nm" in window.panel.form.fields, \
        "the changed plugin was not read again"
    _place(d, window)
    d.shot("Open it from the Plugins menu again, and the new version is "
           "there, with the setting you asked for.",
           spot=[window.panel.form], point=window.panel.run_button, click=True)
    panel, figure = _run(d, window)
    assert "neighbours within" in panel.result.text, panel.result.text
    d.shot("Run: how many neighbours each localization has, and the median. "
           "If it is not what you meant, tell the assistant, and run again.",
           spot=[d.window_rect(figure)])
    figure.hide()
    window.hide()

    d.chapter("Keep it, or share it")
    d.card("Up to you",
           "<p>A plugin for one dataset can be deleted when you are done. One "
           "you keep can have a help page and tests, which the assistant "
           "offers, but nothing requires them.</p>"
           "<p>A plugin others would use can become part of SMAPpy, by a pull "
           "request on GitHub. Then it needs a help page and tests.</p>",
           say="Delete it, keep it, or share it: a plugin others would use can "
               "become part of SMAPpy.")

    d.chapter("Next")
    d.card("What to remember",
           "<ul><li><b>Plugins, New plugin</b>: a working example, in your "
           "plugin folder.</li>"
           "<li>Describe what you need to an assistant started in that "
           "folder.</li>"
           "<li>Your plugins are under <b>Plugins, User</b>; open one again to "
           "get the new version.</li>"
           "<li>The help page <b>Your own plugins</b> has the details.</li></ul>",
           say="That is your own plugin: start from the example, describe what "
               "you need, and run it.")
