"""Chains and batch runs: the same analysis on every file, and a record of it.

Built on what the plugins tutorial shows: a chain is plugins in order, with
their settings, run as one plugin -- one undo step, one entry in the history
-- and a batch runs a chain over a folder of files, one at a time, in a
process of its own, writing each file's table, figures and numbers.

The chain is the smallest that shows the point: drift correction and then
Localization Statistics, over four simulated cells that drift.  It is built in
the Analysis tab with "+ chain", run on one cell, saved, and picked in the
batch window, which is how a user would go about it.  The save and folder
dialogs are answered (`common.answering`) so the clicks run the windows' own
code.  The run is a real ``smappy-batch`` subprocess; the storyboard checks
what it wrote, and that a second run skips what is already done.
"""
from __future__ import annotations

import csv
import time

from .common import answering, menu, menu_item, show_tab

TITLE = "Chains and batch runs over many files"
DESCRIPTION = ("A chain of plugins that runs as one, saved, then run over a "
               "folder of files by the batch window.")

STEPS = (("Analysis/Drift/RCC", "drift"),
         ("Analysis/Measure/Localization Statistics", "statistics"))

_WRITTEN = """
<svg viewBox="0 0 560 190" role="img" aria-label="the output folder: a folder per file with its table, figures and numbers, and a summary of all">
  <g font-size="13" fill="currentColor" font-family="ui-monospace, monospace">
    <text x="20" y="25" font-weight="600">batch output/</text>
    <text x="40" y="50">cell_1/</text>
    <text x="60" y="72">cell_1_proc.h5</text><text x="250" y="72" opacity=".7" font-family="inherit">the processed table, with its history</text>
    <text x="60" y="94">figures/</text><text x="250" y="94" opacity=".7" font-family="inherit">every step's figures</text>
    <text x="60" y="116">results.json</text><text x="250" y="116" opacity=".7" font-family="inherit">every step's numbers and settings</text>
    <text x="40" y="138">cell_2/ ...</text>
    <text x="40" y="160">summary.csv</text><text x="250" y="160" opacity=".7" font-family="inherit">a row per file, a column per number</text>
    <text x="40" y="182">batch_report.json</text><text x="250" y="182" opacity=".7" font-family="inherit">what ran, on which files</text>
  </g>
</svg>
"""


def _data(d):
    """Four cells that drift, as localization files."""
    from ...io.formats import save
    from ...simulate import simulate
    folder = d.data_dir() / "cells"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*"):
        old.unlink()
    for i in range(4):
        save(simulate(8000, seed=10 + i, drift=True), folder / f"cell_{i + 1}.h5")
    return folder


def _wait_for(d, done, what: str, timeout: float = 600.0) -> None:
    """Pump until ``done()`` -- a load or a batch run, neither of which
    `settle` knows about."""
    end = time.monotonic() + timeout
    while not done() and time.monotonic() < end:
        d.pump(0.1)
    assert done(), f"{what} did not finish"
    d.settle()


def _run_batch(d, window) -> None:
    window.run()
    _wait_for(d, lambda: window.process is None, "the batch")


def _statuses(window):
    return [window.table.item(r, 4).text() for r in range(window.table.rowCount())]


def make(d) -> None:
    session, control = d.session, d.control
    cells = _data(d)
    first = cells / "cell_1.h5"
    control.load_paths([str(first)], reset_view=True)
    _wait_for(d, lambda: session.path is not None and len(session.locs) > 0,
              "opening the first cell")

    d.chapter("The same, on every file")
    d.card(TITLE,
           "<p>An experiment is many cells, and each is analysed the same way.</p>"
           "<p>A <b>chain</b> is plugins in order, with their settings, that runs "
           "as one plugin. A <b>batch</b> runs a chain over many files, and "
           "writes what each one gave.</p>",
           say="Every cell should be analysed the same way. A chain puts plugins "
               "in order; a batch runs it over many files.")

    d.chapter("A chain")
    analysis = show_tab(d, "Analysis")
    plus = _button(analysis, "+ chain")
    d.shot("Plus chain, at the top of the Analysis tab, starts an empty chain.",
           spot=[plus], point=plus, click=True, zoom=d.around(plus, 700))
    plus.click()
    d.settle()
    from ...gui.chain_panel import ChainPanel
    panel = [p for p in analysis.findChildren(ChainPanel) if p.isVisible()][-1]
    add = _button(panel.editor, "add...")
    d.shot("Add puts a plugin in, from the same tree as the tabs; up and down "
           "change the order.",
           spot=[add, panel.steps], point=add, click=True, zoom=d.around(panel.editor, 760))
    from ... import plugins
    from ...chain import Step
    spec = panel.current_spec()
    for path, label in STEPS:
        spec.steps.append(Step(plugin=path, label=label, version=plugins.get(path).version))
    assert panel._apply(spec), "the chain did not take its steps"
    d.settle()
    d.shot("Here: drift correction, then Localization Statistics. The tick "
           "turns a step off without removing it.",
           spot=[panel.steps], zoom=d.around(panel.steps, 760))
    panel.edit.setChecked(False)
    d.settle()
    d.shot("Each step is a section with that plugin's own settings, set as in "
           "the plugin itself.",
           spot=[panel.panel.form], zoom=d.around(panel.panel.form, 760))

    d.chapter("Run it once")
    run = panel.panel.run_button
    d.shot("Run it on the open file first, to see that it does what it should.",
           spot=[run], point=run, click=True, zoom=d.around(run, 700))
    run.click()
    d.settle()
    last = session.history[-1] if session.history else {}
    assert len(last.get("steps", ())) == len(STEPS), f"the history has {last}"
    d.shot("It ran as one plugin: one undo step, and one entry in the history "
           "with every step and its settings.",
           spot=[panel.panel.output], zoom=d.around(panel.panel.output, 760))

    from ...chain import SUFFIX
    target = plugins.chains_dir() / f"drift_and_statistics{SUFFIX}"
    target.parent.mkdir(parents=True, exist_ok=True)
    d.shot("Save chain keeps it as a file. Saved, it is a plugin like any other, "
           "under Analysis, Chains.",
           spot=[panel.save], point=panel.save, click=True, zoom=d.around(panel.save, 700))
    with answering(save=str(target)):
        panel.save.click()
    d.settle()
    assert target.exists(), f"the chain was not saved to {target}"

    d.chapter("A batch")
    tools = menu(d, "Tools")
    item = menu_item(d, tools, "Batch")
    d.shot("Tools, Batch opens the batch window.",
           spot=[item], point=item, click=True, zoom=d.around(tools, 700))
    tools.close()
    control.open_batch()
    window = control.batch_window
    window.resize(900, 860)
    d.place(window, 350, 20)
    d.settle()
    add_folder = _button(window, "add folder...")
    d.shot("Add the files: one by one, or a whole folder, searched for "
           "localization files.",
           spot=[add_folder, window.table], point=add_folder, click=True,
           zoom=d.around(window.table, 800))
    with answering(folder=str(cells)):
        add_folder.click()
    d.settle()
    assert window.table.rowCount() == 1, "the folder was not added"
    window.refresh_chains()
    index = next(i for i in range(window.chooser.count())
                 if "Drift And Statistics" in window.chooser.itemText(i).title())
    window.chooser.setCurrentIndex(index)
    window._chosen(index)
    d.settle()
    d.shot("Choose the chain that was just saved. It can be changed here too, "
           "without changing the saved one.",
           spot=[window.chooser, window.chain_panel], point=window.chooser,
           zoom=d.around(window.chooser, 800))
    output = d.data_dir() / "batch output"
    window.out_folder.setText(str(output))
    d.settle()
    d.shot("Where to write, and what: the processed tables, and the figures as "
           "pictures.",
           spot=[window.out_folder, window.figures, window.save_locs],
           point=window.out_folder, zoom=d.around(window.out_folder, 800))
    d.shot("First file only tries the chain on one file before the rest.",
           spot=[window.try_one], point=window.try_one,
           zoom=d.around(window.try_one, 700))

    d.chapter("Run")
    job = d.data_dir() / "cells.batch.yaml"
    d.shot("Run saves the job as a file, and runs it in a process of its own: "
           "a crash costs the batch, never the window.",
           spot=[window.run_button], point=window.run_button, click=True,
           zoom=d.around(window.run_button, 700))
    with answering(save=str(job)):
        _run_batch(d, window)
    rows = list(csv.DictReader(open(output / "summary.csv")))
    assert [r["status"] for r in rows] == ["done"] * 4, rows
    for i in range(4):
        figures = list((output / f"cell_{i + 1}" / "figures").glob("*.png"))
        assert len(figures) == len(STEPS), figures
    d.shot("Each file is loaded, run and saved in turn. The status says which "
           "are done; the log below says what each step did.",
           spot=[window.table, window.log], zoom=None)
    d.card("What is written",
           "<p>A folder per file, with its processed table, its figures and "
           "every number each step gave; and one summary with a row per "
           "file.</p>",
           figure=_WRITTEN,
           say="A folder per file, with its table, figures and numbers, and one "
               "summary with a row per file.")

    d.chapter("Run again")
    _run_batch(d, window)
    assert all("skipped" in s for s in _statuses(window)), _statuses(window)
    d.shot("Run again, and the files already done with the same settings are "
           "skipped. Change a setting, and those files are redone.",
           spot=[window.table, window.rerun], point=window.rerun, zoom=None)
    d.card("Without the window",
           "<p>The job is a small text file. <code>smappy-batch run</code> runs "
           "it on a server, with no window at all.</p>"
           "<p>Chains and jobs are plain text on purpose: an AI assistant can "
           "write one from a description of the analysis.</p>",
           say="The job is a text file: it runs without a window, and an AI "
               "assistant can write one for you.")

    d.chapter("Next")
    d.card("What to remember",
           "<ul><li><b>+ chain</b> in a tab: add steps, set them, run once, "
           "save.</li>"
           "<li><b>Tools, Batch:</b> files, a chain, an output folder, Run.</li>"
           "<li>A folder per file, and <b>summary.csv</b> over all of them.</li>"
           "<li>Running again redoes only what changed.</li></ul>",
           say="That is chains and batch runs: build it once, run it on every "
               "file, and keep the record.")


def _button(parent, text: str):
    """A button by its label, anywhere inside ``parent``."""
    from PySide6.QtWidgets import QAbstractButton
    for button in parent.findChildren(QAbstractButton):
        if button.text() == text and button.isVisible():
            return button
    raise LookupError(f"no {text!r} button; there are "
                      + ", ".join(sorted({b.text() for b in parent.findChildren(QAbstractButton)
                                          if b.text()})))
