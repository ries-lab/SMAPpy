# Working in this repository

SMAPpy (`smappy` in code) is a Python SMLM package: TIFF -> photons -> peak finding -> MLE fit ->
localization table -> analysis, with a Qt GUI over the same functions.  Read
`NOTES.md` for the decisions behind the fitting pipeline and
`docs/plugin-architecture.md` for why the plugin system is shaped as it is,
and `docs/regions.md` for ROIs, clusters and what is measured per region.
`docs/plugin-reference.md` is the reference: the context, the result, the
table's columns.  This file is the practical part: what a session needs before it can write a
plugin, and the traps that cost time.  It is a summary written at one moment
and the code moves, so treat it as a starting point, not as authority -- and
see "Keeping this file true" at the end.

## Before you edit anything

Pull first:

    git fetch origin && git rebase origin/main      # or: git pull --rebase

Work lands on `origin/main` from cloud sessions and other machines, so the
local `main` is routinely well behind it -- 28 commits, on one occasion.
Editing first means rebasing finished work through conflicts in files upstream
has already changed -- `NOTES.md` most often.  If you have already started,
`git stash push -u` first so new untracked files come along, and rebuild
afterwards if `csrc/` moved.

## Getting the checkout to run

A fresh container has **no numpy**.  Before anything else:

    pip install -e .                    # builds the C++ extensions too
    pip install -e .[test]              # pytest, pytest-timeout (a 300 s limit per test), pytest-xdist

Running the tests -- two tiers:

    python -m pytest tests -q -n auto           # after each change: ~1 min, the slow tier skipped
    python -m pytest tests -q -n auto --slow    # before a PR or a push of major work: everything, ~4 min

* **Run the slow tier before you open a PR, and before you call major work
  done** -- a new plugin, a change to the GUI, a tutorial, a page or what an
  algorithm computes.  It is about 60 tests marked `@pytest.mark.slow` (the
  tutorials, the pages' figures, the large numerical checks) that are two
  thirds of the suite's time, and they are where a GUI or docs change breaks.
  The summary's "skipped" count includes them, so a run without `--slow` has
  not checked them; say so if you stop short of it.  No CI runs the tests.
* **A PR description states the `--slow` run's result** on a line of its own
  -- "`--slow -n auto`: 1321 passed, 4 skipped" -- or says plainly that it was
  not run and why, so a reviewer can see from the PR whether it was done.
* Mark a new test `slow` when it takes more than about 4 s.
* `-n auto` needs pytest-xdist (the `test` extra).  `conftest.py` gives each
  worker one BLAS thread (without it four workers are barely faster than one),
  makes work stealing the default (xdist's own piles the tutorials onto one
  worker) and runs the slow tests first.
* **Automatic garbage collection is off, in the tests and in the app.**  The
  collector otherwise runs in whichever thread allocates, and a Qt widget
  left in a reference cycle, destroyed on a fit's reader thread, deadlocked
  a test worker.  `conftest.py` collects on the main thread every 100 tests;
  the GUI's entry points call `gui/collector.collect_on_gui_thread`, a timer
  on the GUI thread.  A test that needs an object gone calls `gc.collect()`
  itself; a new entry point that makes a QApplication installs the collector,
  the icon (`gui/icon.install_icon`) and the session's identity
  (`gui/identity.install_identity`: its name and colour in every window title).
* **A `QTimer.singleShot` callable must never raise.**  PySide leaves its
  exception pending inside `processEvents`, and the next Python override Qt
  calls segfaults (6.10) or hangs.  Catch and print inside it, or connect a
  `QTimer`'s `timeout` signal instead.  Make matplotlib canvases with
  `gui/canvas.canvas_class()`, whose queued redraw survives a deleted canvas.

* The Qt tests need Qt's own libraries, not a display: on a bare container they
  fail with `ImportError: libEGL.so.1`.  Install them once --

      apt-get update && apt-get install -y libegl1 libgl1 libxkbcommon0
      QT_QPA_PLATFORM=offscreen python -m pytest tests -q

  -- and the whole suite runs.  Without them about 30 fail (`test_workspace`,
  `test_calibrate_qt`, `test_roi_pipeline`, parts of `test_context` and
  `test_roi_session`), which is the environment and not your change.
* `SMAPPY_TEST_CAL=/path/to/_3dcal.mat` enables the tests that want real
  calibration data; without it they skip.
* `SMAPPY_UIPSF_PYTHON=/path/to/python` (of an environment with uiPSF, as the
  page of `Localize/uiPSF calibration` installs it) enables the slow test that
  runs uiPSF itself, a few minutes on a CPU; without it it skips.

## Writing a plugin

The framework is the same whatever the plugin does; the shipped ones differ in
shape, so read the one closest to what you are writing:

| shape | example |
| --- | --- |
| measures the selection and draws it | `statistics.py` (settings, selection, several figures in one window) |
| rewrites the table and hands it back | `assign_colors.py` (`Result.locs`, a preview) |
| a long computation with progress | `drift_comet.py`, `drift_rcc.py` |
| loads, saves, exports, simulates | `file.py` (a plugin per format, no input table; the simulator's model is `smappy.simulate`) |
| finds, measures or summarises ROIs | `roi.py` (a segmenter derives from `SegmenterPlugin`, which gives it a Preview, *replace* (`replace_param()`) and a step of the ROI undo; `per_region = True`; analyse; segmenters under `Analysis/Regions/Find`) |
| parts, presets, a C++ backend | `fit.py` (nested settings dataclasses) |
| adds a derived column, no output | `math_parser.py` (an expression, kept with the table as a recipe) |
| removes or flags localizations | `remove_locs.py` (a region, a new table or a filtered flag) |
| fits a model to what is in a ROI | `line_profile.py` (unbinned likelihood, models compared by AIC; one fit per layer, and `live`) |
| reads the session and reports | `history.py` (the log, with an optional export) |
| sets up layers and filters from settings | `chain_layers.py` (bounds as rows per layer, `Result.data["layers"]`) |
| scores a fit against a simulation | `ground_truth.py` (the truth redrawn from the recipe, matched per frame) |
| draws what the microscope recorded per frame | `image_tags.py` (the fit's `metadata["frame_tags"]`) |
| finds a structure by its shape and judges it, then counts it | `npc.py` (a ring filter and circle fit in the segmenter, a per-site count, and a model of whole pores in `smappy/npc.py` fitted to the site table) |
| summarises what evaluators measured | `roi.py`'s `SiteAnalysisPlugin` (an evaluator to read, a *results from* field, a preflight that asks which when there are several; under `Analysis/Regions/Summarize`) |
| runs the ROI manager's segment, evaluate and analyse as one | `npc_analysis.chain.yaml` (a shipped chain: a block runs NPC Corners in every ROI before the next step; `docs/batch.md`, "ROIs: segment, evaluate, analyse") |
| finds clusters and names them in the table | `clusters.py` (DBSCAN: `cluster_id` / `n_in_cluster`; ROIs of them are `region_column.py`'s *Column to ROIs*, for any id column) |
| writes what was measured per region onto its localizations | `region_column.py` (*Regions to column*: `region_runs.values_onto`, NaN outside, never over a column of the localizations' own; a block's *write values* does the same) |
| runs steps once per region, writes columns back | `smappy/blocks.py` (a `chain.Block`; `region_runs.RegionScratch`, `WriteBack`, *write values*; per-cluster results in `Session.id_regions`) |
| opens an interactive window of the GUI's | `bead_calibration.py` (`Plugin.window`: the button opens what the GUI registered in `session.window_openers`) |
| runs an external program in its own Python | `uipsf.py` (`smappy.uipsf`: a worker script under that program's interpreter, progress read from its output, the result converted into smappy's own file) |
| hands the session to a new instance | `smappy.handoff` (*File > Open in a new window*: the table through shared memory, the rest as JSON, `smappy-gui --handoff`) |
| is run by MicroClaw during an acquisition | `gui/live_session.py` (the GUI on a fit another program controls through `Context.stop` and `writer_finished`; the package and its protocol runner are `ries-lab/smappy-microclaw`) |

A chain of plugins that runs as one, and running one over many files
(`smappy-batch`, the batch window): `docs/batch.md`.

A plugin is a settings dataclass plus a `run`:

```python
@dataclass
class ThingSettings:
    radius_nm: float = param(50.0, label="radius", unit="nm", min=0.1,
                             help="what the GUI puts in the tooltip")

@register("Analysis/Measure/Thing")
class Thing(Plugin):
    Settings = ThingSettings
    version = "1"

    def run(self, ctx: Context, settings: ThingSettings) -> Result:
        locs = ctx.selection.apply(ctx.locs)
        return Result(text="...", data={...}, plot=draw, settings=settings)
```

* `param(...)` is a dataclass field carrying the GUI presentation -- label,
  unit, bounds, `choices`, `advanced=True` to hide it under "more".  The GUI
  builds its widgets from these; nothing in a plugin imports Qt.  A field
  that is itself a dataclass is a part, a section of its own
  (`collapsed=True` starts it folded); rows and parts are laid out in the
  order they are declared.
* `version` -- bump it when a change moves the numbers: a chain records it,
  a batch re-runs files whose steps' versions changed, and the plugin's page
  must be re-read and brought along (see "The plugin's page" below).
* `@register("Tab/Group/Name")` places it in the tree.  Without it the *folder*
  decides the path (`<root>/Analysis/Drift/comet.py` -> `Analysis/Drift/COMET`),
  which is how a user drops a plugin in and it appears.
* Discovery *parses* plugin files with `ast` and never imports them, so a
  plugin that imports torch costs nothing until someone opens it.  Keep heavy
  imports inside functions (`from scipy.ndimage import ...`), as the shipped
  plugins do.
* `per_region = True` says `run` hands back `Result.values` (numbers, a
  column each of the region table) and `Result.arrays`, so it can run once
  per ROI -- in `ROIProject.region_context(roi)`: `ctx.locs` the ROI's rows,
  each layer's filter over them, `ctx.region` its shape -- as well as on the
  view like any other plugin.  The evaluation window offers these; in a chain
  it runs once in each region as a step of a block (`chain.Block`,
  `smappy/blocks.py`): over the ROIs or an id column such as `cluster_id`
  (`docs/regions.md`).  Its results belong to
  the ROI, whoever ran them, under the evaluation's name
  (`ROIProject.evaluations`; a renamed one is a second evaluation), so an
  analysis reads `ROIProject.results()` and never the pipeline.
* Override `preview` for a plugin that can show its work before committing;
  the GUI grows a button when it is overridden and nothing when it is not.
  A plugin that changes nothing needs none: Run opens its figures.
* Override `preflight` to put a question before a run whose cost you can know
  in a fraction of a second -- `drift_comet.py` does, rather than letting
  someone find out over the next twenty minutes.  Returning a
  `PreflightQuestion` instead of a string offers named alternatives beside the
  plain yes, each carrying the settings it would run with: COMET offers an RCC
  prepass, because "no" is not a useful answer to "this will take an
  afternoon".
* Override `notice` to say something about the settings as they stand, before
  any run -- a line under the form, and optionally a button that opens a GUI
  window by name (`Notice(text, action, window)`, the same
  `session.window_openers` as `Plugin.window`).  The fitters use it for a file
  whose camera is not in the database, opening the camera editor.  When
  what it says depends on the session, override `notice_for(ctx, settings)`
  instead -- the saver says which region results its settings leave out.
* `changes_locs` -- True if `run` hands back a new table (or files), False
  if not; every shipped plugin declares it (`tests/test_changes_locs.py`),
  and `changes_locs_for(settings)` refines it where it depends on the
  settings.  While a fit is still appending, `Session.apply` refuses a table
  change and the GUI asks before such a run, offering a new window.  A change
  that is a column computed from the others can return a `LiveRecipe`
  instead (`data["live_recipe"]`, with `live_recipe_for`), which follows the
  live table: `math_parser.py`, `assign_colors.py`.
* Set `live = True` on a plugin whose work is tens of milliseconds: the GUI
  grows a *live* tick and measures again while the ROI is dragged (its
  `preview` if it has one, else `run`, never applied), silently (no log line,
  no window, no focus).  It turns a measurement one asks for into one that
  can be aimed.
* **A plugin that measures usually wants one answer per layer**, not one over
  the selection: a line is drawn over two channels and the measurement is how
  they differ.  `ctx.layer_selection(i)` is layer `i`'s (not
  `session.selection(i)`: during a live fit the session's table has grown
  since the context was made), and `line_profile.py`'s `_groups` is the
  shape to copy.

**Put the algorithm in module-level functions that take arrays** and let `run`
be the thin wrapper.  A script, a test and a notebook should be able to get the
same numbers without a session, a plugin or a window.  Every shipped plugin is
written this way; it is also what makes the tests readable.

### The essentials

The full reference -- every field of the context and the result, the column
vocabulary, the picture's axes, simulated data -- is `docs/plugin-reference.md`.
What a plugin gets wrong without it:

* `ctx.locs` is the whole table and **always the ungrouped one**;
  `ctx.selection` is what the user is looking at (filter, ROI, slab) as a mask
  over it.  `ctx.table(layer)` is the table grouped or not as the layer
  decides -- read that in a plugin that should honour grouping.
* `ctx.session` may be `None`: a script says `Context(locs=locs)`.
* `ctx.report` for progress, `ctx.warn` for what must not be missed (progress
  overwrites the one, not the other).
* Return a **new** table in `Result.locs`, never edit `ctx.locs`, and return
  the settings actually used: the undo and the history hang off the result.
  A run that changes the table is logged; one that only measures is not.
* `plot(ax)` gets one axis; for a grid use `Plot(draw, panels=n, size=...)`.
  Never set a figure's size or layout engine (on the *All* page it is a
  `SubFigure`), and make a plot a closure over its data: tabs draw lazily.
* Columns are `<quantity>_<unit>` and precisions `<quantity>_err_<unit>`
  (`x_nm`, `xy_err_nm`, `photons_err`); look a column up rather than assume
  it, and name the columns the table *does* have when one is missing.
* A column that needs a rule for grouping is a recipe (`metadata["derived"]`),
  not plain data: `math_parser.py`.

## Tests

`tests/test_*.py`, pytest, no classes.  Name a test as the sentence it proves --
`test_the_precision_fit_finds_sigma_c_and_both_landmarks` -- and test the
numbers: simulate data with known parameters and assert they come back.
`tests/test_statistics.py` is the pattern.

The discovery and workspace tests (`test_discovery.py`, `test_workspace.py`,
`test_file_plugins.py`) check that the shipped plugins are found and that the
tabs seed by rule -- a floor of plugins that must be there, the Analysis tab
being every favourite under `Analysis/` but the regions, finders before
summaries.  Adding a plugin needs no test edits unless it changes a rule.

`tests/test_control_width.py` builds every shipped plugin's panel and fails if
it is wider than the control column (`CONTROL_WIDTH`): a new plugin with a long
label, a long choice or a wide row of widgets fails there.

`tests/test_tutorial.py` runs every tutorial storyboard against the real GUI
(slow tier: `--slow`)
(`python -m smappy.tutorial`, see `src/smappy/tutorial/__init__.py`), so a GUI
change that breaks a tutorial fails there: fix the storyboard in
`src/smappy/tutorial/topics/`, which usually means a widget that was renamed
or moved.  Before writing or editing a storyboard, read
`src/smappy/tutorial/STYLE.md`: the reviewers' directions for how tutorials
speak.

`tests/test_plugin_docs.py` checks every shipped plugin's page: that it
exists, that its `version` is the plugin's, that every setting is explained, and that it
renders with its maths and figures (the figures in the slow tier).

Plugins reach the shipped workspace by themselves: the Analysis tab seeds from
`Analysis/` except `Analysis/Regions/`, which seeds the Regions tab, and `favorite` defaults to True, so there is no registry to edit.
For a `per_region` plugin, `favorite` also decides whether it is in a
new evaluation pipeline; a specialised one (`NPC Corners`) sets it False.

## The plugin's page

Every plugin has a page, `src/smappy/plugins/docs/<Tab>/<Group>/<Name>.md`
(or `<file>.md` beside a dropped-in `<file>.py`), which the **?** in its title
bar, F1 in its panel and Help > Plugin documentation open, and which
`python -m smappy.docs -o DIR` writes as a static site.  The page is for the
people who use the plugin -- biologists and biophysicists -- so they can
understand the algorithm without reading the code.  The format is in
`src/smappy/docs/__init__.py`; `Analysis/Drift/RCC.md` is the example to copy.

The tabs and windows that are not plugins (the Render tab, the Regions tab,
the ROI manager, the region steps, the 3D view, the bead calibration, the batch window) have pages too,
in `src/smappy/docs/panels/`, with `title`, `summary` and `widget` in the
front matter and a *Controls* section in place of *Parameters*: one
`### label` per control, the label exactly as the panel shows it.
`tests/test_panel_docs.py` builds each panel and fails on a label it does not
show, so a session that renames a control or adds one updates the panel's
page too.

**Writing it is part of the work, and it is yours.**  A session that adds a
plugin writes its page in the same change.  A session that changes what a
plugin computes -- the algorithm, a normalisation, a default that moves the
numbers, anything that bumps `version` -- re-reads the page against the new
code, rewrites what is no longer true, redraws what needs it, and sets the
page's `version` to the plugin's.  The test fails until it does, and bumping
the page's version without reading it defeats the only check there is.  A
refactor that changes nothing a user could see needs neither.

What is generated and must not be written by hand: the title, the menu path,
the version line, the summary (`Plugin.description`) and the settings table
(labels, defaults, units, bounds, choices and the `help` text).  The page
holds everything else.

**How a page is written:**

* **Sections, in this order**: *What it does* (the problem, what the plugin
  does about it, when to use it and when not, what it needs), *How it works*
  (the steps, one bold lead-in each, with figures), *In detail* (the maths,
  the constants, the edge cases), *Parameters* (notes only), *Output* (what
  each number and figure means, what a good and a bad result look like),
  *Differences from SMAP* (only for a plugin ported from SMAP),
  *References*.  Leave out a section that has nothing to say.
* **Two registers.**  The first two sections are for a biologist: plain
  words, one idea per sentence, an equation only where it says something a
  sentence cannot, every symbol explained where it first appears.  *In
  detail* may be as mathematical as the method is, for the reader who wants
  to know exactly what is computed.
* **Describe the code, not the paper.**  Every claim must be true of the
  code as it is: read the functions before writing about them, take the
  constants from them (5 iterations, the 1.4826, the 20th and 80th
  percentiles), and name them in `covers`.  Where the code departs from the
  textbook method, say what it does and why -- the reason is usually in the
  module docstring or NOTES.md.  Check a claim about the GUI (a button, a
  default, what is saved) in the GUI code.  If you are not sure, find out or
  leave it out.
* **Settings by their GUI label**, in italics (*time windows*), in the
  prose.  Column names in backticks (`xy_err_nm`).
* **Parameter notes add to the tooltip, never repeat it.**  The tooltip
  (`param(help=...)`) is a short phrase and is shown in the table anyway; a
  `### field` note says how to choose the value, what goes wrong at either
  extreme, or a typical range.  A setting with neither fails the test, and a
  tooltip is the better place for anything short, because the GUI shows it
  on hover.
* **Figures show the method working** -- for a plugin with a method; a
  loader, a saver or a plugin that only reshapes the table needs none.  They
  are drawn by the plugin's own
  module-level functions on data from `smappy.simulate`, with the known
  truth drawn beside the answer where there is one.  Put the shared
  simulation in one ```` ```figure-setup ```` block; keep a page's figures to
  a few seconds in all.  Simulated tables keep spots down to 10 photons, so
  filter them (at 200 photons, say) before anything about precision.  The
  caption says what to see, not what is plotted.  No screenshots and no
  images checked in: they go stale.
* **Maths is LaTeX, in what both renderers take**: ziamath sets it, and
  matplotlib's mathtext is the fallback, so write the subset mathtext knows
  too -- `\leq`, `\geq` (not `\le`, `\ge`), `\mathrm{}`, `\frac`, `\sqrt`,
  `\sum`, `\left( \right)`, braces around what a `\hat` covers
  (`\hat{N}_0`); no `\texttt`, `align` or matrices.  The test sets every
  formula with both.
* **Differences from SMAP** from the MATLAB itself (`../SMAP`, `jries/SMAP`),
  concretely: what SMAP does, what this does, and why -- again the main
  changes only, the ones that can substantially change the output.  A plugin
  ported from SMAP says so in one line at the start of the section, naming
  SMAP's plugin, and cites SMAP (Ries 2020) there and in *References*.
* **The paper a method comes from is cited in *What it does***, where the
  method is first named ("RCC (Wang et al. 2014)"), with its DOI link, so a
  reader knows at once whose method it is.  *In detail* cites it again and
  says what the code does differently from the publication, and why -- but
  only the main changes, the algorithmic ones that can substantially change
  the output (a different estimator, a step added or dropped, a model that
  differs), not every constant, option or speed-up.  A plugin with no method
  paper behind it cites none.
* **References** that were checked (PubMed or the journal) -- only where
  there is a paper to cite -- with a DOI link;
  write a `(` or `)` in a URL as `%28` / `%29`.  Say in a few words what each
  one is cited for when it is not obvious.
* Spelling as the plugin's module.  Look at the rendered page (the Help
  window, or the exported HTML) before calling it done.

## Porting a plugin from SMAP

SMAP is the MATLAB original, at `../SMAP` in these sessions and `jries/SMAP` on
GitHub.  Its plugins live in `plugins/+Analyze/+<group>/Name.m` and
`plugins/+ROIManager/+{Segment,Evaluate,Analyze}/Name.m`, with the real work
usually in a helper beside them (`private/`, `shared/myfunctions/`) rather than
in the class.

**Port the algorithm, not the structure.**  A SMAP plugin is a
`DialogProcessor` subclass whose `run(obj, p)` reads a GUI struct, and most of
its length is GUI plumbing that has an equivalent here and should not be
transliterated.  The correspondence:

| SMAP | here |
| --- | --- |
| `classdef X < interfaces.DialogProcessor` | `class X(Plugin)` |
| `guidef` / `pard.*` | the `Settings` dataclass and `param(...)` |
| `obj.getPar` / `setPar`, global parameters | `ctx.session`, or a setting |
| `obj.locData.getloc(fields, 'layer', L, 'position', 'roi')` | `ctx.selection.apply(ctx.locs)` |
| `'position', 'all'` | `ctx.locs` |
| `'grouping', 'grouped'` | `ctx.table(layer)` (`docs/plugin-reference.md`) |
| `obj.setPar('status', ...)` | `ctx.report(text)` |
| `obj.showresults` / `initaxis` | `Result.plot` / `Result.plots` |
| `out.clipboard`, the tab-separated summary | `Result.text` and `Result.data` |

Column names differ; `src/smappy/io/formats.py` (`SML_COLUMNS`) is the
authoritative map -- `xnm`->`x_nm`, `phot`->`photons`, `locprecnm`->
`xy_err_nm`, `locprecznm`->`z_err_nm`, `PSFxnm`->`sigma_nm`, `LLrel`->`logl_rel`.  SMAP's
`numberInGroup` is `n_in_group` and no loader writes it: it comes from linking
here.

Deviating is expected and welcome -- simplify, drop options nobody used, fix
what was wrong -- but say so in the module docstring, the way `NOTES.md`
records the departures in the fitting pipeline.  Read the MATLAB carefully
first: the useful part is often a normalisation, an edge case or a constant
buried in a helper, not the shape of the file.

## House style

The code is written to be read.  Docstrings say *why*, not what -- the
trade-off, the failure it avoids, the number that was measured -- and comments
mark the decisions someone would otherwise undo.  Match the density of the file
you are in; see `group.py` (why one column at a time, with timings) or
`assign_colors.py` (the derivation, with the maths) for the register.  British
or American spelling, whichever the file already uses.

Keep changes minimal and finish them: a new plugin means the module, its tests,
and its page.

## Keeping this file true

A stale instruction here is worse than none: it is confidently wrong, and a
session will follow it instead of reading the code.  So, while you work:

* **Check what you use.**  When this file tells you something you are about to
  rely on -- a column name, a signature, which test enumerates what -- confirm
  it in the code as you go.  The code wins, always.
* **If a fact here disagrees with the code, trust the code and correct the
  line** -- a renamed file, function, column or test -- in the same change,
  and say so in the summary.  The same goes for `docs/plugin-reference.md`.
* **Stop and ask only when a line states a rule or an intention** (how
  plugins are written, what is logged, what a test must cover) that the code
  contradicts: it may be the code that is the mistake, and which one changes
  is the user's call.
* **If your own change makes a line here wrong, fix it in the same change.**
  Adding a plugin shape, moving the grouped table or renaming a column lands
  here.  Changing a *rule* -- a section of this file, not a fact in it --
  is proposed first: quote the old line and the new one and ask.
