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
  collector otherwise runs in whichever thread allocates and frees every
  cycle in the process, whatever that thread holds.  PySide sends only plain
  `QWidget`s back to the main thread, not subclasses, so a widget left in a
  reference cycle is destroyed on the worker.  On a fit's reader thread that
  deadlocked a test worker, and with automatic collection switched on again
  in 2026-10, `--slow` segfaulted a worker in 9 runs of 9.  `conftest.py` collects on the main thread every 100 tests;
  the GUI's entry points install `gui/collector.collect_on_gui_thread`, a
  timer on the GUI thread.  A test that needs an object gone calls
  `gc.collect()` itself; a new entry point gets its QApplication from
  `gui.make_app()`, which installs the collector, the icon and the fields'
  style, and one that makes a session installs its identity too
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
| measures drift from beads, a `Drift` the others' way | `drift_fiducials.py` (*Fiducials*: tracks of the bright localizations from *Link tracks*' linker, one linear fit of each bead's offset and COMET's spline in `smappy/fiducials.py`; `KeepsDrift` keeps the curve; beads simulated by `smappy.simulate.simulate_beads`) |
| loads, saves, exports, simulates | `file.py` (one loader with an advanced *format* choice, no input table -- a format is a `Reader` / `Writer` in `io/formats.py`, every column of the file kept; *Load*, a launcher of File > Open; *Export localizations*, whose column list is `kind="columns"`; *Export Image*, the one renderer of the view -- `smappy.picture`, framed by the windows' `Session.view_frames`, which the render and 3D windows' *Save* call too; *Export Stack*, movies and z-stacks on the same renderer, `smappy.movie`; the simulator's model is `smappy.simulate`) |
| reads a container of several measurements and images | `io/msr.py` (Abberior `.msr`, read by *File/Load/Auto*'s *Abberior MSR* part: the OBF container `io/obf.py`, the MINFLUX archive and zarr `io/zarr_lite.py`, blosc/LZ4 `io/blosc.py`, both record layouts `io/minflux.py`; File > Open asks which parts, `gui/msr_import.py`, through a reader's `needs=("measurements",)`; a file per measurement in `Result.files`, images in `Result.image_layers`, each file's own metadata in `FileInfo.metadata`) |
| finds, measures or summarises ROIs | `roi.py` (a segmenter derives from `SegmenterPlugin`, which gives it a Preview, *replace* (`replace_param()`) and a step of the ROI undo; `per_region = True`; analyse) |
| finds ROIs by one of several methods, with presets | `segment.py` (*Segment regions*: one `SegmenterPlugin`, a *method* part each -- ring template (`npc.segment_npcs`), density peaks (`roi.density_peaks`), from an id column (`region_column.column_regions`); a *preset* is a YAML file in `data/segment_presets` or the config directory that fills the settings through `react`; each ROI's `origin["segment"]` names its method, so *replace* removes that method's only) |
| parts, a C++ backend | `fit.py` (nested settings dataclasses; four fitters, thin subclasses of `_FitPlugin`; the *fit region* drawn on the Preview's frame: a figure panel that takes shapes is `Plot.edit`, a `ShapeEdit`, and the figure window adds the tools) |
| adds a derived column, no output | `math_parser.py` (an expression, kept with the table as a recipe) |
| removes or flags localizations | `remove_locs.py` (a region, a new table or a filtered flag) |
| fits a model to what is in a ROI | `line_profile.py` (unbinned likelihood, models compared by AIC; one fit per layer, `follow_roi` and `per_region`: in an ROI the line is one of its named lines, `region_line`); `circle_fit.py` (the NPC segmenter's `fit_circle` with errors) |
| moves the ROIs onto what was measured in them | `roi_centre.py` (*Centre ROIs*: `ROIProject.move_roi` in one ROI undo step) |
| counts in time: blinks, dark times, bursts | `blinking.py` (one region on one time line, the view linked into molecules per file; seconds from the file's frame time, `metadata.file_frame_time`) |
| reads the session and reports | `history.py` (the log, with an optional export) |
| sets up layers and filters from settings | `chain_layers.py` (bounds as rows per layer, `Result.layers`) |
| scores a fit against a simulation | `ground_truth.py` (the truth redrawn from the recipe, matched per frame) |
| draws what the microscope recorded per frame | `image_tags.py` (the fit's `metadata["frame_tags"]`) |
| finds a structure by its shape and judges it, then counts it | `npc.py` (a ring filter and circle fit, the ring template of *Segment regions*, a per-site count, and a model of whole pores in `smappy/npc.py` fitted to the site table) |
| summarises what evaluators measured | `roi.py`'s `SiteAnalysisPlugin` (an evaluator to read, a *results from* field, a preflight that asks which when there are several; under `Analysis/Regions/Summarize`) |
| summarises curves over the regions | `curves.py` (*Curves*: a `SiteAnalysisPlugin` reading each region's `Result.arrays` from its stored entry; groups by an ROI keyword list; pools through the evaluator's `Plugin.pool` where it hands back counts) |
| runs the ROI manager's segment, measure and analyse as one | `npc_analysis.chain.yaml` (a shipped chain: a block runs NPC Corners in every ROI before the next step; `docs/batch.md`, "ROIs: segment, measure, analyse") |
| links localizations into tracks | `link_tracks.py` (*Link tracks*: an id column, `track_id` / `track_length` or an *output column*, of the localizations whatever the layer shows; everything that reads tracks has a *track id*, *auto* by default, resolved by `tracking.columns.track_column` -- never a hardcoded `track_id`; two linkers, an *algorithm* choice, in `smappy/tracking/link.py`; candidates from `csrc/neighbours.cpp`'s `Index.pairs`, a simple linker's loop in C++ (`greedy_links`); tracks simulated by `smappy.simulate.simulate_tracks` -- in the GUI *File/Simulate/Tracks*, a plugin of its own beside *Blinking Structure* -- scored by `tracking.score`) |
| measures how tracks move, per track and pooled | `tracking_analysis.py` (*Tracking analysis*: approaches ticked as in *Spatial statistics*, a part each shown while ticked; an *id column*; vectorised over the whole table in `smappy/tracking/analysis.py`; `per_region`, and per-track columns only on a tick, `changes_locs_for`) |
| diagnoses the instrument from beads, approaches to come | `stability.py` (*Stability*: *vibrations* ticked, its part shown while on; beads by an *id column*, each, their mean or differential; a regular grid with gaps filled and counted, Welch's spectrum, the cumulative RMS and peaks in `smappy/stability.py`; each figure a tick; `per_region`, values pooled over the region's traces) |
| writes a number per localization from what is around it | `neighbourhood.py` (two layers as they show, *around* and *neighbours from*; a recipe column, NaN outside the filter, a blink's value through `group_id`; the search in C++, `csrc/neighbours.cpp`) |
| measures curves over pair distances, with envelopes | `spatial_statistics.py` (*Spatial statistics*: approaches ticked on or off, each one's part shown only while on; pair distances from `Index.pair_histogram` in `csrc/neighbours.cpp`, normalised by the window's set covariance; Monte Carlo envelopes; per region, curves as `Result.arrays` with their numerator and denominator; the directional approaches -- a correlation image, a correlation along a line in a line ROI's own axes -- in `smappy/correlation.py`) |
| finds clusters and names them in the table | `clusters.py` (*Find clusters*: an *algorithm* choice, a part per algorithm, only the chosen one shown -- DBSCAN, Voronoi, HDBSCAN with an optional dependency -- all writing `cluster_id` / `n_in_cluster`; DBSCAN on the C++ cell search of `csrc/neighbours.cpp`, `Index.dbscan_join`; ROIs of them are *Segment regions*' method *from an id column*, for any id column) |
| orders or filters the ROIs by what was measured in them | `roi_order.py` (*Sort ROIs*: `ROIProject.order`, which the list and the results follow, never the numbers; *Filter ROIs*: a parsed condition setting *use* in one undo step, on the measured columns, the keywords and each ROI's place (`roi_x_nm`, `in_drawn_roi`); *Remove double ROIs*: a pair closer than d, the larger kept; `roi_manager/order.py`) |
| superimposes the ROIs into a new table | `average_rois.py` (*Average ROIs*: a centre and a turn per ROI from its centre, its lines (`ROI.lines`) or its results row; one table added as a file through `Result.files` with `append`, each ROI's frames moved apart so merging blinks never joins two) |
| moves a file's localizations or an image layer | `transform.py` (*Transform*: the target -- a file and channel, or an image layer's placement, `ImageData.placement`, never resampled -- and the source -- typed, a named transformation (`Session.transformations`, added through `Result.transformations`), a file (`_2ct.h5`, `_T.h5`) or another file's drift (`Drift.apply(rows=)`); the maths in `smappy/transformation.py`; Register layers opens it with `settings_for(name)` through the GUI's `"transform"` window) |
| makes a new image layer from another | `process_image.py` (*Process image*: background subtraction and local contrast, ticked operations in a list, `_STEPS`; the maths on arrays in `smappy/image_processing.py`; the new image keeps the original's placement and goes back through `Result.image_layers`) |
| measures and hands its result to another plugin | `register_layers.py` (*Register layers*: the shift of a layer against another, localizations or images, by `correlation._xcorr` and RCC's `_peak` -- no third correlation -- in `smappy/layer_registration.py`; kept as a named transformation through `Result.transformations`, and a `Notice` with `window="transform"` and `transform.settings_for` opens Transform on it) |
| writes what was measured per region onto its localizations | `region_column.py` (*Regions to column*: `region_runs.write_values`, NaN outside, never over a column of the localizations' own; a block's *write values* calls the same) |
| runs steps once per region, writes columns back | `smappy/blocks.py` (a `chain.Block`; the regions through `region_runs.regions_of` -- ROIs or an id column, one interface; `RegionScratch`, `WriteBack`, *write values*; per-cluster results in `Session.id_regions`; the ROI project holds one, `ROIProject.block`) |
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
  view like any other plugin.  The per-region steps (the Regions tab) offer these; in a chain
  it runs once in each region as a step of a block (`chain.Block`,
  `smappy/blocks.py`): over the ROIs or an id column such as `cluster_id`
  (`docs/regions.md`).  Its results belong to
  the ROI, whoever ran them, under the evaluation's name
  (`ROIProject.evaluations`; a renamed one is a second evaluation), or to
  the id column (`Session.id_regions`), so an analysis reads them through
  `region_results` (`rows`, the current ones of either) and never the
  pipeline.
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
* Override `describe(ctx, settings)` to say what the form should show about
  the settings as they stand, before any run, as a `FormState`: `hints` (what
  fields on *auto* resolve to), `active` (which fields are read; the rest are
  greyed), `shown` (which are shown at all: the parts of the algorithms not
  selected are hidden), `choices` (lists the session decides) and a `notice` -- a line
  under the form, and optionally a button that opens a GUI window by name
  (`Notice(text, action, window)`, the same `session.window_openers` as
  `Plugin.window`).  `ctx` is None when there is no session.  The fitters
  use it for a file whose camera is not in the database, opening the camera
  editor; the saver says which region results its settings leave out.  A
  chain forwards its steps' with the one hook.
* `changes_locs` -- True if `run` hands back a new table (or files), False
  if not; required, a class that does not declare it is refused when it is
  defined (`tests/test_changes_locs.py` checks the shipped ones say it
  truly), and `changes_locs_for(settings)` refines it where it depends on the
  settings.  While a fit is still appending, `Session.apply` refuses a table
  change and the GUI asks before such a run, offering a new window.  A change
  that is a column computed from the others can return a `ColumnRecipe`
  instead (`Result.recipe`, with `column_recipe_for`), which follows the
  live table: `math_parser.py`, `assign_colors.py`.
* Set `follow_roi = True` on a plugin whose work is tens of milliseconds: the
  GUI grows a *follow ROI* tick and measures again while the ROI is dragged (its
  `preview` if it has one, else `run`, never applied), silently (no log line,
  no window, no focus).  It turns a measurement one asks for into one that
  can be aimed.
* **A plugin that measures usually wants one answer per layer**, not one over
  the selection: a line is drawn over two channels and the measurement is how
  they differ.  `ctx.table(i)` is layer `i`'s table and selection, as the
  layer shows it (not `session.table(i)`: during a live fit the session's
  table has grown since the context was made), and `line_profile.py`'s
  `_groups` is the shape to copy.

**Put the algorithm in module-level functions that take arrays** and let `run`
be the thin wrapper.  A script, a test and a notebook should be able to get the
same numbers without a session, a plugin or a window.  Every shipped plugin is
written this way; it is also what makes the tests readable.

### The essentials

The full reference -- every field of the context and the result, the column
vocabulary, the picture's axes, simulated data -- is `docs/plugin-reference.md`.
What a plugin gets wrong without it:

* `ctx.locs` is **the table the layer shows** -- one row per blink on a
  grouped layer -- and `ctx.selection` is what the user is looking at
  (filter, ROI, slab) as a mask over it.  A plugin that needs the
  localizations (it hands back a new table, or pairs frames as NeNA does)
  declares `grouping = "ungrouped"` and gets them whatever the layer shows;
  a new table made from the grouped one is refused.  `ctx.ungrouped()` is
  the localizations for a plugin that measures blinks and writes onto them
  (`clusters.py`), and `ctx.table(layer)` another layer's table.  Say on the
  plugin's page which table it reads.
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
`test_file_plugins.py`) check that the shipped plugins are found and that
every plugin the shipped GUI names exists.  Adding a plugin needs no test
edits unless it is under `Analysis/Regions/` or offered per region
(`test_roi_pipeline.py` lists both), or a File plugin
(`test_file_plugins.py` lists those).

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

A plugin is in the tree as soon as it exists; which ones a fresh install
pins to its tabs is a list, `src/smappy/data/default.gui.yaml` -- a GUI file
like any other (*File > Workspace > Save* writes the same format), so a new
plugin that belongs in the shipped GUI gets a line there.  Likewise the
per-region steps a new ROI project starts with are a list,
`roi_manager.pipeline.DEFAULT_REGION_STEPS`.

## The plugin's page

Every plugin has a page, `src/smappy/plugins/docs/<Tab>/<Group>/<Name>.md`
(or `<file>.md` beside a dropped-in `<file>.py`), which the **?** in its title
bar, F1 in its panel and Help > Plugin documentation open, and which
`python -m smappy.docs -o DIR` writes as a static site.  The page is for the
people who use the plugin -- biologists and biophysicists -- so they can
understand the algorithm without reading the code.  The format is in
`src/smappy/docs/__init__.py`; `Analysis/Drift/RCC.md` is the example to copy.

The tabs and windows that are not plugins (the Render tab, the Regions tab,
the ROI manager, the per-region steps, the 3D view, the bead calibration, the batch window) have pages too,
in `src/smappy/docs/panels/`, with `title`, `summary` and `widget` in the
front matter and a *Controls* section in place of *Parameters*: one
`### label` per control, the label exactly as the panel shows it.
`tests/test_panel_docs.py` builds each panel and fails on a label it does not
show, so a session that renames a control or adds one updates the panel's
page too.  Beside them is `Glossary.md`, which has no widget (`WIDGETLESS`
in that test): the field's words with the other names they go by, which a
page links (`plugin:Panels/Glossary`) where it first explains one.
`tests/test_plugin_docs.py` checks that every link between pages leads to a
page.

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
* **Jargon** is explained at its first mention on each page: the names
  common in the field and one plain line, linked to the Glossary
  (`plugin:Panels/Glossary`) and worded as it is there.
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
