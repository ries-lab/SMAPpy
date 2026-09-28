# Working in this repository

SMAPpy (`smappy` in code) is a Python SMLM package: TIFF -> photons -> peak finding -> MLE fit ->
localization table -> analysis, with a Qt GUI over the same functions.  Read
`NOTES.md` for the decisions behind the fitting pipeline and
`docs/plugin-architecture.md` for why the plugin system is shaped as it is.
This file is the practical part: what a session needs before it can write a
plugin, and the traps that cost time.  It is a summary written at one moment
and the code moves, so treat it as a starting point, not as authority -- and
see "Keeping this file true" at the end.

## Before you edit anything

Pull first:

    git fetch origin && git rebase origin/main      # or: git pull --rebase

Work lands on `origin/main` from cloud sessions and other machines, so the
local `main` is routinely well behind it -- 28 commits, on one occasion.
Editing first means rebasing finished work through conflicts in files upstream
has already changed, and it is always the same files: `NOTES.md`, and the
plugin-list assertions in `tests/test_workspace.py` and `tests/test_discovery.py`,
which every added plugin extends.  If you have already started, `git stash push
-u` first so new untracked files come along, and rebuild afterwards if `csrc/`
moved.

## Getting the checkout to run

A fresh container has **no numpy**.  Before anything else:

    pip install -e .                    # builds the C++ extensions too
    pip install pytest                  # not a dependency, and not in the container

Running the tests:

    python -m pytest tests -q           # `tests`, not `.`

* Plain `pytest` collects `externaltools/Comet`, which imports numba and errors
  out.  Always name `tests`.
* The Qt tests need Qt's own libraries, not a display: on a bare container they
  fail with `ImportError: libEGL.so.1`.  Install them once --

      apt-get update && apt-get install -y libegl1 libgl1 libxkbcommon0
      QT_QPA_PLATFORM=offscreen python -m pytest tests -q

  -- and the whole suite runs.  Without them about 30 fail (`test_workspace`,
  `test_calibrate_qt`, `test_roi_pipeline`, parts of `test_context` and
  `test_roi_session`), which is the environment and not your change.
* `SMAPPY_TEST_CAL=/path/to/_3dcal.mat` enables the tests that want real
  calibration data; without it they skip.

## Writing a plugin

The framework is the same whatever the plugin does; the shipped ones differ in
shape, so read the one closest to what you are writing:

| shape | example |
| --- | --- |
| measures the selection and draws it | `statistics.py` (settings, selection, several figures in one window) |
| rewrites the table and hands it back | `assign_colors.py` (`Result.locs`, a preview) |
| a long computation with progress | `drift_comet.py`, `drift_rcc.py` |
| loads, saves, exports, simulates | `file.py` (a plugin per format, no input table; the simulator's model is `smappy.simulate`) |
| finds, measures or summarises ROIs | `roi.py` (segment, `scope = "site"`, analyse) |
| parts, presets, a C++ backend | `fit.py` (nested settings dataclasses) |
| adds a derived column, no output | `math_parser.py` (an expression, kept with the table as a recipe) |
| removes or flags localizations | `remove_locs.py` (a region, a new table or a filtered flag) |
| fits a model to what is in a ROI | `line_profile.py` (unbinned likelihood, models compared by AIC; one fit per layer, and `live`) |
| reads the session and reports | `history.py` (the log, with an optional export) |
| sets up layers and filters from settings | `chain_layers.py` (bounds as rows per layer, `Result.data["layers"]`) |
| scores a fit against a simulation | `ground_truth.py` (the truth redrawn from the recipe, matched per frame) |

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
* `scope = "site"` makes it an ROI evaluator, run once per ROI with `ctx.site`
  set.  Otherwise it runs once over the selection.
* Override `preview` for a plugin that can show its work before committing;
  the GUI grows a button when it is overridden and nothing when it is not.
* Override `preflight` to put a question before a run whose cost you can know
  in a fraction of a second -- `drift_comet.py` does, rather than letting
  someone find out over the next twenty minutes.  Returning a
  `PreflightQuestion` instead of a string offers named alternatives beside the
  plain yes, each carrying the settings it would run with: COMET offers an RCC
  prepass, because "no" is not a useful answer to "this will take an
  afternoon".
* Set `live = True` on a plugin whose work is tens of milliseconds and which
  overrides `preview`: the GUI grows a *live* tick and re-previews while the
  ROI is dragged, silently (no log line, no window, no focus).  It turns a
  measurement one asks for into one that can be aimed.
* **A plugin that measures usually wants one answer per layer**, not one over
  the selection: a line is drawn over two channels and the measurement is how
  they differ.  `session.selection(i)` is layer `i`'s, and `line_profile.py`'s
  `_groups` is the shape to copy.

**Put the algorithm in module-level functions that take arrays** and let `run`
be the thin wrapper.  A script, a test and a notebook should be able to get the
same numbers without a session, a plugin or a window.  Every shipped plugin is
written this way; it is also what makes the tests readable.

### The context

* `ctx.locs` -- the whole table, **always the ungrouped one**.
* `ctx.table(layer)` -- `(locs, selection)`, grouped or not as the layer, a
  chain and `Plugin.grouping` decide.  Read this, not `ctx.locs`, in a plugin
  that should honour a chain's grouping; `Session.table` links the grouped
  table on request.
* `ctx.selection` -- what the user is looking at: the layer's filter *and* the
  ROI *and*, while the 3D window is open with "plugins use the slab" ticked,
  the slab (`session.selects_slab`), as a boolean mask.  `.apply(locs)` cuts the table,
  `.require(n, ctx.report, "what")` refuses an empty one and warns about a thin
  one.  A plugin that offers "all localizations" means `ctx.locs` unfiltered.
* `ctx.session` -- may be `None`; a script can say `Context(locs=locs)`, so
  never assume a session exists.
* `ctx.report(text)` for progress, `ctx.emit(event, payload)` to hand partial
  results on.  Both are no-ops when nobody is listening.
* The **grouped** table (one row per blink) is a table of its own, with its own
  filter, at `session.layers[i].state.sets["grouped"]`; it exists only once the
  user has switched that layer to grouped, and `state.grouped_stale` says it no
  longer matches the current table.  Linking costs seconds to minutes -- say
  what is missing instead of doing it behind the user's back.  Once it has been
  built once, `group_id` and `n_in_group` are on the *ungrouped* table too
  (`group.attach`), so a column computed afterwards can be reduced per blink
  with a `bincount` and no relinking.

### The result

`Result(locs=..., text=..., plot=..., plots={...}, data={...}, settings=...)`,
every part optional.  `plot(ax)` is handed **one** axis; for a grid, pass
`Plot(draw, panels=n, size=(w, h))` instead, and `draw` is handed the figure
and lays out its own panels (`statistics.py`, `ROIManager/Analyze/Histograms`
and the fit preview all do this).  Never set the figure's size or its layout
engine: on the window's *All* page a plot is handed a `SubFigure`, which has
neither, and `size` is the hint the window reads instead.  Return the settings
that were actually used -- that is what the history records.

`text` goes into the plugin panel's output box, however long it is; the Text
button opens it in a window on request.  Set `text_window = True` on the
plugin only when the text *is* the result and is long -- `history.py` is the
one that does -- and the window then opens by itself.

A run that **changes** the localizations is logged: `Session.apply` appends
the plugin's path, the result's `text` and its settings to `session.history`,
which is written into the file (`metadata["history"]`, capped at
`MAX_HISTORY`) and read back when it is reopened, so a table says what was
done to it and with which numbers.  A run that only measures is not: it can
be repeated from the file, and logging it would record only that somebody
looked.  What a measurement worked out goes with its *result* instead --
`Plugin.keep`, and the session stores the settings beside it, which for a
measurement is the only record there is.  `Plugin.logged` overrides the rule
(`True` for a run that changes something the log cannot see, such as writing
a file; `False` for never), and `Analysis/Process/History` shows the log and
exports it.  A chain logs once, its steps under `steps`, whether or not it
changed the table.

That is why a plugin returns the settings it actually used, and why one that
changes the table hands back a new one rather than editing `ctx.locs`: the
undo and the record both hang off the result.

`Result.data["bounds"] = {field: (lo, hi)}` asks the session to open the
filter on a column the run has just written -- a plugin cannot set it itself,
because a filter belongs to the table it was built from and the new table
exists only once `Session.apply` has set it.

The figures of one result share a window, a tab each beyond the first, and a
tab is drawn when it is looked at and not before -- so a plugin with six
figures costs what one costs, and a plot must be a closure over its data
rather than something already drawn.

## The table

`Localizations` is a dataclass of `columns: Dict[str, np.ndarray]` plus
`metadata`.  `locs["photons"]`, `"photons" in locs`, `locs.keys()`, `len(locs)`,
`locs[mask]` (a new table).  The column vocabulary:

| column | meaning |
| --- | --- |
| `x_nm`, `y_nm`, `z_nm` | position (a pixel-unit table has `x_pix`, ...) |
| `frame` | camera frame, int64 |
| `photons`, `background` | per localization |
| `xy_err_nm` | lateral precision, the RMS of `x_err_nm` and `y_err_nm`; `xy_err_pix` in a pixel table |
| `z_err_nm` | axial precision, when the fit produced one |
| `n_in_group` | **on-time in frames**; grouping writes it onto both tables |
| `group_id` | which group a localization was linked into, 1-based; on the grouped table, its own row |
| `sigma_nm`, `sigma_y_nm`, `logl_rel` | PSF width and fit quality |
| `filenumber`, `channel` | which file, which channel |

Every precision is `<quantity>_err_<unit>` -- `x_err_nm`, `z_err_nm`,
`photons_err` (no unit, nothing after `_err`).  Names saved before a rename
are read through `columns.RENAMED`, and only there.

Prefer `next((n for n in ("xy_err_nm", "xy_err_pix") if n in locs), None)`
over assuming one spelling, and raise a message naming the columns the table
*does* have when something is missing.

A column that needs a **rule for grouping** is a recipe rather than data:
`mathparse` keeps it in `metadata["derived"]`, and the recipe says what the
field means once the localizations are grouped -- recomputed from its
expression there, or reduced by a rule (`mean`, `sum`, `any`, ...).  It is
`group.combine` that honours it, so a derived column is never averaged by
accident, and it survives a save.  Usually the recipe is an expression
(`math_parser.py`); a column that was *measured* per localization and only
needs the rule -- the `use` flag `remove_locs.py` writes -- is a recipe with
a rule and no expression.

## What the picture's axes are

The render grid is normally the table's positions, and usually a plugin can
forget this -- but it does not have to be.  `RenderSettings.axes`
(`render.RenderAxes`) says which column each axis is and what to divide it by,
so the same renderer draws photons against frame, or one fit parameter against
another, with the same layers, LUTs, filters, ROIs and 3D box; the hidden
"axes" section of the Render tab is where a user turns it on.  A coordinate is
the column over that axis's scale, which keeps the grid square in *render
units* -- that is why nothing downstream knows about it.

What that means for a plugin:

* `ctx.locs` and `render.positions` are unchanged: they are the table and its
  position columns, in nanometres, whatever the picture is showing.  Only the
  render path goes through `axes.coordinates`.
* `ctx.selection` is still a mask over the same rows, and an ROI drawn on a
  versatile picture is in that picture's coordinates, so the mask means what
  the user drew.
* `session.axes()` is the current set and `axes.is_default` is the ordinary
  picture.  A plugin that draws in nanometres, measures a distance or places a
  site should say so -- or check -- rather than assume: the extent the user is
  looking at may be in photons.
* A width per axis: `settings.sigma` and `settings.sigma_y`, in each axis's own
  units, zero meaning plain binning.  The localization precision only blurs an
  axis that is a position (`render.render_sigmas` has the rule).

`NOTES.md` under "Any column against any other" records why it is shaped this
way and what was left out of SMAP's `VersatileRenderer`.

## Simulated data

`smappy.simulate` is one model with two outputs, and the `File/Simulate/
Blinking Structure` plugin is its form: structure -> labelling -> blinking ->
localizations or camera frames.  NOTES.md, "The simulation model", has the
assumptions.  What a session needs:

* Structures are YAML (the syntax is in `simulate/structure.py`); the
  built-ins are the files in `src/smappy/data/structures` -- `demo` (the ring
  and cross the tutorials measure), `npc`, `filaments`, `pie` -- and a new
  one is a file dropped there.
* `simulate(n_frames=..., seed=..., drift=...)` gives a localization table;
  `camera_frames(...)` gives `(frames, truth)`.  Both take the settings
  dataclasses of `simulate/settings.py` for everything else.
* A camera simulation saved as `*.sim.yaml` (`simulate.source.write_recipe`)
  is an acquisition: a fitter's `source.path` may name it, `open_stack` opens
  it, the frames are drawn as they are read, and the camera comes from it.
* The truth columns: `emitter` (which fluorophore), `dye`, `copy` (which
  copy of the structure); camera truth adds `neighbour_nm`, the nearest other
  spot in the same frame.  `metadata["copies"]` has each copy's position and
  turn by its `copy` number (`x_nm`, ..., `alpha_deg`, `beta_deg`,
  `gamma_deg`: R = Rz(alpha) Ry(beta) Rz(gamma)).
* Simulated localizations go down to 10 photons, as SMAP's do, and Ground
  Truth sets anything below its `min_photons` (100) aside, so a check of a
  simulated table's precision should look at the bright ones.
* `Analysis/Measure/Ground Truth` scores a table against its simulation --
  found, false, missed, Jaccard, bias, and error over reported precision --
  finding the recipe by the table's `source` (a fitter writes it) or its
  `simulation_settings`; `plugins/ground_truth.py` has `match` and `compare`
  as functions for a test.  The `simulation` tutorial walks the whole path.
* The camera frames keep every spot, and every fluorophore blinks within the
  stack whatever its length, so a short test stack is dense: pass
  `labelling=LabellingSettings(efficiency=0.1)`, and compare a fit only
  against truth with `neighbour_nm > ISOLATED_NM` -- crowded spots pull z
  towards focus.  `tests/test_camera_frames.py` is the pattern.

## Tests

`tests/test_*.py`, pytest, no classes.  Name a test as the sentence it proves --
`test_the_precision_fit_finds_sigma_c_and_both_landmarks` -- and test the
numbers: simulate data with known parameters and assert they come back.
`tests/test_statistics.py` is the pattern.

One coupling to remember: **`tests/test_discovery.py` and
`tests/test_workspace.py` enumerate the `Analysis/` plugins exhaustively**, so
a new plugin there fails four tests until it is listed in both (the workspace
ones name the seeded Analysis tab, including the order its titles come back
in).  `tests/test_file_plugins.py::test_the_file_tab_ships_with_the_four_it_needs`
asserts the File tab's exact contents, so a new File plugin needs a line there
too (it is a Qt test and skips without PySide6).

`tests/test_tutorial.py` runs every tutorial storyboard against the real GUI
(`python -m smappy.tutorial`, see `src/smappy/tutorial/__init__.py`), so a GUI
change that breaks a tutorial fails there: fix the storyboard in
`src/smappy/tutorial/topics/`, which usually means a widget that was renamed
or moved.  Before writing or editing a storyboard, read
`src/smappy/tutorial/STYLE.md`: the reviewers' directions for how tutorials
speak.

`tests/test_plugin_docs.py` checks every shipped plugin's page: that it
exists or the plugin is in `UNDOCUMENTED` (the plugins written before pages
were -- a list that only shrinks, and a new plugin never joins it), that its
`version` is the plugin's, that every setting is explained, and that it
renders with its maths and figures.

Plugins reach the shipped workspace by themselves: the Analysis tab seeds from
`Analysis/` and `favorite` defaults to True, so there is no registry to edit.

## The plugin's page

Every plugin has a page, `src/smappy/plugins/docs/<Tab>/<Group>/<Name>.md`
(or `<file>.md` beside a dropped-in `<file>.py`), which the **?** in its title
bar, F1 in its panel and Help > Plugin documentation open, and which
`python -m smappy.docs -o DIR` writes as a static site.  The page is for the
people who use the plugin -- biologists and biophysicists -- so they can
understand the algorithm without reading the code.  The format is in
`src/smappy/docs/__init__.py`; `Analysis/Drift/RCC.md` is the example to copy.

The tabs and windows that are not plugins (the Render tab, the ROI tab and
manager, the 3D view, the bead calibration, the batch window) have pages too,
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
  *Differences from SMAP*, *References*.  Leave out a section that has
  nothing to say.
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
* **Figures show the method working**, drawn by the plugin's own
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
* **References** that were checked (PubMed or the journal), with a DOI link;
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
| `'grouping', 'grouped'` | the session's grouped set (see above) |
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
its page, and any exhaustive test that now needs a line.

## Keeping this file true

A stale instruction here is worse than none: it is confidently wrong, and a
session will follow it instead of reading the code.  So, while you work:

* **Check what you use.**  When this file tells you something you are about to
  rely on -- a column name, a signature, which test enumerates what -- confirm
  it in the code as you go.  The code wins, always.
* **If it disagrees with the code, stop and ask.**  Say which line is wrong and
  what you found instead, and let the user decide whether the file or the code
  is the thing to change.  Do not quietly work around it, and do not assume the
  file is describing an intention that the code has drifted from -- it may be
  the code that is the mistake.
* **If your own change makes a line here wrong, say so.**  Adding a plugin
  shape, moving the grouped table, renaming a column or changing how the tests
  enumerate plugins all land in this file.  Propose the edit -- quote the old
  line and the new one -- and ask before making it, in the same message as the
  work it belongs to.  Keeping this file current is part of finishing the
  change, not a separate chore, but it is the user's file and their call.
