# The plugin reference

What a plugin is handed, what it hands back, and the table it works on: the
reference half of what used to be in `CLAUDE.md`, which now keeps only the
practical half.  Like that file it is a summary written at one moment -- the
docstrings of `smappy/plugins/__init__.py`, `smappy/locs.py`, `smappy/render.py`
and `smappy/simulate` are the record, and the code wins where they differ.
`docs/plugin-architecture.md` has why the plugin system is shaped as it is.

## The context

* `ctx.locs` -- the whole table **the layer shows**: one row per blink on a
  grouped layer, one per localization on an ungrouped one -- unless
  `grouping` says which, lowest priority first: the layer, the chain, the
  chain step, the plugin (`Plugin.grouping`).  During a live fit, a snapshot
  taken when the run started.  In a region (an ROI, a cluster) it is the
  region's rows of that table; where no grouped table can be cut to the
  region, the ungrouped one stands in.
* `Plugin.grouping = "ungrouped"` -- for a plugin that needs the
  localizations whatever the layer shows: one that hands back a new table
  (drift, Remove Localizations, Math Parser, Assign colours, ...) or measures
  something of single frames (Localization Precision: NeNA pairs consecutive
  frames).  A new table made from the grouped one is refused
  (`refuse_grouped_table`): there is no way back from one row per blink to
  the localizations.  `"grouped"` is the other requirement; None follows the
  layer.
* `ctx.ungrouped()` -- `(locs, selection)` of the localizations, whatever
  `ctx.locs` is: for a plugin that measures blinks and writes onto the
  localizations (DBSCAN maps each blink's cluster back through `group_id`).
* `ctx.table(layer)` -- `(locs, selection)` of any layer, by the same rule:
  `ctx.locs` and `ctx.selection` for the context's own.  `Session.table`
  links the grouped table on request.
* `ctx.selection` -- over `ctx.locs`, what the user is looking at: the layer's filter *and* the
  ROI *and*, while the 3D window is open with "plugins use the slab" ticked,
  the slab (`session.selects_slab`), as a boolean mask.  `.apply(locs)` cuts the table,
  `.require(n, ctx.report, "what")` refuses an empty one and warns about a thin
  one.  A plugin that offers "all localizations" means `ctx.locs` unfiltered.
* `ctx.session` -- may be `None`; a script can say `Context(locs=locs)`, so
  never assume a session exists.
* `ctx.report(text)` for progress, `ctx.emit(event, payload)` to hand partial
  results on.  Both are no-ops when nobody is listening.
* `ctx.warn(text)` for what must not be missed: kept in `ctx.warnings`, shown
  as its own line that progress does not overwrite.  Put the warnings in the
  result's text if they should reach the history.
* `ctx.region_id` / `ctx.per_region` -- set when the run is one region of
  many (an ROI or a cluster); `ctx.region` is the shape, None for a cluster.
* `ctx.stop` and `ctx.writer_finished` -- `threading.Event`s or None, set from
  another thread: end the run early keeping what it has, and "the live source
  is complete".  The panel's Run is Stop while a run goes, and sets `stop`.
  A long plugin checks `ctx.stopped` between steps of its own and returns
  what it has with `Result(stopped=True)`, or raises `Stopped` when a part
  is worth nothing (RCC); the panel then says "stopped".  The fitters, COMET,
  RCC, blocks over regions and chains (between steps) honour it.
* The **grouped** table (one row per blink) is the session's, one for every
  layer: `session.tables` holds the ungrouped table and its grouped form, a
  cache keyed by the rows and `session.group_settings`.
  `session.tables.peek_grouped()` is what was linked last (None if nothing
  was), `session.tables.is_current` whether it still matches, and
  `session.tables.grouped()` links it if it does not.  A layer has one set of
  bounds (`layer.ranges`), applied to the table it shows --
  `layer.state.filter_for(locset)` is its filter over either -- so on the
  grouped table a bound on `xy_err_nm` is a bound on the combined precision.
  Linking costs seconds to minutes -- say what is missing instead of doing it
  behind the user's back.  Once it has been
  built once, `group_id` and `n_in_group` are on the *ungrouped* table too
  (`group.attach`), so a column computed afterwards can be reduced per blink
  with a `bincount` and no relinking.

## The result

`Result(locs=..., text=..., plot=..., plots={...}, data={...}, settings=...,
title=...)`, every part optional (`title` says what the figures are of, in
the figure window's title).  `plot(ax)` is handed **one** axis; for a grid, pass
`Plot(draw, panels=n, size=(w, h))` instead, and `draw` is handed the figure
and lays out its own panels (`statistics.py`, `Analysis/Regions/Summarize/Histograms`
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

`data` is the plugin's own: numbers and payload for a script or a test,
which nothing reads to decide what to do.  What a result asks of the session
beyond its table is a typed field each, read by `Session.apply` (and by a
chain, a block and the batch); a misspelt keyword raises a `TypeError` where
the result is made, where a misspelt key of `data` used to do nothing:

| field | what the session does with it |
| --- | --- |
| `files` | opens them, as `(locs, info, grouped)`: a loader runs in a worker and must not touch the session |
| `append` | the `files` join the session's rather than replacing them |
| `written` | the file the run wrote (a saved table, an export, a recipe, a fit's output): a table made from nothing is named after it, a chain passes it on, the batch records it |
| `saved` | the table handed back is the one in `written`, so the session is not unsaved |
| `images` | camera frames a fit kept (`smappy.rawframes`); they replace the session's raw frames, since a fit replaces the table |
| `bounds` | `{field: (lo, hi)}`, a filter bound on every layer for a column the run has just written -- a plugin cannot set it itself, because a filter belongs to the table it was built from and the new table exists only once `Session.apply` has set it |
| `layers` | layer set-ups, bounds and grouping per layer (`Session.set_layer_configs`; `Chain/Layers`) |
| `rois` | the ROIs the run added to the ROI manager's project (`[]` when it found none; None when it adds none): the manager redraws and a preview is dropped |
| `roi_preview` | a segmenter's proposals, drawn grey and not added (`Session.show_preview`) |
| `roi_project` | the whole ROI project as a chain left it, loaded into the session's |
| `id_regions` | what was measured per region of an id column, added to `Session.id_regions` |
| `region_tables` | a block's tables per region, kept by a block around it as its region's children |
| `evaluations` | `(name, plugin path)` of the evaluations the run left, so a later step of a chain reads them without asking |
| `recipe` | the table change as a `ColumnRecipe`, which follows a table a fit is still appending to |
| `stopped` | the run was stopped and this is what it had: the panel says "stopped" |

The figures of one result share a window, a tab each beyond the first, and a
tab is drawn when it is looked at and not before -- so a plugin with six
figures costs what one costs, and a plot must be a closure over its data
rather than something already drawn.

## What the form says before a run

`Plugin.describe(ctx, settings)` returns a `FormState`: `hints` (what fields
on *auto* resolve to), `active` (the fields the settings read; the rest are
greyed), `choices` (lists the session decides) and a `notice`.  A
`Notice(text, action, window, settings, kind)` is a line under the form and,
when the GUI has the window named in `window` (`session.window_openers`), a
button `action` that opens it.  `kind` is `"warning"` (the default, orange and
bold) for what goes wrong or is guessed if left as it is -- a camera the
database does not know -- and `"info"` (grey) for what is only worth
knowing: what a save leaves out, a simulation that can be fitted now.

`Plugin.preflight(ctx, settings)` is asked at Run, before the worker starts:
a string is a yes-or-no, a `PreflightQuestion` offers alternatives
(`PreflightChoice`, each with the settings it runs with) beside its
`run_label`.  The dialog's default is the first alternative when there is
one, otherwise Cancel -- never the run the question is about -- and Escape
cancels.

## The table

`Localizations` is a dataclass of `columns: Dict[str, np.ndarray]` plus
`metadata`.  `locs["photons"]`, `"photons" in locs`, `locs.keys()`, `len(locs)`,
`locs[mask]` (a new table).  The column vocabulary:

| column | meaning |
| --- | --- |
| `x_nm`, `y_nm`, `z_nm` | position; what everything reads |
| `x_pix`, `y_pix`, ... | a fit's positions in camera pixels, kept beside the nm ones (`metadata["pixelsize_assumed"]` when the nm ones are on the default scale) |
| `frame` | camera frame, int64 |
| `photons`, `background` | per localization |
| `xy_err_nm` | lateral precision, the RMS of `x_err_nm` and `y_err_nm` |
| `z_err_nm` | axial precision, when the fit produced one |
| `n_in_group` | **on-time in frames** (for a table with a trace id, `tid`, the localizations in the trace); grouping writes it onto both tables |
| `group_id` | which group a localization was linked into, 1-based; on the grouped table, its own row |
| `cluster_id`, `n_in_cluster` | which cluster, 1-based, 0 for none, and the localizations in it -- written by `Analysis/Regions/Find/DBSCAN` (`plugins/clusters.py`) |
| `sigma_nm`, `sigma_y_nm`, `logl_rel` | PSF width and fit quality |
| `filenumber`, `channel` | which file, which channel |

Every precision is `<quantity>_err_<unit>` -- `x_err_nm`, `z_err_nm`,
`photons_err` (no unit, nothing after `_err`).

Read positions and precisions in nm (`render.positions`, `xy_err_nm`): a fit
writes them and a loader gives a pixel-only file nm columns
(`io.formats.with_nm`).  Raise a message naming the columns the table *does*
have when something is missing.  The panel shows a raised error's message
alone -- `str(error)`, without its class -- in red beside Run, with the
traceback behind *details…* (`gui/errors.py`), so write the message for the
reader: `ValueError("no ROI: draw a line")`.

A column that needs a **rule for grouping** is a recipe rather than data:
`mathparse` keeps it in `metadata["derived"]`, and the recipe says what the
field means once the localizations are grouped -- recomputed from its
expression there, or reduced by a rule (`mean`, `sum`, `any`, ...).  It is
`group.combine` that honours it, so a derived column is never averaged by
accident, and it survives a save.  Usually the recipe is an expression
(`math_parser.py`); a column that was *measured* per localization and only
needs the rule -- the `use` flag `remove_locs.py` writes -- is a recipe with
a rule and no expression.

Columns written from what was measured per region (`n_corners` on every
localization of its pore) are listed in `metadata["region_columns"]`, and
only those may be written over from regions again (`region_runs.check_writable`).

A fitted table also carries `metadata["frame_tags"]` (the image tags that
changed, per frame: `smappy.frametags`) and `metadata["acquisition"]` (the
Micro-Manager summary, devices and comment); both are written beside `/locs`,
not in the metadata attribute.

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

NOTES.md under "Any column against any other" records why it is shaped this
way and what was left out of SMAP's `VersatileRenderer`.

## Simulated data

`smappy.simulate` is one model with two outputs, and the `File/Simulate/
Blinking Structure` plugin (shown as *Simulate structure*) is its form:
structure -> labelling -> blinking -> localizations or camera frames.  NOTES.md, "The simulation model", has the
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
