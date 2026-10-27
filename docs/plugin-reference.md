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
  links the grouped table on request, with its progress in `ctx.report`.
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
  A layer shows it by default, but nothing is linked when a file opens: it
  is linked when first read, and in the GUI a large table (`gui.merging`)
  in a worker after the file opens, a layer set to blinks drawing nothing
  meanwhile.  **A run waits for that merge** if it reads the session's
  localizations -- every plugin but one that makes its own table (a fit, a
  loader, a simulation: `SOURCES`, and a chain whose first step is one;
  `plugins.reads_locs`): the panel holds it and starts it when the blinks
  are there, so a plugin never sees localizations standing in for blinks.
  A chain's copy of the session (`Session.scratch`) then shares the merged
  table; what the copy links itself goes into columns of its own.
  `session.tables.peek_grouped()` is what was linked last (None if nothing
  was), `session.tables.is_current` whether it still matches,
  `session.tables.grouped()` links it if it does not, and
  `session.tables.available()` links it only where that is allowed now.  A layer has one set of
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
neither, and `size` is the hint the window reads instead.  The *All* page is
made when two or more figures are on it; `Plot(..., summary=False)` keeps a
figure off it, for one that is already a page of everything
(`statistics.py`).  Return the settings
that were actually used -- that is what the history records.

`text` goes into the plugin panel's output box, however long it is; *Show result*
opens the figures, or, for a result with none, the text in a window.  Set `text_window = True` on the
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
| `layers` | layer set-ups, bounds and grouping per layer (`Session.set_layer_configs`; `Analysis/Process/Layers`) |
| `rois` | the ROIs the run added to the ROI manager's project (`[]` when it found none; None when it adds none): the manager redraws and a preview is dropped |
| `roi_preview` | a segmenter's proposals, drawn grey and not added (`Session.show_preview`) |
| `roi_project` | the whole ROI project as a chain left it, loaded into the session's |
| `id_regions` | what was measured per region of an id column, `{column: {name: RegionTable}}`, added to `Session.id_regions`; a plugin that measures every region at once makes its table with `region_runs.column_table`, signed as a block's run would be, so it reads as current (`plugins/id_statistics.py`) |
| `region_tables` | a block's tables per region, kept by a block around it as its region's children |
| `region_filter` | the region filter's conditions after the run, as dicts (`region_filter.Condition.to_dict`), set by `Session.set_region_filter`; None leaves them (`plugins/region_filter.py`) |
| `evaluations` | `(name, plugin path)` of the evaluations the run left, so a later step of a chain reads them without asking |
| `placements` | `{layer index: 3x3 affine}`, image layers moved (`ImageData.placement`, nm): *Transform*'s image target, one undo step and a log line like a table change |
| `image_layers` | new images (`ImageData`), each added as a source and an image layer after the others (`Session.add_image`): *Process image*'s result; not an undo step and not logged |
| `transformations` | `{name: Transformation}` (`smappy.transformation`) added to the session's named transformations (`Session.transformations`), saved with the localization file (not the workspace) |
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
`run_label`.  A choice with a `window` opens that window of the GUI's
instead of running (the fitters' *Open camera editor* for a camera in no
database).  The dialog's default is the first alternative that runs when
there is one, otherwise Cancel -- never the run the question is about --
and Escape cancels.

`FormState.titles` retitles a part by its dotted name while it is folded:
the fitters' *camera* says what it resolved to.

`FormState.shown` hides fields and parts by dotted name, where `active`
only greys them: for settings of a choice the plugin does not take at all,
such as the part of each algorithm that is not selected.  A row goes with its
label, a part with its title bar, and *more* goes once none of its rows is
left.  Like greying, it is presentation only: a hidden field keeps its value.
Grey what the current settings happen not to read, and hide what belongs to
another choice.

## The table

`Localizations` is a dataclass of `columns: Dict[str, np.ndarray]` plus
`metadata`.  `locs["photons"]`, `"photons" in locs`, `locs.keys()`, `len(locs)`,
`locs[mask]` (a new table).  The column vocabulary:

| column | meaning |
| --- | --- |
| `x_nm`, `y_nm`, `z_nm` | position; what everything reads |
| `peak_x_pix`, `peak_y_pix` | where the peak finder found the spot, in camera pixels.  A fit writes no other pixel columns: the pixel size is `metadata["pixelsize_nm"]` (`"pixelsize_nm_xy"` when not square; `"pixelsize_assumed"` when it is the default), and `locs.in_pixels(locs, "x_pix", ...)` divides the nm columns by it.  A file saved in pixels keeps its own `x_pix`, ... |
| `frame` | camera frame, int64 |
| `photons`, `background` | per localization |
| `xy_err_nm` | lateral precision, the RMS of `x_err_nm` and `y_err_nm` |
| `z_err_nm` | axial precision, when the fit produced one (or `Analysis/After fit/Astigmatic z`, from the widths) |
| `n_in_group` | **on-time in frames** (for a table with a trace id, `tid`, the localizations in the trace); grouping writes it onto both tables |
| `group_id` | which group a localization was linked into, 1-based; on the grouped table, its own row; 0 for a localization left out by `group_use` |
| `group_use` | made by the user (usually by Math Parser), rarely there: where it is, a localization with 0 takes no part in grouping -- in no group, not in the grouped table, kept in the localizations (`group.USE`); e.g. `trace_index > 20` merges MINFLUX traces without their creeping start |
| `cluster_id`, `n_in_cluster` | which cluster, 1-based, 0 for none, and the localizations in it -- written by `Analysis/Regions/Find clusters` (`plugins/clusters.py`) |
| `region_filter` | 1 for a localization in a region the region filter keeps, 0 for one it drops -- written by `Session.refresh_region_filter` from the conditions in the table's metadata; never read by a region's measurement (`region_runs.unmeasured`) |
| `tid`, `trace_index` | MINFLUX: the trace a localization belongs to, and its number in it, 1, 2, 3, ... by time (`io.minflux.trace_index`) -- a filter `trace_index > 20` leaves out the start of each trace, where a still emitter creeps into place (NOTES.md); on merged traces the smallest |
| `track_id`, `track_length` | which track, 1-based, 0 for none (and for a track below the minimum length), and the localizations in it -- written by `Analysis/Tracking/Link tracks` (`plugins/link_tracks.py`, or into its *output column*); a simulation's truth is `track_id_true`, MINFLUX's traces `tid`.  Nothing that reads tracks names one: each has a *track id*, on *auto* the first of `track_id`, `tid`, `track_id_true` the table has (`tracking.columns.track_column`).  Whole numbers when blinks merge -- a blink's largest id and length (`group.id_columns`), whether linked here or read from a file |
| `track_D_um2_per_s`, `track_alpha` | a track's diffusion coefficient and anomalous exponent from its MSD, NaN for none -- written by `Analysis/Tracking/Tracking analysis` with *write per track* |
| `phase_rad`, `z_env_nm`, `fringe_probability`, `cavity_phase_rad` | 4Pi: the interference phase, wrapped to (-pi, pi], and the envelope's z (`psf.Global4PiPSF`, which also writes them as `z_nm`); `Analysis/After fit/4Pi z` replaces `z_nm` with z from the phase and writes the probability that its fringe is right and the cavity's phase in the frame.  `phase_rad` must not be averaged linearly (grouping does, for now) |
| `sigma_nm`, `logl_rel` | PSF width and fit quality; the elliptical Gaussian fit writes `sigma_x_nm`, `sigma_y_nm` and their errors `sigma_x_err_nm`, `sigma_y_err_nm` instead of `sigma_nm` |
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

**Several files.**  The table's metadata is the first file's once several
are open.  Each file's own -- its camera, its acquisition, a MINFLUX
measurement's sequence and beads (`metadata["minflux"]`, `io.msr`) -- is
`metadata.file_metadata(locs.metadata, filenumber)`: the file's entry in
`metadata["files"]` (`FileInfo.metadata`, without what belongs to the
session: `io.formats.SESSION_KEYS`), saved with the session and put back by
the undo.  A plugin that reads a file's acquisition reads it there, not from
the table's metadata.

A column one file lacks is padding in its rows: NaN, or 0 in an integer
column (for an id, "in none").  `FileInfo.columns` is what each file had
when it was read, and `Session.files_without(column)` the files read
without one -- for a plugin that would rather say "file 2 has no `z_nm`"
than compute on the padding.

**Frames.**  Files opened together keep their own frame numbers, so frame
120 of one file and frame 121 of another look consecutive.  Anything that
works along time works per file: linking and *Link tracks* block by
`filenumber`, the drift plugins estimate and correct each file on its own
(`drift_result.per_file`), and what pairs localizations across frames
(NeNA, the channel registration) reads `locs.frames_apart(locs)`, frames
moved apart per file.  *Combine files* makes several files one acquisition,
frames continued.

**Blinks of several files.**  A table of several files is linked one file at
a time, and only the files the layers showing blinks show
(`layer_state.Tables`).  The grouped table holds the files linked so far;
`group_id` on the localizations of a file not linked yet is 0, and the ids
of the others run on from file to file -- look a blink up by its id, never
by its row.  `Session.table(layer, "grouped")` links the files that layer
shows; `session.tables.grouped(files=[...])` the ones named.

**Time.**  A frame is turned into seconds by the file's frame time:
`metadata.file_frame_time(locs.metadata, filenumber)` -- the session's
record of that file (`metadata["files"]`), else the camera's
`metadata["camera"]["frame_time_ms"]`, else its `exposure_ms` -- or
`metadata.table_frame_time(locs)` for the files a table's rows come from
(None when they differ).  A plugin that speaks of seconds reads it there and
offers a *frame time* setting on *auto* to override it (Time Course,
Blinking); MINFLUX's `time_s` is a clock of its own.  When neither says,
its `describe` puts `frame_time_notice()` under the form -- a button to
*File > File properties*, where a file's frame time is set
(`Session.set_frame_time`) -- and its run raises `NO_FRAME_TIME`.  Every
simulator records one.

**The microscope.**  A fitted file's `metadata["camera"]["microscope"]` is
the microscope the database recognised (`microscope_db`), or None: its
`name`, each device's state by name (`states`, e.g. `{"3DA": "on"}`), the
devices the optics follow (`followed`), its `values` (`NA`, `n_immersion`,
`wavelength_nm`, `pixel_nm`, `modality`) and its arms with their halves of
the chip (`channels`).  A plugin that needs the optics reads them there.

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

`smappy.simulate` is one model with three outputs, and the `File/Simulate/
Simulate` plugin is its form: a scene of components (`simulate.scene`), each
structure -> labelling -> blinking (-> diffusion), then localizations, camera
frames or a split camera.  NOTES.md, "The simulation model", has the
assumptions; docs/simulation-redesign.md the scene's design.  What a session needs:

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
* A scene (`SceneSettings`) is a list of `ComponentSettings` -- a structure,
  `homogeneous` labels or `beads` -- each with its own labelling,
  photophysics, diffusion, colour (`channel`) and split ratio; one can be
  placed where another's copies are, or blink as another does.  A
  single-structure `SimulationSettings` is a scene of one (`as_scene`), with
  the same numbers.  Presets are YAML in `data/simulation_presets`.
* The columns: `channel` (the component's colour); the truth `emitter`
  (which fluorophore), `component`, `copy` (which copy of the structure),
  and with diffusion `track_id_true` and `state_true`; camera truth adds
  `neighbour_nm`, the nearest other spot in the same frame.  `metadata["copies"]` has each copy's position and
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
