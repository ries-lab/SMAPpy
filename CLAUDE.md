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

`src/uipsf/` is a second package beside `smappy`: uiPSF v2, the PSF learning
in PyTorch, with its own rules in `src/uipsf/CLAUDE.md` (read it before
working there), its docs in `docs/uipsf/` and its tests in
`tests/uipsf_tests/`.  Where smappy uses it, it imports it inside the
functions that need a model, so that nothing else loads torch.

`src/smappy/assistant/` is the AI assistant's side: SMAPpy's tools served over
MCP (`smappy-mcp`), plain functions over a `Session` in `assistant/tools.py`,
which nothing else imports but the GUI's socket server and pane
(`gui/assistant_server.py`; `gui/assistant_pane.py`, the render toolbar's
button, the proposals and the settings; *Assistant > Start assistant*, which
starts the user's client through `assistant/launch.py`, reached by
`smappy-mcp --attach`), and they only when asked;
the design and the plan are `docs/ai-integration.md`.
A tool's description is what a model reads, so a change to a plugin's
interface that a tool describes (`run_plugin`'s settings, `INSTRUCTIONS`)
updates it too.

## Before you edit anything

Pull first:

    git fetch origin && git rebase origin/main      # or: git pull --rebase

Work lands on `origin/main` from cloud sessions and other machines, so the
local `main` is routinely well behind it -- 28 commits, on one occasion.
Editing first means rebasing finished work through conflicts in files upstream
has already changed -- `NOTES.md` most often.  If you have already started,
`git stash push -u` first so new untracked files come along, and rebuild
afterwards if `csrc/` moved.

## Remind the user to release

Users get SMAPpy from PyPI, and the GUI's update check (`smappy/updates.py`)
offers only a release there -- a change on `main` reaches nobody who
installed with pip until it is released.  So, **at the end of a session
that finished major work, an important fix, or the last of many small
fixes since the last release, remind the user to make a release** --
in one line of the summary, saying what has gathered since the last tag
(`git log --oneline $(git describe --tags --abbrev=0)..origin/main`).
Do not release yourself: a tag publishes to PyPI, which cannot be taken back.

A release is: `version` in `pyproject.toml` raised (and pushed to `main`),
then an annotated tag of that commit, `git tag -a vX.Y.Z -m "smappy-smlm
X.Y.Z"` and `git push origin vX.Y.Z`; `.github/workflows/wheels.yml` builds
the wheels on every platform, tests them and publishes them to PyPI.

## Getting the checkout to run

A fresh container has **no numpy**.  Before anything else:

    pip install -e .                    # builds the C++ extensions too
    pip install -e .[test]              # pytest, pytest-timeout (a 300 s limit per test), pytest-xdist

Running the tests -- three tiers:

    python -m pytest tests -q -n auto           # after each change: ~1 min, the slow tier skipped
    python -m pytest tests -q -n auto --slow    # before a PR or a push of major work: ~4.5 min on four cores
    python -m pytest tests -q -n auto --long    # by hand, when uipsf's learning changes: --slow and the studies

* **Run the slow tier before you open a PR, and before you call major work
  done** -- a new plugin, a change to the GUI, a tutorial, a page or what an
  algorithm computes.  It is about 130 tests marked `@pytest.mark.slow` (the
  tutorials, the pages' figures, the large numerical checks) that are two
  thirds of the suite's time, and they are where a GUI or docs change breaks.
  The summary's "skipped" count includes them, so a run without `--slow` has
  not checked them; say so if you stop short of it.  CI runs only uipsf's
  fast tier (`tests/uipsf_tests`, `.github/workflows/uipsf.yml`) on Linux,
  macOS and Windows, on a push that touches uipsf or its tests -- not on a
  change to `src/smappy/` alone, though those tests use `smappy.calibrate`.
  The rest of the suite is run by hand.
* **A PR description states the `--slow` run's result** on a line of its own
  -- "`--slow -n auto`: 1321 passed, 4 skipped" -- or says plainly that it was
  not run and why, so a reviewer can see from the PR whether it was done.
* Mark a new test `slow` when it takes more than about 4 s, and `long` when
  it takes more than about 20 s: `--long` is uipsf's learning studies (a
  benchmark against the CRLB, a pupil learned in the window, one region size
  against another), which had taken `--slow` from 3.5 to 10.5 minutes.  It
  includes `--slow`; its tests have a 1200 s limit (`LONG_TIMEOUT` in
  `conftest.py`) where every other has 300 s.  A `slow` test that grows past
  20 s is shortened or moved to `long`, not left to stretch `--slow`.
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
  `gui.make_app()`, which installs the collector, the icon, the fields'
  style and the look (`gui/theme.py`: Fusion, light, dark or as the system,
  from Preferences > General), and one that makes a session installs its
  identity too (`gui/identity.install_identity`: its name and colour in
  every window title).
* **A colour is read from the palette, never fixed for a light window**:
  the windows can be dark.  A secondary grey is `widgets.note_style()`,
  which a switch of theme finds and rewrites (`theme._rebake`); a warning
  or error colour is a mid tone that reads on both, `#c06000` or `#e0453a`.
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
* uipsf's tests (`tests/uipsf_tests/`) need torch, the extra `uipsf`:
  `pip install -e .[test,uipsf]`.  Without torch they are not collected at
  all (`collect_ignore` in `tests/conftest.py`), so the count drops by about
  300 and nothing says so; `--gpu` adds their GPU tier, run by hand on a Mac
  and a CUDA machine.

## Writing a plugin

The framework is the same whatever the plugin does; the shipped ones differ in
shape, so read the one closest to what you are writing:

| shape | example |
| --- | --- |
| measures the selection and draws it | `statistics.py` (settings, selection, several figures in one window) |
| rewrites the table and hands it back | `assign_colors.py` (`Result.locs`, a preview; MINFLUX's `dcr` read as the two channels' photons, `DCR_PAIR`) |
| a long computation with progress | `drift_comet.py`, `drift_rcc.py` |
| measures drift from beads, a `Drift` the others' way | `drift_fiducials.py` (*Fiducials*: tracks of the bright localizations from *Link tracks*' linker, one linear fit of each bead's offset and COMET's spline in `smappy/fiducials.py`, on frames, of `smappy/beads.py`'s `fit_offsets_and_spline`, which is also in seconds; `KeepsDrift` keeps the curve; beads simulated by `smappy.simulate.simulate_beads`.  Which beads to trust is `beads.select_beads` (here and in *MBM*; a bead named by where it is, `fiducials.bead_names`, so a choice by hand survives the linker numbering beads otherwise; the bead selector through `bead_view` / `with_overrides`): each bead's noise, wander and jump against the other used beads, a caller's rule (`MadRule`), choices by hand by name never overruled; bead tracks in time with bad ones, `simulate.simulate_bead_tracks`; each bead's localizations named, `bead_id`, and a region table of the beads, `beads.bead_values`) |
| loads, saves, exports | `file.py` (one loader with an advanced *format* choice, no input table -- a format is a `Reader` / `Writer` in `io/formats.py`, every column of the file kept; *Load*, one row of launchers of File > Open -- *Open* and *Add*, `Plugin.window_buttons` -- while the menu's Open asks replace or add; *Save localizations* (`File/Export/Localizations`), whose column list is `kind="columns"`; *Save image* (`File/Export/Image`), the one renderer of the view -- `smappy.picture`, framed by the windows' `Session.view_frames`, which the render window's ⋯ and the 3D window's *Save* call too; *Save stack*, movies and z-stacks on the same renderer, `smappy.movie` -- and the Render tab's *player*, the same window moved live through any coordinate (`smappy.scan`, `gui/scan_player.py`), narrowing the windows' snapshots through `Session.scan`, never the filters, of the layers `Scan.layers` names; its *view path* moves the 3D view and its box along key views, smoothly through them, `smappy.view_path`, interpolated about the screen's centre and set through `Session.set_projection` and `set_slab`; both under the player's *more*) |
| simulates a scene, from presets | `simulate.py` (*Simulate*, `File/Simulate/Simulate`: `smappy.simulate.scene`, a `List[ComponentSettings]` shown as a list with a form per item; *same as* links between components; a `kind="preset"` field filling every setting through `react`, YAML in `data/simulation_presets`; it records a frame time, set by hand for a file in *File > File properties*, `Session.set_frame_time`; and the preset's key parameters at the top, `FormState.keys`; saved components in `data/simulation_components`; a split camera writes its halves' map beside the recipe, `simulate.source.write_transform`, which a two-colour fitter takes in `react` and *Fit this* hands to Gaussian 2D 2C -- or, with a dual-colour *PSF calibration*, that calibration's halves, registration and two PSFs (`simulate.camera.split_layout`) for Spline 3D 2C, the fitter's `react` filling its calibration from the recipe (`simulate.source.recipe_calibration`); the design in `docs/simulation-redesign.md`) |
| draws every ROI as a picture, a PDF gallery or a TIFF stack | `roi_images.py` (*Save ROI images*, `File/Export/ROI images`: a `SiteAnalysisPlugin` for the titles' values; the visible layers cut to each ROI's square and file and turned to a named line before rendering, `roi_layers`; pages through matplotlib's `PdfPages`, the stack as ImageJ's with slice labels) |
| corrects MINFLUX again from its MBM beads, with a bead window | `drift_mbm.py` (*MBM drift correction*: the beads from each file's metadata, `mbm.measurements_of`; judged by `beads.select_beads` per measurement; `x = x_nc - d(t)`, replacing Imspector's correction, on a time-axis `Drift`; a bead file beside it corrected too.  Its notice's *Select beads...* opens `gui/bead_window.py`, one window for any plugin that implements `bead_view` / `with_overrides`: an opener with `replies = True` gets the plugin and a `reply` into the panel's form (`PluginPanel.take_settings`), so a choice by hand is a setting (*by hand*); with the bead file open, its beads named `bead_id` and a region table of them) |
| reads a container of several measurements and images | `io/msr.py` (Abberior `.msr`, read by *File/Load/Auto*'s *Abberior MSR* part: the OBF container `io/obf.py`, the MINFLUX archive and zarr `io/zarr_lite.py`, blosc/LZ4 `io/blosc.py`, both record layouts `io/minflux.py`; File > Open asks which parts, `gui/msr_import.py`, through a reader's `needs=("measurements",)`; a file per measurement in `Result.files`, images in `Result.image_layers`, each file's own metadata in `FileInfo.metadata`; the MBM beads' tracks as a file of their own beside a measurement, `msr.bead_table`, paired with it by `msr.mbm_partner`) |
| finds, measures or summarises ROIs | `roi.py` (a segmenter derives from `SegmenterPlugin`, which gives it a Preview, *replace* (`replace_param()`) and a step of the ROI undo; `per_region = True`; analyse) |
| finds ROIs by one of several methods, with presets | `segment.py` (*Segment regions*: one `SegmenterPlugin`, a *method* part each -- circle template (a ring or a disc over a range of radii, `segmentation.find_circles`), beads (`segmentation.measure_beads` on `fiducials.find_beads`), from an id column (`region_column.column_regions`), positions from a file (`segmentation.read_positions`); every candidate measured -- shared values and its model's own, `segmentation.VALUES` -- and judged by one *keep* condition in Filter ROIs' syntax (`segmentation.judge` on `roi_manager.order.condition`); the values kept in `origin["values"]` and shown as results columns (`core.segment_columns`); a *preset* is a YAML file in `data/segment_presets` or the config directory that fills the settings, *keep* included, through `react`; each ROI's `origin["segment"]` names its method, so *replace* removes that method's only; the array-level helpers, for a user's own `SegmenterPlugin`, in `smappy/segmentation.py`) |
| parts, a C++ backend | `fit.py` (nested settings dataclasses; four fitters, thin subclasses of `_FitPlugin`; the *fit region* drawn on the Preview's frame: a figure panel that takes shapes is `Plot.edit`, a `ShapeEdit`, and the figure window adds the tools; the spline fitters' calibration on *auto*, `smappy/calibration_list.py`, its matches offered in a path field's menu through `FormState.choices`; Spline 3D's *focus*, the calibration's pupil model drawn for the focus in the sample, z the height above the coverslip, the focus on *auto* from the lowest localizations' edge before the fit or *in situ* with every aberration held (`insitu_run`'s *hold*), found once per file and settings and drawn by Preview, `calibrate/depth.py`) |
| fits several channels of one frame as one emitter, 4Pi | `fit_4pi.py` (*Spline 4Pi*: the channel layer -- `channels.ChannelSource` after the loader, through the `_FitPlugin.open_source` and `view_camera` hooks, so everything after it sees (n, C, h, w) and positions are in channel 0's own pixels; the geometry (chip rectangles, mirrors) from the microscope database, `Microscope.channel_layout`, or typed; the registration (maps between channel images) from the model, beads or a file -- `channels.ChannelLayout`; the molecules found on the channels' sum, each fitted in all of them by `multichannel.MultiChannelEngine` with `psf.Global4PiPSF`; 4Pi data read by `io/dcimg.py` and `io/quadrants.py` (quadrant `.mat` series, side by side) through `open_stack`; the maps come with the model, `FourPiCalibration.maps`, or from beads summed over the stack, `channels.register_channels`; the channels' drift from fits with x and y free per channel, `multichannel.measure_channel_drift`; *after the fit*'s *4Pi z*, `Plugin.phase`; the phase grouped as an angle, `group.COMBINE_MODES` "circular") |
| turns a 4Pi fit's interference phase into z | `fourpi_z.py` (*4Pi z*, `Analysis/After fit/`: the cavity's phase followed in time windows from the localizations, each phase put on the fringe its envelope z makes most probable with that probability a column, `fringe_probability`, and the period refined on request -- array functions in `smappy/fourpi.py`, which also holds the IAB model, `FourPiCalibration`, its simulation, readers of uiPSF v1's and SMAP's 4Pi models and a model built from a bead scan without phase steps, `iab_from_beads`; the fit is `psf.Global4PiPSF` on `csrc`'s `CSpline4Pi`; plan and findings in `docs/4pi.md`) |
| turns a fit's widths into z, from a calibration | `astigmatic_z.py` (*Astigmatic z*, `Analysis/After fit/`, beside *Assign colours*: z and `z_err_nm` from the elliptical Gaussian's `sigma_x_nm` / `sigma_y_nm` and their errors, least squares on both started by the `sx^2 - sy^2` lookup, in `smappy/astigmatic_z.py`; the calibration a file from the PSF calibration window's *elliptical Gaussian* method, `calibrate/elliptical.py`, or typed curves; *Fit this* on astigmatic frames opens Gaussian 2D with *elliptical* ticked and its *after the fit* set to it) |
| runs fixed steps over every finished fit, and from a button on the table | `after_fit.py` (*After the fit*: astigmatic z, drift none / RCC / COMET, colours, then a saved chain, in that order -- each the shipped plugin with its settings, a part of every fitter (`fit._Finishing`, `finish`) run before the file is saved; a step shown only where the fitter's table can take it and its settings only while it is on, `after_fit.shown` through `FormState.shown`; one the table cannot take passed over, `applicable`; the fitter's button inside the part, *Run these now*, is `Plugin.part_runs`, the plugin at a path run on the session with that part's settings and applied under its own name) |
| adds a derived column, no output | `math_parser.py` (an expression, kept with the table as a recipe; *on* a region source, an expression over what was measured per region -- evaluations by alias, `npc_corners.n_corners` -- kept as a recipe of the source's `maths` region table and computed again when its inputs are measured again, `smappy/region_maths.py`, optionally onto the localizations) |
| removes or flags localizations | `remove_locs.py` (a region, a new table or a filtered flag) |
| makes the open files one acquisition | `combine_files.py` (*Combine files*: frames and time continued, ids kept apart; the table's `metadata["files"]` is the new file list, which `Session._follow_files` takes when a table change changes how many files there are.  Anything else that works along time works per file -- files share frame numbers: `drift_result.per_file`, `locs.frames_apart`) |
| makes an ROI that is not a shape | `segment.py`'s *from an id column*, *its id alone* (an ROI of one track: `ROI.ids` with `spatial` off, drawn as its line; any ROI's `frames` / `z` range set in the ROI manager's *annotate*, `gui/roi_ranges.py`; membership in `ROIProject.members`, never `rows_in` alone; one track picked by a click in the ROI manager with the Regions tab's *shape* set to *track*, `ROIProject.track_at`) |
| fits a model to what is in a ROI | `line_profile.py` (unbinned likelihood, models compared by AIC; one fit per layer, `follow_roi` and `per_region`: in an ROI the line is one of its named lines, `region_line`); `circle_fit.py` (the NPC segmenter's `fit_circle` with errors) |
| moves the ROIs onto what was measured in them | `roi_centre.py` (*Centre ROIs*: `ROIProject.move_roi` in one ROI undo step) |
| counts in time: blinks, dark times, bursts | `blinking.py` (one region on one time line, the view linked into molecules per file; seconds from the file's frame time, `metadata.file_frame_time`) |
| reads the session and reports | `history.py` (the log, with an optional export) |
| sets up layers and filters from settings | `chain_layers.py` (bounds as rows per layer, `Result.layers`) |
| scores a fit against a simulation | `ground_truth.py` (the truth redrawn from the recipe, matched per frame) |
| draws what the microscope recorded per frame | `image_tags.py` (the fit's `metadata["frame_tags"]`) |
| finds a structure by its shape and judges it, then counts it | `npc.py` (the pores found by *Segment regions*' circle template with the *NPC* preset, a per-site count, and a model of whole pores in `smappy/npc.py` fitted to the site table) |
| summarises what evaluators measured | `roi.py`'s `SiteAnalysisPlugin` (an evaluator to read, a *results from* field, a preflight that asks which when there are several; under `Analysis/Regions/Summarize`) |
| summarises curves over the regions | `curves.py` (*Curves*: a `SiteAnalysisPlugin` reading each region's `Result.arrays` from its stored entry; groups by an ROI keyword list; pools through the evaluator's `Plugin.pool` where it hands back counts) |
| runs the ROI manager's segment, measure and analyse as one | `npc_analysis.chain.yaml` (a shipped chain: a block runs NPC Corners in every ROI before the next step; `docs/batch.md`, "ROIs: segment, measure, analyse") |
| links localizations into tracks | `link_tracks.py` (*Link tracks*: an id column, `track_id` / `track_length` or an *output column*, of the localizations whatever the layer shows; everything that reads tracks has a *track id*, *auto* by default, resolved by `tracking.columns.track_column` -- never a hardcoded `track_id`; two linkers, an *algorithm* choice, in `smappy/tracking/link.py`; candidates from `csrc/neighbours.cpp`'s `Index.pairs`, a simple linker's loop in C++ (`greedy_links`); tracks simulated by `smappy.simulate.simulate_tracks` -- in the GUI *Simulate*'s *Diffusion* presets, a homogeneous component that diffuses, `track_id_true` / `state_true` -- scored by `tracking.score`) |
| measures how tracks move, per track and pooled | `tracking_analysis.py` (*Tracking analysis*: approaches ticked as in *Spatial statistics*, a part each shown while ticked; an *id column*; vectorised over the whole table in `smappy/tracking/analysis.py`; `per_region`, and per-track columns only on a tick, `changes_locs_for`; over the view the per-track fit also goes back as a region table of the track id, `region_runs.column_table`, where the run saw the layer's own filter, `region_runs.layer_mask`) |
| diagnoses the instrument from beads, approaches to come | `stability.py` (*Stability*: *vibrations* ticked, its part shown while on; beads by an *id column*, each, their mean or differential; a regular grid with gaps filled and counted, Welch's spectrum, the cumulative RMS and peaks in `smappy/stability.py`; each figure a tick; `per_region`, values pooled over the region's traces; with *each bead* over the view a region table per bead, `per_bead`; *random scatter*, the Allan deviation of each trace's measured points against the precision, `stability.scatter`) |
| writes a number per localization from what is around it | `neighbourhood.py` (two layers as they show, *around* and *neighbours from*; a recipe column, NaN outside the filter, a blink's value through `group_id`; the search in C++, `csrc/neighbours.cpp`) |
| measures curves over pair distances, with envelopes | `spatial_statistics.py` (*Spatial statistics*: approaches ticked on or off, each one's part shown only while on; pair distances from `Index.pair_histogram` in `csrc/neighbours.cpp`, normalised by the window's set covariance; Monte Carlo envelopes; per region, curves as `Result.arrays` with their numerator and denominator; the directional approaches -- a correlation image, a correlation along a line in a line ROI's own axes -- in `smappy/correlation.py`) |
| finds clusters and names them in the table | `clusters.py` (*Find clusters*: an *algorithm* choice, a part per algorithm, only the chosen one shown -- DBSCAN, Voronoi, HDBSCAN with an optional dependency -- all writing `cluster_id` / `n_in_cluster`; DBSCAN on the C++ cell search of `csrc/neighbours.cpp`, `Index.dbscan_join`; ROIs of them are *Segment regions*' method *from an id column*, for any id column) |
| measures every region of an id column at once, as a region table | `id_statistics.py` (*Id statistics*: per trace (`tid`), track or cluster in one pass -- n, centre, spread, duration, statistics of chosen columns -- in `smappy/id_statistics.py`, regions by a lookup table on the ids, no loop; the table handed back whole in `Result.id_regions`, made by `region_runs.column_table`, which signs it as a block's run so `region_results` finds it current; `RegionTable.from_columns`, `RegionTable.groups` for the statistics of one quantity; the plan it starts, `docs/region-filter.md`) |
| keeps the localizations of the regions whose results pass | `region_filter.py` (*Filter regions*: conditions on region-table values, per region, OR within a source, AND across, out where no region, in `smappy/region_filter.py`; kept in the table's metadata, answered by one column `region_filter` with a bound on every layer (`KEPT_BOUNDS`), recomputed by `Session.refresh_region_filter`; handed back as `Result.region_filter`; summaries honour it through `region_results.rows`; a region's measurement never sees it, `region_runs.unmeasured`; by hand the Regions tab's section, `gui/region_filter_panel.py`, and the Render tab's *regions* button) |
| orders or filters the ROIs by what was measured in them | `roi_order.py` (*Sort ROIs*: `ROIProject.order`, which the list and the results follow, never the numbers; *Filter ROIs*: a parsed condition setting *use* in one undo step, on the measured columns, the keywords and each ROI's place (`roi_x_nm`, `in_drawn_roi`); *Remove double ROIs*: a pair closer than d, the larger kept; `roi_manager/order.py`) |
| superimposes the ROIs into a new table | `average_rois.py` (*Average ROIs*: a centre and a turn per ROI from its centre, its lines (`ROI.lines`) or its results row; one table added as a file through `Result.files` with `append`, each ROI's frames moved apart so merging blinks never joins two) |
| moves a file's localizations or an image layer | `transform.py` (*Transform*: the target -- a file and channel, or an image layer's placement, `ImageData.placement`, never resampled -- and the source -- typed, a named transformation (`Session.transformations`, added through `Result.transformations`), a file (`_2ct.h5`, `_T.h5`) or another file's drift (`Drift.apply(rows=)`); the maths in `smappy/transformation.py`; Register layers opens it with `settings_for(name)` through the GUI's `"transform"` window) |
| makes a new image layer from another | `process_image.py` (*Process image*: background subtraction and local contrast, ticked operations in a list, `_STEPS`; the maths on arrays in `smappy/image_processing.py`; the new image keeps the original's placement and goes back through `Result.image_layers`) |
| measures and hands its result to another plugin | `register_layers.py` (*Register layers*: the shift of a layer against another, localizations or images, by `correlation._xcorr` and RCC's `_peak` -- no third correlation -- in `smappy/layer_registration.py`; kept as a named transformation through `Result.transformations`, and a `Notice` with `window="transform"` and `transform.settings_for` opens Transform on it) |
| puts one MINFLUX measurement onto another by its MBM beads | `mbm_registration.py` (*MBM registration*: each bead's position at the start, the median of its points less the beads' drift curve, `start_positions`; beads paired by a vote on their shifts, never by name -- slots are reused; a translation, or Kabsch's rigid map from three beads, in `smappy/mbm_registration.py`; kept as a named transformation for *Transform*, `Result.transformations`) |
| writes what was measured per region onto its localizations | `region_column.py` (*Regions to column*: `region_runs.write_values`, NaN outside, never over a column of the localizations' own; a block's *write values* calls the same) |
| runs steps once per region, writes columns back | `smappy/blocks.py` (a `chain.Block`; the regions through `region_runs.regions_of` -- ROIs or an id column, one interface; `RegionScratch`, `WriteBack`, *write values*; per-cluster results in `Session.id_regions`; the ROI project holds one, `ROIProject.block`) |
| opens an interactive window of the GUI's | `bead_calibration.py` (`Plugin.window`: the button opens what the GUI registered in `session.window_openers`); `psf_calibration.py` the same for the PSF calibration window, `calibrate/psf_window.py`, whose methods (bead average, uipsf's pupil model, in situ from blinking molecules: `calibrate/insitu_run.py`; and the elliptical Gaussian's width curves for Astigmatic z, `calibrate/elliptical.py`) are Qt-free entries in `calibrate/psf_methods.py` -- each a *Detect beads* and a *Calibrate*, its pages and, after a run, the model against the data in `calibrate/model_views.py` |
| hands the session to a new instance | `smappy.handoff` (*File > Open in a new window*: the table through shared memory, the rest as JSON, `smappy-gui --handoff`) |
| is run by MicroClaw during an acquisition | `gui/live_session.py` (the GUI on a fit another program controls through `Context.stop` and `writer_finished`; the package and its protocol runner are `ries-lab/smappy-microclaw`) |

A chain of plugins that runs as one, and running one over many files
(`smappy-batch`, the batch window): `docs/batch.md`.

### Shipped plugins and the user's own

Everything in this file about pages, tests, `version` and house style is for
what is **shipped**: a plugin under `src/smappy/plugins/`.  A plugin in a
user's own folder (`config.user_plugin_dir`, `<config dir>/plugins`, always
scanned; a file at its top level lands under `User/`) needs only to load --
a page and tests are recommended, never required, and no test enumerates
it.  That folder is made ready by `smappy/user_plugins.py` (*Plugins > New
plugin...*, *Show my plugin folder*, `smappy-plugins new|check`) with an
`AGENTS.md` written from `src/smappy/data/user_plugins/AGENTS.md.in`: the
self-contained guide an assistant working in that folder reads, kept to the
installed version.  A change to the plugin API that it describes (the
context, the result, `param`, the column names) updates that file too, and
its human counterpart, the help page `docs/panels/Your own plugins.md`.  A
user's plugin proposed for the distribution moves under `src/smappy/plugins/`
with `@register` and then meets every rule here.

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
  (`collapsed=True` starts it folded); a `List[X]` of one is a list of parts
  the user adds to, one value of flat maps in a saved file, an item's field
  named `field.<i>.<name>` in `react` and `FormState` (`gui/params.ListField`);
  `beside=True` puts a field on the row before; `kind="preset"` is a name
  from saved settings (`smappy.presets`).  Rows and parts are laid out in the
  order they are declared.
* `version` -- bump it when a change moves the numbers: a chain records it,
  a batch re-runs files whose steps' versions changed, and the plugin's page
  must be re-read and brought along (see "The plugin's page" below).
* `@register("Tab/Group/Name")` places it in the tree; a class's `name = "..."`
  is what the GUI shows in its place (`COMET drift correction` at
  `Analysis/Drift/COMET`), so a plugin is renamed without moving its path,
  which saved chains, GUIs and batches name it by.  Without `@register` the *folder*
  decides the path (`<root>/Analysis/Drift/comet.py` -> `Analysis/Drift/COMET`),
  which is how a user drops a plugin in and it appears.  A standalone file
  (outside the package) is imported again when it changes, so a user's
  plugin is tried without a restart.
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

`tests/test_user_plugins.py` checks the user's folder: where its files land,
and that the template a new plugin starts from loads and runs.

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
shipped plugin writes its page in the same change (for a user's own plugin a
page is offered, not required).  A session that changes what a
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
