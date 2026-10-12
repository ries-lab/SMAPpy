# Simplification plan, October 2026

Steps from `docs/review-2026-10.md`, as decided with the owner.  Each step
is one commit (or a short series) that leaves the suite green, fast and
`--slow` tiers; a step that moves numbers bumps the plugin's `version` and
re-reads its page.  Sizes are rough.  Steps marked *depends* need the one
named first.

**Status.**  Done on `claude/trusting-cray-bhwh1n`: A1-A4, B1-B3, C1-C2,
D1-D2, E1-E3, E5 (`--slow -n auto`: 1644 passed, 6 skipped).  Left: E4
(visible filters), and section F.

## A. Small and independent

**A1. Camera fields stay on *auto*.**  Choosing a file no longer copies the
file's camera values into the form (`plugins/fit.py`, `react` on
`source.path`).  The fields on *auto* keep showing the file's values in grey
-- the machinery is there (`hints`, `ParamWidget.set_hint`) -- and the run
resolves one camera from the file under whatever the user typed.  The
Camera parameters window says "from file" for those values.  Fixes: a later
*camera* choice overridden by old numbers, the next file inheriting the
previous file's values.  *S, low risk.*

**A2. Leftover bugs.**  Gaussian 2D 2C: the preflight's "start anyway"
always failed -> offer *calibrate from this movie* instead.  Statistics'
"(grouped)" printed twice.  `Selection.require` warns through `ctx.warn`,
not progress (which overwrote it).  COMET's *backend* stops offering
"torch", whose module is not in the tree.  *S.*

**A3. One way to declare settings.**  `param()` on the fields of
`DriftSettings`, `RCCSettings`, `FitSettings`, `RegisterSettings`; delete
`Plugin.params`, `Plugin.main` and the dotted-`extra` merge.  `main`
becomes `advanced=False` on those fields.  Same labels, same defaults, same
forms -- a test compares each panel's labels before and after.  *M, low risk.*

**A4. COMET's time-window fit kept as legacy.**  No code removed; a module
note in `drift.py` and `_comet/__init__.py` says what to simplify once GPU
vs CPU is benchmarked (keep only the spline fit, move its two kernels into
`drift.py`, delete the rest of `_comet`).  *XS.*

## B. The table

**B1. Units: pixels kept, nm always there.**  The fitter always writes both
`x_pix`/`y_pix`/`*_err_pix` (the primary data) and `x_nm`/`y_nm`/`*_err_nm`.
With no pixel size known it converts with a default (100 nm), says so in a
warning and in `metadata["pixel_size_assumed"]`, and the Render tab shows
it.  The *units* setting goes (pipeline, GUI, `smappy-fit`, `api.fit`).
Analysis, grouping and rendering read nm only: the `xy_err_pix` fallbacks
in render, viewer, view3d, statistics, line profile, precision, npc, group
go, and grouping's distances are nm by definition.  A loaded table
that has pixel columns only gets nm columns the same way on load.
*M, ~20 files, low risk.*

**B2. One table per session, filters per layer.**  *Depends B1.*
- The session owns the ungrouped table and one grouped table, a cache
  keyed by (table, link settings): rebuilt when either changes, so
  `grouped_stale` and the five places that copy bounds between filters go.
- A layer owns its **bounds** (the user's setting), its file selection, its
  grouped tick and its display.  Its mask is computed on the table it shows
  -- the same bound keeps different localizations grouped and ungrouped, as
  intended.
- Link settings are the session's (they are already, in effect); the
  per-layer copy and the ROI manager's digest copy go.
- `ViewState`/`LocSet` leave `viewer.py` for a small `layer_state.py`.
*L, medium risk: live fit, undo, chains' scratch sessions, ROI manager.*

**B3. Plugins follow the layer's grouping.**  *Depends B2.*  Today only
Statistics, DBSCAN, NPC and the image export follow it; the rest always
measure the ungrouped table.  Default: a measuring plugin gets the table the
layer shows (`ctx.table`).  A plugin that must have one or the other says
so with `Plugin.grouping` (already exists, unused): drift and Remove
Localizations work on the ungrouped table and correct all of it; Precision
(NeNA pairs consecutive frames) needs ungrouped.  Each plugin's choice is
written on its page.  *M, changes numbers of some plugins on grouped
layers -> version bumps.*

## C. Regions

**C1. One interface over the two result stores.**  `region_results.py`:
`Evaluation(source, label)` instead of the `"per cluster_id: label"`
string, and `available`, `table`, `rows`, `steps`, `is_current`.
SiteAnalysisPlugin, Histograms, Regions to column and NPC Labeling
Efficiency use it and stop branching.  Fixes: per-cluster results never
checked for staleness; Histograms' preflight raising on a per-cluster
choice; NPC Labeling Efficiency never offering per-cluster tables.  No file
format change.  *M (~200 lines).*

**C2. Stable ROI numbers.**  An ROI keeps the number it was given; deleting
ROI 3 no longer renumbers ROI 5, and `roi_id` on localizations, in the list
and in result rows is that number.  *S–M; changes saved files' `roi_id`
meaning (no compatibility needed).*

## D. Removals

**D1. The old viewer out, `smappy.show` in.**  *Depends B2 (`ViewState`
moved).*  Delete `viewer.Viewer`, `live.LiveViewer`/`live_view`,
`smappy-view`, `smappy-live`, `DisplaySettings.color_mode` ("sum"),
`imax`, the public `weight_field`, and their tests.  Keep `LiveFit` and
`camera_extent` (moved).  Add `smappy.show(locs | session | path,
wait=False)`: the Qt GUI in its own process through `handoff`, usable from
a script or a notebook.  *M, ~1,100 lines out, ~60 in.*

**D2. `_3dcal.mat` without mirroring code.**  Convert at load: reverse the x
cells of the spline and re-expand each cell's cubic (b_q = (-1)^q
sum_{p>=q} C(p,q) a_p on the x index; odd nx gains a cell).  Then
`LocalizationEngine.mirror`, `ROIStack.mirrored`, `SplinePSF.mirror`,
`GlobalSplinePSF.mirror`, the simulator's mirror and `em_mirror` go.  Tests
on synthetic splines (model equal to float32 rounding, fits equal).  The
`emmirror == 2` question is left marked in the code for the owner's
check on real files (`SMAPPY_TEST_CAL`).  *M, ~55 lines out, ~40 in.*

## E. User experience

**E1. Stop.**  While a plugin runs, Run becomes Stop and sets `ctx.stop`
(honoured today by the fitter and COMET; RCC and long region runs get a
check between steps).  What was done so far is kept, as the fitter does.
*S–M.*

**E2. Drop a file.**  Both main windows accept a dropped localization file
(the loader chosen by extension, as *Load/Auto*) or an image stack (TIFF,
OME-TIFF, NDTiff folder, `*.sim.yaml`): the Localize tab opens with it as
the fitter's *file*.  *S.*

**E3. Unsaved changes on quit, as an option.**  A preference, off by
default; the title shows `*` after a change to the table either way.  *S.*

**E4. Filters you can see.**  The active bounds of the current layer as a
row of small labels above the histogram (click one to edit, × to remove),
and "filters hide 33%" in the status line.  *S–M.*

**E5. Errors.**  A failed run: the message (without "ValueError:") in red
next to Run, a *details* button with the traceback and *Save a bug
report*.  A failed load the same way instead of a modal.  *S.*

## F. Later, by the owner's decision

- The plugin contract: one form-state hook, one effect hook, typed effects
  instead of `Result.data` keys (review §4.3).
- Render threads render a snapshot, not the live session (§4.5) -- the root
  of the GC workaround.
- Naming consolidation, with the old names listed in the docs.
- The Analysis tab's order and grouping; the region summary plots; the 3D
  view (extras into options).
- Fitting: one Fit plugin, one registration geometry, one camera path
  (§4.7).

## Suggested first batch

A1, A2, A3, A4, B1, C1, E1, E2 -- independent of each other and of the
large B2.  Then B2 -> B3 -> D1, with D2, C2, E3–E5 alongside.
