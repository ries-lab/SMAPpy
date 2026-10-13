# Review status, October 2026

Where `docs/review-2026-10.md` stands on `claude/trusting-cray-bhwh1n`
(PR ries-lab/SMAPpy#19): the simplification plan
(`docs/simplification-plan.md`, sections A–F) and a second round from this
document's open list.  83 commits, 378 files, about 7,100 lines fewer
(+15,230 / −22,340).  `--slow -n 3`: 1676 passed, 6 skipped; CI green on
all platforms.

## Done: the plan

| review | what changed |
| --- | --- |
| §3 bugs 1–10 | all fixed: ROI count after a chain, progress fragments, one filter with two meanings (§4.2), pixel-table grouping (§4.1), COMET torch, 2C preflight, "(grouped)" twice, file camera values, `require` warning, ROI numbers stable |
| §4.1 units | a fit's table is nm only (owner's choice, reversing an earlier "keep pixels"): `peak_x_pix`/`peak_y_pix` stay, the pixel size is in the metadata and `locs.in_pixels` gives registration its chip pixels; the *units* setting and every `*_pix` fallback went; no pixel size → 100 nm, said in the metadata and the Render tab |
| §4.2 tables | one ungrouped and one grouped table per session, bounds per layer; `share=`, `grouped_stale` and the bound copying went (`layer_state.py`) |
| §4.3 plugin contract | `param()` the one way to declare settings; `describe() -> FormState` instead of five hooks; `changes_locs` required; typed `Result` fields instead of magic `data` keys; *follow ROI*, `ColumnRecipe`; plugins follow the layer's grouping (`Plugin.grouping`) |
| §4.4 regions | one interface over both result stores (`region_results`, `Evaluation(source, label)`), fixing three bugs; region summary as a titled grid |
| §4.5 render | 2D and 3D workers draw a snapshot taken on the GUI thread, one worker class |
| §4.7 fitting | four fitters kept as thin subclasses of one implementation; one registration geometry with layout detection for movie and beads; one camera path (presets and `*_cameras.mat` gone); dual calibration read once per run; split position in chip pixels everywhere (a real bug: offset ROIs split at the wrong place) |
| §4.8 drift | COMET's time-window fit kept as marked legacy; COMET and RCC merge blinks as the session does |
| §4.9 | matplotlib viewer, `smappy-view`, `smappy-live` gone; `smappy.show` opens the GUI from a script |
| `_3dcal.mat` | mirrored calibrations converted on load; no mirroring code in fitter or simulator |
| §7.1 UX | drop a file (short TIFF = image, long = data to fit), Stop, `*` and an optional quit prompt, errors in red with details and a bug report, active filters as a line with "filters hide 33%" |
| §7.5 words | *ROI* / *region* / *per-region steps*, *merge blinks*, *fit box*, British spelling, px and µm; jargon explained on every page with a Glossary (now a CLAUDE.md rule) |
| §2 CLAUDE.md | trimmed; the reference moved to `docs/plugin-reference.md` |

## Done: the second round

| item | what changed |
| --- | --- |
| CI | the wheels' smoke test no longer passes the removed `units=`; checked on Python 3.9 too |
| segmenters | *replace earlier ROIs* on by default, so a second run finds the pores again |
| cleanup (§5) | `integration/` and most of `scripts/` gone; Line Profile's binned fit gone; dead contract leftovers (`_hint_from_text`, `__smappy_plugins__`, `ParamInfo.step`); one `gui.make_app()`; NOTES.md 2,180 → 1,470 lines (diaries to `docs/drift.md`, `docs/performance.md`); `docs/regions.md` 564 → ~210 lines, the current model |
| small UX (§7) | a hint on the empty view; File > Workspace; render modes and quick filters in words; a colour bar per layer coloured by a column; *Fit this* after simulating; undo entries named in the ROI manager; "workspace" in the title |
| merge blinks (§4.2) | files open at once; a large table merges in the background and the layer switches to merged blinks when ready (small ones at once, as before); scripts merge on first read; the approximate chunked linker gone (exact: 0.8 s for 2 M, 3–6 s for 5 M) |
| regions (§4.4) | no `reviewed` flag; no `slow` options (Statistics' *precision fit* is one choice none / histogram / MLE); one `write_values` for blocks and *Regions to column*, *any* refused there; the ROI project holds a `chain.Block`; one adapter over ROIs and id columns (`regions_of`); the file-backed `ROIProject` and its sidecar gone (scripts use a `Session`); *keep figure data* gone; `smappy/roi.py` → `fit_rois.py` |
| questions before a run | one dialog; Enter takes the faster alternative, else Cancel -- never the long run |
| forms (§7.3) | notices by kind (info grey); optional bools as *auto / on / off*; *boxes per fit* and *threads* scripting-only; the camera part folded under a summary, and at Run a camera nobody entered asks *Open camera editor* / *Cancel*; *cutoff value* resets when *cutoff* changes mode; `cutoff_mode` in `smappy.fit` and `smappy-fit` (a bare cutoff ≥ 20 warns) |
| Line Profile | units and ±; an undetermined parameter says so and leaves the comparison; the winner drawn on top; *Draw line* when there is no line |
| layout (§7.2) | one loader (*Auto* with *format*); the File tab holds Export and Simulate; one *Show result*; "+ chain" in the + chooser; Tools without the duplicate ROI manager / per-region steps / bead calibration entries; the Regions tab header folded; bead calibration asks single/dual once |
| merged blinks wait | while a large file's blinks merge, the view says so (stage and clock) instead of drawing unmerged localizations, and every run that reads the localizations -- plugins, chains, follow ROI, the ROI manager -- waits and then runs on the merged blinks; chains share the session's merge; no `group_id` written from a worker thread |
| regions, follow-ups | `ROIProject` is one class (`SessionROIs` and `link.py` gone); a nested block whose changes to the table would be discarded warns in the editor and when the chain runs |
| garbage collection (§4.5) | tried on again and refused on evidence: with automatic collection on, `--slow` lost a worker to a segfault in 9 runs of 9 -- the collector runs in whichever thread allocates (a fit's TIFF reader, a linking pool) and frees Qt widgets left by earlier tests, and PySide sends only plain `QWidget`s, not subclasses, back to the main thread.  Collection stays off with `collect_on_gui_thread`; `gui/collector.py` says why |

## Open

1. **A `QThread` collected while still running** ("QThread: Destroyed
   while thread is still running"): seen rarely, once with collection off as
   it is, inside conftest's collection on the main thread.  Find which
   thread object is left in a cycle and keep a reference until it finishes.
2. **A percentage while merging blinks** needs a counter inside the C++
   linker (one call today); the view shows the stage and a clock.
3. **`test_handoff.py::test_a_run_handed_over_is_run_by_the_new_instance`**
   failed three times in full fast-tier runs and never alone (32 runs); its
   error was not captured yet.  Catch it with the output kept, then fix.
4. **`cutoff` ≥ 20 without a mode** still means photons, with a warning;
   make the mode required in a later release.
5. **Deferred by the owner**: the Analysis tab (a lean GUI users add to),
   the 3D view redesign with picture settings on the session and the slab as
   a 3D ROI (§4.6), COMET's GPU path (benchmark on Windows, then keep only
   the spline fit), one *Drift correction* plugin.
6. **Dropped on purpose**: one engine for ROI and column runs (per-ROI reuse
   stays separate), removing kept figure data entirely, an *edit* button
   from the fitter to the camera database, *iterations* out of the GUI.

## For the owner to check

- A real `_3dcal.mat` with `SMAPPY_TEST_CAL` (the `emmirror == 2` case).
- Dual-colour calibrations saved before the split fix with a camera ROI
  not at the chip's edge along the split must be made again.
- `ries-lab/smappy-microclaw` still passes `output_unit`; a two-line PR
  removes it.
- Not loaded any more (no aliases, by decision): workspaces and chains
  naming `Dual-Color/AssignColors`, `NPC Labeling Efficiency` or the
  per-format loaders; old settings with *fit precision* off get the
  histogram fit; ROIs saved as not reviewed are measured.
- Scripts with `cutoff=` ≥ 20 now warn: add `cutoff_mode="absolute"`.
