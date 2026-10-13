# Review status, October 2026

Where `docs/review-2026-10.md` stands after the simplification plan
(`docs/simplification-plan.md`, sections A–F) landed on
`claude/trusting-cray-bhwh1n`: 47 commits, 341 files, about 6,000 lines
fewer (+10,950 / −16,920).  `--slow -n auto`: 1672 passed, 6 skipped.

## Done

| review | what changed |
| --- | --- |
| §3 bugs 1–10 | all fixed: ROI count after a chain, progress fragments, one filter with two meanings (§4.2), pixel-table grouping (§4.1), COMET torch, 2C preflight, "(grouped)" twice, file camera values, `require` warning, ROI numbers stable |
| §4.1 units | nm and px columns always both present (owner's choice: keep pixels); the *units* setting and every `*_pix` fallback went; no pixel size → 100 nm, said in the metadata and the Render tab |
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

## Open, with recommendations

In order of value for the effort.

1. **Segmenters keep their earlier ROIs by default.**  *replace earlier
   ROIs* exists but is off, so a second NPC run still finds "0 pores".
   Turn it on by default; undo covers regret.  *XS.*
2. **Cleanup commit (§5)**: `integration/` (now `ries-lab/smappy-microclaw`),
   `scripts/` except `make_icon.py`, `drift_correct.py` and a live-fit
   example (`check_stack.py` is broken), Line Profile's `binned` method,
   `_hint_from_text`, `__smappy_plugins__`, `ParamInfo.step`, one
   `gui.make_app()` for the five QApplication setups, NOTES.md diaries out
   to `docs/`, `docs/regions.md` from a 564-line log to the current model.
   *S, no design questions.*
3. **Small UX wins left from §7**: an empty-view hint ("Open or drop a
   file, or File > Simulate"); *Workspace* in place of "Save GUI / Load
   GUI"; render modes and quick filters in words (*Gaussian (precision)*,
   *histogram*, *diffraction-limited*; *precision*, *fit quality*,
   *photons*); a colour bar when colouring by a column; the simulator's
   "open it in a fitter" as a *Fit this* button; undo entries that name
   what they undo.  *S each.*
4. **Merge blinks lazily (§4.2).**  Every file is still merged on open
   (`GROUPED_BY_DEFAULT`), which is why `_group_chunked.py` (362 lines,
   approximate) exists.  Merge on the first *merge blinks* tick, with
   progress; then the chunking can go.  *M.*
5. **Garbage collection back on.**  The snapshot render (§4.5) removed the
   known cause.  Try it in the test suite and the app under a live fit;
   keep the collector as the safety net if anything hangs.  *S to try.*
6. **Regions, the rest of §4.4**: one runner over ROIs and id columns
   (`RegionSource`), the ROI project holding a `chain.Block`, one
   write-back rule, and the removals (file-backed `ROIProject`/`Source`,
   `reviewed`, `keep_figures`, `slow`/`fast_values`).  Rename `roi.py`
   (the fitter's boxes) to `fit_rois.py`.  *L; do after 1–2.*
7. **Forms (§7.3)**: camera as one line with *edit*; *boxes per fit*,
   *threads* and *iterations* out of the GUI; `api.fit`'s cutoff that
   changes meaning at 20 becomes two arguments; tri-state boxes as combos;
   one vocabulary for which localizations a plugin reads (*selection / per
   layer / all*); a *Draw line* button in Line Profile, and its output with
   units and without degenerate fits.  *M in all.*
8. **Layout (§7.2)**: one entry point each for Load/Save, Bead calibration,
   ROI manager and per-region steps; the Regions tab header collapsible;
   Plot/Text buttons merged into one *Show result*; notices coloured by
   kind.  *M.*
9. **Deferred by the owner**: the Analysis tab (to become a lean GUI users
   add to), the 3D view redesign with picture settings on the session and
   the slab as a 3D ROI (§4.6), COMET's GPU path (benchmark on Windows,
   then keep only the spline fit and delete the rest of `_comet`), one
   *Drift correction* plugin.

## For the owner to check

- A real `_3dcal.mat` with `SMAPPY_TEST_CAL` (the `emmirror == 2` case).
- Dual-colour calibrations saved before the split fix with a camera ROI
  not at the chip's edge along the split must be made again.
- `ries-lab/smappy-microclaw` still passes `output_unit`; a two-line PR
  removes it.
- Saved workspaces and chains naming `Dual-Color/AssignColors` or
  `NPC Labeling Efficiency` no longer find them (no aliases, by decision).
