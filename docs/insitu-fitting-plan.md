# The model in the fit: calibrated or learnt in situ, and kept with the localizations

Agreed with Jonas on 2026-11-26.  Two changes, the first useful without the second:

1. **Every fitted file carries the calibration it was fitted with** -- the pupil (Zernike) model, the
   spline's knots, the registration -- so a `localizations.h5` stands alone, and can itself be loaded as
   the calibration of a new data set.
2. **The spline fitters choose where their model comes from**: a calibration, as now, or learnt in situ
   from the data set being fitted, from a start model, and saved.

## What is there now

| fitter | model from | in situ now |
| --- | --- | --- |
| Spline 3D | *calibration*: a file, *auto* (`calibration_list`), `microscope` | *focus* = `in situ`: `depth.insitu_focus` runs `insitu_run.run_insitu` with *hold* = every aberration: the focus and the blur learnt, kept by `depth.keep` / `kept` |
| Spline 3D 2C | a dual calibration; *in the sample* (`depth.SampleSettings`) auto or typed | none: uipsf has no dual in situ workflow; the PSF window refuses dual |
| Spline 4Pi | a 4Pi model | only in the PSF window (`fourpi_insitu.run_insitu`: a fit with the bead model, then the maps, misalignment and channels learnt) |
| Gaussian 2D 2C | registration from the movie or a file | the PSF is fitted; out of scope |
| DECODE | trains on a calibration | out of scope; its training window gains a fitted file as a calibration through (1) |

A fitted file records only `metadata["calibration"] = {"file": ...}` (`_FitPlugin` before the writer,
`calibration_list` reads it back): move or lose the calibration and the record points nowhere.

## 1. The calibration inside the localization file

**What is embedded** -- small, and exact:

* **The spline's knots, not its coefficients.**  A native calibration keeps both: on
  `astig_v2_upper_3dcal.h5`, `coeff` is 12.8 MB and `psf` (the knots, `(nz+1, ny+1, nx+1)`) 0.22 MB.
  `calibrate.core.spline_coefficients(psf)` rebuilds `coeff` bit for bit (max difference 0.0) in 50 ms.
  The embedded copy keeps the knots and an attribute saying the coefficients are to be rebuilt.  A SMAP
  `_3dcal.mat`, whose `psf` is its bead stack and not the knots, embeds its coefficients (gzip).
* **The pupil model**: the `uipsf` group a calibration carries since 2026-11 (`pupil.MODEL_GROUP`,
  `carries_model`), copied as it is; `uipsf.load_model(path, group=...)` reads a group already.  The
  microscope's kept pupil is in `parameters["pupil"]` and comes with the parameters.
* **A dual calibration**: both channels' knots, its registration and geometry, the same way
  (`calibrate/dual.py` `save_dual_color_calibration`'s content).
* **A 4Pi model**: its `volumes` (already knots only), `transforms`, `layout_json`, parameters -- the
  whole of what `fourpi.save` writes.
* **Field-dependent splines**: the node splines are stored as float16 coefficients
  (`write_field_splines`).  Open: keep the nodes' knots instead (64 × smaller) if the learning has them;
  else embed the coefficients and measure the size on Zenodo's 13 × 13 grid.
* **The context**: the focus and sample index it was drawn at, how it was made (bead average, pupil,
  in situ and from which start, the check of section 2), the source file's name and sha256.

**Where.**  A group `/calibration` in the localization file, laid out exactly as the calibration file it
stands for (the same `format` attribute and datasets), so one reader serves both.  The readers
(`io.calibration._load_native_calibration`, `dual.load_dual_color_calibration`,
`fourpi.load_fourpi_calibration`) are refactored to read from an h5py group, and the path-taking
functions open the file and pass `/` or `/calibration`.

**Reading it back.**  `load_spline_calibration(path)` (and the dual and 4Pi loaders) on a
`smappy-localizations` file take the embedded calibration first, the recorded path only if there is
none (old files).  The fitters' *calibration* field already accepts "a fitted .h5, for the calibration
it was fitted with" -- that now works with nothing else on disk.  `calibration_list` (*auto*) does not
search localization files; it is offered by naming one.

**Carried through.**  The table keeps it across what happens after the fit:

* on load, the group is read into memory as a small in-memory HDF5 image (`h5py` core driver, a few
  hundred kB) kept beside the metadata; *Save localizations* copies it back into `/calibration`.
* several files in one table, each fitted with its own: one per file, `/calibration/<file index>`,
  following `metadata["files"]`; *Combine files* and *add* keep each file's.
* *Open in a new window* (`handoff`) passes it with the rest.
* Picasso and the other exporters do not carry it.

**Tests**: a fit writes the group; loading the fitted file as a calibration gives the same spline
(`coeff` equal) and the same model; the group survives load → edit → save, combine and handoff; an old
fitted file falls back to its path; a dual and a 4Pi fit likewise.

## 2. In situ in the spline fitters

### The settings

A *model* part shared in shape by the spline fitters:

* ***source***: `calibration` (as now) or `in situ`.
* ***calibration***: as now -- with `in situ`, the start model.  Empty: *auto*; `microscope`; or a
  fitted file (section 1).  Without a pupil model (a bead average) in situ starts from the microscope's
  pupil or the nominal astigmatism, as the PSF window's in situ does (`insitu_run.start_astigmatism`).
* ***learn*** (with `in situ`), default `focus`:
  * `focus` -- the focus and the blur, every aberration held (today's *focus in situ*; *hold* = all);
  * `aberrations` -- the pupil from the start, γ held (*hold* = the start's);
  * `everything` -- γ too (*hold* = nothing).
  * Spline 4Pi: `maps, misalignment, channels` (default) or `everything` (`fourpi_insitu.LEARN_CHOICES`).
* the molecules (how many, frames read) under *more*, with `insitu.InSituSettings`' defaults.
* *focus* keeps `auto`, `typed`, `calibrated` for `source` = calibration.  A saved `focus: insitu` is
  read as `source: in situ, learn: focus` in `__post_init__` (`settings_from` has no migration hook and
  would otherwise drop it).

### The run

1. **Learn** from this data set -- the fitter's source, the same frames it fits; a batch item is one data
   set, so a batch learns per item and needs no change.  Spline 3D: `insitu_run.run_insitu` (its own
   detection, `movie_rois`).  Spline 4Pi: `fourpi_insitu.run_insitu`, which fits a part of the movie with
   the start model first.
2. **Check** the learnt model against the start model on held-out molecules (likelihood per molecule,
   as 4Pi §34), plus the twin test's warnings.  If it does not explain them better: **fall back to the
   start model, with a warning** (`ctx.warn`), and record why.  Needs: a held-out split in
   `run_insitu` and a score for the start model at the learnt focus (`uipsf.workflows.scoring`).
3. **Save** the learnt model beside the data set (`<movie>_insitu_3dcal.h5`, model inside, as a PSF
   window in situ saves) -- the reusable copy and the cache: keyed by the data set, the start model's
   sha256 and the learning settings (`depth.keep` / `kept` generalised), so a rerun -- a batch rerun
   for a later step's version, a Preview then Run -- learns once.
4. **Fit** the whole data set with it; the fitted file embeds it (section 1).

* `preflight`: a question with the cost (minutes; torch), offering "use the start model as it is".
* Preview: the model against the data (`insitu_views.draw_model_vs_data`) and the learnt focus, before
  the fit.
* Live fits (watch, MicroClaw): `in situ` refused at first, with the reason in the notice.
* A refit of a fitted file uses its embedded learnt model as a calibration; learning again is asked for
  by choosing `in situ` again.

### Tests

Simulated astigmatic data with a known pupil differing from the start model's (`smappy.simulate`):
`learn = aberrations` recovers the aberrations and the focus within tolerances, and the fit's z bias
drops against fitting with the start model; `learn = focus` reproduces today's *focus in situ* numbers;
a start model already right triggers no fallback, a learning made to fail (too few molecules) falls back
with a warning; the cache is used on a second run.  Slow tier.

## Phases

0. **Embedding** (section 1): single, dual, 4Pi, model; read back; carried through; tests; the pages
   (*Save localizations*, the fitters' *calibration* notes, `docs/plugin-reference.md`).
1. **Spline 3D in situ** (section 2), *focus in situ* folded in; its page and version bumped; the
   PSF calibration window's in situ kept as the place to make a calibration for many data sets.
2. **Spline 4Pi in situ**: the same part over `fourpi_insitu`.
3. **Spline 3D 2C in situ**, when it fits the plan: needs a dual in situ workflow in uipsf (biplane first:
   the same molecules in both channels; colour split later) and its benchmark.

## Open

* Field-dependent calibrations: embed knots or float16 coefficients (size on a 13 × 13 grid).
* The held-out check for single channel: which fraction, and the threshold for "better".
* Whether *auto* should one day also find recent fitted files of the same microscope.
