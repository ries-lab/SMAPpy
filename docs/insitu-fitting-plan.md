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
* **Field-dependent splines**: each node's knots, read off its coefficients (`io.calibration.spline_knots`)
  and splined again on reading (`field_splines_of`).  On `synthetic_x1_3dcal.h5` (7 × 7 nodes) 14 MB in
  place of 368 MB, the nodes back to 5e-9.  Nodes kept in half precision are checked by their values
  (at each cell's centre, `embedded._centre_values`): the knots' spline is 4e-4 of the peak from the
  original, as close as the half-precision coefficients are.
* **The context**: the focus and sample index it was drawn at, how it was made (bead average, pupil,
  in situ and from which start, the check of section 2), the source file's name and sha256.

**Where.**  A group `/calibration` in the localization file, laid out exactly as the calibration file it
stands for (the same `format` attribute and datasets), so one reader serves both.  "The knots rebuild the
coefficients" is checked, never assumed (`io.calibration.knots_rebuild`), within 1e-5 of the largest
coefficient: float32 rounding leaves 1e-6 on dual calibrations and 1e-12 on in situ ones, so exact equality
kept the coefficients of every dual file.  Measured: bead and pupil calibrations 0.4-0.5 MB, an in situ
one 1.5 MB, a dual one 0.5 MB (32 MB as stored).  The readers
(`io.calibration._load_native_calibration`, `dual.load_dual_color_calibration`,
`fourpi.load_fourpi_calibration`) are refactored to read from an h5py group, and the path-taking
functions open the file and pass `/` or `/calibration`.

**Reading it back.**  `load_spline_calibration(path)` (and the dual and 4Pi loaders) on a
`smappy-localizations` file take the embedded calibration first, the recorded path only if there is
none (old files).  The fitters' *calibration* field already accepts "a fitted .h5, for the calibration
it was fitted with" -- that now works with nothing else on disk.  `calibration_list` (*auto*) does not
search localization files; it is offered by naming one.

**Carried through.**  The table keeps it across what happens after the fit:

* on load, the group is read into memory as an HDF5 image in bytes, `metadata["calibration_h5"]`
  (`embedded.KEY`); the writer puts it back as `/calibration`, never in the JSON.
* several files in one table: the table's metadata is the first file's, so `/calibration` is the first
  file's; each other file's is in its `metadata["files"]` entry (`FileInfo.metadata`) as base64
  (`embedded.plain`), the first file's written there once only, as `"the table's"`.
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
   (*Save localizations*, the fitters' *calibration* notes, `docs/plugin-reference.md`).  **Done
   2026-11-26** (`io/embedded.py`, `tests/test_embedded_calibration.py`): the fitter embeds the file its
   *calibration* field names once resolved, before the focus is drawn, so a fitted file loaded as a
   calibration gives the beads' spline and the model, and *focus* draws it again for the new data.
1. **Spline 3D in situ** (section 2), *focus in situ* folded in; its page and version bumped; the
   PSF calibration window's in situ kept as the place to make a calibration for many data sets.
   **Done 2026-11-26**, otherwise than section 2 said in three places:
   * **No *source* field.**  *focus* already had `in situ`, which is what *source* = in situ with *learn* =
     focus would have been; *learn* (`focus and blur` / `aberrations` / `everything`) is shown under it.
     Nothing to migrate, one field fewer, and saved settings keep meaning what they meant.
   * **The start needs a pupil model**: a pupil calibration, or the microscope's pupil (*calibration* =
     `microscope`), which covers "no bead calibration".  Starting from a bead average and the nominal
     astigmatism, as the PSF window can, is left for when someone asks.
   * **Saved where the focus in situ was**: the configuration directory's `sample_depth` cache (keyed by the
     movie, the calibration, the index and *learn*), and -- since phase 0 -- inside the fitted file, which is
     the copy to keep and reuse.  Not beside the movie: *calibration* on *auto* would take it for a bead
     calibration there.
   The check: a focus-only run first (from `InSituSettings`' 2000 molecules), the pupil learnt from the same
   molecules at its focus, both scored on the tenth of the blinks held out of both by their summed deviance
   (`insitu_run.heldout_deviance`); the pupil kept only if better, else the focus alone with a warning
   (`depth.insitu_focus`).  A preflight asks before the first such run.  Not done: Preview's model-against-data
   figure (it draws the heights, as before).
2. **Spline 4Pi in situ**: the same part over `fourpi_insitu`.  **Done 2026-11-26**: *in situ* (off / maps,
   misalignment, channels / everything) with *frames fitted* and *molecules*, resolved before the fit in
   `FourPiFit.with_focus` as Spline 3D's focus is.  A Run fits the first frames with this fitter's own settings
   and the bead model, learns from them (`fourpi_insitu.run_insitu`), and keeps the model only where
   `fourpi_insitu.warnings_of` finds nothing -- held-out molecules no better, molecules on another fringe,
   the period moved -- else the bead model with the warnings; the model kept with its record in the
   configuration directory's `fourpi_insitu` (`fit_4pi.insitu_path`), the registration then the model's.
   A Preview takes what a Run kept and never learns (a fit of 20000 frames is no Preview).  The end-to-end
   test is `long`.
3. **Spline 3D 2C in situ**: needs a dual in situ workflow in uipsf (biplane first:
   the same molecules in both channels; colour split later) and its benchmark.  **Done 2026-11-26** on
   simulated movies, biplane and two dyes alike:
   * **No new uipsf workflow.**  The single-channel rounds (`calibrate_insitu`, `learn_insitu`) run unchanged on
     a model of two channels, since they talk to the model and `rois.batch` only.  uipsf gained what a
     two-channel model needs: `insitu_model` of several channels (each its own held pupil, the transforms),
     `Channel(fit_dz=False)` and **a share of the light per molecule** (`Channel(share_per_emitter=True)`) --
     one share for all is right for biplane and one dye, wrong for two dyes -- and split movies to test on
     (`movie.simulate_movie` with ``region``, ``dyes``).
   * **The molecules** (`calibrate/insitu_dual.py`): each half detected on its own with in situ's steps, units
     of the same frames paired within 2 px of where the map in force puts them -- the calibration's (or the
     file's) with the channel *move* on top when it is measured; a lone unit's partner cut where the map puts it,
     that half's molecules masked; chosen by the main half's Gaussian.
   * **The settings**: *in the sample*'s *focus* gains *in situ* (it was a choice of `SampleSettings` 2C did
     not offer), with *learn* as Spline 3D's and *learn registration* (off: the map held, the channel *move*
     being the tool for a splitter moved since the beads).  *learn* = focus and blur holds both pupils and the
     channels' offset in z (`ch1.dz`); *aberrations* learns the offset too.  The share is per molecule exactly
     when the fit's *link photons* is off: in situ models the photons as the fit does.
   * **The run** (`depth.insitu_dual`): as Spline 3D's, the focus alone first from 1000 molecules, then on
     request the aberrations from the same molecules; both judged by `depth.judged` (shared with Spline 3D now)
     on the tenth of the blinks held out, the learnt kept only if better, else the focus alone with a warning.
     Drawn by `dual_depth_calibration` at the learnt focus, the registration the model's; kept in
     `sample_depth` as a dual calibration with the model inside (`depth.kept(key, dual=True)`); a Preview
     never learns (Spline 4Pi's rule: it takes what a Run kept, else fits with the focus on auto); the
     preflight asks before any in situ not kept, offering the focus on auto.
   * **Measured** (`tests/uipsf_tests/test_insitu_dual.py`, `long`): a biplane movie (150 molecules,
     `ch1.dz` = −300 nm, share 0.45) and a dual calibration with half its astigmatism in both channels;
     *aberrations* brought the astigmatism back to 0.665 / 0.568 rad (0.70 / 0.60), the offset to −297 nm,
     the share to 0.458, the focus to 667 nm (700), 17 nats per held-out molecule better than the focus alone;
     the fit's z scatter about the truth (MAD) 26 nm against 33 nm with the start at the focus on auto.  Two
     dyes (0.2, 0.8): the shares per molecule 0.24 / 0.75.
   * **Not done**: real data (Zenodo `2ch_smlm_nup96_WGA`, M2's two colours); a field-dependent calibration
     is learnt and drawn without its field; the PSF calibration window has no in situ dual sample (the
     fitter was the goal; `insitu_dual.run_insitu_dual` is the entry a window would call).

## Open

* The held-out check: any gain keeps the pupil learnt.  On the 100-molecule test movie only 3 molecules are
  held out; measure on real data (M2) how large the gain is and whether a threshold in nats per molecule is
  needed.
* Whether *auto* should one day also find recent fitted files of the same microscope.
