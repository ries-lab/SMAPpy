# The PSF model in the window and in the fitters: one set of parts

Agreed with Jonas on 2026-11-27; a plan, nothing built.  It continues `insitu-fitting-plan.md`
(the calibration inside the fitted file, in situ in Spline 3D), whose phases 0–3 are done, and
answers its question of how in situ learning can depend less on a start model, or need none.

## The principle

**The PSF window and the fitters are one workflow.**  By hand: the PSF window learns a model,
saves a calibration, the fitter uses it.  A fitter's in situ is the shortcut: the same algorithm,
the same settings, run before the fit on the movie being fitted, and saved with it.  So:

* the fitters run the window's algorithms (Spline 3D and 4Pi already do: `depth.insitu_focus` →
  `insitu_run.run_insitu`, `fourpi_insitu.run_insitu`; Spline 3D 2C's `insitu_dual` is the
  exception the window lacks, and becomes the window's dual in situ);
* a setting that means one thing has one name, one tooltip and one widget wherever it is shown;
* a fitter need not have a calibration: the microscope's optics are a start (*PSF model* =
  *microscope*, the kept pupil or, without one, the theory of its optics);
* everything runs headless: a chain, a batch, the assistant (`form_in_window` for the window).

## The parts

Small parts, each a settings dataclass, composed where they apply -- the code and the widgets
that exist (`OpticsSettings`, `SampleSettings`, `ChannelsSettings`, the move's `MOVE_FIELDS`),
regrouped, not rewritten.

| part | rows | is today |
| --- | --- | --- |
| **optics** | microscope, camera, NA, wavelength(s), immersion and coverslip index | `pupil.OpticsSettings` (the window) |
| **PSF model** | *PSF model*: *auto* / *file* / *microscope* | `SplineModelSettings.calibration`, the window's *start from* |
| **depth** | *focus*, *focus depth*, *sample index* | `depth.SampleSettings`, Spline 3D's own copy of its fields, `InSituModelSettings.focus_nm` / `focus_fit` / `coverslip` |
| **learn** | *learn*; under *more*: molecules, rounds, held-out share, *learn from* (batch) | Spline 3D's *learn*, the window's *hold*, `insitu.InSituSettings`, 4Pi's `FourPiInSituSettings` |
| **channels** | *map*, *learn map*, *move*, *over time* (and the move's fields) | `ChannelMoveSettings`, `ChannelTransformSettings`, `SplineTransformSettings`, `DualModelSettings.learn_registration`, `multichannel.measure_channel_drift` |
| **field** | *field dependence* (tiles) | `InSituSettings.field_tiles`, the window's field option |

Where each is shown:

| | optics | PSF model | depth | learn | channels | field |
| --- | --- | --- | --- | --- | --- | --- |
| PSF window, in situ | ✓ | ✓ (start) | ✓ | ✓ | dual, 4Pi | ✓ |
| PSF window, beads / molecule stacks (stage 2) | ✓ | ✓ (start) | sample index | -- | dual, 4Pi | ✓ |
| Spline 3D | (from the microscope) | ✓ | ✓ | ✓ | -- | ✓ |
| Spline 3D 2C | (") | ✓ | ✓ | ✓ | ✓ | ✓ (later) |
| Spline 4Pi | (") | ✓ | ✓ (later) | ✓ | ✓ (gains the move) | -- |
| Gaussian 2D 2C | -- | -- | -- | -- | ✓ | -- |
| DECODE fit / training | -- | ✓ | ✓ | -- | ✓ (gains *from this movie*) | -- |
| Simulate | -- | ✓ | ✓ | -- | -- | -- |

A fitter's optics come from the microscope database; shown only when the microscope is not in it.

## Names and tooltips

Settled once here, used everywhere; the tooltip is a short phrase, the page explains.

| row | choices: tooltip |
| --- | --- |
| *PSF model* | *auto*: the last calibration of this microscope in this state · *file*: a calibration, or a fitted file · *microscope*: its kept pupil, else the theory of its optics |
| *focus* | *beads' focus*: the calibration as measured, not drawn at a depth · *coverslip edge*: from the lowest molecules; refined when learning · *typed*: the focus depth as typed |
| *focus depth* | how far the objective moved into the sample from the coverslip |
| *sample index* | water / blinking buffer / index matched / typed (`calibrate/medium.py`) |
| *learn* | *nothing*: the model as it is · *focus and blur*: every aberration held · *phase*: the aberrations up to the order, the apodisation held · *phase and apodisation*: γ too; the focus then held at the edge or typed |
| *map* | *the model's* · *from this movie*: each channel fitted with its PSF, then registered · *file* |
| *learn map* | refine the map in situ with the PSF (on by default when learning) |
| *move* | *none* · *typed* · *measured*: the channels' shift in this movie against the map |
| *over time* | the move as a smooth curve over the frames, not one value |
| *learn from* (batch) | *each file* · *the first file* · *all files pooled* |

Renamed by this: the window's *hold* (*every aberration* / *the start's* / *nothing*) and Spline
3D's *learn* (*focus* / *aberrations* / *everything*, `depth.LEARN_HOLDS`) are one setting seen from
two sides; both become the one *learn*: *focus and blur* / *phase* / *phase and apodisation*, with
*nothing* for a fit without learning.  Whether the fitters offer *phase and apodisation*, or only the
window, is decided by the measurements below (it broke M2 August).

### The focus

One field, the same with or without learning; learning only refines it.

* ***beads' focus***: today's *as calibrated*: the beads' spline as it is.
* ***coverslip edge***: today's *auto* and *in situ* together.  The focus estimated from the lowest
  molecules' edge (`focus_from_data`); with *learn* = *nothing* that is the focus (today's *auto*);
  with *focus and blur* or *phase* it is the start, then learnt with the model (today's `focus_fit` = *learnt*); with
  *phase and apodisation* it is set by the edge each round and not fitted (`focus_fit` =
  *coverslip*, forced today in `run_insitu`).
* ***typed***: held at the *focus depth* -- an anchor as good as the edge, for a sample that does
  not touch the coverslip.

Nothing falls back silently.  The form's hint says what the focus resolved to ("edge 692 nm,
learnt 705 nm"), the fitted file records both (`sample_depth`), and **the depth guard** refuses a
learnt focus that moved from the edge by more than its uncertainty allows (the August failure:
692 → 1718 nm, kept): the edge's value is kept with a warning.  With no edge found (sparse near the
coverslip, a sample away from it) the run says so and asks for a typed focus; a fourth choice,
*from the PSF alone* (today's `coverslip=False`, the twin decided only by an index mismatch), is
under *more* if a measurement shows it is ever right.

### Two colours from a wrong map

Most users have a bead calibration whose map is wrong for today's sample (a large shift is
common).  `insitu_dual` pairs molecules within `PAIR_PX` = 2 px through the start's map, so it
fails then.  *map* = *from this movie*: each channel fitted on its own with its PSF from the model,
the localizations registered over the whole chip by `calibrate.transform.register_channels` (the
vote of Gaussian 2D 2C's `_measure_registration`, any shift), then in situ with *learn map*.  With a
map that is right (*the model's*, *file*) the *move* corrects what moved since the beads, constant
or *over time*.  For 4Pi the same part takes three maps; `measure_channel_drift` is its *move over
time*.

### Batch

In situ runs per item and is cached (`depth.keep` / `kept`); the preflight is skipped in a batch,
so learning never waits on a question (a test checks it).  *learn from* sets the scope: *each file*
(default), *the first file* (its model used for the rest), *all files pooled* (molecules drawn
from every file -- several fields of one sample at one focus).  *auto* never takes an in situ model
learnt at another focus.

## The alternative to keep in view: the window as the fitter's editor

Instead of the parts in the fitter's form, the fitter's model part could show its few key rows and
a button *Open in PSF window...* (`Plugin.part_windows`) that opens the window with the same
settings and hands them back (`PluginPanel.take_settings` with a `reply`, as the bead window does),
or hands back a calibration file.  It would remove a second interface to uiPSF.  The parts are what
makes this possible, so they come first.  It is mostly wiring, so stage 2 builds both -- the rows in
the fitter's form and the button to the window -- and Jonas decides by using them which stays, or
whether both do.  Either way a chain and a batch run the settings, not the window.

## Pros and cons of the parts, and how the costs are paid

* **Pros**: one code path, name, tooltip and page per setting; a feature added to a part reaches
  every place that shows it (the move for 4Pi, the depth for 4Pi and DECODE, *from this movie*
  for DECODE's two channels, *microscope* as a start everywhere); tests written once.
* **Cons and their remedy**: settings paths change (`model.sample.focus` → `model.focus`, the move's
  fields into `channels`) -- old saved settings, chains, batches, presets read through
  `__post_init__` migrations and `plugins.MOVED`-style rename maps, with a test per old path; the
  tutorials and the panel pages' label tests follow in the same change; the assistant's tool
  descriptions follow (`run_plugin`'s settings); visibility stays one rule per part
  (`FormState.shown`), never per combination.

## Stages

1. **The parts, the fitters and the window's in situ**: optics, PSF model (with *microscope* →
   theory), depth (the focus as above, the depth guard), learn, channels (*from this movie*, *learn
   map* on), batch's *learn from*.  Spline 3D, Spline 3D 2C, the window's in situ (single and,
   new, dual).  Migrations.
2. **The window's bead and molecule-stack methods** on the same parts (mostly relabelling); the
   *Open in PSF window* button beside the fitter's rows, both tried.
3. **The extensions the parts bring**: the move and *over time* for 4Pi, the depth for 4Pi, *from
   this movie* for DECODE's two channels, field dependence in situ for two colours.
4. **4Pi in situ without a bead model** (the closed-form `fourpi.model_calibration` as a start):
   last.

## How it is measured

| question | data | passes when |
| --- | --- | --- |
| does a theory start match a bead start? | simulations from M2's bead pupil, learnt from theory | z bias < 20 nm, MAD within 10 % of the bead-started fit |
| on real data | M2 Sep / Jan / Aug, `insitu_learn_m2.py --start theory` against `--start beads` (gpuvm) | same, and Aug no longer kept at 1718 nm (the guard) |
| *phase and apodisation*, in the fitters or not | the same runs | it gains on held-out molecules without moving the focus beyond the guard |
| coverslip edge: refined or held | simulations with a known focus, M2 | the refinement is closer to the truth than the edge, else it is held |
| two colours with a wrong map | DECODE-Plex Fig. 2a (Zenodo 21348704, next-day beads); a simulation with a 30 px shift | the map from this movie within 0.1 px of the beads' after learning |
| a showcase | uiPSF Zenodo `2ch_smlm_nup96_WGA` | Nup96 rings' diameter and z as with beads |
| 4Pi (stage 4) | the 2017 4Pi tubulin | the phase and z of the bead model's fit |

## Open

* The depth guard's threshold: the edge's uncertainty from the molecules' count, or a fixed
  distance; set from the M2 runs.
* Whether *from the PSF alone* is ever needed.
* The batch scope *the first file*: whether the first is the right one, or a chosen file.
