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
pupil learnt that moved the focus more than 500 nm from the focus learnt alone
(`depth.DEPTH_GUARD_NM`, `judged`; the August failure: 692 → 1718 nm, kept; the right pupils on M2
moved it by 143--289 nm): the focus alone is kept, with a warning.  Measured against the focus
alone, not the edge, because that is what the M2 runs measured; a check of the edge after learning
is still open.  With no edge found (sparse near the
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

## Progress

* **2026-11-27, stage 1, first slice**: the focus and *learn* in one vocabulary in Spline 3D,
  Spline 3D 2C (its *learn* moved into the sample part), the DECODE training's and Simulate's
  sample part -- *focus* = coverslip edge / typed / beads' focus, *learn* = nothing / focus and blur
  / phase / phase and apodisation; a typed focus held while learning; the depth guard; settings
  saved in the old words read back to what they did (`depth.settled_focus`, `tests/test_focus_learn.py`).
* **2026-11-27, second slice**: the PSF window's in situ *hold* is *learn* in the same words (focus and
  blur / phase / phase and apodisation), a saved *hold* read back as its *learn*.
* **2026-11-27, third slice**: Spline 3D's *calibration* = *microscope* draws the microscope's optics in
  theory where it keeps no pupil (`microscope_psf.theory_pupil`: the entry's NA, its 3D lens's astigmatism,
  γ from the entry, a kept pupil or nominal), said in the menu, the form and the Run; *learn* starts from it.
* **2026-11-27, fourth slice**: the theory's astigmatism, where the entry gives none, from any earlier model of
  the microscope, camera and 3D lens state (`microscope_psf.earlier_astigmatism`), the PSF window's start too.
* **2026-11-27, fifth slice**: Spline 3D 2C's *map* = *from this movie*: Gaussian 2D 2C's measurement (each half
  fitted, the pairs registered over the chip, any shift), the calibration's halves and PSFs kept, the spline's
  *move* still on top.  Built as Gaussian's rather than with the start's PSFs per channel: the move, measured with
  the spline, takes the Gaussian's few-nm offset out; *learn map* (in situ) refines it further.

## Measured: the theory start on M2 (2026-11-27)

`benchmarks/uipsf/real/insitu_theory_m2.py`, rows in `benchmarks/uipsf/results/insitu_theory_m2.jsonl`; 10000 frames
each on gpuvm.  *learn* = *phase* from the bead calibration of the day against the same from M2's optics in theory
(no pupil kept: NA 1.43, the entry's astigmatism for the camera, γ 0.20 from the far-red pupil it keeps) and from
the theory with a nominal γ; z of the localizations matched to the bead start's, its median difference (bias) and
robust sd; blink-pair z (1 = honest errors); the share more than 50 nm below the coverslip.

| data | run | focus (edge) nm | bias, sd vs beads nm | blink z | below |
| --- | --- | --- | --- | --- | --- |
| Jan, CLTA AF647, Evolve | beads/phase | 431 (478) | -- | 1.17 | 5.1 % |
| | theory/phase | 453 (517) | +21, 8 | 1.25 | 2.9 % |
| | nominal/phase | 457 (508) | +25, 9 | 1.26 | 3.0 % |
| | theory, nothing learnt | 498 (edge) | +54, 8 | 1.37 | 2.1 % |
| Sep, CLC Cy3B 590 nm | beads/phase | 568 (640) | -- | 1.34 | 3.6 % |
| | theory/phase | **163** (590), phase refused (-16 nats) | **-245**, 14 | 1.36 | **21.5 %** |
| | nominal/phase | **86** (568) | **-303**, 22 | 1.33 | **32 %** |
| | theory, nothing learnt | 590 (edge) | +64, 32 | 1.64 | 2.7 % |
| Aug, sun1 A647 | beads/phase | 654 (587), **the guard refused** a +2925 nm move | -- | 0.59 | 7.2 % |
| | theory/phase | 521 (604), phase refused (-0.7 nats) | -100, 20 | 0.76 | 7.9 % |
| | theory, nothing learnt | 604 (edge) | -43, 23 | 0.87 | 10 % |

* **Jan passes, just**: 21 nm against the 20 nm asked, the blink pairs 7 % worse than the bead start's.
* **Sep fails, and the guard does not see it**: the first stage, the focus and the blur learnt with the theory's
  pupil held, moved the focus from the edge's 590 to 163 nm (a theory pupil lacks the objective's spherical
  aberration, which the focus then makes up for), the phase learnt from there was refused, and the fit kept the bad
  focus.  The guard compares against that first stage, so it passed; a fifth of the localizations below the
  coverslip shows it at once.
* **Aug** has no reference: the bead start itself was refused by the guard (the phase moved the focus by 2.9 µm),
  as the plan's *everything* was in November.  The theory start lands 100 nm from the bead start's focus alone.
* **What it says for the design**: with a theory start the first stage (every aberration held) is not a safe
  reference -- the held pupil is wrong -- so (1) the phase should start from the edge's focus and fall back to the
  edge, not to a focus learnt with a wrong pupil; (2) the guard should also check the edge: a model that puts more
  than a share of the localizations below the coverslip (Sep 22--32 % against 3--7 % for every model kept) is
  refused.  Not built yet.

## Open

* The depth guard's threshold: the edge's uncertainty from the molecules' count, or a fixed
  distance; set from the M2 runs.
* Whether *from the PSF alone* is ever needed.
* The batch scope *the first file*: whether the first is the right one, or a chosen file.
