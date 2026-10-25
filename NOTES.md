# Design notes

Decisions that are not obvious from the code, and open questions.  Everything
here was decided deliberately; if something looks wrong, read this first.

## Scope

A stripped-down version of SMAP's `fit_fastsimple` workflow: TIFF -> photons ->
filter -> peak finding -> ROI cutting -> MLE fit -> HDF5.  Single channel,
Poisson noise, with the camera's read noise as one variance added to data and
model (see "Read noise" below).  Fit models: `gauss_free`, `gauss_xy`, `cspline`.

Deliberately **not** ported: sCMOS per-pixel variance (one variance for the
chip is used instead), the Anscombe transform
and probability cutoff, background estimation, minimum-distance filtering,
mean-PSF / MIP-PSF detection filters, ROI masks, Zernike and
astigmatic-Gaussian z models, multiple z start values.  Two-channel global
fitting of a split frame came later, and has sections of its own ("Two colours
in 2D" and the dual-colour calibration in `docs/dual_color_calibration.md`).

## Departures from SMAP, and why

* **Read noise.**  The likelihood is Poisson; an sCMOS adds Gaussian read noise
  of about an electron per pixel (0.7 to 1.4, pixel by pixel).  Its variance is
  added to the data and taken up by the fitted background (Huang et al. 2013,
  with one variance for the chip rather than a map), then taken back off the
  background: `CameraMetadata.readout_variance`, `pipeline.with_readout`.
  Default 1 e- without EM gain, 0 with it.  At half a photon of background the
  Poisson likelihood alone read the background as 0.74; with the variance,
  0.53.  It also floors each pixel's weight, 1 / (model + variance), which is
  what keeps a spline's near-zero tail from dominating a fit of a bright spot
  at a low background.

* **No module chain.**  SMAP's WorkflowModules exist to serve a GUI (parameter
  synchronisation, per-frame data packets, global-variable accumulators).  Here
  the stages are plain functions over blocks of frames, and the only state that
  survives between blocks -- the ROI buffer -- lives in `LocalizationEngine`.
* **Chunked, not per-frame.**  Every stage takes `(n, y, x)`.  This is what makes
  the vectorised and threaded implementations possible.
* **No x/y clamp in the fitter.**  SMAP's single-channel kernels clamp x and y to
  the central half of the ROI; the multi-channel fitter dropped that and we
  follow it.  A runaway fit is now visible instead of silently parked on a
  boundary -- reject them with `FitSettings.max_fit_distance` if wanted.
* **Border candidates are dropped, not clamped.**  SMAP shifts a ROI inward and
  fits it anyway, which biases those localizations because the emitter is no
  longer near the ROI centre.
* **No background offset before filtering.**  In SMAP that value (the minimum of
  the image's first column) only compensated the zero padding of the
  convolution at the borders; we pad by edge replication instead.  Verified: the
  interior is unaffected to 2e-5, only a ~3 px rim differs.
* **EM excess noise lives outside the fitter.**  Photons are divided by 2 before
  fitting and photons/background scaled back afterwards, so the fitter assumes
  pure Poisson statistics and knows nothing about cameras.
* **The fit measures pixels; its table is in nm.**  nm is derived at the very
  end by `to_nm()`, and the fit's output keeps the nm columns only, with the
  pixel size in the metadata (`pixelsize_nm`, `pixelsize_nm_xy`), so a pixel
  correction later is a division, not a refit.  `locs.in_pixels` gives a
  column back in camera pixels -- the channel registration is the one reader
  (2026-10: the pixel columns had been kept beside the nm ones, 24 bytes a
  localization for a 2D fit, 20 for 3D, read by nothing else).  The
  candidate positions `peak_x_pix` / `peak_y_pix` stay, as the record of
  where the peak finder found the spot on the chip.  A file saved in pixels
  keeps its own and is given nm columns on load (`io.formats.with_nm`).
  Everything downstream reads nm only.  Without a pixel size the nm columns
  use `DEFAULT_PIXELSIZE_NM` and the table records `pixelsize_assumed`
  (2026-10: the *units* setting, and a fit refusing a stack with no pixel
  size, are gone).  `z_nm` is the exception: it comes from the
  calibration's `dz` and has no pixel equivalent.
* **The background starts at the ROI's border, and the Hessian is the expected
  information.**  SMAP starts from the ROI's minimum and uses the observed
  information `d / mu^2`; together they left a spline fit at a few background
  photons per pixel stuck at no background -- see "A fit that starts at no
  background".  The maximum is the same, and at ordinary backgrounds so are the
  fits.
* **No mirroring outside the loader.**  SMAP flips bead images taken with EM
  gain before it builds the spline, and flips the data the same way; here the
  data are never flipped, so a `_3Dcal.mat` of flipped beads is turned into the
  camera's orientation when it is loaded (`io/calibration._unmirror_x`:
  S'(u) = S(nx - u), the x cells reversed and each cubic re-expanded in 1 - t,
  an odd nx one cell wider), and the ROIs, the fitter and the simulator know
  nothing of mirroring.  It used to flip every ROI instead; the converted model
  equals that to float32 rounding, and fits agree to the size of a one-ulp
  change in the coefficients (`tests/test_unmirror.py`).  Whether SMAP flipped
  is read from `SXY.EMon`, its `isem`, not from `parameters.emmirror`, which is
  a popup (1 from metadata, 2 no mirror, 3 mirror) and was once read as a bool,
  flipping files saved with "no mirror".  That reading is from SMAP's code and
  waits for a real file (`SMAPPY_TEST_CAL`, `tests/test_calibration.py`).

## Rendering

The renderer is deliberately two stages: accumulation into float planes, and a
display step (normalise, gamma, LUT) that turns them into RGB.  Changing the
contrast or the colour map therefore never re-renders, which is what a viewer
needs.  SMAP splits the same way (`renderSMAP` / `drawerSMAP`).

* **One kernel for all three modes.**  The kernel is the *pixel integral* of the
  Gaussian, `erf(right) - erf(left)`, not a point sample of it.  A rendered
  pixel is an integral over its area, and point sampling breaks down exactly
  where sigma approaches one pixel -- which is where the renderer spends most of
  its time.  It also makes the histogram the sigma -> 0 limit of the Gaussian
  rather than a separate model, and both are tested against each other.
* **No template.**  SMAP looks up a 601x601 Gaussian template with
  nearest-neighbour sampling.  The kernel is separable, so a (2d+1)^2 ROI costs
  2(2d+2) `erf` calls plus the outer product, which dominates either way: the
  template buys nothing and costs quantisation error.
* **Normalised by its own sum, not by an erf truncation correction.**  Every
  localization then contributes exactly N, whatever its sigma or subpixel
  position, so histogram and Gaussian images carry the same total intensity.
  The kernel is built over the full ROI even where the image clips it, so a
  localization at the border loses the outside part instead of having it
  redistributed inwards -- and the result does not depend on thread boundaries.
* **Field colouring keeps four planes**, the three colour-weighted ones plus the
  plain weight, so both composites come from one render and the switch between
  them is a re-display.  `sum` is SMAP's: display the colour planes directly, so
  a red and a cyan localization in one saturated pixel **add to white**, as
  additive colour should.  `hue` divides by the weight first, so that pixel
  stays a mid grey at full brightness and the hue never drifts towards white
  with density.  Below saturation the two are algebraically identical -- `hue`
  divides by the weight and multiplies it back in -- so the difference is
  confined to the brightest pixels: on the clathrin data at p = 2.5, 4-8% of
  pixels differ by more than 0.02, and those are the dense cores and the
  fiducials.  Which is right depends on what the image is for: `sum` reads as
  brightness-is-density with colour bleaching out where it is densest, `hue`
  keeps z legible in exactly those cores.  `hue` is the default; the viewer's
  "additive" box switches.
* **Contrast is SMAP's dynamic contrast**, parameterised the way SMAP's
  `imax_min` is: one number p saturates 10^-p of the pixels.  The value
  distribution of a superresolution image is extremely heavy-tailed -- on real
  data a fiducial bead outweighs the structure by three orders of magnitude --
  so an absolute maximum does not transfer between datasets and the useful
  range of p spans decades, which is why the slider moves the exponent.  p = 3
  is the default; SMAP's own scripts use 3.5, which is too dark for the
  clathrin dataset.
* **The pixel grid is stated once.**  A position lands in pixel
  `floor((x - x0) / pixelsize)`.  SMAP rounds and shifts the range by half a
  pixel first, which is the same grid expressed twice.
* Not ported: the transparency/occlusion modes (`gaussrenderT`,
  `gaussrenderTx`), layers and compositing, the DL/tiff modes, grouped
  localizations, `normalizeFoV`, `remout`.  The sigma policy (`gaussfac`,
  `mingaussnm`, `mingausspix`) is kept, in `SigmaSettings`, with SMAP's
  absolute 400 nm cap replaced by one at 10x the median, which means the same
  in any unit, because it is what makes precision-weighted images readable.

### Any column against any other

SMAP's `VersatileRenderer` is a plugin that copies the table, writes
`x = field1/pixelsize1` and `y = field2/pixelsize2`, renders with
`sr_pixrec = 1` and labels the axes -- photons against frame, a fit parameter
against another, with the layers' own LUTs.  Here it is not a plugin but the
renderer itself: `RenderAxes` on `RenderSettings` says which columns the two
(or three) axes are and what to divide each by, and the default -- the table's
own positions, unscaled -- is the picture everybody means and costs nothing.

* **The division is the whole mechanism.**  Coordinates are the column over
  that axis's scale, so the grid stays *square in render units* and the
  anisotropy lives in the two scales.  The field of view, the zoom, the ROI
  shapes, the projection and the GPU kernels are then unchanged and never have
  to learn about aspect ratios.  It is SMAP's trick, kept.
* **One set of axes for the picture, not one per layer**, since the layers are
  composited onto one grid.  They live on each layer's `RenderSettings` all
  the same -- as `white_background` does -- so a render outside a session
  needs nothing else, and the GUI writes all the layers at once.
* **A width per axis.**  The kernel always took `sigma_x` and `sigma_y`; only
  the Python wrapper tied them together.  The localization precision blurs an
  axis that *is* a position and nothing else -- a photon count has no
  precision -- so each axis falls through to an explicit width, zero by
  default, which bins along that axis.  Mode `gauss` is the explicit pair
  everywhere.  An unset `sigma_y` follows x where the two axes are the same
  quantity and is zero where they are not: 10 nm says nothing about photons.
* **A position axis keeps its scale of 1.**  It could be fitted like any
  other, and then an ROI drawn on a picture of x against z would be in tenths
  of a nanometre and a site somewhere else entirely.  A picture of x against z
  is still a picture of a place and the rest of the program may read it as one.
* **The index is not used** while the axes are custom: it answers questions
  about the table's positions, which is not where the picture is looking, so
  the whole filtered table is rendered.  A versatile render is one picture of
  everything rather than a pan over a field, which is the pass SMAP pays too.
* **A scale bar per axis, in that axis's own quantity** ("20000 frame", "2000
  photons"), because one bar in nanometres would be a lie about both.  In the
  3D view it is the tripod that carries them: a rotated view mixes the axes,
  so a bar across the screen is not a length in any of them while an arm is
  its own axis, and each arm is a round number of its own units.
* **A plot hangs the other way up.**  y increases upwards once the axes are
  custom, as it does in SMAP (`axis xy`); a photon count growing downwards
  reads as a lie.  A picture of a place keeps the image convention.
* **Choosing the fields fits them.**  A frame number is in the tens of
  thousands and at scale 1 the first render would be one bright pixel, so
  picking a column scales it -- a round 1/2/5 -- and frames the view on its
  1-99 %, padded, because a ring's brightest arcs sit just outside its own
  99 %.  SMAP does the same from its field callback, with 0.1/99.9 %.
* Not taken from the SMAP plugin: the `plot line` overlay (mean, median, std
  of y binned along x).  It belongs to a measurement rather than to a
  renderer, and there is no line to draw on a 3D box.
* A rotated view has room for one width only, so `projected_settings` folds
  the mapping into the projection: positions at one common scale keep the
  precision with the scale folded into `SigmaSettings`, and mixed axes fall
  back to the mean of the two explicit widths.  The ROI manager's site images
  ignore the axes entirely -- a site is a place.

**Filtering** (`LocFilter`) caches one boolean array per field and recomputes
only the field that changed, which is what SMAP does and the reason interactive
filtering is possible at all; the displayed set is an AND over a handful of
arrays.  `indices` hands the renderer a compacted selection so filtering indexes
three or four columns instead of copying the table.  A range excludes NaN,
because a comparison against NaN is false -- a localization with no z should not
appear in a z slab.  Not ported: ROI/polygon masks and the grouped-localization
filters; `set_mask` takes an arbitrary boolean array under a name, which is
where those, or the viewer's "inside the field of view", would plug in.

Changing the LUT is free -- it is applied to the finished image -- *except* when
colouring by a field, where the colour is baked into the accumulation and the
image has to be rendered again.  That is why `render_locs` takes the display
settings too.

**The viewer** is split into a `ViewState` -- table, spatial index, filter,
settings, and a field of view in, an image out -- and a matplotlib front end, so
another toolkit replaces only the second half.  Three things make it usable on
millions of localizations:

* it renders at *display* resolution, so the pixel size follows from the window
  and the cost of a render is bounded by the canvas rather than by the zoom;
* a gesture only moves the axes limits and matplotlib rescales the image it
  already has; a timer re-renders once the gesture has been still for 150 ms.
  Interaction stays smooth even where a render takes 130 ms;
* the `SpatialIndex` answers what is on screen, so a zoomed view never walks the
  whole table.  Queries are padded by `roi_sigma * max_sigma` so a blob whose
  centre is off screen still contributes its tail -- the index may return extra
  candidates but never loses one, and that is what the tests check.

The filter controls cover only the fields that get used (localization precision,
z or PSF size, `logl_rel`, frame) rather than every column; a slider at either
end means "no bound", so opening a file never silently drops localizations.

Threading is by contiguous row bands: a localization contributes to whichever
bands its kernel reaches and each band writes only its own rows, so there are no
shared pixels, no locks and no per-thread image copies.

## Grouping

A single emitter is localized in many consecutive frames, so a raw table counts
one blink many times.  `group.py` links them (SMAP's `connectsingle2c.c`, greedy
frame-to-frame, first match inside a **box** of half-width `dx` wins, up to `dt`
dark frames) and combines each run into one row using the per-column rules from
`Grouper.m`.

* **The linking bug is fixed.**  The original tested `list[thisentry] > 0`
  before the bounds check, so with the last localization already assigned it
  read -- and could then write -- one past the end of the array.  `Grouper.m`
  carries two workarounds for the unassigned first/last entries ("FIX
  connectsingle doesnt assign last loc. Fix later!"); with the tests in the
  right order every localization gets a group and the workarounds are gone.
* **Blocks are linked separately** rather than by SMAP's trick of zeroing the
  frame at each boundary, which leaves the array no longer sorted by frame and
  lets a group start in one block and search into the next.
* **Linking by precision** is SMAP's `locprec` mode (`connectsinglesigma.c`),
  `GroupSettings(link="precision")`: a localization joins when its distance
  from the running position satisfies `r^2 < k^2 (s_h^2 + s^2)`, held between
  `dx_min` and `dx_max` (10 and 150 nm), with `k = 2.5` as in SMAP.  Two
  departures.  SMAP takes *the seed's* precision for both terms -- `lph` and
  `dxsigma[thisentry]` are the same localization's, and the test one's never
  enters -- so a bright seed held a dim trace to its own precision; here `s`
  is the candidate's and `s_h` the running position's, carried through the
  walk's own mean as `(s_h^2 + s^2)/4`.  And the bounds clamp the distance,
  not each term: SMAP floors every `2.5 s` at `dx_min` first, so its smallest
  radius was `sqrt(2) dx_min`.  With the two precisions right,
  `r^2/(s_h^2 + s^2)` is chi-squared with two degrees of freedom, so `k = 2.5`
  keeps 95.6% of a blink's links -- and cuts the other 4.4%, which a fixed box
  wide enough does not.  A localization with no precision (NaN) links to
  `dx_max`; a table with no precision column falls back to the box, with a
  warning, because every layer merges blinks by default.
* **The error of a summed quantity adds in quadrature.**  This is the rule that
  keeps shot noise consistent: with `e_i = sqrt(N_i)`, `sqrt(sum(e_i^2))` is
  `sqrt(sum(N_i))` exactly -- the shot noise of the summed photons.  SMAP
  applies its `*err -> 1/sqrt(sum(1/e^2))` rule to `photons_err` and
  `background_err` instead, which understates a four-member group by six times.
  Background is summed alongside photons, so the pair stays consistent: a group
  is one measurement over several exposures, with total signal and total
  background.  (Note this makes a grouped `background` n times the per-frame
  level; a filter on it means something different than on ungrouped data.)
* **`logl` is dropped, `logl_rel` is kept.**  A group's raw log-likelihood is a
  sum over its members' fits, so no reduction of it is meaningful across groups
  of different size; the per-pixel `logl_rel`, which is what the filter uses,
  takes the maximum as in SMAP.
* **Each coordinate is weighted by its own error** -- `x_nm` by `x_err_nm`,
  `y_nm` by `y_err_nm`, `z_nm` by `z_err_nm` (`WEIGHT_FOR`).  SMAP weights all
  three with the pooled `locprecnm` because that is the one weight it computes,
  and flags it as a shortcut.  The pooled weight is the correct inverse-variance
  estimate only when the errors are equal, and under astigmatism they are not:
  on the clathrin data `x_err/y_err` runs from 0.42 to 3.15 across
  localizations, since that divergence is exactly what encodes z.  The
  difference in the grouped position is a median 0.11 nm laterally but reaches
  14-21 nm, and in z a median 0.41 nm reaching 487 nm -- far above the
  localization precision, so it is not cosmetic.

Grouped and ungrouped are **two tables kept side by side**, each with its own
spatial index (`layer_state.LocSet`), so that a switch is a single assignment
and only the first one pays for the linking.  There is one of each per
session (`layer_state.Tables`), shared by every layer; the grouped table is a
cache keyed by the rows it was linked from and the link settings, which are
the session's, so it is linked again when either moves and there is no
"stale" flag to keep.  A layer has **one** set of bounds, applied to the
table it shows (October 2026, review section 4.2).  Each table used to carry
a filter of its own, on the argument that a precision cut for single
localizations is the wrong one for groups -- but the bounds were then copied
between the two filters in five places, and each was a place to forget: a
grouped table came out of a correction unfiltered.  So the same bound now
keeps different localizations on the two tables (on the grouped one,
`xy_err_nm <= 25` is the combined precision), which is intended: it is the
user's statement of what is good enough, applied to what is shown.

**A plugin measures what the layer shows** (October 2026, B3).  Until then
`ctx.locs` was always the ungrouped table and only Statistics, DBSCAN, NPC
and the image export read the grouped one, so a measurement on a grouped
layer -- the default -- silently disagreed with the picture.  Now the
context hands out the shown table, and a plugin that needs the
localizations says so (`Plugin.grouping = "ungrouped"`): every one that
changes the table, since no table can be made back from blinks (such a
result is refused, `refuse_grouped_table`, rather than allowed to replace
the localizations), and the few that measure frames -- NeNA, the ground
truth match, the bead pairs of the channel registration, the per-frame
count of Image Tags.  Of the shipped measurements only Line Profile moved.

**Linking writes back.**  `group.attach` puts `group_id` and `n_in_group` onto
the ungrouped table as well.  The on-time is a property of a localization --
SMAP's `numberInGroup` is a field of the ungrouped table and its plugins filter
on it -- and the id is the index a later `bincount` needs, which is what makes
a column computed *after* the linking combinable without linking again.  The
walk is the expensive half and it has already been done; keeping its answer
costs four bytes a localization.  Not on a table that is still growing: a block
arriving from the fitter would not carry those columns, and `extend` refuses a
block whose columns do not match.

## A computed field is a recipe

`mathparse.py` and the `Math Parser` plugin let a user define a column from an
expression in the others -- `on_time_ms = n_in_group * 20`.  What is stored is
not only the numbers but the **expression**, in `metadata["derived"]`, and that
is what makes the field mean the same thing on a table that is derived later.

The grouped table is the case that forces the decision, and there is no single
right answer, so the recipe carries which one was meant:

* **recompute** -- evaluate the expression again on the grouped table.  This is
  what an expression in `n_in_group` says: it is a statement about a blink, and
  averaging a per-localization copy of it would be meaningless.
* **a combine rule** (`mean`, `sum`, `min`, `max`, `any`, `all`) -- reduce the
  localizations' values as grouping reduces every other column.  This is what a
  per-localization measurement says.

`group.combine` honours both: a derived column is excluded from the per-column
rules (so nothing is silently averaged), reduced by its rule, or recomputed,
and a field defined in `n_in_group` gets its first value from the grouping that
creates it.  A field computed over a *selection* is not a function of the table
at all, so its recipe is marked and never recomputed -- only reduced.

The expression is parsed and walked rather than `eval`ed (`mathparse.parse`).
It travels in a file and in a workspace, so it has to be arithmetic over
columns and not a program; the walk also yields the names it reads, which is
what turns a missing column into a message naming what the table does have,
and lets the three numpy traps -- `and`/`or` on a column, `0 < x < 5`, and
`a < 25 & b > 100`, whose precedence MATLAB has the other way round -- be
errors rather than plausible wrong numbers.

## Conventions

Fixed once, in `io/calibration.py`, so nothing downstream deals with MATLAB's
layout:

* images `(y, x)` C-contiguous; spline coefficients `(64, nz, ny, nx)`, x fastest
* fit parameters `(x, y, photons, background, [z | sigma | sigma_x, sigma_y])`
* MATLAB's first spatial axis is the image **row**, which is why `P(:,1)` is y
  there; both the spatial axes and the 64 monomial indices are swapped on load

## Verification that is not circular

* the loaded spline coefficients reproduce the bead stack stored beside them in
  the same file (correlation 0.9999, vs 0.82/0.82/0.83 for transposed,
  x-mirrored and z-flipped controls) -- `scripts/check_calibration.py`, since
  removed (`git show ec09f06:scripts/check_calibration.py`)
* the Gaussian models are checked against independently simulated data: unbiased
  to <0.01 px, scatter matches the returned CRLB
* the mirror flip was confirmed on real data: the flipped orientation has the
  better log-likelihood for 65.8% of ROIs (median dlogL +2.82).  That was the
  ROI flip; the conversion on load that replaced it is the same model, and
  `scripts/check_fit.py` (removed; `git show ec09f06:scripts/check_fit.py`) kept
  the x-flipped fit as a control
* threaded output is bit-identical to serial, at every stage
* the render kernel's second moment is `sigma^2 + 1/12`, the variance the pixel
  itself contributes -- a point-sampled kernel would give `sigma^2`
* the C++ renderer matches a numpy reference to float32 precision in all three
  modes, coloured and not
* the whole renderer runs on real data: 844k localizations from the clathrin
  dataset, filtered to 575k, render to structure that looks like clathrin pits,
  and zooming to a 150 nm field resolves the individual localizations as
  ellipses of their own precision
* the spatial index is checked against brute force on random rectangles: it may
  return extra candidates, never lose one
* grouping on the clathrin dataset: 844k -> 377k rows, photon sum conserved
  exactly, `frame` is the group start, and the precision of every multi-member
  group is better than its best member (median 7.0 -> 4.5 nm)
* the summed-error rule is checked against the shot-noise identity: for
  `e_i = sqrt(N_i)` the grouped error is `sqrt(sum(N_i))` to float precision

### Crowding compresses z

Simulated beads -> calibration -> Spline 3D on simulated astigmatic frames gave
fitted z against true on a slope of 0.93 (0.95 through Spline 3D 2C), with no
offset.  It looked like a scale error in the calibration.  It was not, and
measuring one thing at a time is what showed it:

| what was fitted | slope |
| --- | --- |
| the calibration's own beads (`fit_bead_diagnostics`) | 1.002 |
| noise-free isolated spots of the same PSF, z smoothing 0 / 20 / 60 nm | 0.988 / 1.001 / 1.005 |
| the same, Poisson noise, gamma photons around 2000, one start at z = 0 | 0.99 |
| the frames' own ROIs cut at the true positions, all spots | 0.907 |
| the same, only spots with no emitter on within 1 um in that frame | 0.994 |
| the whole pipeline, all matched localizations | 0.933 |
| the whole pipeline, those with no neighbour within 600 nm / 1 um | 0.974 / 0.996 |
| Spline 3D 2C, all / no neighbour within 1 um | 0.951 / 0.999 |

So the calibration, its z smoothing, the registration and the fitter are
right, and the compression was entirely the spots with a neighbour.  A second
emitter inside the 13 px ROI is light the one-emitter model has to account
for, and an astigmatic model accounts for extra light around a spot as a
rounder spot -- z nearer focus.  A neighbour 2-4 px away gave a slope of 0.5,
6-10 px still 0.89.  `camera_frames` promised frames that fit cleanly and was
sparse on average, seven spots on 100 x 100 pixels, but the structure's
emitters cluster, and a quarter of the blinks had another emitter on within a
micrometre.

The simulators first dropped both blinks of such a pair, as `simulate` does
for the localizations it pretends to have fitted.  Both paths gave 1.002.
Since the simulation model was unified (below) the camera frames keep every
spot -- crowding is what a fitter is tested *on*, not something a simulator
should hide -- and the truth says instead how far each spot's nearest
neighbour in its frame was (`neighbour_nm`).  The tests compare only spots
with none within `ISOLATED_NM` (1 um: the ROI's half-width plus a defocused
neighbour's spot) and hold the slope to 0.02 there.

For real data the lesson is that this is a density effect, not a calibration
one: z from a crowded acquisition is pulled towards focus by the overlapping
fits, and they are the ones with a poor `logl_rel` (median -1.6 against -0.5
for isolated spots; keeping the best 70 % by `logl_rel` gave 0.996 on the
unfiltered frames).  Filtering on it is the remedy, as it is in SMAP.

### A fit that starts at no background

On an sCMOS a background of a few photons per pixel is realistic -- a dim
sample, TIRF, a short exposure -- and Spline 3D failed there.  At 5 photons/px
and 2000 photons, 38 % of the pipeline's fits ended with a background of 0.01,
the clamp's floor, their photons inflated and z frozen at the start value: the
slope of fitted z against true was 0.53, the good fits' `logl_rel` median
-1.37 instead of -0.5, and a `logl_rel` cut kept half of the isolated spots.
Below 2 photons/px the fits went to NaN.  The Gaussian fitter never showed it:
a Gaussian's tails never reach zero.

Two things combined, measured on isolated synthetic spots (Poisson plus 1.5 e-
read noise, 200-5000 photons, 0.5-20 photons/px background):

* **The start.**  The background started at the ROI's minimum pixel, which at a
  few photons is 0, so the floor.  Starting at the mean of the border pixels
  fixed 1000-photon spots, but bright ones still stalled (4-40 % at 5000
  photons below 2 photons/px).  Starting *higher* than the data says -- two or
  three times the border -- was worse, up to 75 % stalled: a bright spot's excess
  over a guessed background is explained by lowering the background.
* **The Hessian.**  With the observed information, a pixel's curvature is
  `d / mu^2`, and once the background nears the floor `mu` is tiny where the
  spot is not, the curvature explodes and the Newton step for the background
  shrinks to about `mu`.  From 0.01 it could at most double per iteration, and
  LM's damping and step halving took the rest.  The expected information,
  `1 / mu`, steps by about `d - mu` and climbs straight back.  It has the same
  maximum -- the gradient is unchanged -- and is guarded at `mu >= 1e-3`
  because a spline's tail can dip below zero.

With both, no fit stalls from 0.5 to 20 photons/px except 3 % at 5000 photons
and 0.5 photons/px, and those may be honest estimates of almost nothing.
Through the pipeline at 5 photons/px: none at the floor, slope 0.997 for
isolated spots, `logl_rel` median -0.55; at 1 photon/px 0.992.  At 20
photons/px the fits agree with the old ones to 1 nm in z and 5e-6 in photons,
at the same speed, and at low background they are three times faster, since
they converge in 9 iterations instead of running out at 50.

The two-colour global fit takes the same Hessian and the same start per
channel, for consistency rather than because it failed: at 3 photons/px per
half it gave a slope of 1.004 before and 1.000 after.  Its empty-channel
pairs had come back NaN only because the model went non-positive; the guard
ended that, so a channel with no light now marks the pair NaN explicitly.
`test_a_spline_fit_at_a_few_background_photons_does_not_stall_at_no_background`
failed on the old fitter with 43 % stalled.

## Performance

Measurements and their investigations: `docs/performance.md`.

## Two windows

The image has a window to itself and the controls a second, small one.  The
image window is then free to be as large as the screen allows, and since the
rendered pixel size follows the canvas rather than the zoom, enlarging it
renders a *finer* image rather than scaling one up -- on the clathrin file, the
same view went from 60.3 to 35.5 nm per rendered pixel simply by not sharing
the window with a control strip.  The controls keep their size instead of
stretching across a wide window, and their rows are laid out in inches with the
window sized to fit, so the spacing does not change with the number of filter
fields or colour choices.  Closing the image closes both; closing the controls
leaves the image alone.

A side effect worth having: the text boxes are no longer on the image canvas,
so a click on the image cannot reach them at all (see below).

The image window may be any shape.  The axes is never shrunk to the proportions
of the view -- `set_aspect("equal", adjustable="datalim")` -- so it is the view
that widens to the proportions of the window: a wide window shows more x rather
than growing bands beside the image.  One pixel size still serves both axes,
which is not negotiable; an SMLM image with non-square pixels would be a lie.
`FieldOfView.fit` already worked this way (it sizes a render from the canvas
and picks the pixel size from the larger span), so the two agree by
construction: on a 1300x600 window the render is 1248x558 pixels of 46.5 nm,
covering 58.0 x 26.0 um.  Empty area beside square data stays black, as part of
the image, rather than grey figure background.

Filter bounds now have defaults -- `xy_err_nm` below 25 nm, `logl_rel`
above -1.5, `z_nm` within +-500 -- because those are right almost every time
and an unfiltered first view is not what anyone wants to look at.  Every one of
them throws localizations away, so each is written into its box: the earlier
rule that opening a file must never *silently* drop localizations still holds,
it is the silence that mattered, not the absence of a default.  They apply once
per layer (the grouped table shares them), a bound already set is never
overridden, and a bound the user clears stays cleared.  Only fields with a
fixed unit get one, since 25 nm means nothing in pixels.

## Interaction latency

The matplotlib viewer's notes on drag latency (a deferred re-render, a patched
`TextBox`) went with it in 2026-10: the Qt GUI renders on its own thread and
never had either problem.

## Online analysis

Fitting an acquisition while the microscope writes it and watching the image
build up (`smappy.live`, `smappy.io.watch`).

**The bridge between the fitter and the viewer is a queue in one process, not
the HDF5 file.**  The file is still written, and it is still the result -- but
using it as the transport would mean SWMR mode (every column declared before
the first block knows what they are, `refresh()` on every read) and re-reading
and decompressing a tail that is already sitting in memory.  The engine already
hands finished blocks to a `sink`; the live path adds a second sink, which is a
queue.  A separate process only earns that cost when the fitter and the viewer
are on different machines, and then the change is additive: same
`ViewState.append`, a different feeder.

**The fitter runs in a thread, and the GIL is not in the way.**  Detection,
fitting, rendering and grouping all release it in the C++ (`csrc/`), and frame
reading already overlaps through `prefetch`, so the window stays responsive
while a block is being fitted.  Only the viewer's own thread touches the view,
on a timer, so nothing in the state needs a lock: a block of localizations is
handed over and never looked at again by the thread that made it.

**An update must not change a setting.**  This is structural, not a discipline:
appending *extends* the table, each filter's cached mask and the spatial index,
and re-derives none of them.  What that took:

* `Localizations.extend` grows each column in a buffer with 50% headroom and
  exposes an exact-length view, so 500 updates over an hour do not each copy
  the whole table.
* `LocFilter.append` evaluates the bounds already set on the new rows only.  A
  bound therefore keeps meaning exactly what was typed; nothing re-quantiles
  the larger table.  A hand-set mask has no rule to extend it, so its new rows
  are excluded until it is set again -- visible, rather than silently included.
* `GrowingIndex` gives each appended block its own `SpatialIndex` and answers a
  query as the union.  Segments are merged occasionally (the tail when there
  are too many of them, everything when the tail reaches the size of the head),
  so the merge work is O(N log N) over an acquisition rather than a 0.3 s
  argsort per update.  Segments deliberately do not share a grid: a query is
  answered against each segment's own origin, so a block landing outside the
  area seen so far needs no rebuild.
* The index takes an `extent` -- the camera's field of view -- so `bounds`, and
  with it the viewer's full view, is the same before the first localization and
  after the last.  The image does not rescale under the user.
* The median localization precision, which caps the rendering sigma, is
  recomputed once the table has grown by a fifth rather than on every block.

**Colouring by a field is a re-render, and its range is explicit.**  Each
localization contributes its own colour, so the LUT is applied per localization
and baked into the accumulation -- unlike contrast and gamma, which re-display
what is already there.  The range is never taken from the data at render time,
which `render_locs` would otherwise do: the visible set changes with every zoom,
every filter and, online, every block, so an implicit range would give the same
z a different colour each time.  Choosing a field therefore fills the range in
from that field's 1--99% quantiles and leaves it there; a range typed for one
field is not carried to the next, since a z slab means nothing for a frame
number.  The LUT follows the choice (`hot` for intensity, `turbo` for a field),
because a coded field cannot be read off a brightness ramp.

**Grouping is the one thing that cannot be extended.**  It links localizations
across frames, so after an append the grouped table no longer matches the
rows it was linked from (`Tables.is_current`) and it is linked again the next
time it is asked for; the bounds are the layer's and stay.  The title says
"(stale)" rather than the view quietly lagging.

**Knowing when the acquisition ended.**  Micro-Manager does not say, and the
declared frame count is what was planned.  So: read what is there, wait, look
again, and stop when nothing new has appeared for `timeout` seconds.  The
newest page of the file being written is held back until another follows it
(its pixels may still be arriving) and read on the way out, once nothing can
follow it.  Every read is retried rather than trusted -- a file caught
mid-write raises instead of returning half an image -- and the file list is
re-globbed on each poll, so the `_1.ome.tif` that appears when the current file
fills up is picked up without reopening anything.

**Unless somebody says when it ended.**  A program that drives the microscope
(MicroClaw) knows when the writer has finished, and then a pause -- a refocus,
a laser change, a time-lapse interval -- must not end the fit: `finished`, an
event handed to the watch, replaces the timeout altogether, and once it is set
one more pass reads everything and the stream ends.  A longer timeout would
only move the pause that ends the fit too early, which is why MicroClaw's
protocol (its `design/84`) asks for the event and not for a number.  The
Fit plugin takes it from `Context.writer_finished`, and `Context.stop` ends a
run early with its file closed and nothing finished (no drift correction of
half an acquisition; the caller allows ten seconds).
`smappy.gui.live_session` is the GUI opened on such a run: the same session
and windows, the run started without its preflight question, and the fit's
file added to `Session.protected`, because the caller hands it on with a
digest while the window stays open and a Ctrl+S would change what that digest
vouches for.

**A live fit fits each read at once.**  The engine's buffer holds about 9000
ROIs (15000 at a 10 px box); the live fit no longer waits for it, nor for a
timer, but flushes after every block (`drive(flush_seconds=0)`), and the
NDTiff watch looks again at once after reading new frames instead of sleeping
its poll (`LIVE_POLL_SECONDS`, 0.05 s, applies only when nothing arrived).
Measured on a 4-core container (2026-10-13, an NDTiff written at 100 frames/s,
3000 frames): frame written to its localizations, median / max, went from
2.8 / 6.5 s to 0.04 / 0.11 s at 0.3 localizations a frame, and from 3.3 / 6.7 s
to 0.08 / 0.17 s at 7.8 a frame -- the same localizations either way.  It
costs no throughput: a camera writing 1000 frames/s that the fitter could not
follow took 15.0 s to fit, against 15.7 s with the old timer.  A fitter that
falls behind is handed up to *frames per block* frames at a time, so it
catches up in large blocks; the progress line says how many frames it is
behind, and it warns once when that is more than 5 s of acquisition
(`LIVE_BEHIND_WARN_SECONDS`).  This needed the index read incrementally
(`read_index(previous=...)`): parsing it whole took 0.06 s at 10,000 frames
and 0.6 s at 100,000.  `LiveFit`, which keeps its own loop for its stop
event, still flushes every `LiveSettings.flush_seconds`.

Checked against the offline path on 300 frames of the astigmatic dataset,
replayed frame by frame into a growing two-file series: same 28724
localizations, every column identical.

### Plugins while the fit is still appending

The live table is one object that `Session.append` extends, block by block,
and four things broke on it before any rule was written:

* **A plugin's context held the growing table.**  `Context` read
  `session.locs` once, but that is the object `extend` keeps replacing the
  columns of, so a plugin saw a mask of 50 rows over a table of 100, or
  `x_nm` read before a block and `y_nm` after it.  During a live fit the
  context is now *pinned* (`Session.context`): a snapshot of the table
  (`Localizations.snapshot`, a new dictionary over the same arrays -- `extend`
  writes only past their end or into a new buffer) and every layer's
  selection cut to its rows, all taken on the GUI thread.  A plugin that reads
  several layers asks `Context.layer_selection`, not the session.
* **A table change replaced the growing table** with one that could not take
  the next block, and the fit's result then replaced it in turn: the view
  stopped, the change was lost, and the log kept a line for it.
  `Session.apply` now refuses one (`LiveTableError`) except the streaming
  fit's own result.
* **A copy of the session linked it.**  `Session.scratch` (a chain's copy)
  built layers over the shared table, and linking writes `group_id` into the
  table it is given: the next block failed on a column mismatch.
* **A second layer took every block twice**: each set over the table
  extended it.  Now there is one table per session (`layer_state.Tables`):
  it takes the block once, and each layer's filter catches up with the new
  rows.

What a user wants during an acquisition, then, comes in three kinds, and
`Plugin.changes_locs` says before the run which one a plugin is:

* **A measurement** runs on the pinned snapshot, and is simply stale by the
  next block.
* **A recipe** -- the Math Parser's expression, the colour assignment with
  the modes it found -- is columns computed from the others, and follows the
  live table (`LiveRecipe`): computed for the table so far, for every block
  as it arrives, and once more for the finished table.  An expression with a
  reduction is recomputed over the whole table on every append, since its
  earlier rows move with the table.  Colours are *not* found again per block,
  which would recolour what is on screen every few seconds; the finished
  table gets its modes found over all of it.
* **Anything else** -- a drift correction, a filter that removes, a loader --
  needs the whole acquisition or a table that holds still, and is offered a
  new window instead: *File > Open in a new window*, which `PluginPanel` asks
  before the run (after it, for a plugin that does not declare
  `changes_locs`), and in which the new instance runs the plugin with the
  same settings.

**Why a new instance rather than a frozen view.**  Freezing the live view --
blocks queued, the table modified, the modification discarded on unfreeze --
was weighed and dropped: every feature of the session would have had to know
about the frozen state (undo and history cut back on unfreeze, the display
carried over to the restored layers, the fit's result held back while
frozen), and the user's work was lost by design.  A copy has none of that:
the live path is untouched, nothing is discarded, and the copy is useful
without a live fit too.  And a copy in a *process* of its own rather than a
second set of windows in this one: closing or quitting the copy, or a crash
in it, leaves the acquisition's fit alone, and a COMET run in the copy does
not compete with the live view for the GIL.

**The handoff goes through shared memory** (`smappy.handoff`): 0.36 s for 5M
localizations x 15 columns, against 11 s to write the HDF5 file as
`Session.save` does and 2.6 s to read it back.  The fit's own file cannot be
read instead: HDF5 locks a file open for writing, and with locking off and no
SWMR a reader saw torn data in 22 of 25 tries while blocks were appended.
SWMR would need the writer to stop touching attributes until it closes, newer
HDF5 to read our files, and a local disk -- it is not supported on network
shares, where acquisitions are often written.  The JSON beside the block
carries the rest; its deletion by the new instance is the signal to free the
block, which on Windows must outlive the taker's attach.

With several instances open, every window title carries the session's name
and a coloured mark (`gui.identity`, one filter on the application, since a
second session is a second process), and a second instance writes a log of
its own (`diagnostics`, `smappy-<pid>.log`).

## Drift correction

Estimation is not ours: COMET (vendored, trimmed, under `src/smappy/_comet`) is called
as a library. `drift.py` is the seam between it and the rest of the package,
and it is deliberately thin -- an array in, a per-frame drift table out.

* **Estimate on a subset, correct everything.** The estimator wants clean
  localizations; a badly fitted one contributes noise to the overlap cost and
  nothing else. So `estimate_drift` takes the same `select` a render does. But
  the correction is a coordinate change and reaches every row, filtered out or
  not -- otherwise changing a filter afterwards would mean re-running it.
* **COMET's file I/O is not wrapped.** It can load ThunderSTORM CSV and write
  molecule sets; we hand it a `(N, 4)` array and read the drift back, so the
  corrected file stays an ordinary SMAPpy HDF5 that the viewer opens unchanged.
  The drift curve rides along in a `/drift` group, which readers ignore.
* **The drift table is indexed by frame.** COMET interpolates onto
  `arange(0, max_frame + 1)`, so applying it is one fancy-index per coordinate.
  `min_max_frames` is passed explicitly because the subset the estimate is made
  from need not reach the last frame of the table it is applied to.
* **Or by time, where there are no frames.**  `Drift(..., times_s=...)` is
  sampled at those times and applied by interpolating linearly at each
  row's `time_s`, held at the ends (`Drift.at`).  MINFLUX has no frames --
  its `frame` is the rank in time, so a per-frame curve would be one value
  per localization and true of one ordering of one table -- and its MBM
  beads give a point every two seconds.  A time curve takes no
  transformation, adds (`then`) only to one at the same times, and writes
  `time_s` in place of `frame` in the `/drift` group.
* **`peak_*` is left alone.** Those are detection positions -- the record of
  what was found where -- not measurements of the structure.
* **Units survive.** COMET works in nm; a table in pixels is converted for the
  estimate and the drift divided back by the pixel size when applied.
* **A new file, not an in-place edit.** `_driftc.hdf5` beside the input: drift
  correction is a hypothesis (segmentation width, maximum drift) that is worth
  redoing with other parameters, and the raw fit must stay untouched to redo it.

Where the time goes, what was tried, how estimates were compared, RCC and
the spline fit, measured: `docs/drift.md`.

## What a tool leaves behind

A figure is part of a result, and a result outlives its session.  A drift
correction subtracts a curve and the corrected table no longer says what the
curve was, so `Plugin.keep` names what of a result is worth saving and
`Plugin.restore` reads it back: the session collects it in `Session.results`,
keyed by plugin path, and writes it into the localization file, which is what
re-enables *Plot* a week later.  Most plugins keep nothing and are unaffected.

It goes into a dataset of its own (`results/saved`) rather than into the
metadata attribute, for the same reason the GUI state does: an HDF5 attribute
is bounded by the object header, about 64 kB in practice, and a per-frame
drift curve passes that on any real acquisition.  What is kept is the estimate
and not the run -- the curve, the settings, the counts -- because the
localizations it was made from are in the same file.

Separate from the figures is the **log**: `Session.apply` appends the plugin's
path, the line it reported and the settings it actually used to
`session.history`, which is written into the file's metadata and read back
when the file is reopened (`session.file_history`), so the record continues
across sessions instead of starting again at each load.  It is capped at
`MAX_HISTORY` entries, since all of the metadata goes in that one bounded
attribute.

What belongs in it is **what cannot be worked out again from the file in front
of you**.  A correction moved every localization and the file no longer says
by how much, so it is logged, with its settings; a measurement can be repeated
from the same data, so logging it would record only that somebody looked --
and on a plugin that prints the log, it would store a copy of the log inside
the log on every run.  So the default is to log a run that changed the
localizations and nothing else (`Plugin.logged` overrides it: `True` for a run
that changes something the log cannot see, such as writing a file or exporting
a picture).  What a measurement worked out is not lost by this: it goes with
its result, and the session keeps the settings beside it, which for a
measurement is the only record there is.

`Analysis/Process/History` is where it is read: the log on screen, optionally
only the entries that changed the localizations, and an export (`.csv` for a
spreadsheet, `.yaml`/`.json` to keep the settings as values) because the
record is most useful outside the program -- in a methods section or in an
issue about a file that looks wrong.  Reading the log is not an event in it
(`Plugin.logged`); otherwise every look would store a copy of the whole log
inside it.  Relinking is, since which `dx` and `dt` produced the grouped table
cannot be recovered from the file afterwards.

Two things had to be fixed for that record to be true.  Opening a file cleared
the session's log, and saving then wrote the cleared one back over the file's
-- so the provenance went no further back than the current session.  And the
HDF5 writer merged the *table's* metadata over what the caller passed, which
meant the log that was saved was the one the table had been loaded with rather
than the one with the run that had just finished in it.  The caller's copy is
the newer of the two; the table's now fills in only what was not given.

### The camera frames, kept with the table

A fit also leaves behind what the camera saw (`smappy.rawframes`), as SMAP's
CameraConverter does: the average of every fitted frame, then the first
fitted frame and the rest spaced evenly over the range -- 50 by default,
*raw frames* in the fitter's output settings -- in photons, each with its
frame number in the table's numbering.  SMAP keeps every `diffrawframes`-th
frame instead; a number rather than a spacing keeps a 500,000-frame run from
filling its file with them.  A live fit does not know its length, so it keeps
every k-th frame and doubles k, dropping every second one, whenever there are
more than asked for.  The average is summed in ADU as the blocks go past and
converted once, which is the same number because the conversion is linear.

They go under `/images` in the localization file, one group per image with
the placement (pixel size, the edge of pixel 0 in the table's units) in its
attributes, and are read a plane at a time (`io.hdf5.StoredStack`): fifty
frames of a full sCMOS chip are 800 MB, which opening a table should not
cost.  Everything else that writes into the file (`save_localizations`,
mode "w") starts it again, so the images are written *after* the table --
by the fitter after its finishing steps have rewritten it, and by
`Session.save`, which first reads in full any image still read from the file
it is about to replace.  A session saves every file's frames and every image
a layer shows, and on opening they are sources for an image layer rather
than layers: most of the time one looks at the localizations.

## What is still current, per step

A record used to carry one signature, over the ROI's inputs -- the source, the
geometry, the filters, the grouping.  So moving an ROI marked its numbers out
of date and editing an evaluator's parameter did not, which is the case where
it matters most: the numbers on screen were measured with parameters nobody
was using any more, and nothing said so.

Every step now carries its own signature, over the inputs *and* the step's
identity (plugin, version, parameters -- not its label, which names the
columns and is not part of the measurement).  That is what makes
"re-evaluate what changed" possible: editing one evaluator costs that
evaluator over the sites, and moving one ROI costs that ROI's steps, instead
of either costing the whole pipeline over everything.

Stored entries are matched by signature first and by label second, so a step
that was renamed or moved up the pipeline keeps its results.  A state is one
of four, and the difference between the last two is worth keeping: `current`,
`stale` (a signature that no longer matches), `unverified` (an evaluator
that is not installed here, so it cannot be checked) and `missing`.
`results()` drops a row with anything stale or missing in it, and reports an
unverified one: it was true when it was written, and dropping it would lose
the file's results on a machine without that evaluator.

## Evaluating the ROI being looked at

Selecting an ROI shows what is already known and measures only what it has to.
The numbers come from the stored records -- that is what makes a long list
scrollable -- and a step is run again only when its signature no longer
matches, which is `evaluate_one(reuse=True)`.

The figures are the exception, and cannot not be: a plot is a closure over the
data it drew, so nothing stored brings one back and the evaluator whose tab is
open is always run for the site (`force`).  That is one step of one site, a
few hundred localizations; everything else is stored numbers.  What ran is
kept as a one-site run, so the next visit is free -- except that a step run
only for its figure has told nobody anything new, and storing it again would
grow the file per click, so only steps whose result actually changed are
written.

(This reverses the first version of this, which recorded nothing on the
grounds that a run is a pipeline over every site.  Recording nothing meant the
same stale step was re-measured on every visit and the site table never caught
up with what was on screen.  A one-site run says `scope: "site"`, which is
honest about what it was.)

Outer tabs are the evaluators, inner ones their figures, and both are lazy, so
a pipeline of five evaluators with three figures each costs one figure per
site rather than fifteen.  *Re-evaluate when out of date* is the switch for
the rest: off, a stale step that is not being looked at is left as it is and
shown labelled, for scrolling a large project without it computing behind you.

## The camera database

Micro-Manager records a great deal and not the two numbers a fit needs most.
The e-/ADU conversion is never in the metadata and depends on the readout
mode, which is; an iXon reports no baseline at all; an EMCCD's conversion
changes with the pre-amp gain, the readout rate and the output amplifier.  So
the numbers come from a database, chosen by what the metadata does say --
which is what SMAP keeps in `settings/*_cameras.mat` and what
`smappy/data/cameras.json` holds here, ported from the Ries lab's file.

A camera is identified by one tag (a serial number, a camera ID) and carries,
per parameter, where its value comes from: `fixed`, `metadata` (this tag, read
this way) or `state` (it depends on the readout mode, which is recognised by a
set of tags that must all match).  Tags may be written `{prefix}-Gain`, so a
lab that renames its Micro-Manager device edits one field.

Two rules decide the rest:

* **The file wins.**  The database says what a parameter is when the file does
  not; a value the acquisition recorded beats a value stored months ago, and
  the user's own setting beats both.
* **Nothing is guessed.**  A camera that cannot be identified resolves to
  nothing rather than to a plausible default -- a wrong conversion is
  invisible in the fit -- and the camera is then chosen by hand.  SMAP's own
  `Default` entry, which it falls back to for an unrecognised camera, is
  therefore not converted: its numbers stand for no camera in particular.

Every value is attributable: `Resolution.sources` says, per parameter, whether
it came from a tag (which one, and what it read), from the readout mode (which
one), from the database, or from the user.  That is what the parameter view
shows, and the reason it exists: a conversion that is wrong by a factor of two
looks exactly like a conversion that is right until something says where it
came from.

A camera is added from a file it took (`gui/camera_editor.py`, over
`camera_db.draft`, `with_mode` and `save_camera`).  The fitters say a file's
camera or readout mode is unknown as soon as the file is chosen
(`Plugin.notice`), not when the run fails for want of a conversion, and offer
the editor there.  The draft fills in everything the tags say -- the device as
the prefix, the serial number, which tag is the EM gain, which tags set the
mode -- and leaves the conversion and offset empty, by the same rule as
above.  Modes are added one file at a time: saving replaces only the mode this
file is in and keeps the others, each of which was measured on its own.  What
is saved goes into the user's `cameras.json`, never the shipped list an
update replaces; a shipped camera edited there is saved as the user's copy
under its own name, which `joined_with` puts in its place.

One camera path: the database, the file, and the fitter's typed fields, met
in one place (`io.tiff.resolve_camera`, which `CameraSettings.resolution`
calls for every fitter, its parameter view, the camera editor and the uiPSF
calibration).  The camera YAML presets -- a second database of a camera per
file, merged in three places -- and the reading of SMAP's `*_cameras.mat`
went with it.  A plain YAML file of the typed fields stays for scripts
(`smappy-fit --camera CONFIG.yaml`, `smappy.fit(camera="cam.yaml")`): it is
the fields written down, not another source of cameras.

A pixel size is one number or two.  Almost every microscope has square pixels;
where x and y differ the database keeps both, `to_nm` scales each axis by its
own, and a width or a precision -- which is neither x nor y -- by the mean.

## Two colours in 2D

The 2D two-colour workflow (`Localize/Gaussian 2D 2C`) is the 3D one with the
PSF model taken out, and it departs from SMAP in what replaces it.

**A global fit, where SMAP reads intensities.**  SMAP's
`fit_wavelet_dualcolorratiometric` fits the whole frame single-channel, finds
the transformation, then goes back to the movie and *measures* the partner
intensity at the transformed position (`Get2CIntImagesWF` ->
`Intensity2Channel`).  SMAP does have a global Gaussian -- `MLE_global_spline`
has a "PSF free" mode through `mleFit_LM_global`, and
`shared/mexfiles/mleFit_LM_global_gauss.m` sits beside it -- but its
ratiometric workflow does not use it.  Here it is the whole workflow, because
`global_fit` in `csrc/global.hpp` was written as a template over the model and
instantiating it for `GaussFree` is a binding, not an algorithm.  So this is one pass instead of two, a proper linked
MLE instead of a readout, and the photon ratio falls out of the fit rather than
being assembled afterwards.

This works only because of a coincidence worth stating: `GaussFree` and
`CSpline` both have five parameters, with `(x, y, photons, background)` first
and one model-specific parameter fifth -- z for the spline, sigma for the
Gaussian.  The link array is therefore the same shape for both, and
`dualfit.build_link` leaves that fifth slot at offset 0, factor 1 in either
case: correct for z because the channels are checked to share a z grid, and
correct for sigma because sigma is free per channel and the link is never
consulted.  `combine_peaks`, `cut_paired_rois` and `build_link` are shared
verbatim; `paired_to_localizations` is the single place the two part.

**Sigma free per channel.**  The two halves of a ratiometric splitter see
different wavelengths and rarely share a focus.  Linking the width would push
the misfit into the photon numbers -- the very quantity whose ratio carries the
colour -- so the default is free, and linking it is an advanced option for
checking, not for production.

**Registration by voting on pairs, not by cross-correlating images.**  See
`docs/channel_registration.md`.  The short version: `RegisterLocs2` renders both
channels and cross-correlates, which needs the split position to render around;
voting on within-frame pair vectors is the same cross-correlation computed on
the points, and needs nothing.  The offset comes first and the split follows.
That matters because an initial shift, a magnification guess and a wrongly
assumed image centre are between them the usual cause of a registration that
fails without saying so.

**The split is a position on the chip, everywhere.**  The fitter
(`dualfit.which_channel`, the per-half detection threshold), a movie
registration, the bead calibration and uiPSF's profile all give and read
`split_position` in camera-chip pixels.  The bead calibration once stored the
index in the bead image's own pixels, which agreed only while the ROI began at
the chip's corner along the split: a right-left splitter recorded with its ROI
at x = 91 recorded 220, and the fitter split the localizations at chip
x = 220 instead of 311.  A bead stack is now cut where the chip position falls in it.

**One geometry for beads and movies.**  The bead calibration's split used to
be settings of its own, with a fixed `up-down mirrored` default that a
left-right splitter failed on.  It is now the movie registration's
`RegisterSettings` as a part of `DualColorSettings`, its defaults set by
`bead_registration` rather than by a second class: 8 pairs, not 20 (a slide
gives tens); a 2 px RANSAC radius and a 0.15 px dx/dy screen, not 1 and 0.5
(beads are all bright, and that screen is where a bead calibration's accuracy
comes from); and pairing within 12 px, not 20 -- the numbers the bead
calibration had, so a calibration with its layout chosen is what it was.  On
*auto* the layout is found by the same vote, taken on the beads with each
stack as one frame, and the beads are then paired as before.  Two things did
not merge.  `ChannelTransform` and `DualColorCalibration` stay two classes:
the second is the first plus two PSFs, they share the mapping and the
geometry check already (`_Mapping`, `check_geometry`), and one class would
mean PSF fields that are None for a transformation, or a new constructor
for every caller and reader.  And a regular bead grid centred in its half is
its own mirror image, so positions alone cannot say whether it is mirrored:
a simulated grid needs its layout chosen, a slide does not.

Two stages of pair-and-refit, as in SMAP, and measurably: the tight second round
is worth a factor of three to seven in accuracy over the coarse one alone, and a
third intermediate round is worth exactly nothing (identical transformations,
identical residuals).  The table is in the doc.

**What cannot be measured, and is said so.**  For a mirrored layout the seam
between the halves and the optical mirror line are not separable from pair data
-- a seam at `c` with residual shift `t` and one at `c + t/2` with no shift
predict the same pairs.  The mirror line is used and reported as an estimate.
For an unmirrored layout with a large misalignment there is a band that pairs
with nothing, and the split is then bracketed rather than known; the bracket is
recorded as `split_interval_px` and a wide one warns.  Neither is a bug to be
fixed by a cleverer estimator; both are answers the data does not contain.

**Keeping the edges of the field.**  A tight refit round is where a
localization registration quietly goes wrong: the first round is worst at the
edges, those pairs get screened out, the transformation is refitted on the
middle, and the tight round never reaches the edges again.  Nothing in the
residual says so -- it is only ever measured on what survived.  Two things
stop it, and both were found by measuring rather than by reading the code.

* The tight tolerance is derived from the coarse round's residual over *every*
  pair it matched, not over the ones it kept.  With a field distortion a
  projective map cannot absorb, edge recovery goes from 0% to 99.9%; on a
  clean field it changes nothing, to the pair.
* `robust_projective` iterated until its inlier set *repeated exactly*.  With
  a bead calibration's tens of pairs that converges; with thousands it never
  does, because a few pairs always sit within noise of the threshold and flip
  each iteration -- so it raised instead, and every field it could not fit
  perfectly failed outright rather than degrading.  It now settles for a set
  that has stopped changing meaningfully.

This was not hypothetical: on the NPC benchmark dataset it was costing three
quarters of the field.  Before the fixes the pairs covered 27% of the
reference channel and sat in a band; after them they span it entirely.

**The transform fit is inverse-variance weighted; the bead one is not.**
`fit_dual_transform` took the soft-L1 loss as its only defence, which is right
for beads -- they are all bright -- and wrong for localizations, which span
orders of magnitude in precision.  It matters most exactly where the
two-colour workflow lives: the dim channel's threshold has to come down to
find partners at all, and what that buys is imprecise pairs.  Measured against
ground truth, lowering the threshold unweighted makes the registration seven
times worse (0.0017 -> 0.0125 px) and more pairs do not compensate, because
they win by count; weighted, the same pairs cost nothing (0.0016) and help
slightly.  The weight is `1/(s_ref^2 + s_sec^2)`, since it is the *difference*
whose variance matters.  RANSAC stays unweighted -- a precise outlier is still
an outlier.

**The adaptive tight tolerance is guarded, not capped.**  It protects the
edges of a distorted field, where the projective fit misses by more than any
fixed tolerance allows, and on a dense dataset it misfires: at twenty pixels
over a hundred thousand localizations the coarse round's pairs are
contaminated by chance partners, so a percentile of its residual measures the
junk rather than the periphery.  On the NPC dataset it read 18 px of misfit
that did not exist and doubled the final residual.

The two cases look identical in the residual and are obvious in the *inlier
fraction*: a distorted field keeps nearly all its coarse pairs, a contaminated
one about a third.  So the adaptation runs only above `MIN_CLEAN_FRACTION`,
and is then allowed all the way up to the coarse tolerance.  A cap tight
enough to contain the contaminated case would also be too tight to reach the
periphery, which is the whole point -- the guard has to be on the diagnosis,
not on the size of the remedy.

**A looser coarse tolerance does not cost anything.**  Worth recording because
the opposite is intuitive, and because it is what the real-data residuals
appear to say.  Measured against ground truth instead, the true error after
the tight round is flat from 6 px to 30 px at every density tried -- 0.0011 px
either way at fifteen pairs per frame, coverage 100% throughout -- because the
tight round removes the chance matches the loose one admitted.  Only at 150
pairs per frame *and* 30 px does it show (0.0045 against 0.0006).  The coarse
tolerance is therefore set by the rotation it has to cover and by nothing
else.

**Residuals are not comparable across runs that kept different pairs.**
Widen the tolerance, the edges come back, and the reported spread grows -- an
improvement reported as a regression.  On a clean field, widening the tight
round from 1.5 px to 20 px leaves the true accuracy at 0.0014 px while the
reported residual inflates fivefold.  So the residual is recorded separately
for the middle of the paired region and its edge: flat is noise, rising is the
wrong model, and the two have opposite remedies.

**Which frames to calibrate on.**  Not the first ones.  The opening frames of
a movie are not single molecules -- everything is on, often saturated -- and
fitting them gives a mat of spurious positions; on the NPC dataset, including
frame 0 does not degrade the registration but *fails* it outright.  The rest
are taken as evenly spaced blocks across the whole movie, which on the same
data more than doubled the vote contrast (19 -> 47) for half the
localizations.  Blocks and not a stride, because reading is sequential.

**Order three, and no option for two.**  Radial distortion is
`r' = r(1 + k1 r^2 + ...)`, i.e. `u' = u + k1(u^3 + u v^2)`: the leading term
is cubic.  A quadratic spans no cubic monomial, so it cannot describe it at
all -- measured, order two is indistinguishable from projective while order
three is thirty times better.  The polynomial is refused when the pairs cannot
support twenty coefficients, because outside its pairs a cubic is not a
transformation but an extrapolation, ~85x worse than the projective map it
would have replaced.  Both directions are fitted, since a cubic has no
closed-form inverse; the backward cubic is only a seed, refined by Newton
against the forward map, because a 0.5 px round-trip error would be handed to
the fitter as a link offset as if it were real.

**A file name that must not change.**  The registration plugin lives in
`plugins/registration.py` and not `plugins/register.py`, because every plugin
module does `from . import register` for the decorator and a sibling module of
that name silently replaces it on the package.  The failure surfaces as
"module object is not callable" in whichever *other* plugin happens to be
imported next, which is nowhere near the cause.  There is a comment saying so
at the top of the file.

### Finishing the fit

A fit is not finished when the last frame is: the localizations want z
from the widths, the drift taken out, the colours put on, and only then is
the file worth opening.  These are all plugins already, so every fitter runs
them -- the *after the fit* part (`finish`, `plugins/after_fit.py`), in this
order: astigmatic z (off), drift none / RCC / COMET (none), colours (on), then
a saved chain (none).

**Fixed steps, then any chain.**  Not a chain of anything with switches: the
known steps have typed settings, so choosing COMET shows COMET's settings and
nothing else, a step is shown only in a fitter whose table can take it
(colours after two channels, z after an elliptical Gaussian), and saved
workspaces keep naming the fields.  What the lab does beyond them is a chain
run last.  z comes before the drift so that the drift corrects z too.

**One part for every fitter, judged by the table.**  When the steps run, a
step whose columns are missing is passed over rather than failed
(`after_fit.applicable`), so the default colours do nothing to a one-colour
fit and the same settings serve the plugin *After the fit* on any table.

**Again, from a button.**  A step not ticked during the fit is run later
with *Run these now* inside the part: `Plugin.part_runs` runs the plugin
*After the fit* on the session with the part's settings, one undo step under
its own name.  RCC, COMET, Assign colours and Astigmatic z are no longer
pinned to the default Analysis tab; they are in the tree, and + pins them.

**At the end, not per block.**  No step can see one block.  A drift curve
is measured *across* the acquisition, and the modes of the colour histogram are
a property of the sample that the first two hundred localizations do not yet
show; running either per package would also cost the fit its throughput, which
is the thing the block pipeline exists for.  So they run once, over the
finished table, and the raw fit is what streams.

**The file is rewritten, not corrected.**  The writer is append-only and
streams while the frames come in, which is what makes a crash cost one block;
the finishing steps add columns, which an append-only writer cannot do to what
it has already written.  So the streamed file is the raw fit, and it is
replaced by the finished table once the steps have run -- the saved file is the
one somebody should open, and the raw fit is what survives a crash.  Every step
goes into the file's `history` in the same shape `Session.log` uses, so a file
says which drift estimator corrected it and with which numbers.

**A failed step is a note.**  These run when the frames are already fitted.  A
colour histogram with one mode, or a drift estimate on too few localizations,
must cost its own step and not the twenty minutes that produced the table -- so
each step is caught, said in the result's text, and the table goes on to be
saved regardless.

**Drift off by default.**  Colour assignment is seconds and is what the fit was
for.  A drift estimate is minutes on a dataset whose size the form cannot see,
and COMET's own preflight -- the question it asks before a long run -- is not
asked here, because the fit has already been agreed to.  Choosing RCC or COMET
in the section is the agreement.

## Chains and batch runs

`docs/batch.md` has the whole design; the decisions someone would otherwise
undo:

* **A chain runs on a scratch copy of the session** (`Session.scratch`), and
  only its one result reaches the real session: one undo step, one log entry
  holding every step.  Applying each step to the real session would give a
  chain of five steps five undo steps and a half-finished table on failure.
* **A chain is logged even when it only measured** (`ChainPlugin.logged =
  True`), unlike a plain measurement: in a chain, and above all in a batch,
  the numbers a measurement step produced are the point of the run.
* **Validation is strict and reading is not.**  `settings_from` drops unknown
  names so that an old workspace or file still opens; `batch.validate`
  refuses them, because in a file someone wrote this morning an unknown name
  is a typo and would otherwise become a silent default.
* **`ctx.locs` stays ungrouped.**  The grouping a chain decides reaches a
  plugin only through `ctx.table()`, so no existing plugin changed behaviour.
* **Linear, and one file per input.**  Pooling across files, several files per
  input and parallel files are recorded under "Not yet" in `docs/batch.md`.

## One name for a precision

Every precision is `<quantity>_err_<unit>`, the unit last and nothing after
it when there is none: `x_err_nm`, `z_err_nm`, `x_err_pix`, `photons_err`.
The fitter had always written those; the two exceptions were the lateral
precision, `loc_precision_nm` (`_pix`), and the axial one,
`loc_precision_z_nm`.  They are now `xy_err_nm` (`xy_err_pix`) and `z_err_nm`.

The second was a bug and not only a spelling.  SMAPpy's Spline 3D fit writes
`z_err_nm`, while the renderer, Statistics, the precision plugin and the line
profile looked for `loc_precision_z_nm`, which only files from other programs
carried: after a SMAPpy 3D fit there was no z precision in Statistics and a
side view was blurred along z by the lateral precision.  Those three plugins'
versions are bumped for it.

`xy_err` is the RMS of the two errors, `sqrt((x_err^2 + y_err^2) / 2)`, and
wherever the pair exists it is *derived* from them rather than carried
alongside (`columns.add_xy_err`): after the fit, after a conversion to nm with
pixels that are not square, and after grouping, which combines `x_err` and
`y_err` each by the precision rule and then derives `xy_err` from the pair.
Grouping it by the rule directly gives a slightly different number, and the
grouped table would disagree with itself.  A table with only a lateral
precision (SMAP's `locprecnm`, ThunderSTORM's `uncertainty`, MINFLUX's
estimate) keeps the one it has.

## The simulation model

`smappy.simulate` is one model with two outputs, localizations and camera
frames, so a dataset looked at as localizations can be fitted from its frames
by changing nothing but the output.  The stages and what each assumes:

* **Structure**: label positions from a YAML (`simulate/structure.py` has the
  syntax): explicit points or a CSV, the same in every copy, or lines,
  circles, polygons and grey-value images sampled at a density, Poisson in
  number and drawn afresh per copy; copies placed at random (with a minimum
  distance) or on a grid, turned in-plane or in 3D, tilted by up to a
  given angle and jittered, and each copy's position and turn kept with the
  truth (`metadata["copies"]`, by the `copy` column).  The built-ins are YAML
  files in `data/structures`, so each is also an example: `demo` (the ring
  and cross the tutorials use), `npc` (Nup96), `filaments`, and `pie`, a
  density resolution target of eight segments from 25 to 3200 labels per
  um^2.  A new one is a file dropped there.
* **Labelling**: a label is labelled with a probability and then carries
  exactly one fluorophore (or a set number, or a Poisson number).  Linkage
  error comes in SMAP's two kinds: fixed, one Gaussian offset per
  fluorophore shared by its blinks, and free, drawn anew for every blink.
* **Blinking**: exponential on- and off-times in continuous time, and a
  bleaching probability after each blink -- one model for STORM and PAINT,
  since PAINT's docking strands are lost too.  The off-time is set from the
  mean number of blinks *in the measurement*, which is what one knows about
  a dye, by inverting `E = sum_k (1-p)^(k-1) P(K >= k)` with `K` the
  Poisson number of activations; asking for more than `1/p` is refused.  By
  default the activation is ramped against the bleaching: the blinks of the
  fixed-rate model, their times mapped monotonically so they fall evenly
  over the measurement (the density is flat to within 5 %).  The fixed rate,
  with most blinks early, is one option; SMAP's "Dye" model -- every blink a
  fluorophore has until it bleaches, `blinks` on average whatever the
  number of frames, spread evenly in each one's order -- is the other.
* **Photons**: a total per blink from a gamma distribution with a given mean
  and standard deviation (0: exact), emitted at a constant rate while on, so
  a frame gets its share by the time on in it and the first and last frames
  of a blink are dim.  Background likewise, per localization or per frame,
  flat over the frame.
* **Localizations**: Poisson photons, a detection limit of 10 photons (the
  dim ones are kept, as SMAP keeps them, and Ground Truth leaves them out of
  its score), Mortensen's maximum-likelihood
  precision from the photons, the background, the PSF and the pixel (times
  sqrt 2 for an EMCCD), and noise drawn from it, so `xy_err_nm` is honest.
  Emitters on in one frame within the separation are removed, or merged at
  their photon-weighted mean.
* **Camera frames**: pixel-integrated Gaussians (the fitter's own model),
  astigmatic optionally, or a measured spline PSF from a bead calibration;
  shot noise, an EMCCD's multiplication noise when there is EM gain (the
  electrons a gamma of the Poisson count: twice the variance), gain,
  offset and Gaussian read noise; every spot drawn, however dim.  Saved as the recipe (`*.sim.yaml`), which `open_stack`
  opens as an acquisition that draws its frames as they are read, seeded per
  frame, and which tells the fitter its camera.

What changed from the first simulators, which each had their own blinking:
blinks were uniform in time with geometric lengths in whole frames, the
photons of every frame independent, the precision `150 / sqrt(N)` whatever
the background, and a fluorophore could overlap itself in time (two
localizations in one frame, then both dropped as "too close").  The number of
blinks now follows from the kinetics rather than being drawn, and the demo's
structure is the same ring and lines as before, written as a YAML.

**Scoring a fit** (`Analysis/Measure/Ground Truth`): the truth is redrawn
from the recipe rather than stored, localizations and true spots are paired
one to one per frame (Hungarian, within a radius, as the SMLM challenge
does), and the Jaccard index sums up detection.  Every spot is matched
against but only some are *counted* -- above a photon count, optionally
only the isolated -- and a fit of a spot that does not count is set aside,
not called false: otherwise tightening what counts turns good fits into
false positives.  With only the isolated counted, what lies within half the
isolation distance of a crowded spot is set aside too, since a crowded pair
is fitted as one spot between them, or twice.  On 300 simulated frames the
isolated spots come back with a recall of 1.00, two false positives in 1300
and an error over reported precision of 1.00 and 1.02; counting every spot
above 300 photons gives a Jaccard index of 0.77, the rest crowding.

Frames drawn with a spline PSF from simulated beads and fitted with Spline
3D and the same calibration give z on a slope of 1.001, biases below 0.4 nm
and error over reported precision 0.94-1.03; with an EM gain of 100 the
lateral error grows from 7.5 to 10.7 nm, the sqrt(2) of the excess noise, and
the fit's reported precision grows with it.  Ground Truth can also require a
z match (SMAP asks 300 nm), and leaves localizations dimmer than its photon
threshold (100 by default) out of the score when they matched nothing: a dim
localization that found a counted spot is a true positive, and one that found
nothing is too imprecise to be called false.

### Against SMAP's simulator

Compared with `shared/simulatelocs.m` (SimulateSites), `simulatecamera.m`
(SimulateCameraImages) and `CompareToGroundTruth`, the physics within a blink
is the same -- exponential on-time starting at a random point in a frame,
photons per blink spread by the time on, Poisson per frame, a Mortensen
precision with sqrt 2 for EM and z three times lateral -- and so is the
labelling.  What differs, on purpose:

* **The precision is Mortensen's eq. 5, maximum likelihood; SMAP's
  `MortensenCRLB` is his eq. 6**, `16/9 + 8 pi sa^2 b / (N a^2)`, which is the
  error of an *unweighted least-squares* fit (the paper's "Gaussian mask
  estimator", 1.5x the variance in its own measurement).  SMAPpy copied it
  until the plugin pages were written; against SMAPpy's own fitter on
  simulated spots it was 10-23% too pessimistic, from 200 to 5000 photons,
  where eq. 5 agrees to 1-2% (`test_the_mortensen_precision_is_what_the_fitter_achieves`).

* SMAP's default blinking (`simple`) puts every blink at an independent
  uniform frame, 1 + round(Exp(n)) of them, with no kinetics; here the
  number follows from off-time and bleaching, set by the blinks wanted in the
  measurement, with SMAP's `Dye` as an option.
* SMAP keeps close emitters in its localizations; here they are removed or
  averaged, since a fitter could not have separated them (the camera frames
  keep them, as SMAP's do).
* SMAP's PSF width in the precision is fixed at 100 nm; here it is a setting.
* SMAP places sites on a grid, one per ROI; here copies are placed at random
  or on a grid.  SMAP normalises an image structure by 255, here by its
  brightest pixel.
* SMAP matches nearest-first; here the assignment of least total distance.

And three things in SMAP that look like bugs, noted here and not fixed
there: the `Dye` model assigns the sort permutation instead of the
ranks when spreading the blinks, which scrambles each fluorophore's order;
SimulateCameraImages draws `phot` that `simulatelocs` had already
Poisson-sampled and applies shot noise again, doubling its variance; and its
random source draws a new position for every frame of a blink.

Left out for now: a non-flat background, the precision from a spline PSF's
CRLB (SMAP has it) rather than Mortensen's, and using a loaded table as the
structure.

## uiPSF calibrations

**The v1 bridge is retired** (October 2026, `docs/uipsf/plan.md` §7.5).  Until
then `Localize/uiPSF calibration` (`smappy.uipsf`, last at 0164d42) ran uiPSF
v1 under its own TensorFlow Python as a worker script and turned its voxels
into a spline calibration.  It went before v2 does in situ, which is accepted:
in situ is v2's M5.  What it taught, for v2 and its window:

* **smappy reads the data, not the PSF learner.**  Camera to photons, the
  stacks' z, the split of a two-channel camera: all as smappy's own
  calibration and fitter do them, so the channel transform comes back in
  chip pixels, the map `dualfit` reads.  v2 keeps this: uipsf takes arrays
  from `smappy.calibrate.pupil`, and its own reader, finder and pairing are
  gone.
* **The spline is made as the bead calibration makes it**: clipped,
  brightest plane summing to one, `positive_pair_models` for two channels,
  so a fit cannot tell the calibrations apart.  v2's samples go the same way
  (plan §6).
* **In situ was never validated.**  On simulated frames v1's in situ model
  had z running 0.76 of the truth and photons 10 % high, and from another
  start converged to a model whose z ran backwards.  v2's M5 starts from
  that, with real data and a bead calibration beside it.
* **uiPSF v2, in this repository.**  `src/uipsf/` is uiPSF re-implemented
  in PyTorch, moved here from its own private repository (ries-lab/uiPSF2)
  so that it can use smappy's loading, camera database, bead detection and
  fitters rather than copies of them.  It is a package of its own, `uipsf`,
  beside `smappy`.  Two rules keep it separable: it never imports Qt, and only it
  imports torch.  torch is the extra `smappy-smlm[uipsf]`, not a dependency:
  on Linux its default wheel brings NVIDIA's libraries, 4 GB to download
  (torch 2.9.1), for a fitting machine that may never learn a PSF.  Its
  plan, conventions and what is built are in `docs/uipsf/`, its rules in
  `src/uipsf/CLAUDE.md`; the *PSF calibration* plugin that uses it is
  planned in `docs/uipsf/plan.md` §7.5.
* **The PSF calibration window** (`Localize/PSF calibration`,
  `calibrate/psf_window.py`): one window for a PSF calibration, by a
  *method* -- the bead average or the pupil model -- each an entry in
  `calibrate/psf_methods.py` that says which sections it reads, how it runs,
  its pages, its table and how it saves; the window knows none of them.  In
  situ will add a *source* choice and an entry, not a branch.  It replaces
  *Bead calibration* once it does everything that one does; until then both
  are shipped.  It calls rather than copies: the bead average's run, pages
  and fit quality are the bead calibration's, and the pupil model finds
  beads with its *find* half (`core.find_stack_beads`), so both list the
  same beads.  The pupil model's run is `calibrate.pupil.run_calibration`,
  headless.  Both save with *save model* beside the stacks under the bead
  calibration's name, never over another file (the bead calibration asked
  for a path each time; *Save as...* still does).  The pupil model's fit
  quality (`pupil.fit_quality`) fits every slice of every bead with the
  saved spline by Spline 3D, as SMAP's bead calibration does: the check is
  the fitter the data will meet, not uipsf's own localisation.  *Install
  PyTorch* (`torch_install`) picks the build per machine; Windows with an
  NVIDIA card gets PyTorch's CUDA 12.6 index, for older drivers.  The pupil model drops
  the bead average's *too dim* beads before learning (a pair by its light in
  both channels together): on the Zenodo beads 24 of 84 detections were
  specks a fifth as bright as a bead, whose fits were noise; with them gone
  the rejection, no longer skewed by them, caught the one bead whose fit
  quality curve ran away.  An absolute SNR floor was tried and dropped: real
  beads in a ratiometric pair's dim channel have an SNR of 15, specks up to 8.
  Isolation, for both methods (`core.select_beads`): a neighbour within
  *min distance* counts only if it is bright enough to be a bead (its local
  height in the smoothed projection, not its box, which beside a bead holds
  the bead's light) or saturated.  "Bright enough" is the typical bead,
  the median weighted by each spot's own light, over √*brightness range*:
  the plain median, even iterated, fell among the specks once a low
  threshold found four times more of them than beads.  On Zenodo the
  pupil model keeps 56, 60 and 62 beads at 6, 4 and 3 σ (before: 59, 35,
  and the specks) and 88 pairs at each; the bead average accepts 55 at the
  default 6 σ, where it accepted 36.  Its verdict keeps the plain median:
  on three beads the weighted one moves too much.
  `BeadSettings` (the model's choices) lives in smappy, not uipsf: every
  `uipsf` import loads torch through the package, and the window must open
  without it to say how to install it.  Until the microscope database, the
  optics are a form whose NA and wavelength have no default: 0 means "not
  given" and a run refuses it (None would show as *auto*).  No *diagnosis*
  mode: every run draws every figure, and *save model* off leaves no file.
* **The spline samples the model as it was learnt** (October 2026, the M6
  checks, `docs/uipsf/architecture.md`): `pupil.UPSAMPLE` is the model's own
  sampling (pixel centres), no longer each pixel integrated 4 × 4.  The
  learning fits pixel centres, so on a camera's integrated pixels the pixel's
  width goes into the learnt blur; integrating that model again counted it
  twice.  Plan §6.5's check found it: Spline 3D and uipsf's own localisation
  of the same Zenodo 1ch beads were 5.0 nm rms apart with a ±7 nm bend, and
  are 0.6 nm apart now (CRLB 2.5).  Learning with integrated pixels instead
  would make the blur the optics' own, at four times the pixels per fit.
* **The dual fit quality is Spline 3D 2C's global fit** (`pupil.fit_quality`):
  one x, y and z per pair through the learnt transform, the photons free per
  channel, the link built by `dualfit.build_link` -- as the bead average's
  `fit_paired_bead_diagnostics` does, and as the data will be fitted.  Each
  channel on its own half of the calibration never tried the transform, and
  on Zenodo showed the dim channel wandering by 32 nm rms where the pair's
  fit is within 1 nm of the bright channel's.

## Astigmatic z from elliptical widths

3D without a spline: Gaussian 2D's *elliptical* fit measures a width per
axis, and `Analysis/After fit/Astigmatic z` (`smappy/astigmatic_z.py`) reads
z off them through one defocus curve per axis, Huang et al. 2008's with its
A and B.  Not a port of SMAP's Gauss fitter or its `sx^2 - sy^2` lookup; the
decisions:

* **Least squares on both widths, weighted by their CRLB.**  The lookup of
  `sx^2 - sy^2` uses only the difference, so it finds a z for a spot too wide
  in both axes (an overlap) as readily as for a good one, and gives no error.
  It survives as the start: one `np.interp` on the curve's one-to-one stretch,
  then six Gauss-Newton steps.  0.6 s a million; a grid-search start for
  everyone was 3.5 s.  The fitter now writes `sigma_x_err_pix` /
  `sigma_y_err_pix` (the CRLB it already had) for the weights; merged blinks
  combine them as precisions.
* **No flag column: the error carries it.**  chi^2 at the minimum (one
  degree of freedom) inflates `z_err_nm` by sqrt(chi^2) above 1, and a z on
  an edge of the calibrated range gets the whole range.  A filter on
  `z_err_nm` then does what a flag would; on fitted simulated frames the
  error comes out about a fifth larger than the scatter.
* **z = 0 is the round spot**, for a typed calibration and one from beads
  alike, and each bead is moved to its own round spot before the curves are
  fitted (a tilted coverslip puts beads tens of nm apart).  z is the sample's,
  the objective's position turned round, as `simulate.bead_stacks` and the
  spline calibration have it.
* **The calibration is the PSF calibration window's fourth method**
  (`psf_methods.EllipticalGaussian`, `calibrate/elliptical.py`), fitting the
  beads with the same elliptical Gaussian in the data's fit box -- widths of
  a Gaussian on a non-Gaussian PSF depend on both.  Its `_astigz.h5` is never
  listed for the fitters' *auto*; `use_calibration` sends it to Astigmatic z.
* **A finishing step of the fit.**  Gaussian 2D with *elliptical* ticked
  offers it in *after the fit* ("Finishing the fit"), before the drift.

## Tracking

*Link tracks* (`smappy/tracking/link.py`, wave 4 step 19a) writes
`track_id` and `track_length`.  Single-particle tracking here is fitted from
single molecules, so it is sparse: a jump comparable to the distance between
molecules would mean motion blur or spots that cannot be fitted.  Linking is
not hard; bridging the frames where the fluorophore was dark, dim or badly
fitted is what matters -- which is why u-track was never used.  Not
`group.py`'s linker: that one follows a running mean position and takes the
first match in a box, right for one blink of a still molecule and wrong for
a moving one.

* **Candidates from one index.**  Neighbourhood's C++ cell index got a
  `pairs` query (`csrc/neighbours.cpp`).  x is shifted by the frame times a
  stride wider than the field plus twice the jump, so one index holds every
  frame and a query shifted by a lag meets only that lag's frame: memory + 1
  queries, no loop over frames.
* **Two linkers.**  *simple*: the candidates in the order (frame, distance,
  gap) and a link taken when both ends are free -- frame by frame, nearest
  first, in one C++ loop (`greedy_links`).  *global*: per frame the least
  sum of squared jumps with an unlinked end costing R^2, Crocker and Grier's
  criterion (SMAP's `mytrack.m`, trackpy), by `linear_sum_assignment` on each
  group of competing candidates.  The same R for a gap as for one frame, as
  theirs.  The design's optional cost scaled by the precisions was left out:
  where the linkers differ at all, both are already unreliable.
* **The benchmark** (`studies/tracking/bench.py`; tracks from
  `simulate.simulate_tracks`, 10% dark frames of 1.5 on average, 20 nm
  precision, memory 3, R three times the measured rms jump; 0.02 to 1
  particles per um^2, jumps 50 to 400 nm).  Below a jump of 0.1 of the
  nearest-neighbour spacing the two find 98.6-99.6% of the true links, the
  same to 0.3%, and whole tracks within 1.6%; at 0.1-0.4 global finds 0.3-3%
  more links; above that both are wrong on 20-40% of links.  Global gave
  exactly trackpy 0.7's links; trackpy raised `SubnetOversizeException`
  from 0.4 on.  Speed: simple 1.4-2.5 us per localization, global 4-31 us,
  trackpy 7-25 us; 880 k localizations over 4000 frames: 2.3 s, 4.6 s,
  5.5 s.  **The default is simple**: the same tracks where SPT of single
  molecules is possible, faster, and easier to reason about.  Global stays
  as the option for the crowded case, and as SMAP's behaviour.
* **Ids.**  0 for a localization in no track and for a track below the
  minimum length; 1, 2, ... in the order tracks start; files always and
  channels optionally linked apart; recipes with the rule `max`, so a merged
  blink keeps its track.  The plugin links localizations, never blinks:
  merging would join a slow particle's whole track.

### Tracking analysis

*Tracking analysis* (`smappy/tracking/analysis.py`, step 19c): MSD and
jump distances of an id column, per track and pooled, `per_region`.

* **Vectorised.**  The table is sorted by track then time once; the MSD's
  sums and counts per (track, lag) are `bincount`s over row offsets 1..50
  (a pair k frames apart is at most k rows apart).  1 M localizations in
  54 k tracks: about 10 s, most of it the MSD sums.
* **The offset from the precisions** by default (`xy_err_nm`^2, mean per
  track), so D is the slope through it; *fitted* is Michalet's case.
* **The number of lags, computed rather than looked up.**  Michalet (2010)
  gives an empirical formula for the optimal number of MSD points, but the
  equations are missing from every text reachable from these sessions
  (PubMed's full text of his paper and of Michalet & Berglund 2012), so the
  optimum is computed from the model: the MSD points' covariance for a
  Brownian track with Gaussian errors follows exactly from Isserlis'
  theorem (`msd_covariance`, against 20 000 simulated tracks within a few
  percent), and the variance of either estimator of D for each p from it.
  The x-independent sums are cached per track length (lengths bucketed by
  1.25 beyond 40, capped at 2000), the optimum tabulated over x, and
  Michalet's iteration runs on all tracks at once.  With the offset known
  the optimum is 1 lag for x << 1 and grows to ~35 at x = 100, N = 50.
* **Jumps by EM**, not a general optimiser: one minute became under a
  second for a million jumps.  EM crawls when two states are fitted to one
  population, so at most 200 000 jumps are fitted (a fixed random subset).
* **MINFLUX's clock**: `time_s` put on a grid of the median within-track
  interval.  Not checked on real MINFLUX data.

### The tracks render mode

`RenderSettings(mode="tracks")` (step 19b): each track of an id column
(`track_column`, `track_id` by default) drawn as antialiased segments
between its localizations in time order, `line_width` pixels wide
(`render.render_tracks`, the kernel `render_segments` in `csrc/render.cpp`).

* **A raster, not a vector layer**, so exports, overlays and the sum of
  layers needed nothing; the design's choice.
* **Culled per track, not per localization.**  `LayerSnapshot.select` culls
  points to the view with the spatial index, which would drop a segment
  whose ends are both outside it; in this mode it hands over the filter's
  rows and `render_tracks` drops tracks whose bounding boxes miss the view.
* **The kernel's work is a segment's length times its width**: each row it
  can reach is visited only between where it enters and leaves that row's
  band.  Coverage is `clamp(w/2 + 1/2 - d)`, round ends; a joint is up to
  twice as bright over a pixel (summing, not taking the larger, which
  would need to know which segments are one track's).  1 M localizations
  in 54 k tracks: 0.3 s for the whole field at 1000 x 1000 pixels, 0.12 s
  zoomed in; counting the drawn localizations with `union1d` had been
  0.8 s of it.
* The 3D view draws a tracks layer as lines in the volume
  (`view3d.render_tracks_3d`, 2026-10; it had drawn the localizations as
  points): the segments found once over the filter's rows and cached, only
  their ends projected per frame, a segment shown when both ends are in the
  slab.  Always on the CPU, whichever engine draws the points -- the GPU
  engines have no lines, and a tracks table is small.  Image layers are a
  plane at the pivot's z (`render_image_3d`), each screen pixel's ray
  intersected with it; a z-stack is drawn as a volume (see "Image
  stacks").  The movie export's *tail* is wave 5's.

### Which column names the tracks

Nothing that reads tracks hardcodes `track_id` (2026-10, after the tracks
render mode drew nothing on a simulation, which has only `track_id_true`,
and the view kept the old picture in silence).  `tracking.columns`: a
*track id* field everywhere (the Render tab's tracks mode, Tracking
analysis; the movie export and step finding to come), `""` meaning *auto*,
the first of `track_id`, `tid`, `track_id_true` the table has, said in the
list; any integer column by choice.  `group_id` and `cluster_id` are never
*auto*: nearly every table has a `group_id`.  Link tracks gained an
*output column* and writes only new columns or ones it wrote before
(`metadata["track_columns"]`).  A render that fails now says why over the
view (`RenderView.error_hint`), not only in the status line.
Found on the way: merging blinks averaged `track_id_true` (no rule, so
*mean*), so a blink of two particles passing close took a third track's
id and the tracks mode drew a line across the field to it.  The truth's
ids now merge by `min` and stay integers (`group.COMBINE_MODES`,
`INTEGER_COLUMNS`), as `tid` did; `track_id` was already a recipe with
`max`.

## Image stacks

2026-10.  A multi-page image is a stack of planes, and what the planes are
is kept in `ImageData.metadata` -- `axis` "z" (with `z0_nm`, the first
plane's centre, and `z_step_nm`), "t" (`frame_time_s`) or "" -- so that it
is saved with the image (`io/hdf5.save_images` writes the metadata) and
seen by every shallow copy a render thread is handed, with no new field
for the reader, the saver or the snapshot to carry.

* **Channels are split when a file is opened, never an axis.**  A layer
  has one LUT, and localization channels are layers already; a channel
  axis inside an image would have needed a channel chooser on every image
  layer and in every plugin that reads one.  `images.load_images` returns
  an image per channel (and per time point of a stack in both z and t),
  and `Session.open_images` a layer each, coloured `CHANNEL_LUTS`.
* **A file that does not say is asked about** (`gui/dialogs.StackAxesDialog`
  on `axes_unknown`), not guessed: two interleaved channels taken for a
  z-stack still make a picture, wrong every other plane, and nothing would
  say so.  The axis and the step can be corrected in the Render tab; the
  channels only by opening the file again.  Read from the file: ImageJ
  hyperstacks (slices, frames, channels, *spacing*, *finterval*), OME
  (dimension order, PhysicalSizeZ), Micro-Manager's z step, `.msr` stacks.
* **2D**: a slice, or a max / mean / sum over a range of planes, cached on
  the image per range (`ImageData.projection`); *follow z filter* resolves
  the plane in `snapshot_layers` from the first localization layer's z
  bounds (`layer_state.focus_z`), so the render threads see a plain plane.
* **3D**: *plane* (the plane at the pivot's z), *max* or *volume*
  (`view3d.render_volume_3d`): rays marched in numpy with
  `map_coordinates`, linear, about two samples per voxel up to 30 M
  samples a frame (3 M dragging) -- 1.3 s for 700 x 530 and 0.1 s
  dragging, bead stack 81 x 96 x 96.  *volume* subtracts the stack's
  median first: a bead stack's camera background, summed along the rays,
  was a fog brighter at the box's edges than the beads.  A GPU ray-marcher
  (wgpu, a 3D texture) is the next step; the CPU one is its reference.
  Plugins that read an image layer (Process image, Register layers) still
  read its plane `frame`, not its projection.

## Several files in one table

Files opened together are one table with a `filenumber` column, not a table
per file.  A table per file was weighed again (2026-10) for what it would
give -- columns of their own, no padding, filtering one file without a pass
over the others -- and kept as it is.  Each of those has a cheaper answer
inside one table, and splitting would cost every reader:

* **Columns of their own.**  A column one file lacks is NaN in its rows (0
  in an integer column, which for an id is "in none", the convention of
  `cluster_id` and `track_id`).  What was missing is knowing it is padding:
  `FileInfo.columns` is what each file had when it was read, and
  `Session.files_without` the files without a column.
* **Size.**  On disk the padding is a constant column under LZF, close to
  nothing; in memory it is 4-8 bytes a row per padded column, which matters
  only for files that are different datasets (MINFLUX beside a camera fit),
  and those belong in separate sessions anyway.
* **Speed.**  The file mask is one `np.isin` over `filenumber`: measured in
  `docs/performance.md` ("Large tables") with what else a plugin and a
  render pay for, which is where the time goes.
* **The cost of splitting.**  Almost every reader wants the files together,
  so a combined table would be a cached concatenation beside the files: a
  second copy and an invalidation problem -- the kind `layer_state.Tables`
  removed (the grouped table shared between layers, bounds copied between
  filters in five places).  Rows are also addressed by their index into the
  one table: the undo, ROI membership, `group_id`, recipes, the handoff.

What one table does get wrong is **time**: files keep their own frame
numbers, so frame 120 of one and 121 of another look consecutive, and in
the same camera coordinates.  So whatever works along time works per file:
linking and *Link tracks* block by `filenumber`; the drift plugins estimate
and correct each file of the selection on its own (`drift_result.per_file`
-- they used to estimate one curve over all the selected files and move the
whole table by it, files outside the selection included); NeNA and the
channel registration pair within a file (`locs.frames_apart`).  Files that
*are* one acquisition saved in pieces are made one by *Combine files*,
frames and time continued, ids kept apart.

What one table cost at 10^8 rows (`docs/performance.md`, "Large tables")
was paid in two places, and both now pay for the files shown rather than
for the table:

* **The filter.**  Every bound and the file choice was a pass over every
  row.  A file is one block of rows (`add_file` appends; `LocSet.file_rows`
  finds the blocks in one pass and says None when a plugin has reordered
  them), so a layer showing some files restricts its filter to their spans
  (`LocFilter.restrict`): the ranges are tested there and the rest of the
  mask is zeros nobody writes.  The masks stay as long as the table, so
  nothing that combines them with an ROI or a slab changed.
* **Linking.**  The grouped table was linked whole before a layer showing
  blinks drew anything -- 86 s and 13 GB at 10^8.  `Tables` now links a
  table of several files one file at a time, only the files the layers
  showing blinks show, and keeps what it linked while the layers move
  between files.  Linking never joined two files, so the blinks are the
  same; the cost is that the grouped table is a different object each time
  a file joins it, and that `group_id` on the localizations of a file not
  linked yet is 0.  The ids run on from file to file, in the order the
  files were linked, so they stay the grouped table's ids as files are
  added -- every reader looks them up by value (`IdLookup`, `searchsorted`),
  none by row.  A save links every file, as the file keeps all the ids.

So that the default view pays for one file, a session of several files
opens on the first, as in SMAP, and a file added keeps the layers on the
files they showed (`Session._on_files`); the others are ticked in the Render
tab.

## MINFLUX beam-line monitoring beads

What Imspector does with its MBM beads, measured on a file of 2026-10 (two
measurements, four beads each) and checked by
`test_imspectors_correction_is_the_mean_displacement_of_the_used_beads`:

* **The correction is the plain mean displacement of the used beads**,
  each from its first point, interpolated linearly to every localization's
  time and subtracted: ``loc = lnc - mean_b (bead_b(t) - bead_b(t_0))``.
  It leaves 1.3 / 1.6 / 0.08 nm rms (x / y / z) of a correction varying by
  6 / 11 / 2.6 nm; holding the last point instead, a delay of 1-2 s, a
  weight of 1/sigpos^2 or the median all fit worse.  The 1-2 nm left in xy
  is probably beads measured more often than they are saved.  So a bead
  that wanders by 20 nm among four moves every localization by 5 nm, which
  is why the tracks are worth reading at all.
* **The tracks are `zarr/grd/mbm/points`**, a point every ~2 s per bead.  A
  file of 2026-06 kept only the points from before each measurement started
  (and its ``loc - lnc`` was a constant): a bead file can be just those.
* **`gri` is a slot, not a bead**: R0 is gri 61 in every measurement, and
  in two measurements of one file it was two beads 20 um apart.  The bead
  table therefore gives each (measurement, gri) its own `tid`.
* **The 3D beads are noisy, and the mean passes it on**: in the 3D
  measurement each bead scattered by ~10 nm from one point to the next
  (2.5 nm in 2D), and Imspector's mean adds that, divided by two for four
  beads, to every localization.  A spline through the beads
  (`beads.select_beads`) with a knot every 8 s is Imspector's curve to 0.7
  nm in x and y (0.2 nm in 2D) and smooths that noise away; a knot every
  2 s, one per point, rings (2.6 nm).
* **A bead is judged against the others, leaving it out**: with four beads
  one in the fit pulls the curve a quarter of the way to itself.  Each
  bead's metrics (noise, wander, jump) come from its residual about the fit
  of the other used beads (`beads.select_beads`), the rule is the caller's
  (`beads.MadRule` by default: above the median by k robust sigmas *and* by
  a margin in nm, because four beads that agree have a tiny spread), and a
  choice made by hand is never overruled.  None of the eight real beads was
  rejected; the defaults are to be set from many files.
* **The correction replaces Imspector's, never adds to it**
  (`mbm.replace_correction`: `x = x_nc - d(t)`): the bead tracks are raw,
  so a curve from them holds all Imspector corrected, and on top of `x_nm`
  it would count twice.  Run again, it gives the same table.  An earlier
  plan had a mode "on top of Imspector's"; it was dropped for this reason.
* **A trace's first localizations are biased** (2026-10, from the lab's
  experience): in MINFLUX a still fluorophore's first localizations of a
  trace creep into place -- the first iterations do not converge fast
  enough, an estimator bias -- in 10-30 % of traces and over 5 to 30
  localizations depending on the sequence, so a trace's last points
  measure its position better than its first.  It is within a trace only:
  the rest of the trace is good, so the starts are filtered, never whole
  traces.  The reader numbers each trace's localizations, `trace_index`
  (1, 2, ...), so a filter such as `trace_index > 20` leaves the starts
  out of whatever reads the layer (*Id statistics* does).  The grouped
  layer ignores layer filters, so merging takes its own column,
  `group_use` (`group.USE`, general, usually absent): `group_use =
  trace_index > 20` in Math Parser merges each trace without its start,
  while the localizations keep every row.  **For the MINFLUX tutorial**
  (to be written once the MINFLUX functionality is in): show `trace_index`,
  the filter, and `group_use` for the merged traces.  The MBM beads are not traces -- one position
  per bead and time point, from the beam monitoring -- so nothing is left
  out of them: a first version of *MBM registration* skipped each bead's
  first 30 points and was undone.
* **The beads are a file of their own** in the one table (`msr.bead_table`),
  with only columns the localizations have, so padding costs nothing; it is
  paired with its measurement by the `.msr`'s path and the measurements'
  `did`s (`msr.mbm_key`), never by file number or name.

## Open questions

* **Two colours from MINFLUX's DCR, untried on real data.**  *Assign
  colours* reads `dcr` and `photons` as the two channels' photons
  (`assign_colors.DCR_PAIR`) and was tested only on simulated binomial
  splits.  Imspector's split includes the detectors' crosstalk and
  background, so whether the shot-noise model holds -- and whether the
  *probabilistic* method's posteriors mean what they say -- is to be checked
  on a two-colour `.msr` when one is available (none was, 2026-10).  Compare
  against minflux-viewer's per-trace EM on the same file.
* **Astigmatic z, untried on real beads.**  The elliptical Gaussian
  calibration and Astigmatic z were tested on simulated Gaussian spots, where
  the model is exact.  On real beads the widths of a Gaussian on the real PSF
  are what the curves describe; whether A and B are enough far from focus, and
  how far z from widths falls behind Spline 3D, is to be checked.
* **MBM registration, untried on real washes.**  `mbm_registration` matches
  beads by a vote on their shifts, not by name (slots are reused), and was
  tested on simulated beads; a real sequential measurement should be checked.

* Fitted x sits ~0.24 px from the peak-finder position, and the sign flips with
  the mirror (-0.22 unmirrored, +0.26 mirrored) while y stays at -0.09.  A
  round trip (render the model, fit it back) is exact, so this is a property of
  the data/calibration, not of the code.  Worth comparing against SMAP's own
  output on the same dataset.
* Camera offset: taken from the image metadata, but an iXon does not report one,
  so the settings file is used with a warning.  Should that be a hard error?
* RCC's slice width, z bin width and number of time windows are SMAP's values,
  which were set by hand rather than optimized; there is likely room there.
## Loading only the columns that are used

The next thing to do, and a memory problem before it is a speed one.  Opening
the 57 M localization `_sml.mat` peaks at **20 GB**: 30 columns are read and 13
are read by anything downstream, and four of the unused ones (`bg1`, `bg1err`,
`bg2`, `bg2err`) are float64, 457 MB each.  Microscope PCs have 8-16 GB, so
that file does not open on one at all.  Keeping only the used columns takes the
peak to about 8 GB, the read from 31 s to about 13, and `combine` -- one pass
per column -- down with it.

Which columns those are is **per format**, not global: a SMAP `_sml.mat` names
them `xnm`, `locprecnm`, `LLrel`; a MINFLUX export names them `loc`, `efo`,
`cfr`; a csv names whatever the operator called them.  `SML_DROP` in
`io/formats.py` is today's version of this and is the wrong shape -- a
hard-coded list of eight names to skip, where what is wanted is a keep-list
that can be seen and changed.

So it wants a settings file and a dialog, one section per reader: the columns
the format offers, which are kept by default, and a tick per column, remembered
between sessions.  The reader takes the set it is given and reads no more;
opening a file it has never seen shows the list first.  Worth doing together
with the readers' `needs` mechanism (`Reader.needs`), which already exists for
asking the user something before a file can be read -- the csv column mapping
goes through it.
* `group()` numbers groups from 1, a leftover from the MATLAB original, so the
  row of the grouped table is `group_index - 1`.  Worth rebasing to 0 at some
  point, together with the C++ `connect`.
* Why the ungrouped estimate still wins slightly in the axial projection when
  its z drift is three times noisier than the grouped one's.
* A configuration layer that merges file metadata with user settings once, up
  front, so downstream code never deals with partial metadata.
* A filter bound cannot be set on a live table before the first block has
  arrived, because the column does not exist yet: `LocFilter.set` raises and
  the viewer restores the box.  Holding such a bound as pending and applying it
  on the first append would be friendlier.
* Online drift correction, and an online estimate of the labelling density:
  both want the whole table, which is what an appended view already has.
* The EM readout mirror between a bead calibration and two-colour data.  An
  EMCCD's EM register reads out mirrored in x relative to its normal port, so
  a dual-colour calibration from beads on one port does not line up with data
  on the other: the transformation pairs almost no peaks and the photon ratio
  means nothing.  Seen on the 2019 NPC4C data (Evolve512, beads on `Normal`,
  data on `Multiplication Gain`): flipping the data in x about the chip before
  pairing lined up nearly every peak of the dimmer channel, and the ratios
  spread the way four ratiometric dyes should.  Deliberately not handled.  The
  calibration does not record the bead port, and the correction is not an
  image flip alone -- the camera ROI has to be mirrored on the chip too, and
  which axis and which chip width differ between cameras.  SMAP records
  `emmirror` with a bead calibration and has a `mirrorem` loader option.  It
  belongs with a camera manager; until then, take beads and data on one port.

## Environment

`python3` on this machine is x86_64 conda under Rosetta.  Use the native arm64
venv:

    .venv/bin/python -m pytest tests/
    .venv/bin/python setup.py build_ext --build-lib src

The extension builds universal2.  `setup.py` contains a workaround for a
Command Line Tools install with incomplete libc++ headers; it is gated on a
test-compile and stays silent on a healthy toolchain.
