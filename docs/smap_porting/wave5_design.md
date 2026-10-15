# Wave 5: the design, one topic at a time

Wave 5 is designed with Jonas a topic at a time, each when work on it
starts, so this file grows.  Each step is one session: read this file,
`SMAP_porting_plan.md` (wave 5) and CLAUDE.md, then implement the step with
its tests and its page, run `--slow`, update the plan's *Done* list,
commit, push, open a PR and merge it.

Topics: **1. coordinates** (decided 2026-10-15, below) · **2. stability**
(decided 2026-10-15, below) ·
3. rendering and movies · 4. figures of regions · 5. LocMoFit and NPC
geometry · 6. MINFLUX.  AssignColors for more than two colours (C4) was
found done: `assign_colors.py` takes up to six colours with both methods.

---

## Topic 1: coordinates

One idea underneath C1, C2, D1, D2 and D3: a transformation of a file's
coordinates -- constant in time (shift, rotate, scale, mirror, affine, the
projective map of *Calibrate transform*) or changing with it (a drift
curve, a magnification).  Applying one is one plugin, *Transform*;
measuring one is another (*Register layers*, the drift plugins).  Nothing
applies a transformation twice in code: a measuring plugin hands its
result to Transform.

Order: 5a, 5c and 5d in parallel; 5b after 5a.  Left for later: the
channel alignment diagnostic (C3) becomes *Check overlay*, on paired
coordinates (the 2C fitters' `x1`/`y1` and `x2`/`y2` with linking off, the
pairs of *Calibrate transform*, MINFLUX's dual-colour positions):
displacement against time and over the field of view.  It is designed with
MINFLUX (topic 6), where most of its inputs come from.

### Step 5a -- Transform

`Analysis/Process/Transform`, `changes_locs = True`, with a Preview and one
undo step.  Replaces SMAP's ApplyTransform, TransformLocsManually, Mirror
and ChangeCamPixelsize.

* **Target**: a file's localizations (optionally one `channel`), the
  default: the coordinates are changed (logged, undoable), never only the
  display, so measurements, exports and ROIs agree with the picture.  Or
  **an image layer**: sometimes the TIFF is what should move.  An image is
  moved by its **placement** -- `ImageData` gains an affine placement
  (today `x0`, `y0` and the pixel size), which its resampling onto the view
  honours -- so nothing is resampled and a move is lossless and undoable.
  For images only shift, rotation, scale and affine: the projective map is
  refused there.
* **Source**: *typed* (shift x, y, z; rotate about a centre; scale;
  mirror), a **named transformation** of the session, a **transformation
  file** (`_2ct.h5` and the new transformation file below), or **the drift
  of another file** (D1).
* **Named transformations**: the session keeps a list, saved with the
  localization file (decided after 5a: not in the workspace, the GUI file --
  a transformation belongs to the data) and recorded in the table's
  metadata when applied.  *Register
  layers* (5b) adds its result there; Transform chooses one by name.  A
  transformation can also be **written to a file** (and read back), so
  explicit transform files exist for scripts and other sessions.
* **The drift of another file** (D1): the usual case is simultaneous
  acquisition (two cameras): both files cover the same frames, and the
  curve applies frame by frame.  Frame ranges are checked; another case is
  refused with a message.  For sequential files, SMAP let a drift be
  referenced to the **first or the last frame**, so that two consecutive
  files corrected separately match where they meet: add that *reference*
  (first frame / last frame / as now -- read what "as now" is in
  `drift.py`) to the `Drift` result and to COMET and RCC (and 5c uses it).
  Bump their versions if the default changes numbers (it should not).
  *Fiducials* (5c, done first) fixes its curve to zero at the first frame
  (`fiducials.fit_drift`, the constraint row): give it the same choice.

Tests: each typed operation and their composition against hand-computed
coordinates; a `_2ct.h5` applied to the second channel of a simulated
two-camera pair matches the first; an image layer moved by placement lands
where the localizations moved by the same transformation land; a
transformation round-trips through its file; drift of another file applied
frame by frame, and refused on mismatched frames; a drift referenced to the
last frame is zero there.

### Step 5b -- Register layers

`Analysis/Register/Register layers`, beside *Calibrate transform*,
`changes_locs = False`.  Measures the shift of a target layer against a
reference layer by cross-correlation: of localization layers as rendered
(bin a setting), or of image layers.  Reuse what exists: the peak fit of
`rcc.py` and the correlation of `smappy/correlation.py` (step 17c); one
correlation, not a third.  x and y; **z optional, off by default** (SMAP's
XYZdisplacement: the shift in z from slices).  No rotation or scale for
now.  The result is added to the session's named transformations (5a) and
shown with the correlation and its peak; a button **applies it with
Transform** (opens Transform with that transformation as the source, or
runs it -- no second copy of the applying code).  Optionally limited to a
maximum displacement (SMAP's 2500 nm).

Tests: a layer and a shifted copy (known x, y, z) give the shift; an image
layer against a rendering of the same structure; the button's
transformation applied by Transform aligns the layers.

### Step 5c -- Fiducial drift

`Analysis/Drift/Fiducials`, a `Drift` result like COMET's and RCC's
(`drift_result.py`: the same figure, apply, save, and "drift of another
file" in 5a).  The tracks of beads give one dx(t), dy(t) and, where the
table has z, dz(t):

* **Beads** are the bright, long-lived tracks: link with *Link tracks*
  (`smappy/tracking/link.py`) with a generous memory, keep tracks above a
  minimum length (a fraction of the frames) and brightness, after the
  layer's filter.  A bead's track may be interrupted, and a bead may land
  on the glass during the measurement (a track starting late).
* Each track relative to itself (its own offset, fitted, not its first
  point), all of them **one spline** fitted to every bead's positions --
  COMET's spline representation, so a gap or a late bead costs nothing.
  All filtered beads are used; outlier rejection and a robust fit later.
* The *reference* of 5a (first / last frame).
* Beads stay in the table (removing them: later).

Tests: simulated beads with a known drift (with gaps and a late bead) give
the curve within their precision; the curve applied removes it; 2D and 3D.

### Step 5d -- Magnification over time in COMET (D3)

A time-dependent transformation in COMET beyond the shift: at least a
magnification changing with time (thermal or focus drift of the optics),
SMAP's `driftcorrectionM`.  COMET is vendored, its cost function in C++.
**The session starts with a short discussion with Jonas** (scale only or
rotation too; how it varies in time -- a spline like the drift; about which
centre; 2D only or 3D), then implements: the model, the cost, the `Drift`
result carrying it (and Transform applying it, 5a), the page, a simulated
test with a known magnification change.

**Decided with Jonas (2026-10-15).**  What it is for: samples that change
during imaging -- they expand, or drift differently in different places --
where a local drift correction works and a global shift does not; a small
rotation can happen too.

* **A choice of models**, the *transformation* setting, from the shift alone
  (as now, the default) through *xy scale*, *xy scale + rotation*, *xy
  affine* to *3D affine*.  Each is a per-frame matrix A(t) = I + sum of
  theta_p(t) E_p over the model's generators E_p, every theta_p a spline in
  time; simulations show which are robust.  In the 2D models z is only
  shifted.  SMAP's `driftcorrectionM` fitted the scale alone, in blocks
  correlated pairwise and splined afterwards, and no shift with it; here
  shift and transformation are one fit, in COMET's spline fit only (the
  legacy time-window fit refuses a transformation).
* **Then, with Jonas**: nothing turns about x or y, so *xy affine + z
  scale* -- the xy map with a scale of z of its own -- sits between *xy
  affine* and *3D affine*; the cost divides x, y and z by their own scales.
  Each parameter's variable is in nm at the edge of the data *along its
  own axis* (a z entry over the few hundred nm of depth, not the field's
  width).  A *transformation penalty* (advanced, 0 by default) pulls the
  relative change theta towards none, so the z entries stay near 1 on a
  thin sample.  Measured: z is as well determined as xy in nm (thousands
  of ppm of 300 nm is a nanometre); 1e-3 keeps z below 700 ppm on
  unchanging data and leaves xy alone, but also flattens a real 5% z
  change over 600 nm, so it is not on by default.
* **Time**: the same cubic B-spline as the drift by default; an advanced
  *transformation knots* setting gives it its own (coarser) spacing, for a
  transformation that varies much more slowly than the drift.
* **Centre**: the midpoint of the bounding box of the localizations the
  estimate used, as SMAP; recorded in the result.  The shift is fitted
  freely, so the centre does not change the correction, only how it splits
  into shift and transformation.
* **The cost** compares the two transformed positions of a pair after
  dividing their difference by the geometric mean of the two frames'
  linear scales -- det(A_xy)^(1/2) for x and y, z's own for z, det(A)^(1/3)
  for all three in *3D affine*: without it the overlap grows when
  everything shrinks, and the fit collapses the image.  The theta splines
  are kept at zero mean (the global transformation is not measurable from
  overlaps, only its change), and the *reference* makes A the identity and
  the shift zero at the first or last frame.
* **Result**: `Drift` carries A(t) (n_frames, 3, 3), the centre and the
  model; `apply` moves x, y, z together (corrected = c + A(x - c) - d);
  saved with the curve, in the `_driftc` file and in the session, so
  Transform's "drift of another file" applies it too.  The figure gains a
  second panel, the transformation's parameters in ppm (nm per mm from the
  centre; a rotation in microradians is the same unit), and the text a line
  with their range.

---

## Topic 2: stability

SMAP's *Vibrations* (one bead in the ROI: x, y, z against time, their SD
beside the mean precision, the amplitude spectrum 2|FFT|/L with its highest
peak) becomes the first approach of a diagnostic plugin, **Stability**:
later approaches are long-term drift (how a bead wanders over the
acquisition) and the dual-channel drift (*Check overlay*, topic 1, may land
here as an approach rather than a plugin of its own -- decide with topic 6).
It stays apart from fiducial drift correction (5c): that corrects, this
diagnoses; they share only the bead tracks.

### Step 5e -- Stability: vibrations

`Analysis/Measure/Stability`, `changes_locs = False`, `per_region`, with
approaches ticked on or off and each one's part shown only while on, as
*Spatial statistics* and *Tracking analysis* do; this step builds
*vibrations*.  An *id column* (`track_id` by default, from *Link tracks*), so
it runs on one bead in an ROI or on every bead track.

* **Input**: per bead, and the mean over beads; optionally **differential**
  -- two beads' difference, or each bead minus the mean of the others -- so
  motion common to the sample (stage, drift) cancels and what is left is
  relative motion and noise.  Frames where a bead is missing are filled by
  interpolation for the spectrum, and counted.  Time is the file's frame
  time (`metadata.file_frame_time`); MINFLUX's irregular times are
  interpolated onto a regular grid, as SMAP did.
* **Output per axis** (x, y, z), **each figure an option** (a tick), since
  there is a lot of it:
  * the **trace zoomed in time**, the first second or so (about 1000
    points, a setting), with a smoothed overlay -- where one sees the
    vibration directly; and the whole trace with slow drift removed (a
    high-pass above a chosen frequency), its SD beside the mean precision
    (SD^2 - precision^2 is the vibration's variance);
  * the **amplitude spectrum** in nm, as SMAP's, averaged over segments
    (Welch) so the peaks stand out;
  * the **cumulative RMS against frequency**: which band holds how much of
    the jitter;
  * a **table of the main peaks**, frequency and amplitude.
* **Values per region**: per axis the SD, the vibration RMS, and the
  largest peak's frequency and amplitude, so Histograms and Curves
  summarise many beads.

Tests: a simulated bead with a known sinusoidal vibration (frequency,
amplitude) plus localization noise gives the peak and its amplitude, and
the RMS; the differential mode cancels a vibration common to two beads;
gaps are filled and counted.
