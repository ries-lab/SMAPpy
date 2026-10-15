# Wave 4: spatial statistics, blinking, tracking -- the design

Decided with Jonas on 2026-10-15 (the questions and his answers are below
each step's decisions where they shaped them).  Each step is one session:
read this file, `SMAP_porting_plan.md` (wave 4), `wave4_research.md` (what
SMAP's plugins compute) and CLAUDE.md, then implement the step with its
tests and its page, run `--slow`, update the plan's *Done* list, commit and
push.

The decisions behind all of it:

* **Reuse what wave 2 built.**  Ripley's K at r is the mean neighbour count
  within r over the density: Neighbourhood's count (`csrc/neighbours.cpp`,
  `Index`) with its two-layer convention.  Pair correlation, K, L and H are
  one computation -- a histogram of pair distances, normalised by the
  window -- so they are one plugin, and it gets its pairs from the same C++
  index.  Spatial statistics measures curves and changes nothing;
  Neighbourhood writes a column per localization.  So they stay two
  plugins, sharing the engine.  Thresholding a local L(r) and joining what
  is dense (Getis-Franklin clustering) is what DBSCAN already does: no new
  clustering algorithm.
* **One plugin with a list of approaches** where SMAP had several
  (`SpatialPointPattern`, `correlation`, `Ripley`): each approach ticked on
  or off, with its own few settings shown only while it is on
  (`FormState.shown`).
* **Keep it simple** for counting: plain, well-named functions with a clear
  use.  The NPC labelling efficiency already fits LE from corners; a
  general counting fit (copy number N0 unknown) waits.
* **Tracking lives in smappy**, not in a separate package: the renderer is a
  core change anyway, a track is a region (an id column), and MINFLUX needs
  the same.  Single-particle tracking is fitted from single molecules, so it
  is **sparse**: a jump comparable to the distance between molecules means
  motion blur or overlapping spots that cannot be fitted.  The linking
  problem is therefore easy -- what matters is bridging frames where the
  fluorophore was dark, dim or badly fitted.  This is why u-track was never
  used in the end; the linker is simple, and a global assignment is only
  benchmarked against it.
* **Time**: every file gets a frame time (step 18); everything in seconds
  reads it.

Order: 17a, 18, 19a and 19b can run in parallel.  17b and 17c after 17a;
19c after 18 and 19a.

Later (recorded, not planned): a counting fit for the copy number on top of
the NPC model's blinking and LE (`smappy/npc.py`: `_copy_pgf`, the FFT over
roots of unity); DNA-PAINT and qPAINT, after looking at what Picasso does
and how it is used; step finding (AutoStepFinder, for MINFLUX) and changes
of motion within a trajectory, as their own plugins; hidden Markov states;
Ripley in 3D; the movie export (wave 5, plan section I), which must offer
the tracks (step 19b's render mode, with a *tail*: the frames before "now")
over the raw frames.

---

## Step 17a -- Spatial statistics: the radial approaches

`Analysis/Measure/Spatial statistics`, module `plugins/spatial_statistics.py`,
`per_region = True`, `changes_locs = False`.  Ports SMAP's
`SpatialPointPattern`, `myRipleysK` and the 2D pair correlation of
`correlation.m` (`paircrosscorrelationfunction`).

**Layers**: every visible localization layer as it shows (its filter, its
*merge blinks*: `ctx.table(i)`), the auto-correlation of each and the
cross-correlation of each pair, as SMAP's `correlation.m`.  Lateral (x, y)
only.

**The computation.**  One pass of a new mode of the C++ `Index`
(`csrc/neighbours.cpp`, beside `Count`, `Gauss`, `Ball`): the histogram of
the distances of every pair closer than r_max, in bins of dr, per file,
summed over threads.  Normalised by the **window's set covariance** (the
FFT autocorrelation of the window's mask, on a coarse grid): this is Ohser's
translation edge correction, and it is what Veatch's normalisation in
SMAP's `paircrosscorrelationfunction` does, for any window shape.  That
gives g(r); K is its cumulative integral, L = sqrt(K / pi), and the plugin
shows L - r and H.  SMAP binned the localizations at 2 nm and Fourier
transformed them (400 M pixels for a 40 um field); counting pairs costs
N rho pi r_max^2 (about 10^9 pairs for 10^6 localizations at r_max =
500 nm, about a second) -- time it, and add a `preflight` above a few
seconds.  Self-pairs are left out (g(0) is not an outcome).

**The window** (Q2): in a region the ROI's shape (`ctx.region`); else the
drawn ROI; with neither, the **occupied area** -- a mask of the localizations
binned at about 200 nm and closed -- or the bounding box (a choice).  SMAP
used the field of view, which makes a uniform cell look clustered.

**The null model** (Q3): Monte Carlo envelopes, n simulations (99 by
default), the same number of points in the same window, a pointwise band
and one global test (the MAD test).  Two nulls: **CSR** of what the layer
shows (the page says: merge blinks first), and **keep blinks**: each blink's
localizations moved together to a random place (through `group_id`), so the
null has the same blinking.  For a cross-correlation layer A stays fixed and
B is redistributed.  Sengupta's PC-PALM fit is not ported now.

**Values per region** (Q4): per curve, the position and height of the first
maximum of g(r) and of L(r) - r; g and L - r at a chosen **period** (a list
of distances gives one value each); whether the curve leaves the envelope.
**Arrays**: r and each curve, with the pair counts and the expected counts
(so step 17b pools them), and the envelope.

**Neighbourhood, a small addition** (same step): the *count* quantity gains
*express as* -- *count*, *local L(r)* (Getis and Franklin 1987: sqrt(A n /
(pi (N - 1))), the L function of each localization), or *relative to the
mean density* -- using the same window normalisation.  Its page explains
local L and that DBSCAN thresholds the same thing.  Bump its version.

Tests: CSR in a square, a disc and an L-shaped window gives g = 1 and
L - r = 0 within the envelope; simulated clusters of known radius give the
maximum of L - r near it; a cross-correlation of two independent layers is
1, of a layer with a copy shifted by d peaks at d; the C++ histogram equals
a brute-force one.  Page with figures (CSR and clusters with their
envelopes).

## Step 17b -- Curves summary

`Analysis/Regions/Summarize/Curves`, a `SiteAnalysisPlugin`: reads the
arrays of any evaluation (Spatial statistics, Line Profile, step 17c, later
MSD) that share an x axis, and plots the **mean with its spread** (SD or
SEM, a choice) over the regions, **grouped by an ROI keyword** so conditions
are compared in one plot.  Where the plugin hands back a numerator and a
denominator (pair counts and expected counts), the pooled estimator (sum
over sum) instead of the mean of ratios.  Curves on different x grids are
interpolated onto the first one's, or refused (decide, and say why).
`Result.data` holds the summary curves; values per group (the mean of each
value column) go in `Result.text`.

Tests: regions with known curves give their mean and SD; the pooled
estimator equals g of the union of equal-window regions.

## Step 17c -- Spatial statistics: the directional approaches

Two more approaches of the same plugin (Q1: one plugin, not a Line
correlation of its own):

* **2D correlation image**: the auto- and cross-correlation of the rendered
  layers (FFT, bin dr), not averaged over angle, with its normalisation by
  the window as in 17a.
* **1D correlation along a line**: SMAP's `correlationtools`: the profile
  along the line, its autocorrelation and spectrum, the **period** (the
  peak nearest a guess, as SMAP's `selectpeak`) and its magnitude,
  cross-correlation of two layers.  Values: period and magnitude per layer.

**Axes**: with a line or a rectangle ROI, x and y are the ROI's own
directions (along and across; Line Profile's `project` and `region_line`,
the named lines per ROI included).  The axes are selectable: x-y, x-z or
y-z for 2D, and x, y or z for 1D.  The z range is taken from the data, as
robust quantiles, clipped to -500 .. 500 nm, so a few far-out
mislocalizations do not stretch it.

Tests: a periodic structure (simulated rings every 190 nm along a line)
gives its period in x and, turned, along a tilted line ROI; a z-periodic
one in z.

---

## Step 18 -- Frame time and Blinking

**Frame time** (Q9): each file has a frame time, `metadata["camera"]
["frame_time_ms"]` (or a field of `CameraMetadata`: decide by where the
other per-file numbers live), read from the file's metadata where it has a
frame or cycle time, otherwise the exposure; editable where the camera's
metadata is edited (`gui/camera_editor.py`).  Per-frame timestamps (NDTiff
has them) later.  Time Course and Localization Statistics' on-time should read it
(they take the exposure now).

**Blinking**: `Analysis/Measure/Blinking`, `per_region = True`, plain
module-level functions.  Per region: localizations, blinks (as the layer
merges them), bursts at a dark time T (runs of frames separated by more
than T: SMAP's `locsdf`), mean on-time, mean dark time, in frames and
seconds.  On the view: per file, the histograms of on-times, dark times and
blinks per molecule (SMAP's `dyeblinking`, a molecule being the
localizations linked with the long dark time T).  No counting fit, no
cluster finding of its own (Find clusters, Segment regions and Filter ROIs
do that), no qPAINT (later).

Tests on simulated kinetics (`simulate/kinetics.py`) with known on- and
off-rates.

---

## Step 19a -- Track simulation and linking

**Simulation**: Brownian tracks in `smappy.simulate`, one or more diffusive
states with switching, blinking (dark frames inside a track), a
localization precision and a density; the truth (`track_id_true`) kept, so
tests and the benchmark score against it.

**Link tracks**: `Analysis/Tracking/Link tracks`, module
`smappy/tracking/link.py` plus the plugin.  Writes `track_id` (1, 2, ...;
0 for unlinked localizations and tracks shorter than the minimum) and
`track_length`, an id column: every `per_region` plugin runs per track.
Settings: maximum jump (nm), memory (dark frames bridged), minimum length,
2D or 3D, per file and per channel.  Two linkers, an *algorithm* choice:

* **simple** (the baseline and likely the default): frame by frame, the
  nearest candidate within the maximum jump, track ends kept open for
  *memory* frames; conflicts resolved nearest first.  SMAP's SimpleTracking
  (Crocker-Grier) is the reference for the settings.
* **global assignment**: per frame a linear assignment
  (`scipy.optimize.linear_sum_assignment` on the sparse candidates) over the
  open ends, the same gap bridging, cost the squared jump (optionally
  scaled by the precisions).

Candidates come from the C++ `Index`.  **Benchmark** both (and trackpy, in
a study script only, not a dependency) on the simulations: fraction of
correct links, whole tracks recovered, and speed, over density and jump
size; the result (and the choice of default) goes on the page and in
NOTES.md.

## Step 19b -- The tracks render mode

A fifth render mode, *tracks*, next to the SMLM modes (`render.MODES`,
`gui/render_tab.py`'s `RENDER_MODES`): the segments between consecutive
localizations of each track (`track_id`, or a chosen id column such as
`tid`; sorted by frame), drawn antialiased in C++ (`csrc/render.cpp`), a
line width, coloured as the layer colours (by any column: time, D, the
shuffled id), culled per track by its bounding box.  A raster, so the
export, the overlays and the sum of layers work unchanged (a vector layer
kind was rejected: exports would miss it).  A static view draws each track
whole within the frames the layer's filter keeps; the *tail* is the movie
export's (wave 5).

Tests: a known track renders its segments; culling does not lose a track
crossing the view; the Render tab offers the mode (`test_panel_docs` and the
Render page).

## Step 19c -- Tracking analysis

`Analysis/Tracking/Tracking analysis`, approaches ticked as in Spatial
statistics; an *id column* setting (`track_id`, or MINFLUX's `tid`);
vectorised over the whole table (sorted by track, then frame) for 10^5
tracks, and `per_region` so it runs in an ROI or per track.

* **MSD**: per track and over all tracks, fitted with the localization-error
  offset (4 sigma^2, from the precisions) and the number of lags chosen as
  Michalet (2010); D and alpha.  Uses the frame time (step 18).
* **Jump distances**: the distribution of jumps at a lag, fitted with one to
  three diffusive states with the localization error; D and fraction per
  state.

Per-track D (and alpha) written as columns onto the localizations, so the
tracks can be coloured by them (19b).  Tests on step 19a's simulations with
known D, states and precision.
