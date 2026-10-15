---
version: "1"
covers: [smappy.plugins.spatial_statistics.pair_counts, smappy.plugins.spatial_statistics.expected_counts, smappy.plugins.spatial_statistics.curves_of, smappy.plugins.spatial_statistics.window_of_region, smappy.plugins.spatial_statistics.window_of_points, smappy.plugins.spatial_statistics.Window.covariance, smappy.plugins.spatial_statistics.keep_blinks, smappy.plugins.spatial_statistics.mad_test, smappy.plugins.spatial_statistics.pointwise_band, smappy.plugins.spatial_statistics.first_maximum, smappy.plugins.spatial_statistics.spatial_statistics, smappy.plugins.spatial_statistics.curve_values, smappy.plugins.spatial_statistics.SpatialStatistics.run, smappy.plugins.spatial_statistics.SpatialStatistics.preflight]
---

## What it does

Asks whether the localizations are [clustered](plugin:Panels/Glossary) --
closer together than chance would put them -- and at what distance.  It
writes nothing to the table; it draws curves and, per region, gives
numbers.

The curves all come from one count: how many pairs of localizations lie at
each distance *r* from each other.  Divided by what [complete spatial
randomness](plugin:Panels/Glossary) (CSR: every place in the window equally
likely, independently of the others) would give in the same window, it is
the **pair correlation** $g(r)$: 1 for random positions, above 1 at the
distances where localizations come together, below 1 where they keep
apart.  Added up from 0 to *r* it is **Ripley's K(r)**, the mean number of
neighbours within *r* over the density (Ripley 1977); made
straight, $L(r) - r$ is 0 for random positions and above 0 for clusters
(many papers call it $H(r)$).  g shows each distance on its own, and so
which distances matter; L - r adds up all the distances below *r*, which
makes it smoother and is why its maximum is the usual estimate of a
cluster's size ([Kiskowski et al. 2009](https://doi.org/10.1016/j.bpj.2009.05.039)).
These are the *approaches*; tick the ones to see.  More are to come
(a correlation image, a correlation along a line), each with its own
settings.

Every visible layer is measured as it shows -- with its filter, with or
without [merged blinks](plugin:Panels/Glossary) -- and with two or more,
also every pair: the **cross-correlation** of two channels is the same
count with one end of each pair in each, 1 when the two are independent
and above 1 at the distances where they meet.  This is the pair
correlation of [Veatch et al. 2012](https://doi.org/10.1371/journal.pone.0031457)
and PC-PALM ([Sengupta et al. 2011](https://doi.org/10.1038/nmeth.1704)),
counted without a picture.

**Blinking is clustering.**  One molecule seen ten times is ten
localizations within a few tens of nm of each other: g is far above 1 at
small *r* even for molecules placed at random.  Either measure a layer
that merges blinks, or compare with the *keep blinks* envelope (below),
which blinks the way the data do -- and still remember that a molecule
that blinks again after a long dark time is two blinks.

Use it on a region with a uniform background -- a patch of membrane, a
nucleus -- or on every ROI or cluster as a per-region step.  It is lateral
(x and y) only.

## How it works

```figure-setup
import tempfile, yaml
from pathlib import Path
from smappy.group import group
from smappy.simulate import simulate
from smappy.simulate.settings import (BlinkingSettings, SimulationSettings,
                                      StructureSettings)
from smappy.plugins.spatial_statistics import (Points, draw_curves, draw_window,
                                               expected_counts, first_maximum,
                                               pair_counts, spatial_statistics,
                                               window_of_points)
R = 60.0
disc = [[float(R * np.cos(t)), float(R * np.sin(t))]
        for t in np.linspace(0, 2 * np.pi, 33)[:-1]]
square = [[0, 0], [8000, 0], [8000, 8000], [0, 8000]]

def simulated(parts, seed, **blinking):
    path = Path(tempfile.mkdtemp()) / "structure.yaml"
    path.write_text(yaml.safe_dump({"name": "spatial", "parts": parts}))
    settings = SimulationSettings(n_frames=2000, seed=seed,
                                  structure=StructureSettings(file=str(path)))
    settings.blinking = BlinkingSettings(**blinking) if blinking else settings.blinking
    locs = simulate(settings)
    locs = locs[np.asarray(locs["photons"]) > 200]
    group(locs)                                     # writes group_id
    return np.column_stack([locs["x_nm"], locs["y_nm"]]), np.asarray(locs["group_id"])

# clusters of radius 60 nm over a few scattered molecules
clustered, clustered_blinks = simulated(
    [{"elements": [{"polygon": disc, "density": 0.004}],
      "copies": {"n": 100, "field": [0, 0, 8000, 8000], "min_distance": 300}},
     {"elements": [{"polygon": square, "density": 2e-5}]}], seed=4)
# molecules at random, each blinking once for a few frames
scattered, scattered_blinks = simulated(
    [{"elements": [{"polygon": square, "density": 6e-5}]}], seed=5,
    blinks=1.0, activation="all", on_time=4.0)
window_c = window_of_points(clustered[:, 0], clustered[:, 1])
window_s = window_of_points(scattered[:, 0], scattered[:, 1])
edges, (by_csr,) = spatial_statistics([[Points(clustered, clustered_blinks)]], [window_c],
                                      300, 5, null="keep_blinks", n_sims=19)
mid = 0.5 * (edges[1:] + edges[:-1])
_, (blink_csr,) = spatial_statistics([[Points(scattered, scattered_blinks)]], [window_s],
                                     300, 5, null="csr", n_sims=19)
_, (blink_kept,) = spatial_statistics([[Points(scattered, scattered_blinks)]], [window_s],
                                      300, 5, null="keep_blinks", n_sims=19)
```

**1. The window.**  Where localizations could have been: the region's
shape when it runs in an ROI; else the ROI drawn in the view; else, as
*without an ROI* says, the **occupied area** -- the localizations binned in
squares of about 200 nm, a square kept when it holds one, the gaps between
molecules closed -- or their bounding box.  The window matters: g compares
the data with random positions *in the window*, and a window larger than
the cell (the field of view around a cell, which is what SMAP used) makes
any cell look clustered, since all its localizations are in one part of it.

**2. The pairs.**  The distance of every pair of localizations closer than
*up to r*, counted in rings *bin width* wide.  A localization is never
paired with itself.  Each file is counted on its own.

**3. What chance expects.**  The same number of localizations placed at
random in the window would give, in each ring, a number of pairs that
depends on the window's shape: near the edge part of every ring is outside
it.  This expected count is worked out exactly for any window shape
(*In detail*), and g is the count over it.

**4. The envelope.**  The localizations are placed at random in the window
again and again -- 99 *simulations* -- and the same curves computed each
time.  The shaded band holds the middle 95 % of them at each *r*; the
dotted lines are the band of the **global test**, which takes all distances
at once: the data leave it somewhere only one time in 20 by chance, so a
curve outside it is clustered (or dispersed) with a 5 % error, where a
curve that pokes out of the shaded band at one *r* out of 60 means little.
*Envelope* says how they are placed: **random (CSR)** puts every row the
layer shows anywhere; **random, keep blinks** moves each blink -- the
localizations one molecule gave in consecutive frames -- as a whole, so the
simulations blink like the data and only the arrangement of the molecules
is tested.  For a cross-correlation the first layer stays where it is and
the second is placed at random.

```figure Clusters of radius 60 nm: g (left) is far above 1 up to about twice the radius, L - r (right) peaks at 100 nm, between one and two radii. Both leave the envelope of whole blinks placed at random (shaded: pointwise 95 %; dotted: the global test).
fig.set_size_inches(7.4, 3.0)
left, right = fig.subplots(1, 2)
draw_curves(left, mid, [by_csr], "g", ["C3"])
draw_curves(right, mid, [by_csr], "L_r", ["C3"])
right.axvline(R, color="0.5", linewidth=0.6)
right.axvline(2 * R, color="0.5", linewidth=0.6)
```

```figure Molecules at random, each seen in a few consecutive frames. Against random localizations (left) the blinks alone make the curve leave the envelope: it looks clustered at 20 nm. Against random blinks (right) it stays inside: the molecules are random.
fig.set_size_inches(7.4, 3.0)
left, right = fig.subplots(1, 2, sharey=True)
draw_curves(left, mid, [blink_csr], "L_r", ["C0"])
draw_curves(right, mid, [blink_kept], "L_r", ["C0"])
left.set_title("envelope: random (CSR)", fontsize=9)
right.set_title("envelope: random, keep blinks", fontsize=9)
```

## In detail

**The estimator.**  With $n$ localizations in a window $W$ of area $|W|$,
the count $H(r_0, r_1)$ of ordered pairs at distances between $r_0$ and
$r_1$ (each pair twice for a layer with itself, a localization never with
itself) is divided by its expectation under CSR,

$$E(r_0, r_1) = \frac{n (n - 1)}{|W|^2} \int_{r_0 \leq |h| < r_1} \gamma_W(h) \, dh ,$$

where $\gamma_W(h) = |W \cap (W + h)|$ is the window's **set covariance**,
the area the window shares with itself shifted by $h$.  For two layers the
numerator is $n_A n_B$, less the rows the two layers share.  So

$$\hat{g}(r) = \frac{H(r_0, r_1)}{E(r_0, r_1)} , \quad \hat{K}(r) = \int_0^r 2 \pi s \, \hat{g}(s) \, ds , \quad \hat{L}(r) = \sqrt{\hat{K}(r) / \pi} .$$

This is the translation edge correction averaged over directions, and it is
what Veatch's normalisation does in Fourier space -- the picture's
autocorrelation divided by the mask's.  $\gamma_W$ is the autocorrelation of
the window's mask, by FFT: on a raster of 512 pixels along the longer side
for a drawn shape (a rectangle exactly), on the occupied area's own 200 nm
squares, padded so no lag up to *up to r* wraps.  The set covariance of a
raster is exactly bilinear between whole-pixel lags, so it is read there at
90 directions over half a turn ($\gamma_W(h) = \gamma_W(-h)$) and averaged.
K is summed ring by ring and read at the bin centres; L - r is shown
there.  Rings beyond the window's size, where nothing can be expected, are
NaN.  Several files are pooled: their counts and expectations are added,
and g is the ratio of the sums.

```figure The window's correction on random points in a disc: divided by what an infinite plane would give (grey), g falls with r because the pairs that would cross the edge are missing; divided by the disc's set covariance (black) it stays at 1.
fig.set_size_inches(4.2, 3.0)
ax = fig.subplots()
from smappy.plugins.spatial_statistics import window_of_region
from smappy.regions import Region
rng = np.random.default_rng(3)
disc_roi = Region.circle((0, 0), 2000)
w = window_of_region(disc_roi)
p = w.uniform(4000, rng)
e = np.arange(0, 1505, 25.0)
h = pair_counts(p, None, 1500, 25)
plane = len(p) * (len(p) - 1) / w.area * np.pi * (e[1:] ** 2 - e[:-1] ** 2)
r = 0.5 * (e[1:] + e[:-1])
ax.plot(r, h / plane, color="0.6", label="no correction")
ax.plot(r, h / expected_counts(w, len(p) * (len(p) - 1.0), e), color="k",
        label="set covariance")
ax.axhline(1, color="k", linewidth=0.5)
ax.set_xlabel("r (nm)")
ax.set_ylabel("g(r)")
ax.legend(fontsize=7, frameon=False)
```

**The pairs** come from Neighbourhood's index: the localizations sorted
into square cells *up to r* wide, each one testing the nine cells around
it, on every core, the distances binned as they are found and never
stored.  A million localizations over 40 µm to 500 nm -- half a billion
pairs -- take about 1.6 s on four cores.  A run is estimated before it
starts (the simulations repeat the count each time), and one expected to
take more than 5 s asks first, offering to run without the envelope or
with 19 simulations.

**The occupied area** is the localizations of all the layers binned on a
grid from their smallest to their largest x and y, squares of about
*occupied bin*; a square holding a localization is in, and the mask is
closed (dilated and eroded twice by one square), so the gaps between
molecules are not holes.  A uniform rectangle's occupied area is the
rectangle.

**Simulations.**  CSR places $n$ points uniformly in the window's raster.
*Keep blinks* takes the blinks from `group_id` (merge blinks once on the
layer, so the column exists), and puts each blink's centre uniformly in
the window, its localizations keeping their offsets from it; a blink that
then sticks out of the window is placed again, up to 30 times.  A layer
that shows merged blinks has one row per blink, and keep blinks is CSR.

**The tests.**  Pointwise: the 2.5th and 97.5th percentiles (at *level*
0.95) of the simulated curves at each r.  Globally, Diggle's maximum
absolute deviation (MAD) test on L - r, over r from *from r*: $T = \max_r
|\hat{L}(r) - r - \bar{L}(r) + r|$, the deviation from the simulations'
mean $\bar{L}$; each simulation's $T_i$ is taken from the mean of the
others, and $p = (1 + \#\{T_i \geq T\}) / (n + 1)$.  The global band is the
mean $\pm$ the $\lceil 0.95 (n + 1) \rceil$-th smallest $T_i$, so the curve
leaves it exactly when $p \leq 0.05$ ([Baddeley et al. 2014](https://doi.org/10.1890/13-2042.1)).
With 99 simulations the smallest p is 0.01; with 19, 0.05.  The test is on
L - r rather than g because its spread is about the same at every r, while
g's is largest at small r.

**The first maximum** of a curve is read off the curve smoothed over 5
bins: the first local maximum at r from *from r* (the first bin counts
when the curve falls from it) that rises at least half as far above the
random value (1 for g, 0 for L - r) as the highest one does.  So a small
bump of noise before the real peak is passed over, and of two equal peaks
-- rings of a periodic structure -- the first is taken.

## Parameters

### pair_correlation
The one to read for a distance: g's peaks are where the localizations come
together, and for a periodic structure each period gives one.

### ripley_k
K grows as $\pi r^2$ for random positions, so its curve says little by
eye; it is here for comparison with other software.  L - r is the same
information, drawn readably.

### ripley_l
The smooth one, and the usual measure of a cluster's size: for clusters of
radius R its maximum lies between R and 2R, depending on how dense the
clusters are against the background.

### radial.r_max_nm
A few times the size of the structures; the time grows with its square.
Beyond about a third of the window's size few pairs fit and the curves are
noisy.

### radial.dr_nm
About the localization precision.  Finer bins hold fewer pairs each and the
curve is noisier; K and L - r, being sums, are not affected much.

### radial.cross
Off, only each layer with itself.

### radial.at_nm
Distances to read g and L - r at, as numbers for the region table --
the expected period of a structure, the radius of a cluster -- so ROIs or
conditions can be compared at one distance.

### radial.window
The occupied area for a cell or a tissue, whose edge is the edge of the
data; the bounding box for data that fill a rectangle (a simulation, or a
field cut to a rectangle).  With a region or a drawn ROI, that shape is the
window and this is not used.

### radial.occupied_bin_nm
Large enough that most squares inside the cell hold a localization (a few
times the typical distance between neighbours), small enough to follow the
outline.  Too small and the gaps between molecules count as outside: the
window shrinks to the localizations and g drops below 1.

### radial.null
*Random (CSR)* on a layer that merges blinks; *random, keep blinks* on one
that does not.  *None* to see the curves fast.

### radial.simulations
99 gives a test at 1 % and a smooth band; 19 is the fewest for a 5 % test.
Each costs as long as the measurement.

### radial.level
0.95: a 5 % test.  0.99 needs at least 99 simulations.

### radial.r_min_nm
Raise it past the blinking (two or three times the precision) to look for
the first maximum and test only at the distances above it.

### radial.seed
The same seed gives the same envelope.

## Output

**Figures**: one per approach ticked -- each layer's curve in its colour,
the cross-correlations dashed in grey, the pointwise band shaded, and for
L - r the global band dotted; the legend gives each curve's p.  The
*window* figure shows the window and the localizations measured in it
(the first file's): check it before trusting a curve.

**Text**: the window, its area, the number of files, and per curve the
first maximum of g and of L - r and the test.

**Values per region** (a column each in the region table; the first layer
with no suffix, then `_L2`, ..., and `_L1xL2` for a pair):

* `g_max_r_nm`, `g_max` -- where the first maximum of g is, and its height;
* `lr_max_r_nm`, `lr_max_nm` -- the same of L - r;
* `g_at_<d>nm`, `lr_at_<d>nm` -- both at each of *at distances*;
* `p_csr` -- the global test's p, and `leaves_envelope` 1 when the curve
  leaves the global band (p at most 1 - *level*), else 0.

**Arrays per region**: `r_nm` (the bin centres), and per curve `pairs`
and `expected` (the counts, so a summary over regions can pool them as
the sum of the pairs over the sum of the expectations), `g`, `K_nm2`,
`lr_nm`, and the bands `g_lo`, `g_hi`, `lr_nm_lo`, `lr_nm_hi`.

A good result: g near 1 and L - r near 0 at large r, a clear peak inside
the window's size, the curve outside the envelope at the distances that
matter.  A bad one: g drifting away from 1 at large r -- the window does
not fit the data (the occupied area's squares too small or too large, a
drawn ROI reaching past the cell), or the density is not uniform at the
scale of the window (a gradient looks clustered at every r).

## Differences from SMAP

Based on SMAP's `SpatialPointPattern` with its helper `myRipleysK`, and the
2D pair correlation of `correlation` (`paircrosscorrelationfunction`)
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).  The main changes:

* **Pairs instead of pictures.**  SMAP binned the localizations at 2 nm and
  took the Fourier transform of the whole picture -- 400 million pixels for
  a 40 µm field.  Here the pairs are counted at their exact distances; only
  the window is Fourier transformed, on a coarse raster.
* **The window.**  `SpatialPointPattern` normalised by the drawn ROI or,
  without one, a rectangle twice the size of the view; `myRipleysK` kept
  only the localizations farther than r from the edge of the bounding box
  (minus sampling), which throws most of a small ROI away.  Here the
  window is the region, the drawn ROI or the occupied area, and the same
  set covariance corrects every shape.
* **Envelopes and a test** are new: SMAP drew the curves alone.
* **No self-pairs**: SMAP's first bin held them and its plots dropped the
  first bins by hand.
* `SpatialPointPattern`'s cross term multiplied the magnitudes of the two
  layers' transforms, which loses where the two are relative to each
  other; it was not a cross-correlation.  Here the cross-correlation is a
  count of pairs, one end in each layer, as in `correlation`.
* Every visible layer and every pair, per file, and per region; SMAP had
  one or two layers chosen in the dialog.

## References

* Ripley BD. Modelling spatial patterns. *J R Stat Soc B* 39, 172-212
  (1977) -- the K function.
* Getis A, Franklin J. Second-order neighborhood analysis of mapped point
  patterns. *Ecology* 68, 473-477 (1987).
  [doi:10.2307/1938452](https://doi.org/10.2307/1938452) -- L, and its
  local version (Neighbourhood's *local L(r)*).
* Kiskowski MA, Hancock JF, Kenworthy AK. On the use of Ripley's K-function
  and its derivatives to analyze domain size. *Biophys J* 97, 1095-1103
  (2009). [doi:10.1016/j.bpj.2009.05.039](https://doi.org/10.1016/j.bpj.2009.05.039)
  -- the maximum of L - r and the size of a cluster.
* Sengupta P, Jovanovic-Talisman T, Skoko D, Renz M, Veatch SL,
  Lippincott-Schwartz J. Probing protein heterogeneity in the plasma
  membrane using PALM and pair correlation analysis. *Nat Methods* 8,
  969-975 (2011). [doi:10.1038/nmeth.1704](https://doi.org/10.1038/nmeth.1704)
  -- pair correlation of localizations, and blinking as clustering.
* Veatch SL, Machta BB, Shelby SA, Chiang EN, Holowka DA, Baird BA.
  Correlation functions quantify super-resolution images and estimate
  apparent clustering due to over-counting. *PLoS ONE* 7, e31457 (2012).
  [doi:10.1371/journal.pone.0031457](https://doi.org/10.1371/journal.pone.0031457)
  -- the normalisation by the window.
* Baddeley A, Diggle PJ, Hardegen A, Lawrence T, Milne RK, Nair G. On tests
  of spatial pattern based on simulation envelopes. *Ecol Monogr* 84,
  477-489 (2014). [doi:10.1890/13-2042.1](https://doi.org/10.1890/13-2042.1)
  -- the global envelope and the MAD test.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
