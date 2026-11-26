---
version: "1"
covers: [smappy.pixel_equalise.fit_smooth, smappy.pixel_equalise.fit_smooth_axis, smappy.pixel_equalise.SmoothAxis, smappy.pixel_equalise.fourier_cdf, smappy.pixel_equalise.keep_monotone, smappy.pixel_equalise.place, smappy.plugins.pixel_equalisation.PixelEqualisation, smappy.plugins.pixel_equalisation.equalise_table, smappy.plugins.pixel_equalisation.revert_table, smappy.plugins.pixel_equalisation.draw_places]
---

## What it does

A neural network that localizes molecules -- [DECODE](plugin:Localize/DECODE)
and its optimised version -- tends to put a molecule it is unsure of near the
centre of a camera pixel.  The localizations then crowd toward the pixels'
centres instead of lying evenly within them:
[pixelation](plugin:Panels/Glossary) (also called pixel locking), the
localizations crowding toward the centres or edges of the camera's pixels.
In a rendered image it draws the camera's grid, a fine pattern of dots or
stripes about 100 nm apart, most visible where the network is least sure:
far from focus and for dim molecules.

The sample knows nothing of the camera's grid, so the true positions lie
evenly within the pixels.  This plugin uses that fact.  It moves each
localization within its own pixel so that, among localizations of similar
z and similar uncertainty, the places within the pixel become even again.
The grid disappears from the picture.

**What it does not do: make the localizations more precise.**  On simulated
data with the true positions known, the moves are as often away from the
truth as toward it.  On real data, two localizations of one molecule in
frames that share nothing end up slightly further apart, not closer (*In
detail*).  The crowding is mostly how the network's errors are spread, not
a shift of the molecules that could be undone.  So use it for images and
for analyses that a grid would disturb, such as correlations and spatial
statistics at the scale of a pixel.  Do not use it to improve a precision,
a cluster size or a distance.

**Correct and revert.**  The button says *Correct* on a table that has not
been equalised and *Revert* on one that has.  Each localization's move is
kept in the table as two attributes, `x_pixel_shift_nm` and
`y_pixel_shift_nm`, so *Revert* puts every localization back exactly, even
after a drift correction.

It needs the positions as they were fitted, because only then is the place
within a pixel the camera's.  So run it **before drift correction** -- or
let DECODE run it: its *after the fit* has it as the first step, *pixel
equalisation*, off by default.  On a table that records a drift correction
it refuses to correct.  It uses
`z_nm` and the uncertainties `x_err_nm` and `y_err_nm` where the table has
them, and works without them on a 2D fit.

## How it works

```figure-setup
from smappy.pixel_equalise import fit_smooth, place, fourier_cdf
from smappy.plugins.pixel_equalisation import draw_places
rng = np.random.default_rng(1)
n = 300_000
true = rng.uniform(0, 40, (n, 2))           # in pixels; the sample knows no grid
z = rng.uniform(-600, 600, n)
err = rng.uniform(0.05, 0.2, (n, 2))       # each localization's uncertainty, in pixels
# what a network does where it is unsure: a pull toward the pixel's centre that grows with
# the uncertainty and the distance from focus, and a little random error on top
edge = np.round(true) - 0.5
u = true - edge
strength = 0.05 + 2.0 * (err - 0.05) * (0.1 + np.abs(z)[:, None] / 600)
found = edge + u + strength * np.sin(2 * np.pi * u) / (2 * np.pi) + rng.normal(0, 0.02, (n, 2))
eq = fit_smooth(found[:, 0], found[:, 1], z, err[:, 0], err[:, 1])
fx, fy = eq(found[:, 0], found[:, 1], z, err[:, 0], err[:, 1])
equalised = np.c_[fx, fy]
```

**1. Where each localization sits within its pixel.**  For each
localization the plugin takes its place within the camera pixel, from 0 at
one edge to 1 at the other, in x and in y separately.  If the network did
not pixelate, these places would be spread evenly.

```figure A uniform sample localized by a network that pixelates, every localization folded into one camera pixel (place within the pixel in x and y, 20 bins each, the same grey scale).  As localized, the pixel's centre is crowded: in a rendered image, a grid of bright dots one pixel apart (left).  Equalised, the pixel is even (right).
fig.set_size_inches(7, 3.4)
axes = fig.subplots(1, 2)
bins = np.linspace(0, 1, 21)
hists = [np.histogram2d(place(xy[:, 0]), place(xy[:, 1]), (bins, bins))[0] for xy in (found, equalised)]
top = max(h.max() for h in hists)
for ax, h, title in zip(axes, hists, ("as localized", "equalised")):
    ax.imshow(h.T, origin="lower", cmap="gray", extent=(0, 1, 0, 1), vmin=0, vmax=top)
    ax.set_title(title)
    ax.set_xlabel("place within the pixel, x")
axes[0].set_ylabel("place within the pixel, y")
```

**2. How uneven they are, for alike localizations.**  The crowding is
strongest where the network is least sure, so localizations are not pooled
blindly.  How uneven the places are is described as a smooth function of
two things: the localization's z and its uncertainty along that axis.

**3. Each moved to where it would be if they were even.**  Among alike
localizations, the one at the 30th percentile of places is moved to 0.3 of
the pixel, the one at the median to its middle, and so on.  This is the
place's cumulative distribution, which turns any distribution into an even
one.  The move never takes a localization out of its pixel, and never
swaps the order of two alike localizations within it.

```figure The places within the pixel, x and y pooled, before and after (1 is even).  The network's crowding toward the centre is gone.
fig.set_size_inches(5, 3)
ax = fig.subplots()
draw_places(ax, found.ravel(), equalised.ravel())
```

**4. Only what stands out from noise.**  A few thousand localizations are
never exactly even: counting noise alone makes some places more frequent
than others.  The plugin corrects only the parts of the unevenness that
stand out from that noise, so a fit that does not pixelate, or too few
localizations, is left exactly as it was.

```figure How far the map moves a localization, by its place within the pixel: little near the focus, more far from it, and more for an uncertain localization than for a sure one.  Positive is a move to the right; the pull toward the centre is undone by moving localizations out toward the edges.
fig.set_size_inches(6, 3.2)
ax = fig.subplots()
grid = np.linspace(0, 1, 201)
for zz, ss, style in ((0.0, 0.12, dict(color="C0")), (500.0, 0.12, dict(color="C1")),
                      (500.0, 0.18, dict(color="C3"))):
    a, b = eq.x.coefficients(np.full(len(grid), zz), np.full(len(grid), ss))
    ax.plot(grid, (fourier_cdf(grid, a, b) - grid) * 100, label=f"z {zz:.0f} nm, uncertainty {ss * 100:.0f} nm", **style)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("place within the pixel")
ax.set_ylabel("move (nm, at 100 nm pixels)")
ax.legend(frameon=False, fontsize="small")
```

## In detail

**The place.**  Pixel centres are at whole numbers of pixels throughout
SMAPpy, so the place within the pixel is $u = x + 0.5 - \lfloor x + 0.5 \rfloor$, with $x$
in pixels (`x_nm` over the pixel size the table records).  x and y are
equalised separately, each with its own uncertainty (`x_err_nm`,
`y_err_nm`; `xy_err_nm` for both where only that exists).

**The density as a Fourier series.**  The density of $u$ among alike
localizations is written as

$$p(u) = 1 + \sum_{k=1}^{K} a_k \cos 2\pi k u + b_k \sin 2\pi k u$$

with $K$ = *harmonics*.  Its coefficients are averages:
$a_k = 2 \langle \cos 2\pi k u \rangle$ and
$b_k = 2 \langle \sin 2\pi k u \rangle$.  The cumulative distribution then
has a closed form,

$$F(u) = u + \sum_{k=1}^{K} \frac{a_k \sin 2\pi k u + b_k (1 - \cos 2\pi k u)}{2\pi k}$$

and each place $u$ is moved to $F(u)$.  $F(0) = 0$ and $F(1) = 1$, so no
localization leaves its pixel.

**Smooth in z and uncertainty.**  Each coefficient is a smooth function of
z and of the uncertainty's rank among the table's: a product of cubic
B-splines, *z splines* along z and *uncertainty splines* along the rank.
These functions are fitted by least squares to every localization's own
$2 \cos 2\pi k u$ and $2 \sin 2\pi k u$ at once, so there are no cells
whose edges would make the correction jump.  z is used with its sign,
because an astigmatic spot is sharp in x on one side of the focus and in y
on the other.  The range of z is its 0.1st to 99.9th percentile; beyond it
the end values are used.

**Noise.**  Each $2 \cos 2\pi k u$ has a variance of about 2.  So for one
harmonic, the fitted coefficients $c$ of its $P$ basis functions have the
covariance $2 G^{-1}$, where $G = X^T X$ is the basis evaluated at every
localization.  Where the harmonic is noise alone, $\chi^2 = c^T G c / 2$
follows a chi-squared distribution with $P$ degrees of freedom.  A harmonic
is kept only where $\chi^2 - P$ exceeds *significance* times
$\sqrt{2 P}$, its standard deviation.  A kept harmonic is shrunk by
$1 - P / \chi^2$ (James-Stein shrinkage), which takes off what the noise
adds to its size.  A table with fewer than four localizations per basis
function is left as it is.

**Monotone.**  Where the coefficients would make the density fall below
0.2 somewhere in the pixel, they are scaled down until it does not, so that
$F$ rises everywhere and two localizations never swap places.

**Per file and channel.**  Each combination of `filenumber`, `channel`
and `camera_channel` gets its own map, since files may come from different
models, cameras or depths, and the two halves of a split chip have pixels
of their own.  A two-channel network fits each molecule once, in the first
channel's pixels, so its table is one grid and one map.

**How the settings were chosen** (`benchmarks/decode/pixelation/estimators.py`).
The method was compared with quantile maps in cells of |z| and uncertainty
(40 quantiles per cell), on netfit's models on simulated test frames and on
DECODE's example movie of microtubules:

* **Map noise.**  The map's own noise was measured as the difference
  between two maps learnt from separate halves of the data.  It was
  0.6 -- 0.9 nm, against the cells' 0.7 -- 1.4 nm, with as flat a result.
* **Nothing to correct.**  On a Spline 3D fit, which does not pixelate, it
  moved localizations by 0.45 nm against the cells' 0.77 nm.
* **Basis size.**  Finer bases than 6 x 4 flattened less at these numbers
  of localizations (50 000 to 600 000), because the shrinkage grows with
  the number of parameters.

**What it costs.**  On simulated frames, the localizations moved 2 -- 5 nm.
About half of the moves went away from the truth, and the error against
the truth did not fall.  On the example movie, the robust spread between
two localizations of one molecule, in frames that share no frame of the
network's input, grew by 0.01 -- 0.06 nm on fits that pixelate little.  On
the most pixelated one (modulation 0.15) it grew by 0.6 nm, from 16.5 to
17.2 nm.  Far from focus on that fit, some unevenness (0.07) is left that
neither more harmonics nor finer z splines remove.  It depends on something
the map does not group by, such as the photons.

**Which table.**  The localizations, whatever the layer shows: the move is
computed for every row from its own place, z and uncertainty, and a
grouped layer's blinks then take the mean as usual.

## Parameters

### action
*auto* is what the button does.  A chain or a batch that should always
correct, and fail rather than revert a table already corrected, names
*correct*.

### z_splines
6 resolves changes over about a third of a typical ±1 µm range.  1 makes
the correction the same at every z.

### sigma_splines
4 tells sure localizations from unsure ones; with few localizations,
fewer is steadier.

### harmonics
The data keep only the harmonics they support, so a larger value costs
little.  The leftover unevenness on real data did not change between 10
and 20.

### significance
3 corrects nothing that counting noise could have made.  Lower values
correct more of a weak pixelation but move more localizations at random.
0 keeps every harmonic, shrunk.

## Output

The table, with `x_nm` and `y_nm` moved and their moves kept as
`x_pixel_shift_nm` and `y_pixel_shift_nm`.  The maps are recorded in its
metadata (`pixel_equalisation`), one per file and channel, with the number
of harmonics each axis kept.

The result line gives the pixel modulation before and after, and the RMS
move.  The modulation is how uneven the places are, as the amplitude of a
cosine of the same variance: 0 is even, and 0.05 makes the pixel centres
10 % more crowded than the edges.  After a correction it should be at most
0.01 -- 0.02.  A large RMS move (several nm) on a modulation that was small
means a correction to think twice about.  If it says there was nothing to
equalise, the table was even within counting noise and nothing moved.

The figure shows the places within the pixel before and after.
