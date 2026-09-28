---
version: "2"
covers: [smappy.plugins.assign_colors.ratios, smappy.plugins.assign_colors.find_modes, smappy.plugins.assign_colors.unclaimed_modes, smappy.plugins.assign_colors.assign_by_minima, smappy.plugins.assign_colors.log_likelihoods, smappy.plugins.assign_colors.variances, smappy.plugins.assign_colors.deviations, smappy.plugins.assign_colors.posteriors, smappy.plugins.assign_colors.assign_by_probability, smappy.plugins.assign_colors.region_polygons, smappy.plugins.assign_colors._mode_width, smappy.plugins.assign_colors.AssignColors]
---

## What it does

In a **ratiometric** two-colour experiment the two dyes are not imaged one
after the other or through different filters.  Both are imaged at once, and a
dichroic mirror splits the light of every molecule between two halves of the
camera.  The dyes emit at slightly different wavelengths, so they split their
light differently: one might send three quarters of its photons to the first
half, the other only a third.  Which dye a molecule is shows only in *how its
photons divide* between the two halves.

The two-colour fitters (*Gaussian 2D 2C*, *Spline 3D 2C*) fit each molecule in
both halves at once, with one position and a photon number per half.  This
plugin turns those two photon numbers into a colour.  For every localization
it computes the **photon split**

$$r = \frac{N_1 - N_2}{N_1 + N_2} ,$$

$N_1$ and $N_2$ the photons in the two halves.  $r$ runs from $-1$ (all light
in the second half) to $+1$ (all in the first).  Over a sample with two dyes,
the histogram of $r$ has two peaks, one per dye.  The plugin finds them and
writes a `channel` column: 1 for the peak at lower $r$, 2 for the next, and 0
for a localization it will not decide on.

It offers two ways to decide:

* **split at the minima** -- the boundary between two colours is the lowest
  point of the histogram between their peaks.  Every localization is judged
  by its $r$ alone, and a fixed band around the boundary is left undecided.
* **probabilistic** -- every localization is judged by its own photon
  numbers.  A bright localization measures its split precisely and can be
  given a colour close to the boundary; a dim one cannot.  It also refuses a
  localization that is too far from *every* colour to be either.

Both fitters run this plugin at the end of every fit (*assign colours*, in
their *after the fit* section, on by default, with the minima method), so a
two-colour table normally arrives with its colours.  Run it again from the
Analysis tab to see how it decided, or to decide differently.

**What it needs.**  Two columns of photons per localization: `photons_ch0`
and `photons_ch1`, as the two-colour fitters write them, or another pair (see
*channel 1 column*).  The fit's errors on them, `photons_err_ch0` and
`photons_err_ch1`, are used when they are there.  The peaks are looked for in
the **current selection** -- the layer's filter and the ROI -- and the colours
are then given to every localization of the table.  A filter on `channel`
itself is left out of that selection: after a first run the layers are
usually set to one colour each, and the histogram still has to see both
dyes.  It warns below 50 selected localizations.

## How it works

```figure-setup
from smappy.locs import Localizations
from smappy.simulate import simulate
from smappy.simulate.settings import BlinkingSettings
from smappy.plugins.assign_colors import (
    AssignColorSettings, _intensity_plotter, _plotter, assign_by_minima,
    assign_by_probability, find_modes, ratios)

# the demo structure: its ring is dye 1, its lines and scattered points dye 2
locs = simulate(n_frames=3000, seed=1,
                blinking=BlinkingSettings(photons=1200, photons_std=600))
rng = np.random.default_rng(0)
dye = np.asarray(locs["dye"])
# dye 1 splits as r = -0.35, dye 2 as r = +0.25, and each molecule's own
# split wanders around its dye's by 0.04; the photons are then split
# between the halves binomially
RHO, SPREAD = np.array([-0.35, 0.25]), 0.04
_, molecule = np.unique(np.asarray(locs["emitter"]), return_inverse=True)
own = RHO[dye - 1] + rng.normal(0, SPREAD, molecule.max() + 1)[molecule]
total = np.asarray(locs["photons"]).round()
first = rng.binomial(total.astype(int), np.clip((1 + own) / 2, 0, 1)).astype(float)
table = Localizations({"x_nm": locs["x_nm"], "y_nm": locs["y_nm"],
                       "frame": locs["frame"], "photons_ch0": first,
                       "photons_ch1": total - first}, {})
values = ratios(table)
seen = values.valid
modes = find_modes(values.r[seen])
minima = AssignColorSettings(mode="minima")
probabilistic = AssignColorSettings(mode="probabilistic", spread=0.04)
by_minima = assign_by_minima(values.r, modes, minima.exclusion)
by_probability = assign_by_probability(
    values.r, values.n_eff, modes, crosstalk=probabilistic.crosstalk,
    spread=probabilistic.spread, tolerance=probabilistic.tolerance)[0]
```

**1. The split.**  For every localization the plugin computes $r$ from the
two photon numbers, and how precisely that localization measures it.  Photons
arrive at random, so even a molecule of a known dye does not split its light
exactly as the dye does on average: with $N$ photons in all, the measured
$r$ scatters around the dye's own value $\rho$ by

$$\sigma_r = \sqrt{\frac{1 - \rho^2}{N}} .$$

A localization with 1000 photons measures $r$ to about $\pm 0.03$; one with
50 photons only to about $\pm 0.14$.  This is the whole difference between
the two methods: the first ignores it, the second uses it.

**2. The peaks.**  The histogram of $r$ over the selection (*histogram bins*,
200 over the range $-1$ to $1$) is smoothed (*smoothing*, 2 bins), and its
highest peaks are the colours (*colours*, 2).  Two peaks closer than three
smoothing widths count as one.  Each peak's position is refined between the
bins.  If the ratios of the dyes are known -- measured on samples with one
dye only, or computed from the spectra -- they can be typed in (*expected
r*), and only the boundaries and the share of each colour are then read off
the histogram.

**3. The boundaries.**  Between two neighbouring peaks, the boundary is the
lowest point of the smoothed histogram.  When the valley is flat -- two well
separated dyes leave a stretch of empty bins -- it is the middle of the
longest empty stretch.  The number of localizations on each side of the
boundaries is each colour's **share** of the sample.

```figure The histogram of $r$ for a simulated sample of two dyes ($\rho = -0.35$ and $+0.25$), as the plugin draws it.  Left: split at the minima.  The dotted lines are the peaks, the solid line the boundary, and the grey band around it is left undecided.  Right: probabilistic.  The dashed curves are the probability of each colour for a localization of median brightness, and the shaded stretches are what such a localization would not be given: the valley between the dyes, and the far tails.
fig.set_size_inches(8.0, 3.4)
left, right = fig.subplots(1, 2)
_plotter(values, modes, by_minima, minima, seen)(left)
_plotter(values, modes, by_probability, probabilistic, seen)(right)
for ax in fig.axes:
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_fontsize(8); ax.yaxis.label.set_fontsize(8)
    ax.title.set_fontsize(8)
fig.subplots_adjust(wspace=0.45)
```

**4a. Split at the minima.**  A localization gets the colour of the side of
the boundary it falls on.  Within *exclusion dr* of a boundary it gets 0.
The band is the same for every localization, however bright.  It has to be
wide enough for the dim localizations, whose $r$ is uncertain, and it then
throws away bright ones that were perfectly clear.

**4b. Probabilistic.**  Each localization is given the **probability of
each colour**, worked out from its own two photon numbers.  The question is:
given that the dyes split as the peaks say, how likely is it that *this*
molecule, with these photons on each side, is dye 1 rather than dye 2?  How
bright the molecule is drops out of the answer; only how its photons divide
matters, and how many there are to go by.  The answer is weighted by each
colour's share of the sample (*use abundances*).

A colour is then given only if two tests pass:

* **Clear enough.**  The best colour's probability must reach $1 - c$, $c$
  the *allowed crosstalk* (5%).  Among the localizations that pass, the
  expected fraction given the wrong colour is then at most $c$.  This test
  refuses a narrow band around the boundary, narrow for a bright
  localization and wide for a dim one.
* **Close enough.**  The localization's $r$ must lie within *sigma* (3)
  standard deviations of the chosen dye's own ratio, the standard deviation
  being that localization's own $\sigma_r$ plus the *extra spread*.  This
  test refuses a localization that is neither dye: two molecules of
  different dyes on at the same place, or a fit that went wrong.  The
  probability cannot do this, because it only compares the colours with each
  other: a localization exactly between two far-apart peaks still gets a
  probability near 1 for whichever side it leans to.

**5. The intensity plane.**  The histogram of $r$ divides the brightness
out, and the brightness is what the second method depends on.  So the
plugin draws a second figure (the *intensities* tab): the photons of one half
against those of the other, on log scales.  A dye is a straight diagonal line
there, because a fixed split is a fixed ratio $N_1/N_2$.  The coloured
regions are where each colour is given; grey is where nothing is.

```figure The same localizations in the intensity plane.  Left: split at the minima gives each colour a band of fixed width in $r$, the same for dim and bright molecules.  Right: the probabilistic method gives each colour a band that is wide at few photons, where the split is uncertain, and narrow at many, and leaves the space between the dyes grey.
fig.set_size_inches(8.0, 3.3)
left, right = fig.subplots(1, 2)
_intensity_plotter(values, modes, minima, seen)(left)
_intensity_plotter(values, modes, probabilistic, seen)(right)
for ax in fig.axes:
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_fontsize(8); ax.yaxis.label.set_fontsize(8)
    ax.title.set_fontsize(8)
fig.subplots_adjust(wspace=0.5)
```

**6. How wide the peaks really are.**  Real peaks are usually wider than the
photon statistics alone make them: a dye's split varies between molecules,
with its surroundings, or across the field of the splitter.  The probabilistic
method only knows about the photon statistics unless it is told (*extra
spread*), and without it it refuses bright localizations that sit a little
off their dye's ratio.  The text therefore compares the strongest peak's
measured width with the width the photons explain, and prints the extra
spread that would account for the rest.

```figure Against the truth, in bins of total photons.  Left: the fraction of localizations given a colour.  Right: of those, the fraction given the wrong one (points) and, for the probabilistic method, the fraction it expected to get wrong (line).  The minima method keeps nearly everything and makes its mistakes among the dim localizations; the probabilistic method refuses the dim ones it cannot tell apart, and with *extra spread* 0 it also refuses bright localizations whose own split is off their dye's by more than photon noise.  The simulated molecules wander by 0.04, which is the spread given to the red curve.
from smappy.plugins.assign_colors import posteriors
edges = np.array([10, 30, 60, 100, 200, 400, 800, 1600, 5000])
middle = np.sqrt(edges[:-1] * edges[1:])
which = np.digitize(values.total, edges) - 1
no_spread = assign_by_probability(values.r, values.n_eff, modes, tolerance=3.0)
with_spread = assign_by_probability(values.r, values.n_eff, modes, spread=0.04,
                                    tolerance=3.0)
runs = (("minima, dr = 0.05", by_minima, None, "0.4"),
        ("probabilistic, spread 0", *no_spread, "#1f77b4"),
        ("probabilistic, spread 0.04", *with_spread, "#d62728"))
fig.set_size_inches(8.0, 2.9)
left, right = fig.subplots(1, 2)
for name, channel, top, colour in runs:
    kept, wrong, claimed = [], [], []
    for b in range(len(middle)):
        here = which == b
        given = here & (channel > 0)
        kept.append(given.sum() / max(here.sum(), 1))
        wrong.append(np.mean(channel[given] != dye[given]) if given.any() else np.nan)
        claimed.append(np.mean(1 - top[given]) if top is not None and given.any()
                       else np.nan)
    left.plot(middle, kept, "o-", color=colour, ms=3, lw=1, label=name)
    right.plot(middle, np.maximum(wrong, 1e-5), "o", color=colour, ms=4)
    if top is not None:
        right.plot(middle, np.maximum(claimed, 1e-5), "-", color=colour, lw=1)
right.axhline(0.05, color="0.5", ls=":", lw=0.8)
right.text(1500, 0.06, "allowed crosstalk", fontsize=7, color="0.4", ha="right")
for ax in (left, right):
    ax.set_xscale("log")
    ax.set_xlabel("photons in both halves", fontsize=8)
    ax.tick_params(labelsize=7)
left.set_ylabel("fraction given a colour", fontsize=8)
left.set_ylim(0, 1.05)
left.legend(fontsize=7, frameon=False, loc="lower right")
right.set_yscale("log")
right.set_ylim(5e-5, 0.3)
right.set_ylabel("wrong among those given", fontsize=8)
right.text(4500, 7e-5, "none wrong: drawn at the floor", fontsize=6, color="0.4",
           ha="right")
fig.subplots_adjust(wspace=0.35)
```

## In detail

**The split and its precision.**  With $N = N_1 + N_2$, a localization is
used if both counts are finite and $N > 0$ (and $N \geq$ *minimum photons*
when that is set); $r$ is clipped to $[-1, 1]$, because a fit can put a
channel slightly below zero.  If each photon of dye $k$ goes to the first half
with probability $p_k = (1 + \rho_k)/2$, then $N_1$ is binomial and the
variance of $r$ is exactly $(1 - \rho_k^2)/N$.

With the fitted errors $\sigma_1$ and $\sigma_2$ of the two counts (*use
fitted errors*, and both `photons_err_*` columns present), the variance of
$r$ at the measured point is propagated,

$$\sigma_r^2 = \frac{4 \left( N_2^2 \sigma_1^2 + N_1^2 \sigma_2^2 \right)}{N^4} ,$$

and turned into an **effective photon number**

$$N_{\mathrm{eff}} = \frac{1 - r^2}{\sigma_r^2} .$$

It equals $N$ when the errors are pure shot noise and is smaller when the
background under the spot or an EMCCD's excess noise has cost information.
Everything below uses $N_{\mathrm{eff}}$ in place of $N$, so the variance keeps
its $1 - \rho^2$ form under each dye's hypothesis instead of the plug-in
value, which would go to zero at $|r| = 1$ and make the most extreme, dimmest
localizations look the most certain.  Where $N_{\mathrm{eff}}$ is not a positive
finite number -- an error missing, or $r$ exactly $\pm 1$ -- the total $N$ is
used.  Without error columns, $N_{\mathrm{eff}} = N$ throughout.

**Finding the peaks.**  The histogram has *histogram bins* bins over
$[-1, 1]$ and is smoothed with a Gaussian of *smoothing* bins (0: not
smoothed).  A bin is a candidate peak if it is non-zero and at least as high
as both its neighbours.  Candidates are taken highest first, skipping any
within $\mathrm{round}(3 \times \mathrm{smoothing})$ bins (at least one) of a
peak already taken, until there are *colours* of them; fewer is an error that
names the remedies.  Each is refined by a parabola through its bin and the
two beside it, moved by at most one bin.  With *expected r*, those values are
the peaks and nothing is searched.

**Boundaries.**  Between two peaks, the bins at the valley's minimum are
found, and the longest contiguous run of them is taken -- the one nearest the
midpoint of the two peaks if two runs are equally long.  The boundary is the
middle bin of that run, refined by a parabola.  Two peaks less than two bins
apart are split halfway.  Each colour's share $\pi_k$ is the fraction of the
selection between the boundaries on either side of it.

**The probability of each colour.**  The two counts of a molecule of dye $k$
with expected brightness $\lambda$ are independent Poisson numbers, and their
joint probability factorises into a Poisson in the total and a binomial in
the split,

$$P(N_1, N_2 \mid \lambda, k) = \mathrm{Poisson}(N \mid \lambda)\, \mathrm{Binomial}(N_1 \mid N, p_k) .$$

Only the second factor depends on the dye and only the first on the
brightness, so the brightness, whatever its distribution, cancels from the
posterior.  What is left, evaluated at the effective counts
$n_1 = N_{\mathrm{eff}} (1 + r)/2$ and $n_2 = N_{\mathrm{eff}} - n_1$, is

$$P(k \mid N_1, N_2) = \frac{\pi_k\, p_k^{n_1} (1 - p_k)^{n_2}}{\sum_j \pi_j\, p_j^{n_1} (1 - p_j)^{n_2}} ,$$

with $\pi_k = 1$ for all $k$ when *use abundances* is off.  For two dyes the
log-odds are linear in the counts: every photon adds the same amount of
evidence.  The binomial is used rather than a Gaussian in $r$ because the two
part company when one half collects few photons -- the case of a dye that
sends almost all its light to one side.

**Extra spread.**  With *extra spread* $\epsilon > 0$, a dye's $p_k$ is itself drawn
from a Beta distribution of mean $p_k$ and concentration

$$\kappa_k = \max\left( \frac{1 - \rho_k^2}{\epsilon^2} - 1,\ 10^{-3} \right) ,$$

which gives the ratio a standard deviation $\epsilon$ in $r$.  The split is then
beta-binomial -- still exact, still independent of the brightness -- and the
variance of $r$ under dye $k$ becomes

$$s_k^2 = \frac{1 - \rho_k^2}{N_{\mathrm{eff}}}\ \frac{N_{\mathrm{eff}} + \kappa_k}{\kappa_k + 1} ,$$

which is shot noise and spread added in quadrature to first order, and exact
at both ends.  With $\epsilon = 0$, $s_k^2 = (1 - \rho_k^2)/N_{\mathrm{eff}}$.

**The two tests.**  With $k^\ast$ the most probable dye, a localization is
given $k^\ast$ if

$$P(k^\ast \mid N_1, N_2) \geq 1 - c \qquad \mathrm{and} \qquad z = \frac{|r - \rho_{k^\ast}|}{s_{k^\ast}} \leq t ,$$

$c$ the *allowed crosstalk* and $t$ the *sigma* tolerance; $t = 0$ turns the
second test off.  The first is Chow's rule, a Bayes classifier with a reject
option.  The chance that an assigned localization is wrong is
$1 - P(k^\ast)$, so requiring $P(k^\ast) \geq 1 - c$ bounds the expected
error among the assigned by $c$ -- under the model, which is only as good as
the claim that each peak is as wide as its photons and its spread make it.
The mean of $1 - P(k^\ast)$ over the assigned localizations is reported as
the expected crosstalk, and is usually far below $c$, since most
localizations are nowhere near a boundary.  With *keep the tails*, the second
test is not applied to a localization with $r$ below the first peak or above
the last: there is no other dye out there to be confused with.

**How the two methods compare.**  For two dyes of equal share and the same
$s_k = s$, the first test alone refuses a band around the midpoint $r_{\mathrm{mid}}$
of half-width

$$\delta r = \frac{s^2 \ln\left( (1 - c)/c \right)}{|\rho_2 - \rho_1|} ,$$

the exclusion zone of the minima method, except that its shot-noise part
shrinks as $1/N_{\mathrm{eff}}$.  The text reports it for the median localization, with
$s^2 = (1 - r_{\mathrm{mid}}^2)/N_{\mathrm{eff}} + \epsilon^2$, as the
*exclusion dr* the minima method would need to match.

**Mode width.**  The strongest peak's width is read off the smoothed
histogram: the distance from the peak to where it falls to half its height,
interpolated between bins, on each side that crosses half before reaching
the neighbouring boundary (both sides averaged when both do).  That
half-width at half maximum is divided by 1.1774 to give a standard deviation,
and the smoothing, in the same units, is taken off in quadrature.  It is
compared with the median $\sqrt{(1 - \rho^2)/N_{\mathrm{eff}}}$ of the
localizations on that peak's side of the boundaries, $\sigma_{\mathrm{shot}}$;
when the peak's width $w$ is larger,
$\sqrt{w^2 - \sigma_{\mathrm{shot}}^2}$ is printed as the *extra spread* to
try.

**A population between the peaks.**  A bump in the smoothed histogram that
lies between the first and the last peak, is more than one bin from every
chosen peak, and is at least 5% of the height of the lower of the two peaks
beside it is named in the text as a warning.  It is a population the model
does not have -- two dyes on in one spot, a third dye -- and the minima method
hands it whole to one side; the *sigma* test of the probabilistic method
refuses it.

**The regions in the intensity plane.**  The rule depends on the two counts
only through $r$ and $N_{\mathrm{eff}}$, so it can be evaluated anywhere.  A
grid point has no fitted error, so $N_{\mathrm{eff}}$ there is its total times
the table's median $N_{\mathrm{eff}}/N$ (shown in the figure's title, kept
between 0.001 and 1).  Along 260 totals, 2001 values of $r$ are decided and
each colour's first and last are its region's two edges.  The view is limited
to the 0.1th and 99.9th percentile of each half's photons, the histogram of
$r$ to the 0.1th and 99.9th percentile of $r$.

## Parameters

### mode
Start with the minima.  Switch to *probabilistic* when the peaks overlap, when
the brightness varies a lot, or when a number for the crosstalk is needed.
Each method greys out the settings it does not read.

### colors
More than two works the same way, with a boundary between each pair of
neighbouring peaks, up to 6.  The peaks are found in the histogram, so every
colour needs a visible peak of its own; otherwise give *expected r*.

### exclusion
In units of $r$.  A good value is a little more than $\sigma_r$ of the dim
localizations (step 1): at 100 photons, about 0.1.  0 decides every
localization.

### tolerance
3 refuses about 0.3% of the localizations that really are the dye, and
anything further out.  Much smaller refuses good localizations; 0 gives
every localization to its likelier dye, however far away.

### keep_tails
Worth turning on when a dye's peak has a long tail on its outer side -- a
splitter whose ratio changes across the field, say -- and those molecules
should be kept.

### crosstalk
The band it refuses grows only as $\ln(1/c)$: going from 5% to 0.1% makes it
about 2.4 times as wide.  With well separated dyes it hardly matters; it is the *sigma*
test that does most of the refusing.

### spread
Read it off the text after a first run (*colour k is ... wide against ...*),
and run again.  Too small, and bright localizations are refused for being a
little off their dye's ratio; too large, and the *sigma* test lets through
what lies between the dyes.

### use_errors
Leave it on.  Off, $N_{\mathrm{eff}} = N$: a localization over a high
background is then trusted more than it should be.

### use_prior
Off treats the colours as equally likely beforehand, which gives the
rarer colour a little more of the localizations near a boundary.  It matters only close to a boundary.

### min_photons
The photons of both halves together.

### bins
Raise it with many localizations and narrow peaks; lower it for a sparse
selection, whose histogram is otherwise noisy.

### smoothing
If two dyes are found as one peak, lower it; if one dye is found as two,
raise it.

### expected
One value per colour, in any order; they are sorted.  Use it when the peaks
are not clear in the histogram, or to keep the ratios fixed between samples.

### channel1
The column $r$ counts positive.  Name both columns or neither; auto takes
the first pair the table has of `photons_ch0`/`photons_ch1`,
`photons_ch1`/`photons_ch2`, `intensity_ch0`/`intensity_ch1` and
`intensity_ch1`/`intensity_ch2`.

### cmap
Only the look of the intensity plot.

## Output

**The table** gains these columns, for every localization, selected or not,
and the run is logged in the file's history:

* `channel` -- the colour, 1 for the peak at the lowest $r$, 2 for the next,
  and 0 where none was given or the photons are missing.
* `color_ratio` -- $r$.
* `channel_p1`, `channel_p2`, ... -- the probability of each colour, from the
  probabilistic model, whichever method decided `channel`.  A probability is
  only a comparison of the colours with each other: it is near 0 or 1 almost
  everywhere, also for a localization that is neither.
* `channel_p` -- the probability of the colour given; 1 for every assigned
  localization with the minima method, and 0 where `channel` is 0.
* `channel_sigma` -- how many standard deviations the nearest dye is from
  this localization's $r$.  This is the column that tells a localization on a
  peak from one between the peaks, and a useful filter.

Beside the table:

* **The histogram of $r$** (the first figure above), coloured by what was
  given.
* **The intensity plane** (*intensities*, the second figure above).
* **The text**: how many localizations were given a colour; each colour's
  ratio, count and share; the boundaries; for the probabilistic method the
  expected crosstalk against the allowed one, how many were refused by each
  test, and the equivalent *exclusion dr*; the width of the strongest peak
  against its photon noise; and warnings about a population between the
  peaks or missing photon errors.

A good result has clear, separate peaks, a boundary in an empty or nearly
empty valley, and a strongest peak not much wider than its photon noise (or
an *extra spread* that accounts for the difference).  A warning about a
population between the peaks, or a boundary on the flank of a peak, means
the two colours are not what the histogram holds.

**Preview** draws both figures and writes the text without changing the
table.

## Differences from SMAP

SMAP's `Intensity2Channel` (with `get_intensity2ch`) draws the two channels'
intensities against each other, on log scales by default, and the user types
in the lines that divide them: a *slope* (1 by default), an *offset*, an
*edge* excluded on either side (0.5 in $\log_{10}$, a factor of about three
in the ratio) and a minimum intensity per channel.  Localizations between the
lines get channel 5, and those with no partner intensity channels 3 and 4,
which *associate unassigned* folds into 1 and 2.  `Intensity2ManyChannels`
lets the user draw a polygon per colour in the same plane, and
`intensity_ratios_channels` finds the peaks of a photon-weighted histogram of
the ratio, refined by a parabola.  Here:

* **The boundaries are found**, not typed in: from the peaks and valleys of
  the histogram of $r$, or from *expected r*.  The minima method's band
  is, like SMAP's *edge*, the space between two lines of constant ratio.
* **The probabilistic method is new.**  It gives each localization a
  probability of each colour from its own photons and their fitted errors, a
  crosstalk bound, and a test against each dye's ratio.
* **Unassigned is 0**, not 5, and there are no channels 3 and 4: the
  two-colour fit always gives both photon numbers.
* **The histogram is not photon-weighted.**  It counts localizations, so the
  share of each colour is a share of localizations.
* SMAP measures the partner's intensity after a single-channel fit; here both
  photon numbers come from one global fit of the pair (see NOTES.md, "Two
  colours in 2D").

## References

* Bossi M, Fölling J, Belov VN, et al. Multicolor far-field fluorescence
  nanoscopy through isolated detection of distinct molecular species. *Nano
  Lett* 8, 2463 (2008).
  [doi:10.1021/nl801471d](https://doi.org/10.1021/nl801471d)
  -- colours of single molecules from the split between two detection
  channels.
* Testa I, Wurm CA, Medda R, et al. Multicolor fluorescence nanoscopy in
  fixed and living cells by exciting conventional fluorophores with a single
  wavelength. *Biophys J* 99, 2686 (2010).
  [doi:10.1016/j.bpj.2010.08.012](https://doi.org/10.1016/j.bpj.2010.08.012)
  -- up to four dyes told apart with two channels.
* Lampe A, Haucke V, Sigrist SJ, et al. Multi-colour direct STORM with red
  emitting carbocyanines. *Biol Cell* 104, 229 (2012).
  [doi:10.1111/boc.201100011](https://doi.org/10.1111/boc.201100011)
  -- spectral-demixing dSTORM with red dyes.
* Winterflood CM, Platonova E, Albrecht D, Ewers H. Dual-color 3D
  superresolution microscopy by combined spectral-demixing and biplane
  imaging. *Biophys J* 109, 3 (2015).
  [doi:10.1016/j.bpj.2015.05.026](https://doi.org/10.1016/j.bpj.2015.05.026)
