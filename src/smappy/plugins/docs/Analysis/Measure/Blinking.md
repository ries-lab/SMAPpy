---
version: "1"
covers: [smappy.plugins.blinking.blinks_of, smappy.plugins.blinking.dark_frames, smappy.plugins.blinking.bursts, smappy.plugins.blinking.molecules, smappy.plugins.blinking.region_blinking, smappy.plugins.blinking.molecule_blinking, smappy.plugins.blinking.view_values, smappy.plugins.blinking.draw_histograms, smappy.plugins.blinking.Blinking.run, smappy.metadata.frame_time_ms, smappy.metadata.file_frame_time, smappy.metadata.table_frame_time]
---

## What it does

Measures how the fluorophores blink: how long a
[blink](plugin:Panels/Glossary) lasts (the consecutive frames in which one
fluorophore stays on), how long it stays dark before it comes back -- its
[dark time](plugin:Panels/Glossary), also called the off-time -- and how
many times one molecule blinks.  These numbers decide how a dye should be
imaged (how long a blink is, against the exposure), how the localizations of
one molecule are merged, and whether the localizations in a cluster can be
turned into a number of molecules.

It does two things, depending on where it runs:

* **On the view**, per file, it links the localizations of each molecule
  over a long *dark time* and draws three histograms: the on-times, the dark
  times between the blinks of a molecule, and the blinks per molecule.  Use
  it on a sample of sparse, isolated molecules -- a dye on a coverslip, a
  structure with known single copies -- to characterise a dye.
* **In a region** -- an [ROI](plugin:Panels/Glossary) of the ROI manager, or
  a cluster of an id column (*Regions to column*, the Regions tab, a chain's
  block) -- it counts the localizations, the blinks and the
  [bursts](plugin:Panels/Glossary): the blinks joined across dark times of
  at most the *dark time*.  With a dark time longer than the dye's and
  shorter than the time between two of its molecules coming on, a burst is a
  molecule, and the bursts per region are a count of molecules.

Times are in frames and, with the file's
[frame time](plugin:Panels/Glossary) (the time from one frame to the next),
in seconds.

It does not find the clusters to count in (*Find clusters*, *Segment
regions* and *Filter ROIs* do), and does not fit the number of molecules to
the counts.

## How it works

```figure-setup
from smappy.simulate.kinetics import blink, emission
from smappy.simulate.settings import BlinkingSettings as Kinetics
from smappy.plugins.blinking import (blinks_of, dark_frames, bursts,
                                     molecule_blinking, draw_histograms)
ON, OFF, P = 2.0, 30.0, 0.25
rng = np.random.default_rng(2)
kin = Kinetics(on_time=ON, off_time=OFF, bleaching=P, activation="decay",
               photons=1000, photons_std=0)
out = emission(blink(1500, 1_000_000, kin, rng), 1_000_000)
order = np.lexsort((out.frame, out.owner))
frame, owner = out.frame[order], out.owner[order]
x, y = (owner % 50) * 1000.0, (owner // 50) * 1000.0
```

**1. Blinks.**  A blink is what the [layer](plugin:Panels/Glossary) merges:
on a layer that shows [merged blinks](plugin:Panels/Glossary), the
localizations merged into one; on one that does not, a run of localizations
in consecutive frames.  It is on from its first frame to its last, so its
on-time counts a frame the fit missed in the middle of it.

**2. Dark times.**  The dark time before a blink is the number of frames
between the end of the blink before it and its start.  Two blinks that are
on at the same time -- two fluorophores in one region -- have none between
them.

**3. Bursts.**  The blinks are joined wherever the dark time between two of
them is at most the *dark time* $T$; a longer one starts a new burst.  In a
region, that is all: the blinks of the region on one time line, cut at the
long dark times.

```figure One region holding three molecules that came on at different times.  Each bar is a blink; the gaps between them are dark times.  With a dark time T of 200 frames the long gaps between the molecules split the blinks into three bursts (colours), and the short gaps within a molecule do not.
sel = np.isin(owner, [3, 4, 5])
f = frame[sel] + (owner[sel] - 3) * 2000 - frame[sel].min()
found = blinks_of(f)
burst = bursts(found, 200)
fig.set_size_inches(7.0, 1.8)
ax = fig.subplots()
colours = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")
for first, last, b in zip(found.first, found.last, burst):
    ax.broken_barh([(first, last - first + 1)], (0, 1), color=colours[b % 5])
ax.set_yticks([]); ax.set_xlabel("frame")
ax.set_title(f"{len(found)} blinks, {burst.max() + 1} bursts at T = 200 frames",
             fontsize=9)
```

**4. Molecules, on the view.**  On the view a field holds many molecules on
at once, so time alone cannot tell them apart.  The localizations are first
linked into molecules by the same walk that merges blinks, but across up to
$T$ dark frames, within a box of *within*, per file and channel: a molecule
is a blink merged with a long dark time.  Within each molecule its blinks,
the dark times between them and their number make the three histograms, one
line per file.

```figure Simulated kinetics with an on-time of 2 frames, an off-time of 30 frames and a bleaching probability of 0.25 after each blink.  Black is what those rates predict: a mean on-time of 3 frames (the frames a 2-frame blink touches), dark times one frame plus a geometric number with the off-time's decay, and a geometric number of blinks per molecule with mean 4.
from smappy.plugins.blinking import integer_edges
found = molecule_blinking(frame, x, y, dark_time=1000, distance_nm=50, name="simulated")
fig.set_size_inches(5.5, 6.0)
fig.set_layout_engine("constrained")
draw_histograms(fig, [found], log=True)
on_axis, dark_axis, count_axis = fig.axes
on_axis.axvline(ON + 1, color="k", ls="--", lw=1)

def geometric(axis, values, q, start, first_edge):
    """P(k) = (1 - q) q^(k - start) for k >= start, summed over the bins
    the histogram has (from first_edge, as draw_histograms makes them)."""
    edges = integer_edges(values, start=first_edge)
    low = np.maximum(np.ceil(edges[:-1]), start)
    high = np.floor(edges[1:])
    inside = q ** (low - start) - q ** (high - start + 1)
    axis.stairs(np.where(high >= low, inside, np.nan), edges, color="k", ls="--", lw=1)

geometric(dark_axis, found.dark_frames, np.exp(-1 / OFF), 1, 0)
geometric(count_axis, found.blinks_per_molecule, 1 - P, 1, 1)
```

## In detail

**Blinks.**  Without merged blinks the localizations of a molecule are
sorted by frame, and a new blink starts wherever the frame number jumps by
more than one.  With them, a blink is the localizations that share a
`group_id`; its first and last frame are their smallest and largest
`frame`.  Blinks are ordered by their first frame within each molecule.

**Dark times.**  For blink $i$ of a molecule, after blinks that ended at
frames $e_1 \ldots e_{i-1}$,

$$d_i = s_i - \max_{j < i} e_j - 1 ,$$

with $s_i$ its first frame: the frames in between, none of them on.  A
negative $d_i$ -- the blink began before an earlier one had ended -- is no
dark time and is left out, as is the first blink of a molecule.

**What the simulation predicts.**  An exponential on-time of mean $\tau$
frames that begins anywhere within a frame touches $\tau + 1$ frames on
average.  A dark time that leaves no frame wholly dark is not seen at all:
the two blinks are one run.  Past the first dark frame an exponential dark
time starts afresh, so the dark frames seen are $1 + \lfloor r \rfloor$, with
$r$ exponential of the same mean $\tau_d$: on average
$1 + 1/(e^{1/\tau_d} - 1) \approx \tau_d + 0.5$ frames.  The blinks of a
fluorophore that bleaches with probability $p$ after each one are geometric,
of mean $1/p$, less the few pairs too close to tell apart.

**Bursts.**  Blink $i$ starts a new burst when it is the first of its
molecule or when $d_i > T$.  SMAP counts $\Delta\mathrm{frame} > T$, which is
a dark time of $T$ frames or more; here a dark time longer than $T$ splits,
the same rule as the layer's *dark frames* when it merges blinks, so $T$ is
the number of dark frames a burst bridges.

**Molecules.**  The localizations are linked by `group.connect`, the walk
that merges blinks, with a box of half-width *within* and $T$ dark frames,
separately per file and channel.  The blinks within a molecule are found as in a region, with the
molecule kept apart.

**The view's numbers** (the `values` of a run on the view) are the
molecules': `n_bursts` is the number of molecules, `n_blinks` their blinks,
and the means are over all of them.  Per region there is no linking: the
region is taken to be one structure.

**Seconds.**  A time in frames times the frame time.  The frame time is the
file's (`metadata["camera"]["frame_time_ms"]`, or the session's record of
each file): the cycle time the file recorded where it recorded one, else the
exposure.  With files of different frame times in one region, there is no
one frame time and the seconds are NaN; set *frame time* to give one.

## Parameters

### dark_time
Longer than the dye's own dark times, so that the blinks of one molecule
stay together, and shorter than the time between two molecules coming on,
so that two molecules are not one.  The dark-time histogram of a sparse
sample shows the dye's: take $T$ where it has fallen to nearly nothing.
Too short counts a molecule several times; too long, on a dense sample,
counts several molecules as one.

### distance_nm
A few times the [localization precision](plugin:Panels/Glossary): wide
enough for the scatter of one molecule's localizations, narrow enough that
two molecules are not linked.  On a dense sample the molecules overlap
whatever it is set to, and the view's numbers mean little.

## Output

**Per region**, a row of the region table:

* `n_localizations` -- the localizations of the selected blinks in the
  region.
* `n_blinks` -- the blinks, as the layer merges them.
* `n_bursts` -- the bursts at the *dark time*: the molecules, if $T$ is
  chosen well.
* `mean_on_frames`, `mean_on_s` -- the mean on-time of the blinks.
* `mean_dark_frames`, `mean_dark_s` -- the mean dark time between blinks
  within a burst (dark times longer than $T$ are between bursts and are not
  counted).

The seconds are NaN without a frame time.

**On the view**, the same numbers over the molecules, a line per file in the
text (its molecules, their blinks, the mean on- and dark time in frames and
seconds) and three histograms, each a line per file and each the fraction of
that file's total.  The on-times and the dark times of a dye with one
on- and one off-rate fall in a straight line on the logarithmic axis; a bend
in the dark times says there are two kinds of dark state, or that the dark
time $T$ is cutting the long ones off.  Blinks per molecule much larger
than the dye's means the molecules are too dense to be linked apart.

## Differences from SMAP

A port of the counting part of SMAP's `Analyze/cluster/dyeblinking`, and of
the `locsdf` count of its counting plugins
(`Analyze/counting/private/cluster_mmaple_makehist.m`)
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).  The main changes:

* SMAP's maximum dark time is in seconds; here it is in frames, the unit
  the linking runs in, and the same for every file.
* A blink here is what the layer merges, which may bridge a missed frame;
  SMAP split blinks at any missing frame.
* SMAP's histogram of the number of blinks shows the number minus one; here
  it is the number of blinks.
* SMAP's dark-time count splits at $\Delta\mathrm{frame} > T$; here at a dark
  time of more than $T$ frames, which is $\Delta\mathrm{frame} > T + 1$.
* SMAP also drew the photons, the on-time and the blinks against a series of
  dark times, the on-fraction and the duration of a molecule; they are left
  out.

## References

* J. Ries, "SMAP: a modular super-resolution microscopy analysis platform
  for SMLM data", *Nature Methods* 17, 870-872 (2020),
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
  -- the MATLAB original this is ported from.
