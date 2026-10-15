---
version: "1"
covers: [smappy.tracking.link.candidates, smappy.tracking.link.link_simple, smappy.tracking.link.link_global, smappy.tracking.link.track_ids, smappy.tracking.link.link_tracks, smappy.tracking.link.score, smappy.plugins.link_tracks.LinkTracks.run, smappy.simulate.tracks.simulate_tracks]
---

## What it does

In [single-particle tracking](plugin:Panels/Glossary) (SPT) a molecule is
localized again in frame after frame as it moves, and its
[track](plugin:Panels/Glossary) -- the localizations of one particle in
time order -- shows how it moves: freely, slowly, confined, or switching
between the two.  This plugin decides which localizations belong to one
particle and names each track, writing two columns onto the table:
`track_id` (1, 2, ... in the order the tracks start, 0 for a localization
in no track) and `track_length` (how many localizations the track has).

Here the tracks are fitted from single molecules, so they are **sparse**:
in one frame the molecules are far apart compared with how far each moves
(its [jump](plugin:Panels/Glossary)).  A jump as long as the distance
between molecules would blur the spot or make two spots overlap, and
neither could be fitted.  So which localization continues a track is
rarely in doubt.  What matters is the frames in which the fluorophore was
dark, too dim to be found, or badly fitted: a track must survive them.
*memory* says how many such frames in a row are bridged.

Two linkers, the *algorithm*:

* **simple**, the default: frame by frame, each localization continues the
  nearest track that ended within the last *memory* + 1 frames and within
  the *maximum jump*; where two want the same one, the nearest pair wins.
* **global assignment**: per frame, the links that together make the
  smallest sum of squared jumps -- the criterion of Crocker and Grier
  ([Crocker & Grier 1996](https://doi.org/10.1006/jcis.1996.0217)), which
  SMAP's *SimpleTracking* and the Python package trackpy use.

On sparse tracks the two give the same links to within a few in a
thousand, and the simple one is several times faster (the benchmark
below).

A track is a [region](plugin:Panels/Glossary): `track_id` is an id column
like `cluster_id`, so any measurement offered per region runs once per
track (a **block** of a [chain](plugin:Panels/Glossary) set to *for each*
`track_id`, or the [per-region steps](plugin:Panels/Regions tab)), a
filter on `track_length` keeps the long tracks, and
[Segment regions](plugin:Analysis/Regions/Segment regions), *from an id
column*, makes an ROI of each track to review.

It links the localizations the layer's [filter](plugin:Panels/Glossary) and
ROI keep; the rest get `track_id` 0.  It always links the single
localizations, never [merged blinks](plugin:Panels/Glossary): merging joins
the localizations in consecutive frames that lie close together, which for
a slow particle is its whole track.

## How it works

```figure-setup
from smappy.simulate import simulate_tracks, jump_nm, spacing_nm
from smappy.tracking import link_tracks, score
locs = simulate_tracks(n_frames=60, field_um=6.0, density_um2=0.15, d_um2_s=(0.3,),
                       track_frames=25.0, p_dark=0.15, dark_frames=1.5, seed=4)
x, y = np.asarray(locs["x_nm"], float), np.asarray(locs["y_nm"], float)
frame, truth = np.asarray(locs["frame"]), np.asarray(locs["track_id_true"])

def draw(ax, ids, title):
    order = np.lexsort((frame, ids))
    for t in np.unique(ids[ids > 0]):
        rows = order[ids[order] == t]
        ax.plot(x[rows] / 1000, y[rows] / 1000, lw=0.8, color=f"C{t % 10}")
    ax.scatter(x[ids == 0] / 1000, y[ids == 0] / 1000, s=4, c="0.5", zorder=3)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
```

**1. Candidates.**  For every localization, the localizations of the next
*memory* + 1 frames within the *maximum jump* of it are its candidates: the
places its particle may have gone, the next frame or after a few dark ones.
They are found with the cell search of
[Neighbourhood](plugin:Analysis/Process/Neighbourhood), on every processor
core.

**2. Links.**  Taking the frames in order, each localization is joined to
one open track end among its candidates, or starts a track of its own.  A
track end stays open for *memory* frames: a particle that was dark for that
long can still continue its track.  The *simple* linker takes the nearest
pairs first; the *global assignment* takes, per frame, the set of links
with the smallest sum of squared jumps.

**3. Tracks.**  The chains of links are numbered 1, 2, ... in the order
they start.  A chain of fewer than *minimum length* localizations -- a lone
localization included -- gets 0.

```figure Simulated tracks over 60 frames, the fluorophores dark in 15% of the frames for 1.5 frames on average. Left, the truth; middle, linked without memory: every dark frame cuts a track in two (and lone localizations, grey, are in none); right, with memory 2 the dark frames are bridged and most tracks come out whole; where three particles cross (bottom left) some are still mixed up.
fig.set_size_inches(9, 3.4)
axes = fig.subplots(1, 3, sharex=True, sharey=True)
jump = 3 * np.hypot(jump_nm(0.3, 10.0), 40.0)
draw(axes[0], truth, f"the truth: {len(np.unique(truth))} tracks")
for ax, memory in zip(axes[1:], (0, 2)):
    ids, _ = link_tracks(x, y, frame, max_jump_nm=jump, memory=memory)
    found = score(ids, truth, frame)
    draw(ax, ids, f"memory {memory}: {ids.max()} tracks, "
                  f"{100 * found['tracks_whole']:.0f}% whole")
axes[0].set_ylabel("y (µm)")
```

## In detail

**Candidates.**  Localization $j$ is a candidate to continue localization
$i$ when

$$1 \leq f_j - f_i \leq m + 1 \quad \mathrm{and} \quad |\mathbf{r}_j - \mathbf{r}_i| \leq R ,$$

with $f$ the frame, $\mathbf{r}$ the position (x and y, or x, y and z with
*in 3D*), $m$ the *memory* and $R$ the *maximum jump*.  The same $R$ holds
for a link across dark frames as for one between consecutive frames, as in
Crocker and Grier's code and trackpy.  A particle diffusing for $k$ frames
goes $\sqrt{k}$ times as far as in one, so a fast particle's dark frames
are bridged less often than a slow one's: set $R$ for the longest gap that
should be bridged.

All candidates of all frames come from one search: x is shifted by the
frame times a stride wider than the field plus $2R$, so a search around
$x + (f + k) \cdot \mathrm{stride}$ meets only the localizations of frame
$f + k$ -- one search per gap length, no loop over frames.

**Simple.**  The candidates are visited by the frame they lead to, then by
distance, then by gap length (at equal distance the shorter gap first),
and a link is taken when its earlier localization has no successor yet and
its later one no predecessor.  This is exactly "frame by frame, nearest
first", done in one pass in C++.

**Global assignment.**  Per frame, among the open track ends $i$ and the
frame's localizations $j$, the links are those that minimise

$$\sum_{\mathrm{linked}} d_{ij}^2 + R^2 \cdot (\mathrm{ends\ left\ open}) ,$$

Crocker and Grier's criterion: a track end not continued costs as much as
the longest allowed jump.  It is the same as maximising
$\sum (R^2 - d_{ij}^2)$ over the links, which
`scipy.optimize.linear_sum_assignment` solves for each group of candidates
that compete; a link nobody else wants is taken without a solve.  It
differs from the simple linker where two particles pass close by: the
simple one gives the closest pair first and may leave the other particle
with no link, where the global one may prefer two slightly longer jumps.
On the benchmark below, global assignment found exactly trackpy's links.

**Groups.**  Each file is linked on its own, and with *per channel* each
channel too, so that two files or two channels are never one track.  Track
ids continue from one group to the next.

**Benchmark.**  Simulated tracks (`smappy.simulate.simulate_tracks`): 300
frames of a 20 µm field, particles lasting 20 frames on average, dark in
10% of the frames for 1.5 frames on average, 20 nm precision; *memory* 3
and the *maximum jump* $3\sqrt{\langle J^2 \rangle + 4\sigma^2}$ -- three
times the root-mean-square jump $J$ as it is measured, with the precision
$\sigma$ of both ends.  Over densities of 0.02 to 1 particles per µm² and
jumps of 50 to 400 nm, compared by the ratio of the jump to the mean
distance between nearest neighbours, $1/(2\sqrt{\rho})$ at density
$\rho$ (the study `studies/tracking/bench.py`):

| jump / spacing | true links found | wrong links | whole tracks |
| --- | --- | --- | --- |
| below 0.1 | 98.6-99.6%, the two within 0.3% | under 1% | simple within 1.6% of global |
| 0.1 to 0.2 | simple 0.3-1% fewer | 0.5-5% | simple 1-3% fewer |
| 0.25 to 0.4 | simple 1-3% fewer | 5-15% | under half, both |
| 0.6 to 0.8 | 63-81% | 20-40% | almost none, both |

The simple linker took 1.4-2.5 µs per localization (up to 11 µs at the
densest), the global assignment 4-31 µs.
trackpy (0.7, the same jump and memory) gave the global assignment's links
exactly, at 7-25 µs per localization, and gave up (a subnetwork too large)
from a ratio of 0.4 on.  On 880,000 localizations over 4000 frames: simple
2.3 s, global 4.6 s, trackpy 5.5 s.  Where a single-molecule tracking
experiment can be -- jumps well below the distance between molecules --
the two linkers agree, and the simple one is the default for being faster
and easier to reason about.  Where global assignment is better, by a few
percent, both get too many links wrong to be trusted: lower the density.

```figure The two linkers against the truth at 0.5 particles per µm², over the ratio of the jump to the distance between molecules: the true links each finds (solid) and the fraction of tracks it finds whole (dashed). They agree while the ratio is small; past about 0.2 the global assignment is a little better, and both lose most whole tracks.
fig.set_size_inches(6, 3.4)
ax = fig.subplots()
ratios = {"simple": [], "global": []}
for jump in (50.0, 100.0, 200.0, 400.0):
    sim = simulate_tracks(n_frames=150, field_um=10.0, density_um2=0.5,
                          d_um2_s=(jump ** 2 / 0.04 / 1e6,), p_dark=0.1,
                          dark_frames=1.5, seed=7)
    for algorithm in ratios:
        ids, _ = link_tracks(sim["x_nm"], sim["y_nm"], sim["frame"],
                             max_jump_nm=3 * np.hypot(jump, 40.0), memory=3,
                             algorithm=algorithm)
        found = score(ids, sim["track_id_true"], sim["frame"])
        ratios[algorithm].append((jump / spacing_nm(0.5), found["links_found"],
                                  found["tracks_whole"]))
for k, (algorithm, values) in enumerate(ratios.items()):
    r, links, whole = np.array(values).T
    ax.plot(r, links, "o-", color=f"C{k}", label=f"{algorithm}: true links found")
    ax.plot(r, whole, "s--", color=f"C{k}", label=f"{algorithm}: whole tracks")
ax.set_xlabel("jump / distance between molecules")
ax.set_ylabel("fraction")
ax.set_ylim(0, 1.02)
ax.legend(fontsize=7, loc="lower left")
```

## Parameters

### algorithm
*simple* unless the particles come close to each other within a jump --
then *global assignment* gets a few percent more links right, at a few
times the time (still seconds for a million localizations).

### max_jump_nm
About three times the root-mean-square jump as measured, precision
included: for a particle with diffusion coefficient $D$ (µm²/s) over a
frame time $t$ (s), $J = \sqrt{4 D t}$ µm in 2D, and the maximum jump about
$3\sqrt{J^2 + 4\sigma^2}$.  Too small, and the fast steps (and the
bridged gaps) are cut, splitting tracks; too large -- near the distance
between molecules -- and a track ending next to another particle's start
is joined to it.  For a population with fast and slow particles, set it
for the fast ones.

### memory
As many frames as the fluorophore's dark runs usually last; a longer run
still cuts the track.  Each frame more lets a track continue after a
longer gap, but also lets an ending track be joined to a particle that
appears near its end.  0 cuts a track at every missed frame.

### min_length
2 drops only the lone localizations.  An analysis of motion wants more:
an MSD needs several steps per track, so 5 or more.  It
changes nothing about the links, only which tracks get an id.

### column
`track_id` unless there is a reason: link a second time with other
settings into `track_id_strict`, say, to compare the two, or link a MINFLUX
file without touching its `tid`.  The lengths go to the same name with
`_length` for `_id` (`track_id_strict_length`).  It writes a new column,
`track_id`, or one it wrote before -- never over a column of the table's
own, so a typo cannot overwrite `frame`.  What reads tracks (the *tracks*
render mode, Tracking analysis) has a *track id* to choose the column.

## Output

The table with `track_id` and `track_length`, a line with the number of
tracks, the localizations in them, the mean length and the settings, and
a figure: the tracks drawn as lines (at most 3000 of them), and how many
localizations the tracks have.  Good tracking shows long tracks with a
length distribution that falls off smoothly; many tracks of 2-3
localizations, where long ones are expected, mean the maximum jump or the
memory is too small.  Both columns keep their value on merged blinks (the
largest, a recipe with the rule `max`), so a layer that merges blinks
still shows the tracks.

## Differences from SMAP

Based on SMAP's *SimpleTracking*
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)), which links with
Crocker and Grier's code (`mytrack.m`).  The main changes:

* Two linkers.  SMAP's is the *global assignment* here, solved there by
  enumerating the possible links of each group of competing particles, which
  stops with "excessive combinatorics" when a group is large.  Here the
  default is the *simple* nearest-first linker, which gives the same links
  on sparse tracks, and the global one is solved by a linear assignment
  with no such limit.
* The default maximum jump is 500 nm, not 1000: a jump that long between
  single molecules is rarely one that could be fitted, and a long limit
  invites wrong links.
* Every file is linked on its own, and each channel with *per channel*;
  SMAP linked all, per layer, or per channel of one file.
* A localization in a track shorter than the minimum gets `track_id` 0, and
  `track_length` is written with the ids in one run.

## References

* Crocker JC, Grier DG. Methods of digital video microscopy for colloidal
  studies. *J Colloid Interface Sci* 179, 298-310 (1996).
  [doi:10.1006/jcis.1996.0217](https://doi.org/10.1006/jcis.1996.0217) --
  the global assignment's criterion.
* Jaqaman K, Loerke D, Mettlen M, et al. Robust single-particle tracking in
  live-cell time-lapse sequences. *Nat Methods* 5, 695-702 (2008).
  [doi:10.1038/nmeth.1237](https://doi.org/10.1038/nmeth.1237) -- u-track,
  for dense tracking, not used here.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
