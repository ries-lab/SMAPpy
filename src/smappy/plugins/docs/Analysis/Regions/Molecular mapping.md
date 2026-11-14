---
version: "1"
covers: [smappy.molecules.centres, smappy.molecules.events_in, smappy.molecules.g5m, smappy.molecules.fit_mixture, smappy.molecules.kmeans_pp, smappy.molecules.resolvable, smappy.molecules.molecule_values, smappy.molecules.keep, smappy.molecules.g5m_clusters, smappy.plugins.molecular_mapping.MolecularMapping.run, smappy.plugins.molecular_mapping.MolecularMapping.preflight, smappy.plugins.molecular_mapping.molecule_table]
---

## What it does

In [DNA-PAINT](plugin:Panels/Glossary) every binding site is localized
again and again: each time an imager strand binds, the site gives a few
localizations, and over the acquisition it gives tens to hundreds, scattered
about the site by the [localization precision](plugin:Panels/Glossary).
*Molecular mapping* turns these clouds into **molecules**: one position per
site, much more precise than any single localization.

It works on clusters, an id attribute that *Find clusters* writes
(`cluster_id`), and has two *methods*:

* **centres** -- one molecule per cluster, at the mean of its
  localizations.  This is the step **RESI** (resolution enhancement by
  sequential imaging, [Reinhardt et al. 2023](https://doi.org/10.1038/s41586-023-05925-9))
  is made of: when each cluster holds exactly one site -- *Find clusters*'
  *local maxima* aims for that -- its mean is the site, known to the
  precision divided by the square root of the localizations.
* **G5M** ([Kowalewski et al. 2026](https://doi.org/10.1038/s41467-026-70198-5)),
  a Gaussian mixture modified for molecular mapping: inside each cluster it
  finds how many molecules there are and where, even when they are closer
  together than the clouds are wide.  Use it on clusters that may hold
  several sites -- a whole DNA origami, a protein complex -- found by
  DBSCAN, say.

Both are ported from [Picasso](https://doi.org/10.1038/nprot.2017.024),
the Jungmann lab's DNA-PAINT software.

The molecules come out as **a new file**, a molecule per row, with its
position, the precision of it (`xy_err_nm`), its localizations and binding
events: they render, filter and measure like localizations, so
[Spatial statistics](plugin:Analysis/Measure/Spatial statistics), SPINNA or
[Register layers](plugin:Analysis/Register/Register layers) (for RESI's
imaging rounds) run on them.  Or, with *output* set to the localizations,
each localization gets its molecule's number, `molecule_id`, so
[Binding kinetics](plugin:Analysis/Measure/Binding kinetics) or
[Id statistics](plugin:Analysis/Measure/Id statistics) can measure each
molecule.

It reads the localizations, not merged blinks, whatever the layer shows,
and only those the layer's filter and ROI keep.  G5M needs each
localization's precision (`xy_err_nm`, and `z_err_nm` in 3D); the centres
do not.

## How it works

```figure-setup
from smappy import molecules as mol
from smappy.plugins.clusters import dbscan, local_maxima
rng = np.random.default_rng(3)
sites = np.array([(i * 20.0, j * 20.0) for i in range(4) for j in range(3)])
parts, frames = [], []
for s in sites:
    n = int(rng.integers(30, 80))
    starts = np.sort(rng.integers(0, 19990, 12))
    frames.append(np.sort(starts[rng.integers(0, 12, n)] + rng.integers(0, 5, n)))
    parts.append(rng.normal(0, 3.0, (n, 2)) + s)
xy = np.vstack(parts)
frame = np.concatenate(frames)
lp = np.full(len(xy), 3.0)
```

### Centres

**1. One cluster, one molecule.**  The molecule is the mean of the
cluster's localizations.

**2. Its precision.**  The standard deviation of the localizations about
that mean, divided by the square root of their number: 50 localizations at
3 nm give a molecule known to about 0.4 nm.

**3. Its binding events.**  The cluster's localizations split wherever more
than 3 frames pass without one.  A real site binds many times over the
acquisition; a cluster of one or two events is suspect.

### G5M

**1. A mixture of Gaussians.**  Inside each cluster the localizations are
modelled as the sum of a few round Gaussian clouds, one per molecule, each
with its own centre, width and share of the localizations.  The model is
fitted by expectation maximisation, from several starting points chosen by
k-means++ (seeded, so a run gives the same answer every time).

**2. Clouds as wide as the precision.**  At every step a cloud's width is
held between 0.8 and 1.5 times the precision of the localizations it
explains.  A cloud cannot grow to cover two sites, nor shrink onto a few
localizations.  This is what stops the mixture from inventing molecules or
merging them.

**3. Molecules that can be told apart.**  A model is refused when two of
its clouds overlap so much that their summed density has no dip between
them (the Sparrow criterion): two molecules closer than that are one.  A
cloud with fewer than *minimum size* localizations is not a molecule.

**4. How many.**  One molecule, then two, then three ... are fitted, and
the number with the lowest **BIC** (Bayesian information criterion: the
fit's likelihood, penalised for every parameter added) is kept.  The search
stops after three numbers in a row without a better BIC.

**5. Each molecule** gets its position and its precision, its width, its
localizations and binding events, the spread of its frames over the
acquisition, and a p-value (below).  *Picasso's filter*, on by default,
drops molecules whose frames spread over less than a tenth of the
acquisition (a sticky imager), with a p-value of 0.015 or less, or with
3 binding events or fewer.

```figure A simulated DNA origami: 12 sites on a 20 nm grid, 3 nm localization precision (left, a colour per localization's site, the sites as black crosses). DBSCAN with a 10 nm radius makes it one cluster; G5M finds 12 molecules in it (right, coloured by molecule, the molecules as crosses), each within a nanometre or two of a true site.
model = mol.g5m(xy, lp)
labels = model.predict(xy)
truth = np.repeat(np.arange(12), [len(p) for p in parts])
fig.set_size_inches(8, 3.6)
left, right = fig.subplots(1, 2, sharex=True, sharey=True)
left.scatter(xy[:, 0], xy[:, 1], s=2, c=truth % 20, cmap="tab20", linewidths=0)
left.scatter(sites[:, 0], sites[:, 1], s=40, c="k", marker="+")
left.set_title(f"12 sites, {dbscan(xy, 10.0, 9).max()} DBSCAN cluster", fontsize=9)
right.scatter(xy[:, 0], xy[:, 1], s=2, c=labels % 20, cmap="tab20", linewidths=0)
right.scatter(model.means[:, 0], model.means[:, 1], s=40, c="k", marker="+")
right.set_title(f"G5M: {len(model)} molecules", fontsize=9)
for ax in (left, right):
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
left.set_ylabel("y (nm)")
```

```figure How close two molecules can be. Two sites of 100 localizations each at 3 nm precision, 12 simulations at each separation: G5M finds both from about 7 nm on, a little more than twice the precision -- the separation at which two such clouds first show a dip between them (dashed) -- and below that calls them one.
seps = [3, 5, 7, 9, 11, 13, 15]
found = []
for sep in seps:
    two = 0
    for k in range(12):
        r = np.random.default_rng(100 + k)
        pts = np.vstack([r.normal(0, 3.0, (100, 2)), r.normal(0, 3.0, (100, 2)) + [sep, 0]])
        two += len(mol.g5m(pts, np.full(200, 3.0))) == 2
    found.append(two / 12)
fig.set_size_inches(5, 3.2)
ax = fig.subplots()
ax.plot(seps, found, "o-", color="0.3")
ax.axvline(6.0, color="#c06000", lw=1, ls="--")
ax.text(6.2, 0.05, "2 precisions", color="#c06000", fontsize=8)
ax.set_xlabel("separation (nm)")
ax.set_ylabel("fraction found as two")
ax.set_ylim(-0.05, 1.05)
```

## In detail

### Centres

For a cluster of $n$ localizations $x_i$, the molecule is
$\bar{x} = \frac{1}{n}\sum_i x_i$ and its precision per axis is
$s / \sqrt{n}$, with $s$ the standard deviation (divided by $n - 1$).
`xy_err_nm` of the molecule is the mean of the x and y values.  The frame
written is the mean frame of its localizations, rounded; `mean_frame` and
`std_frame` are the mean and standard deviation of the frames.

### G5M

**The model.**  In 2D each component $k$ is a round Gaussian of variance
$\sigma_k^2$ with weight $w_k$ and centre $\mu_k$; in 3D it has one
variance in x and y and another in z.  The expectation-maximisation step
computes each localization's responsibilities $r_{ik}$ and then
$\mu_k$, $\sigma_k^2$ and $w_k$ as their weighted means.  The width is then
clipped,

$$0.8 \, \bar{p}_k \leq \sigma_k \leq 1.5 \, \bar{p}_k , \quad \bar{p}_k = \frac{\sum_i r_{ik} \, p_i}{\sum_i r_{ik}} ,$$

$p_i$ the localization's precision -- in 3D separately for the plane and
for z.  Each number of components $K$ is fitted from $\max(K, 3)$ starts,
k-means++ with seeds 42, 43, ..., at most 100 iterations, stopping when the
mean log-likelihood changes by less than $10^{-3}$.  The start with the
highest likelihood whose components pass the Sparrow test is kept; a
component with $\mathrm{round}(w_k n) < m$ localizations ($m$ the *minimum
size*) is dropped from it.

**The Sparrow test.**  For every pair of kept components, their weighted
densities summed are sampled at 40 points along the line between the
centres; there must be a dip.

**BIC.**  With $\mathcal{L}$ the log-likelihood of the localizations under
the kept components and $P$ their parameters (2 or 3 for the centre, 1 or 2
for the widths, and the weights less one),

$$\mathrm{BIC} = P \ln n - 2 \mathcal{L} .$$

$K$ runs from 1 to $\min(100, n / m)$ and stops after 3 values without a
lower BIC.  A cluster with more than *largest cluster* localizations is
passed over.

**Per molecule.**  The precision of the centre is
$\sigma_k / \sqrt{w_k n}$, Picasso's approximation, which treats each
molecule on its own; its bootstrap is not ported.  `sigma_rel` is the width
over $\bar{p}_k$.  The binding events are the cluster's localizations split
at gaps of more than 3 frames, each event's mean position given to the most
likely molecule.  `mean_frame` and `std_frame` are responsibility-weighted.

**The p-value.**  A molecule's localizations' responsibility-weighted mean
log-likelihood, compared with what draws from that component would give,

$$\mathrm{E} = \ln \frac{w_k}{(2\pi)^{d/2} \sqrt{\det C_k}} - \frac{d}{2} , \quad \mathrm{sd} = \sqrt{\frac{d}{2 \, w_k n}} ,$$

$d$ the dimensions and $C_k$ its covariance, as a normal cumulative
probability.  Where the fitted width is free, the two are equal and the
p-value is 0.5.  It moves only where the width is held at a bound: a cloud
wider than 1.5 precisions -- two molecules taken for one, or no molecule
at all -- gets a p-value near 0, which the filter drops.

**Where it departs from Picasso.**  The algorithm is Picasso's, with its
constants.  It is written with numpy instead of numba, every start of a
component count fitted at once.  In 3D the widths are held to the
precisions along each axis, where Picasso takes the shape of an astigmatic
spot from its z calibration.  And the Sparrow test also counts a dip with a
flat bottom, which two equal molecules give with an even number of samples
and Picasso's test misses.

**Speed.**  About 0.3 s for a 12-site origami of 600 localizations on four
cores; a run expected to take more than 20 s asks first.  Clusters of one
site each, from *local maxima*, are much faster.

## Parameters

### method
Centres when every cluster is one site -- *Find clusters* with *local
maxima* and a radius of about twice the precision.  G5M when a cluster can
hold several, or to check that it does not.

### id_column
Any id attribute whose regions are clusters of localizations; *auto* is
`cluster_id`.

### output
The file is what the rest of the analysis reads.  `molecule_id` on the
localizations is for measuring each molecule's own localizations -- its
binding kinetics, its time trace.  Both runs find the same molecules,
numbered alike.

### use_z
Off maps in x and y only, which merges molecules above one another.

### g5m.min_locs
Below the localizations a site gives over the acquisition and above what
the background gathers in a cluster.  Picasso's default is 10.

### g5m.sigma_min
Picasso's 0.8.  Higher refuses molecules that are narrower than the
precision says -- a precision that is overestimated.

### g5m.sigma_max
Picasso's 1.5.  Higher accepts clouds widened by residual drift or a
linker, and with them pairs of molecules taken for one.  The localization
precision must be right for G5M to be: correct drift well before.

### g5m.max_rounds
More tries more molecules past the best before giving up: slower, and
rarely different.

### g5m.max_locs
A bead or a sticky spot gives thousands of localizations in one cluster,
and costs most of a run.

### g5m.filter
Off keeps every molecule G5M found, with its `p_val`, `n_events` and
`std_frame` to filter by hand.

### g5m.min_std_frame
A site visited at random through the acquisition has frames spread over
about 0.29 of it (a uniform distribution's standard deviation); one that
bound for a short while, much less.

### g5m.min_p
Raise it to drop more of the molecules whose width hit the upper bound.

### g5m.min_events
Short acquisitions with few binding events per site need a lower value.

## Output

* **A file of molecules** per file of the session, *molecules of* its name:
  `x_nm`, `y_nm` (`z_nm`), `xy_err_nm` (`z_err_nm`) the precision of the
  position, `sigma_nm` the width of its cloud, `n_locs`, `n_events`,
  `frame` (the mean, rounded), `mean_frame`, `std_frame`, the cluster it
  came from and `molecule_id`; with G5M also `p_val` and `sigma_rel`.
* **Or** `molecule_id` on the localizations, 0 for none.
* **Text**: how many molecules in how many clusters, and how many the
  filter dropped.
* **Figure**: the localizations coloured by molecule with the molecules as
  crosses, and how many molecules the clusters hold.  For DNA origami of
  known design, the count per cluster is the number of sites times the
  labelling efficiency; a peak at one when sites were expected means the
  clusters are single sites and centres would do.

## References

* Kowalewski R, Reinhardt SCM, Pachmayr I, Xu S, Masullo LA, Jungmann R.
  Molecular mapping in DNA-PAINT via modified Gaussian Mixture Modeling.
  *Nat Commun* 17 (2026).
  [doi:10.1038/s41467-026-70198-5](https://doi.org/10.1038/s41467-026-70198-5)
  -- G5M.
* Reinhardt SCM, Masullo LA, Baudrexel I, Steen PR, et al.
  Ångström-resolution fluorescence microscopy. *Nature* 617, 711-716
  (2023). [doi:10.1038/s41586-023-05925-9](https://doi.org/10.1038/s41586-023-05925-9)
  -- RESI and its cluster centres.
* Schnitzbauer J, Strauss MT, Schlichthaerle T, Schueder F, Jungmann R.
  Super-resolution microscopy with DNA-PAINT. *Nat Protoc* 12, 1198-1228
  (2017). [doi:10.1038/nprot.2017.024](https://doi.org/10.1038/nprot.2017.024)
  -- Picasso, where both are ported from.
