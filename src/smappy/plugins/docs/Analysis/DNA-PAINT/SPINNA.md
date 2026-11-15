---
version: "1"
covers: [smappy.spinna.parse_structure, smappy.spinna.neighbour_orders, smappy.spinna.search_space, smappy.spinna.proportions, smappy.spinna.farthest_points, smappy.spinna.around, smappy.spinna.simulate, smappy.spinna.nn_distances, smappy.spinna.ks_statistic, smappy.spinna.nnd_score, smappy.spinna.totals_of, smappy.spinna.simulated_distances, smappy.spinna.fit, smappy.spinna.fit_efficiency, smappy.spinna.efficiency_from_proportions, smappy.plugins.spinna.Spinna.run, smappy.plugins.spinna.Spinna.preflight, smappy.plugins.spinna.Spinna._gather]
---

## What it does

Many membrane proteins work as complexes: a receptor pairs up when its
ligand binds, two proteins meet in a signalling complex.  DNA-PAINT with
RESI or *Molecular mapping* resolves the single proteins, one position each, a few nanometres apart.  It
does not count them into complexes directly: most complexes miss a label
(the [labelling efficiency](plugin:Panels/Glossary) is often 30-60 %), each
label sits a few nanometres off its protein, and two complexes can lie as
close to each other as the molecules of one.

SPINNA (*single-protein investigation via nearest-neighbour analysis*,
[Masullo, Kowalewski et al. 2025](https://doi.org/10.1038/s41467-025-59500-z),
ported from Picasso) answers by simulation.  You say which **structures**
the molecules may be in -- monomers, dimers 20 nm long, trimers, a complex of
two proteins -- and SPINNA asks which mixture of them, placed at random,
labelled and blurred as the experiment labels and blurs, gives
[nearest-neighbour distances](plugin:Panels/Glossary) like the data's.  The
answer is each structure's share of the molecules, with an error from a
[bootstrap](plugin:Panels/Glossary).

Use it on a table of **molecules**, one row each: the centres of
*Molecular mapping* or any table of single-protein positions (`x_nm`,
`y_nm`, and `z_nm` for 3D).  Each visible layer is one **target** (one kind
of protein): A the first visible layer, B the second -- so two proteins
imaged in two rounds or channels are two layers, each showing its file or
channel.  It is not for raw localizations: a molecule seen in many frames
would look like a cluster of itself.  It needs the labelling efficiency and
the label uncertainty, measured beforehand (the second fit, *labelling
efficiency*, measures the first: below), and the molecules spread evenly
over the window -- in a cell whose density changes, draw ROIs where it is
even and fit each (*per region*).

## How it works

```figure-setup
from smappy import spinna as sp
from smappy.plugins.spatial_statistics import Window
side = 6000.0
box = Window(np.ones((1, 1), bool), 0.0, 0.0, side, side)
monomer = sp.parse_structure("monomer", "A 0 0")
dimer = sp.parse_structure("dimer", "A -10 0; A 10 0")
model = sp.Model([monomer, dimer], [0.5], [5.0])
rng = np.random.default_rng(104)
# 4000 molecules, 30 % of them in dimers 20 nm long; half of them labelled
data = sp.simulate([monomer, dimer], [2800, 600], box, [0.5], [5.0], rng)
field = sp.Field(box, data)
fit = sp.fit(model, [field], n_sims=5, bootstraps=10, seed=0)
orders = fit.orders
observed = sp.data_distances([field], orders)
best = sp.simulated_distances(model, [field], fit.counts, orders, 5,
                              np.random.default_rng(1))
```

**1. The structures.**  An [oligomer](plugin:Panels/Glossary) or any
other complex is a structure.  Each structure is a short list of molecules about
its centre, in nm: a dimer 20 nm long is `A -10 0; A 10 0`, a heterodimer
of the two targets `A -5 0; B 5 0`.  A *model* (a preset) fills the list;
any structure can be edited, added or ticked off.

**2. The candidates.**  The labelling efficiency says how many molecules
there are in all: the observed ones over the efficiency.  Every candidate
mixture uses up exactly that many, so with monomers and dimers, choosing the
number of dimers fixes the number of monomers.  The free numbers are put on
a grid of *grid steps* values, from none to as many as the molecules allow.

**3. A simulation per candidate.**  The structures are placed uniformly
at random in the window, each turned at random, each molecule moved by a
Gaussian of the [label uncertainty](plugin:Panels/Glossary) (the label is not on the protein's
centre, and its position has an error), and only the *labelling efficiency*
of the molecules kept.  This is repeated *simulations* times.

```figure The same density of molecules as monomers alone (left) and as dimers 20 nm long (middle), labelled at 50 % and blurred by 5 nm -- a square of 1 µm.  Right: the distance from each molecule to its nearest neighbour.  Dimers add a peak at their length, from the dimers with both molecules labelled; that peak is what SPINNA weighs.
rng2 = np.random.default_rng(3)
alone = sp.simulate([monomer], [4000], box, [0.5], [5.0], rng2)[0]
pairs = sp.simulate([dimer], [2000], box, [0.5], [5.0], rng2)[0]
fig.set_size_inches(8.0, 2.8)
axes = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 1.5]})
for ax, pts, title in zip(axes, (alone, pairs), ("monomers", "dimers")):
    inside = (pts[:, 0] < 1000) & (pts[:, 1] < 1000)
    ax.plot(pts[inside, 0], pts[inside, 1], ".", color="#2880C4", markersize=3)
    ax.set_xlim(0, 1000); ax.set_ylim(0, 1000); ax.set_aspect("equal")
    ax.set_title(title, fontsize=9); ax.tick_params(labelsize=7)
for pts, colour, label in ((alone, "0.5", "monomers"), (pairs, "#2880C4", "dimers")):
    d = sp.nn_distances(pts, pts, 1, True)[:, 0]
    axes[2].hist(d, bins=np.arange(0, 200, 4), density=True, histtype="step",
                 color=colour, linewidth=1.4, label=label)
axes[2].set_xlabel("nearest-neighbour distance (nm)", fontsize=8)
axes[2].legend(fontsize=7, frameon=False); axes[2].tick_params(labelsize=7)
```

**4. The score.**  For each pair of targets the distance from every
molecule to its 1st, 2nd, ... nearest neighbour is measured, in the data and
in the simulation, up to as many neighbours as a structure gives a molecule
(a dimer one, a trimer two).  Each distribution of the simulation is
compared with the data's by the [Kolmogorov-Smirnov
statistic](plugin:Panels/Glossary), the largest gap between their
cumulative distributions, and the statistics are averaged: 0 for the same
distributions, 1 for the most different.

**5. The search.**  Trying every candidate (*every candidate*) is safe and
slow.  *Coarse to fine* first tries candidates spread evenly over the grid,
then every candidate near the best of them.

**6. The errors.**  The best mixture is simulated again, *rounds* times,
each simulation taken as the data and fitted, among the candidates near the
best.  The spread of the answers is the error.

```figure Left: the data's nearest-neighbour distances (bars) and those of the best-fitting simulation (line), for simulated data with 30 % of the molecules in dimers.  Right: the score of every candidate tried, coarse (crosses) then fine (dots); the best is ringed, with its bootstrap error.
fig.set_size_inches(8.0, 3.0)
left, right = fig.subplots(1, 2)
sp.draw_histograms(left, observed[0], best[0], "A to A")
sp.draw_scores(right, fit)
right.axvline(30.0, color="0.6", linestyle="--", linewidth=1)
right.text(31, right.get_ylim()[1] * 0.92, "truth", fontsize=7, color="0.4")
for ax in (left, right):
    ax.tick_params(labelsize=7)
```

## In detail

**Which table.**  Each target's layer as it shows: its rows, its filter, the
drawn ROI.  A layer that merges blinks gives a row per blink, which is
seldom what is wanted: SPINNA is meant for molecules, one row each.

**Proportions are of molecules.**  A structure's share is the number of
molecules in it over all molecules, all targets together: 300 dimers and
1400 monomers are $2 \cdot 300 / 2000 = 30\,\%$ in dimers, not 18 % of the
structures.  This is the paper's and Picasso's convention.

**The candidates.**  With $a_{ts}$ the molecules of target $t$ in
structure $s$ and $N_t$ the observed molecules of $t$ over its efficiency
(rounded down), the numbers of structures $n_s$ solve

$$\sum_s a_{ts}\, n_s = N_t \quad \mathrm{for\ every\ target}\ t .$$

The first structures that make the system solvable are dependent, the rest
free.  Each free $n_s$ takes *grid steps* values evenly from 0 to the most
the molecules allow, rounded; the dependent ones are solved, and a
candidate that needs a negative number is dropped.  One target and two
structures is one free number, *grid steps* candidates; three structures
two free numbers, about half of *grid steps* squared.  So a structure added
multiplies the run time.

**The simulation.**  Each structure's centres are uniform in the window;
the turn is uniform in the plane (*rotation*: or uniform in 3D, or none); for
*3D* the centres are uniform in z over the molecules' z range.  The label
uncertainty is a Gaussian of that standard deviation in each coordinate.  Of
each structure's molecules of a target exactly $\lfloor n\,\eta \rfloor$
are kept, $\eta$ the efficiency, drawn without replacement -- Picasso's
choice, which takes the binomial noise of labelling out of the simulation.
Molecules of a structure near the window's edge may fall just outside it;
they are kept, as in Picasso.

**The neighbours compared.**  For targets $t_1 \leq t_2$: the 1st to
$k$-th nearest neighbours, $k$ the largest number of partners a structure
gives -- $m-1$ for a structure of $m$ molecules of one target, the smaller
count for two targets -- and none where no structure has two molecules
that could be neighbours.  Two targets are measured one way, from A to B.
A molecule is never its own neighbour.

**The score.**  With $F$ and $G$ the cumulative distributions of a
distance in the simulation and the data,

$$D = \max_r \left| F(r) - G(r) \right| ,$$

and the score is the mean of $D$ over every pair of targets and every
order.  A candidate's simulations are pooled before $D$ is taken.  The
score is noisy -- two simulations of one candidate do not give the same
$D$ -- which is what *simulations* reduces.

**Coarse to fine.**  The coarse pass takes the candidate nearest the middle
of the grid, then again and again the one farthest from those taken
(farthest-point sampling), at least 10 % of the candidates and until every
candidate is within 15 percentage points of one taken.  The distance between
two candidates is the sum over structures of the differences of their
proportions (an L1 distance).  The fine pass tries every candidate within 30
points of the coarse winner and keeps the best.

**The bootstrap.**  The best mixture is simulated once per round, the
distances of that simulation taken as the data, and the candidates within
30 points of the best scored against it; the standard deviation of the
proportions found over the rounds is the error, and that of their scores
the score's.  It is a *parametric* bootstrap: it says how far the answer
moves with the randomness of placing and labelling, not whether the model
is right.  An error below half a grid step means the rounds seldom left the
best candidate; more grid steps resolve it.

**Labelling efficiency** (*fit*: *labelling efficiency*;
[Hellmeier et al. 2024](https://doi.org/10.1038/s41592-024-02242-5)).  A
protein carries two tags, A (the first visible layer) and B (the second),
each labelled and imaged on its own.  The model is A alone, B alone and an
A-B pair at the *tag distance*, simulated with every label detected; the
proportions found are then read as efficiencies:

$$\eta_A = \frac{p_{AB}/2}{p_B + p_{AB}/2}, \qquad
\eta_B = \frac{p_{AB}/2}{p_A + p_{AB}/2} ,$$

the fraction of the B molecules that have an A partner, and the other way
round.  Several tag distances are each fitted and the best-scoring kept.
The bootstrap's proportions give the efficiencies' errors.

**Several files** are several windows.  Each file gets the candidate's
structures in proportion to its share of the molecules, and the distances
of all files are pooled before they are compared.

**What limits it.**  Structures smaller than the nearest-neighbour
distance at the density measured show; a dimer longer than the typical
distance between complexes hides among them.  Low labelling efficiency
leaves few complexes with two labels: at 30 % one dimer in eleven shows
both.  And the model has to hold the right structures: a mixture fitted
with only monomers and dimers will call trimers dimers.  Compare models by
running each and reading their scores.

## Parameters

### fit
*proportions of structures* needs the efficiency known; *labelling
efficiency* measures it on a sample made for it, two tags on one protein.

### structures
Each structure's molecules are written as `x y` or `x y z` in nm about its
centre, separated by `;`, each after an optional target letter: `A -10 0;
A 10 0`.  Every structure ticked is a dimension more of the search, unless it
holds a target of its own.

### efficiency
Measure it (the *labelling efficiency* fit, or a reference such as DNA
origami) rather than guess it: the proportions move with it.  Too low a
value overestimates the complexes: fewer of their molecules are taken to be
seen, so more complexes are needed for the close pairs the data show.

### label_unc_nm
The size of the label (antibody or nanobody, linker) and the precision of
the molecule's position together: 3-8 nm for nanobodies and RESI, more for
antibodies.

### granularity
21 is a step of 5 points for monomers and dimers.  A finer grid costs a
candidate per step for one free structure, but squares for two.

### simulations
Ten gives a smooth enough score for a few thousand molecules; fewer
molecules want more.

### distance_nm
Around the size of the two binders together; several values separated by
commas are each fitted and the best kept.

## Output

**Text**: the molecules of each target, their density and the window;
which neighbours were compared, how many candidates were tried and how
long it took; each structure's share of the molecules with its bootstrap
error and the number of structures; the score.  In the *labelling
efficiency* fit, the efficiency of A and of B and the tag distance.

**Figures**: *nearest neighbours*, the data's distances (bars) against the
best simulation's (lines) for each pair of targets and each order -- they
should lie on top of each other; a gap that no mixture closes says the
model lacks a structure or the efficiency or label uncertainty is off.
*Scores*, the score of every candidate tried against its proportion: a
clear minimum is a good fit; a flat floor means the data cannot tell those
mixtures apart.  *Window*, the window and the molecules in it.

**Values** (per region, a column each of the region table):

* `<structure>_pct` and `<structure>_err_pct` -- each structure's share of
  the molecules and its error, the name in lower case with spaces as `_`
  (`dimer_pct`);
* `efficiency_A`, `efficiency_err_A`, ... and `distance_nm` -- from the
  labelling-efficiency fit;
* `ks` and `ks_err` -- the score;
* `n_A`, `density_A_um2`, ... -- the molecules of each target and their
  density per µm².

*Stop* ends the search; nothing is kept.

## Differences from Picasso

SPINNA is Picasso's (`spinna.py`); the constants are its own -- 20 bootstrap
rounds, neighbourhoods of 30 points (L1), its default grid of 21, 10
simulations, 5 nm and 50 % -- and so is the convention of proportions of
molecules.  The changes that can move an answer:

* **The window**: the region, the drawn ROI or the area the molecules occupy,
  uniform inside.  Picasso's density masks (a blurred image of the data,
  centres drawn in proportion to it) are not ported; fit regions of even
  density instead.
* **The coarse pass covers the grid**: Picasso samples 10 % of the
  candidates, which for one free structure and 21 steps is two candidates,
  and its fine pass can then not reach every answer.  Here the sampling goes
  on until every candidate is within half the fine pass's radius of one
  sampled.
* **Several files** are fitted together, each in its own window.
* Not ported: the Bayesian search, fitting the label uncertainty, the
  comparison of several models in one run, and the masks.

## References

* Masullo LA, Kowalewski R, Honsa M, et al. Spatial and stoichiometric in
  situ analysis of biomolecular oligomerization at single-protein
  resolution. *Nat Commun* 16, 4202 (2025).
  [doi:10.1038/s41467-025-59500-z](https://doi.org/10.1038/s41467-025-59500-z)
  -- SPINNA.
* Hellmeier J, Strauss S, Xu S, et al. Quantification of absolute labeling
  efficiency at the single-protein level. *Nat Methods* 21, 1702-1707
  (2024). [doi:10.1038/s41592-024-02242-5](https://doi.org/10.1038/s41592-024-02242-5)
  -- the labelling efficiency from a reference tag.
* Schnitzbauer J, Strauss MT, Schlichthaerle T, Schueder F, Jungmann R.
  Super-resolution microscopy with DNA-PAINT. *Nat Protoc* 12, 1198-1228
  (2017). [doi:10.1038/nprot.2017.024](https://doi.org/10.1038/nprot.2017.024)
  -- Picasso, where this implementation comes from.
