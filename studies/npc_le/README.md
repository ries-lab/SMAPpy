# NPC labelling efficiency: how biased is corner counting?

A simulation study of `ROIManager/Evaluate/NPC Corners` and
`ROIManager/Analyze/NPC Labeling Efficiency`: how far the effective labelling
efficiency (ELE) they give is from the truth, and whether a different rule for
telling an occupied corner from a gap does better.

## Set-up

* **Structure**: the Nup96 pores of `smappy/data/structures/npc.yaml` (8 corners
  of 4 copies: 2 per spoke, 2 rings 50 nm apart), each tilted at random by up
  to 10° (`npc_tilt.yaml`).  No linkage error, no background localizations.
* **Blinking**: imaged until every fluorophore has bleached (activation
  "every blink": a geometric number of blinks with mean *blinks*), the blinks
  spread evenly; 1000 frames per blink, so that the density of emitters in a
  frame is the same in every condition.  Photons per blink with a spread of
  half the mean.
* **Always grouped** (`smappy.group`, 50 nm, 1 frame).
* **Segmentation**: `segment_npcs` with *min localizations* 3.  It is not
  under study, so only sites within 20 nm of a true pore are kept, one per
  pore; the rest is counted as `junk`.
* **Truth**: per pore, the corners with a labelled copy (`labelled`), and
  the corners with a copy that left at least one grouped localization
  (`detected`).  `detected` is what a perfect counter could see, and the
  reference for every bias below.  The ELE is fitted from either the way the
  plugin fits it (likelihood, 3–8 corners).
* **Grid**: ELE 0.2 / 0.35 / 0.5 / 0.7 × photons per blink 500 / 5000 ×
  blinks 1 / 3 / 10, 3 seeds, about 150 pores each.  The spread between seeds
  is about 0.9 points.

Run it with `python studies/npc_le/bench.py` (45 s), then
`python studies/npc_le/summary.py` for the tables; `fraction.py`, `cutoff.py`
and `selfcal.py` make the curves and estimates of the later sections (each
under a minute).

## The counters

All on the localizations in the ring band (30–70 nm from the fitted centre),
the rotation as the weighted circular mean of 8θ:

| name | rule |
|---|---|
| `hard` | the plugin: a segment with at least one localization of precision < 15.7 nm |
| `hard all` | the same without the precision filter |
| `dead k` | ignore a localization within k·σ/ρ of a segment border (`all`: no fixed filter) |
| `soft t` | each localization shared among the corners by its probability of coming from each (precision widened by a spoke's spread); a corner needs evidence ≥ t |
| `relative f` | evidence ≥ f × the pore's median occupied corner, so the threshold grows with the localizations per corner |
| `hard true phase`, `hard true centre+phase` | `hard` with the true rotation, and the true centre as well |

## Findings

**1. At 5000 photons the plugin's rule is within a few points, with a bias
of either sign.**  Bias in percentage points of ELE (method − detected):

| blinks | ELE 0.2 | 0.35 | 0.5 | 0.7 |
|---|---|---|---|---|
| 1 | −1.7 | −2.8 | −3.7 | −5.1 |
| 3 | +0.2 | 0.0 | −0.1 | −2.0 |
| 10 | +2.4 | +1.7 | +2.5 | +1.7 |

The stray localizations you expected are there: with 10 blinks per
fluorophore they add 0.15 corners per pore on average, about +2 points.
But with one blink the error has the opposite sign, and it grows with the
ELE: corners whose only localization is lost to the precision filter or to
grouping.

**2. No other rule does better everywhere: every threshold trades one error
for the other.**  At 5000 photons, the worst |bias| over the conditions is
5.1 points for `hard`, 6.1 for `dead k=0.5 all`, 9.9 for `dead k=1`, 15.8 for
`soft t=1` and 17.7 to 25.3 for `relative`.  A stricter rule removes the
strays at 10 blinks (`dead k=0.5`: +0.2 to +0.6) and loses more single-blink
corners at 1 blink (−2.8 to −6.9).  A rule that looks at one pore at a time
cannot tell a corner seen once from a stray seen once.

**3. The fit is not the problem.**  The true centre and rotation change the
result by at most about a point.

**4. At 500 photons, counting fails.**  The median precision of a grouped
localization is 35 nm (3.8 nm at 5000), and only a quarter pass the 15.7 nm
filter.  Corners 42 nm apart are not resolved.  With the filter the ELE comes
out 11 to 40 points low.  Without it, it is up to 32 points high at 10
blinks.  The segmentation suffers too: it finds 35 to 133 of the about 150
pores (148 at 5000 photons), and proposes 60 to 260 sites besides them.  For data like this, counting corners does not
measure the ELE, whatever the rule.

**5. The fraction-of-data curve is a diagnostic, not an estimator.**  ELE of
the plugin's rule on a random fraction f of each pore's blinks (5000 photons):

| blinks | ELE | f=0.1 | 0.3 | 0.5 | 0.7 | 1.0 | detected |
|---|---|---|---|---|---|---|---|
| 1 | 0.35 | 0.044 | 0.092 | 0.154 | 0.222 | 0.318 | 0.344 |
| 3 | 0.35 | 0.079 | 0.188 | 0.255 | 0.299 | 0.349 | 0.348 |
| 10 | 0.35 | 0.179 | 0.283 | 0.321 | 0.345 | 0.364 | 0.351 |

With 1 to 3 blinks the curve is still rising steeply at f = 1: it is shaped by
detection, and the strays do not show.  Only at 10 blinks does it level off,
and the slope that is left at the end (+0.019 from 0.7 to 1) contains the
strays and the last of the saturation together.  Separating them needs a
model of how many times each fluorophore blinks.  What the curve does say is
whether the ELE is limited by detection: a curve still rising at the end
means more blinks (or frames) would have found more corners.

**6. The −5 at one blink and ELE 0.7 is amplification, not a new loss.**
There the counter loses the fewest corners of all (0.08 per pore, against
0.19 to 0.28 at lower ELE), but the corner histogram hardly moves with the
ELE once nearly every corner shows: $dp_c/dp = 4(1-p)^3$ is 0.11 at 0.7 and
2.0 at 0.2.  Above about 0.6 any small error in the counts becomes a large
error in the ELE.  Of the 49 corners lost in 3 seeds, 31 had only
localizations less precise than 15.7 nm (filtered, or outside the ring band),
16 were put in a neighbouring segment, half of them because grouping (50 nm)
had merged two emitters of neighbouring corners (1.5 % of grouped rows mix
emitters), and 2 had other causes.

## The bright localizations, and extrapolating to all of them

`cutoff.py`, `selfcal.py`.  Count only localizations more precise than a
cutoff *c*.  At a tight cutoff no stray lands in a gap, but only the
fluorophores with a blink that passes are seen.  With *q* the fraction of
grouped localizations that pass (measurable), and a geometric number of
blinks with mean 1/*p* (imaging to full bleaching), a fluorophore is seen
with probability $d = q / (q + p(1-q))$, and the measured ELE is
$E = LE \cdot d$.

**7. The model holds at tight cutoffs, and strays come in above them.**
Measured ELE minus $LE_{labelled} \cdot d(q, p_{true})$, points:

| photons | blinks | 6 nm | 8 nm | 10 nm | 15 nm | 25 nm |
|---|---|---|---|---|---|---|
| 500 | 1 | 0.0 | 0.0 | −0.2 | −1.2 | −4.6 |
| 500 | 3 | 0.3 | 0.3 | 0.1 | −0.2 | 0.0 |
| 500 | 10 | 0.4 | 0.7 | 2.4 | 6.8 | 14.4 |
| 5000 | 10 | 0.5 | 0.8 | 1.0 | 2.2 | 3.9 |

(mean over ELE 0.35 to 0.7.)  At 8 nm and below the bright fluorophores give
a clean ELE, even at 500 photons.  With many blinks, every nanometre of
cutoff above that lets strays in.

**8. The blinks cannot be fitted from the curve, but the pores give them.**
Fitting *LE* and *p* together to *E* over several cutoffs fails: the two are
nearly degenerate, and the cutoffs that would separate them are the ones
with strays.  With *p* known, the fit is good.  And *p* is in the data: a
pore of 32 copies holds on average $N = 32 \, LE / p$ grouped localizations.
Together with *E* at one tight cutoff this gives

$$p = \frac{E q}{N q / 32 - E (1 - q)}, \qquad LE = \frac{N p}{32} .$$

The **integrated** version fits one LE to the ELE at all cutoffs up to
10 nm (4 to 10 nm), with $p = 32 \, LE / N$ tied to it.

Bias against the labelled ELE, points, mean of 3 seeds, for ELE 0.35 / 0.5 /
0.7 (counting is the plugin as it is):

| photons | blinks | counting | single cutoff 10 nm | integrated ≤ 10 nm | moments (9.) |
|---|---|---|---|---|---|
| 5000 | 1 | −3.7 / −6.2 / −9.6 | 0.1 / −0.8 / −1.8 | −0.1 / −0.9 / −2.2 | −0.8 / −1.8 / −5.3 |
| 5000 | 3 | −0.5 / −1.0 / −3.1 | 0.4 / 0.8 / −0.4 | 0.4 / 0.6 / −1.2 | −1.2 / −0.6 / −0.2 |
| 5000 | 10 | 1.6 / 2.2 / 1.4 | 0.9 / 1.2 / 1.3 | 0.6 / 0.8 / 1.3 | 2.3 / 2.3 / 2.0 |
| 500 | 1 | −24 / −33 / −49 | −10 / −11 / −23 | −10 / −11 / −25 | −2.4 / −4.1 / −11 |
| 500 | 3 | −14 / −19 / −30 | 0.4 / −2.0 / −5.5 | −0.3 / −1.3 / −4.6 | −2.0 / −0.1 / −0.9 |
| 500 | 10 | 2.6 / −0.6 / −4.4 | 5.2 / 5.0 / 2.6 | 4.1 / 4.4 / 0.4 | 6.2 / 3.5 / 2.4 |

The cutoff estimates recover *p* (1.03, 0.345, 0.105 at 5000 photons for 1,
3, 10 blinks) and the *labelled* ELE, including the fluorophores whose blinks
were too dim to count.  At 5000 photons they are within about 2 points
everywhere, where counting is off by up to 10.  The integrated fit is a
little steadier than one cutoff (spread between seeds 0.2 to 1.6 points at
5000 photons, 1 to 5 at 500), but not more accurate.

**Why one blink at 500 photons fails, whatever the fit.**  When few
localizations pass ($q \ll p$), a fluorophore is seen with $d \approx q/p$,
so $E \approx LE \, q / p = q N / 32$: the ELE of the bright ones no longer
depends on LE at all.  Written out, the blinks follow from
$b = 1/p = 1 + (N/32 \div E/q - 1)/q$, and at $q = 0.1$ an error of 5 % in
$E$ makes one blink look like 1.5, and LE a third too low.  The data only
say how many dim fluorophores were missed when the average fluorophore has
at least one blink that passes, $q \, b \gtrsim 1$.  Both $E/q$ and $N/32$
are upper bounds on LE, and with one blink they coincide with it, but there
is no lower bound to go with them.

**9. How much N varies between pores says how often a fluorophore blinks.**
With exactly one blink, the localizations per pore are binomial; with a
geometric number of mean *b* they are overdispersed,
$\mathrm{var}(N)/\overline{N} = 2b - 1 - LE \, b$.  The mean and the
variance of N alone then give *b* and LE, without a cutoff or a corner.  On
the grid, *b* comes out 1.02 / 2.96 / 9.17 at 5000 photons and 1.20 / 3.21 /
9.76 at 500, and LE as in the last column above: it is the only estimate
that does not fail at one blink and 500 photons, but with about 150 pores
the variance is noisy (spread between seeds up to 5 points), and on real
data anything that varies between pores -- the labelling itself, background,
a neighbour in the window -- adds variance and reads as more blinks.

What all of these assume, and the simulation gives them for free:

* every pore holds 32 copies, and its localizations are its own: no
  background, no neighbours inside the 110 nm window (a slight excess shows
  at 500 photons, 12.6 rows per pore where 32 LE predicts 11.9, and *b*
  reads 1.2 for one blink);
* the same LE and the same blinking in every pore;
* a geometric number of blinks;
* blinks whose brightness is independent of the fluorophore they come from;
* one grouped row per blink.

## All pores together: the histograms fitted jointly

`pooled.py`.  Everything is fitted to histograms over all pores, the way the
plugin fits the corners.  Each estimate pools 7 simulations (about 1000 pores;
fewer at 500 photons, where fewer are found) and is repeated on 5 independent
pools; the spread between pools is the precision at that size.

* **counting**: the plain corner histogram (the plugin's rule, 15.7 nm),
  fitted from 4 corners up, since pores with fewer are the ones a
  segmentation misses.
* **localization histogram (N)**: the localizations per pore.  A pore's
  labelled copies are binomial in LE, and each has a geometric number of
  blinks of mean 1/p, so N given M labelled copies is M plus a negative
  binomial.  Its mean and its spread both enter the fit.
* **joint**: the N histogram with the corner histograms at 6, 8 and 10 nm,
  each at efficiency $LE \cdot d(q, p)$; LE and p both free.  The cutoff
  histograms are fitted over 0..8: which pores are in the sample is decided
  by all their localizations, not by the bright ones.
* **joint, plain histogram**: the N histogram with the plain corner
  histogram (4..8), at $LE \cdot d(q, p)$ with q the fraction better than
  15.7 nm.

**10. With 1000 pores the joint fits are good to about a point where the
corners are resolved.**  Bias against the labelled ELE, points, ELE 0.35 /
0.5 / 0.7, mean of 5 pools:

| photons | blinks | counting | N alone | joint | joint, plain histogram |
|---|---|---|---|---|---|
| 5000 | 1 | −4.3 / −6.1 / −9.7 | −0.8 / −1.5 / −3.1 | −0.5 / −1.0 / −2.7 | −0.9 / −1.4 / −3.1 |
| 5000 | 3 | −0.6 / −1.3 / −3.0 | 1.2 / 1.2 / 2.3 | 0.4 / 0.1 / −0.1 | 0.8 / 0.6 / 0.3 |
| 5000 | 10 | 2.5 / 2.3 / 1.3 | 2.2 / 2.9 / 4.7 | 1.0 / 0.9 / 1.0 | 2.9 / 2.9 / 2.7 |
| 500 | 1 | −25 / −34 / −48 | −4.2 / −5.0 / −9.8 | −4.7 / −5.5 / −10.4 | −4.6 / −5.9 / −11.7 |
| 500 | 3 | −13 / −19 / −28 | 1.6 / 1.1 / 0.8 | 0.6 / −0.1 / −0.3 | 2.0 / 0.9 / −1.0 |
| 500 | 10 | 1.9 / −0.9 / −4.6 | 3.6 / 4.2 / 5.2 | 3.7 / 3.6 / 4.1 | 9.5 / 8.9 / 8.1 |

The spread between pools is 0.1 to 2 points: at 1000 pores what is left is
bias, not noise.  The fitted blinks are right (p = 0.99, 0.35, 0.11 at 5000
photons for 1, 3, 10 blinks).

* The joint fit with tight cutoffs is within about a point at 5000 photons
  (−2.7 at one blink and ELE 0.7) and at 500 photons with 3 blinks.
* With the plain histogram instead, the strays it lets in come back: +3 at
  5000 photons and +9 at 500 with 10 blinks.  The model has no term for them,
  which is why the tight cutoffs do better.
* 500 photons, 10 blinks: +4, the strays already at 8 and 10 nm (finding 7).
* 500 photons, 1 blink: −5 to −10.  The N histogram reads 1.2 blinks for 1
  (p = 0.82): N is overdispersed, most likely by the imprecise localizations
  of neighbouring pores in the 110 nm window.  That is a problem of how N is
  measured, not of the model.

## Filtering less and correcting for the spillage

`spill.py`.  Two changes to the joint fit of section 10.

**Grouping by precision** (`bench.group_by_precision`, now the bench's
default).  smappy's grouping links within a fixed 50 nm, and at 500 photons a
dim frame at the end of a blink (precision 100 nm and more) lands beyond it:
one blink became 1.55 grouped rows, which the blinking model reads as more
blinks.  Linking consecutive frames within $3\sqrt{\sigma_1^2 + \sigma_2^2}$
(at most 200 nm) leaves 1.14 rows per blink (1.0 at 10 blinks), with 1 % of
rows mixing two emitters; a fixed 200 nm reaches 1.11 but mixes 3.4 %.

**N below a loose cutoff.**  The localizations per pore are counted within
100 nm of the centre and better than 30 nm, which keeps out the scattered
localizations of neighbouring pores; each copy's count is then a thinned
geometric (zero-modified), with q the fraction of all localizations of the
field that pass.

**The spillage model.**  The corners are counted with every localization
better than an explicit 20 nm, which does not depend on what the peak finding
lets through.  A localization lands in a given neighbour's segment with
probability $\varepsilon$, computed from the precisions of the counted
localizations and the margin to the border ($\pi r/8$, $r$ the ring's
radius from the precise localizations, less the $\pm 5.6$ nm of a corner's
two copies, with 4 nm added for tilt and the rotation fit): 0.06 at 500
photons, 0.01 at 5000.  Localizations spill independently, so a corner's
split into stay / left / right with $(1 - 2\varepsilon, \varepsilon,
\varepsilon)$, and the distribution of the segments seen around the ring is
exact through a $64 \times 64$ transfer matrix over consecutive corners.  It
is the binomial at $\varepsilon = 0$, and matches a Monte Carlo of its own
assumptions to the third decimal.

**11. With the spillage modelled, a 20 nm cutoff gives the labelled ELE to
about 2 points everywhere, one blink at 500 photons included.**  Bias against
the labelled ELE, points, ELE 0.35 / 0.5 / 0.7, 5 pools of about 1000 pores:

| photons | blinks | 20 nm, no spillage | 20 nm, spillage | 20 nm, spillage, ≥ 4 |
|---|---|---|---|---|
| 5000 | 1 | −0.6 / −1.3 / −2.1 | −0.5 / −1.0 / −1.7 | −0.6 / −1.0 / −1.7 |
| 5000 | 3 | 1.0 / 0.7 / 0.4 | 0.1 / −0.1 / −0.1 | 0.1 / −0.1 / −0.1 |
| 5000 | 10 | 3.6 / 3.7 / 3.3 | 0.3 / 0.3 / 1.0 | 0.3 / 0.2 / 1.0 |
| 500 | 1 | 1.5 / −1.7 / −7.0 | 2.5 / 0.4 / −3.3 | 2.9 / 1.4 / −2.7 |
| 500 | 3 | 4.1 / 2.8 / 1.2 | 0.2 / −1.5 / −2.3 | 0.1 / −1.6 / −2.3 |
| 500 | 10 | 11.8 / 11.4 / 9.5 | 0.7 / −0.3 / 1.2 | 0.6 / −0.2 / 1.2 |

The spread between pools is 0.2 to 2.3 points.  The fitted blinks are right
(p = 0.99 / 0.96 / 0.92 at 500 photons and one blink, for ELE 0.35 / 0.5 /
0.7; 0.31 to 0.34 for 3 blinks; 0.10 for 10).  Without the spillage term the
same cutoff is up to 12 points high; with it, what is left is at ELE 0.7,
where the histogram is least sensitive (finding 6).  Fitting the corners
only from 4 up changes nothing: the localizations carry the rest.

## What follows

* **The method to take into the plugin**: the corner histogram at an explicit
  cutoff (20 nm) with the spillage model, fitted jointly with the histogram
  of localizations per pore (better than 30 nm, within 100 nm), on grouping
  that links by precision.  It gives the labelled ELE and the blinks per
  fluorophore.
* Before any of it goes into the plugin, break each assumption above in the
  simulation: other blink-number distributions, brightness that varies
  between fluorophores, background localizations, labelling that varies
  between pores, grouping that splits blinks.
* Plain counting stays good to a few points at high photon numbers, but has
  a bias of either sign that depends on the blinks, grows above ELE 0.6, and
  measures the detected rather than the labelled ELE.
