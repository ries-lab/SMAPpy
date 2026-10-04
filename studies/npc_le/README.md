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
`python studies/npc_le/summary.py` for the tables; `fraction.py` makes the
fraction-of-data curves.

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

**6. Not yet explained**: the −5 points at 1 blink and ELE 0.7.  Candidates
are corners whose only blink is grouped with a blink of the neighbouring
corner (the grouping distance, 50 nm, is more than the 42 nm between
corners), and emitters of one pore on in the same frame, which the
simulation removes as unresolvable (`close: remove`).

## What follows

* For data like 5000 photons per blink, the plugin's rule is good to a few
  points, and its bias depends on the blinks and the ELE in a smooth way.  A
  correction by simulation matched to the data (blinks per fluorophore and
  photons, both measurable) would take out most of what is left.
* A rule applied to one pore at a time cannot do much better.  What could is
  pooling: the strays and the lost corners are both rates that can be
  measured over all pores, and put into the corner model.
* Finding 6 is worth a short look first, because a cause in the grouping
  would be cheap to fix.
