# Wave 6: Segment regions, simpler and with more structures

Decided with Jonas on 2026-10-25.  This covers porting-list rows #109
(segmentBeads), #111 (segmentCME and other templates) and #135
(ImportSitePositions), together with the NPC cleanup that SMAP did after
segmenting (`NPCsegmentCleanup`).

Do it in one separate session, in three steps (6.1-6.3 below), each step
with its tests and the page.  Before starting, read this file,
`src/smappy/plugins/segment.py`, `src/smappy/plugins/npc.py` (the ring
template's `segment_npcs` / `ring_quality`) and CLAUDE.md.

## The aim

**One plugin, kept as simple as possible**, even where that means losing a
feature or being less general.  A segmentation that is very specific to one
structure gets a plugin of its own, built on the helpers here.

A method is a **model of the structure**.  A **preset** is a set of values
for a method, as now: a YAML file in `data/segment_presets`, or the
user's own in the config directory.  Filtering the candidates is a
**condition** on what was measured on each candidate, *keep*, which the
preset fills in.

## Methods

| method | finds | what it measures on each candidate |
|---|---|---|
| **circle template** | a ring or a disc, at a radius or over a range of radii | radius, roundness, and whatever the shape adds |
| **beads** | bright spots seen in many frames | the fraction of frames, photons, spread |
| **from an id column** | unchanged | -- |
| **positions from a file** | a csv of x, y (nm), with a file column optionally | -- |

**Density peaks is removed.**  A disc template at a small radius does its
job.  What is lost is the Gaussian blur's tolerance for blobs that have no
particular shape; that was accepted.  The *ring template* method becomes
the circle template with *shape* = ring.  No aliases for the old settings
(as in wave 2): the NPC preset, `npc_analysis.chain.yaml`, the tests and
the docs move to the new names.

### Circle template

* **Settings** (detection only): *shape* (ring | disc), *min radius* and
  *max radius* (equal for one radius), *ring width* (ring only), *bin*
  (advanced).  The separation is not a setting: two candidates are kept
  apart when their circles overlap by more than half the smaller radius.
  The implementing session decides the exact rule and says what it is on
  the page.
* **Filter**: a zero-sum kernel, as the ring filter already is.  The ring
  is a Gaussian ring minus a disc of the same weight.  The disc is a filled
  disc minus a surrounding annulus of the same weight.  Over a radius
  range, the filter runs at radii spaced by about the ring width or 15 % of
  the radius, whichever is larger.  Each response is a number of
  localizations above the local background, so responses at different
  radii compare directly.  At each position the best radius wins, and it
  is the candidate's first size estimate.
* **Then per candidate**, on its localizations within a window of about
  1.5 times its radius:
  * the centre, from a circle fit with the radius fixed (soft-L1, as now);
  * `radius_nm`: for a ring, the radius fitted free; for a disc,
    `sqrt(2)` times the RMS distance from the centre (the radius of a
    uniform disc), from localizations better than the precision
    cut-off;
  * `roundness`: the SD along the short axis divided by the SD along the
    long axis, from the second moments (Cluster shape's functions);
  * ring only: `n_ring`, `n_inside` and `n_outside`, plus `far_p`, the
    current binomial far test's p value, the test itself unchanged;
  * disc only: `centre_ratio`, the density inside half the radius divided
    by the density in the outer half.  For CME, pits that look like rings
    late on show as a low ratio: a value to sort or filter on, not a
    second class.
* **Presets:**
  * `NPC`: ring, 55 nm, *keep* as the current checks, written as a
    condition;
  * `CME`: disc, 30-150 nm, can be changed later;
  * `Circles`: ring, 100-500 nm, for any larger ring-like structure.

### Beads

`fiducials.find_beads` as it is (min photons, min length as a fraction of
the frames, maximum jump, memory): an ROI at each bead's median position,
the bead's localizations its members.  Values: `frames_fraction`,
`photons` (median), `spread_nm` (SD of the positions).  The use is
**stability and drift**: *Fiducials* already takes each used ROI as a bead
(*beads from* = ROI manager), and *Stability* runs per ROI.  Bead calibration has its own
bead finding; using bead ROIs for PSF learning is an option for later.

### Positions from a file

A csv (or tsv) with `x_nm` and `y_nm` (the csv reader's unit and column
conventions where they apply), optionally a file number or name.  An ROI
at each position, given the project's ROI size.  Small.

## The values and *keep*

Every method measures each candidate with one shared set of values, plus
its own:

* shared: `n_locs` (within the window), `radius_nm`, `roundness`,
  `psf_nm` (the mean `sigma_nm` when the table has it: SMAP's in-focus
  check), `touching` (1 when another candidate's circle overlaps this
  one's, else 0);
* ring: `n_ring`, `n_inside`, `n_outside`, `far_p`;
* disc: `centre_ratio`;
* beads: `frames_fraction`, `photons`, `spread_nm`.

**keep** is one text field, in *Filter ROIs*' syntax (the parser is
`roi_manager.order.condition`, reused as it is).  It replaces the ring's
check settings (*min localizations*, *min / max fitted radius*, *max far*,
*far test level*).  The NPC preset writes the current checks as
`n_ring >= 4 and 40 <= radius_nm <= 70 and far_p >= 0.05`, with
`psf_nm <= 150` added where there is a PSF column.  An empty *keep* keeps
every candidate.

* A candidate that fails is dropped, or kept as an unused ROI when *keep
  rejected* is ticked; its origin names the clauses it failed.
* The values are stored on each ROI and show as columns of the results
  table, so *Filter ROIs*, *Sort ROIs* and the summaries read them like
  any measured value.  The implementing session decides whether they come
  from `origin` or from a "segment" evaluation, whichever keeps the
  results table's code simplest.
* The Preview draws a histogram of each value *keep* uses, with the
  condition's bounds marked, and the candidates kept and rejected on the
  image as now.  The form's `describe` lists the value names the chosen
  method provides.
* A name in *keep* that the method does not provide is an error naming
  the values it does provide (`psf_nm` without a `sigma_nm` column
  included: the run says so instead of passing every candidate silently).

**Touching structures** are flagged (`touching`) and not split.  A preset
may exclude them (`touching == 0`).

**z and frame ranges** need no setting: Segment searches the table as the
ROI manager filters it, so a z or frame filter there limits the search.

## Helpers

The building blocks are module-level functions on arrays in one module,
`smappy/segmentation.py`, so that a script, a notebook or a user's plugin
gets them without the GUI: the density image, the zero-sum ring and disc
kernels, the filter over a radius range with the best radius per pixel,
peak finding with the overlap rule, the per-candidate window, the circle
and moment fits, and evaluating *keep* over a list of candidates.  A user's
own segmenter subclasses `SegmenterPlugin`, which already gives it Preview,
*replace* and an ROI undo step.  Users cannot add methods to *Segment
regions* itself: that machinery waits until it is asked for.  Matching
against an arbitrary template (for example the output of *Average ROIs*)
fits these helpers and is left for later.

## The numbers can change

The ring segmenter may find slightly different pores from now on: Jonas
agreed that the numbers can move.  `version` goes to 2.  Re-run
`studies/npc_le`'s selection check (`studies/npc_le/selection.py`) once
and note the result in its README if it moved materially.  The NPC page
and *NPC labelling efficiency*'s page are re-read for what they say about
the segmentation.

## Steps

| step | what | done when |
|---|---|---|
| **6.1** | `smappy/segmentation.py`; the circle template (ring and disc, radius ranges) with its values and *keep*; density peaks removed; the `NPC`, `CME` and `Circles` presets; the chain, the tests, `docs/regions.md`, `docs/roi_manager.md`, CLAUDE.md's shape table and the Segment regions page updated | the NPC tests pass through the new settings; a simulated field of discs of 30-150 nm returns their radii within 15 %; ellipses return their roundness; touching pairs are flagged; `--slow` passes |
| **6.2** | beads | simulated beads (`simulate.with_beads`) become ROIs with `frames_fraction`; Fiducials with *beads from* ROIs gives the same drift as finding them itself |
| **6.3** | positions from a file | a round trip: ROI centres exported with *Export table*, read back as the same ROIs |

Afterwards, update `SMAP_porting_plan.md`'s *Done* list and the porting
spreadsheet's status column for #109, #111 and #135.
