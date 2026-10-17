# Other SMLM software: what SMAPpy could take from it

A survey written on 2026-10-17 to choose what to implement next.  The
repositories were cloned and their code read; nothing was run.  Facts carry
`path:line` in each tool's repository and the commit read; claims from a
paper, a README or memory are marked *(unverified)* or *(from the paper)*.
ThunderSTORM and DECODE were left out by request, and the MINFLUX tools are
in a survey of their own, [minflux.md](minflux.md).

| file | area | tools read |
| --- | --- | --- |
| [picasso.md](picasso.md) | Picasso and its satellites | Picasso 0.11.3 (localize, render, qPAINT, G5M, RESI, SPINNA, AIM, average, server) |
| [tracking.md](tracking.md) | tracking and track analysis | TrackMate, trackpy, laptrack, u-track, Spot-On, saSPT, vbSPT, ExTrack, AutoStepFinder, BNP-Step, andi_datasets |
| [clustering.md](clustering.md) | clusters, spatial statistics, counting, colocalisation | Locan, PoCA / SR-Tesseler, ClusterViSu, Clus-DoC, RSMLM, BaClAva, FOCAL3D, SMoLR, pointpats, the Nieves 2023 benchmark |
| [platforms.md](platforms.md) | large platforms and user experience | PYME, GDSC SMLM, locan / napari-locan, viewers, commercial tools (documentation only) |
| [minflux.md](minflux.md) | MINFLUX: readers, MBM, trace analysis (written earlier, 2026-10) | PYME-extra, MBM-inspector, pyMINFLUX, minflux-viewer, minflux-analysis, napari-storm, Pyflux, matFLUX, MFX_data_analysis, ... |
| [drift_averaging_fitting.md](drift_averaging_fitting.md) | drift, registration, resolution, averaging, model fitting, fitting | DME / photonpy, NanoJ, ImDecorr, smlm_datafusion2d/3d, smlm_classification2d, LocMoFit, fit3Dcspline, storm-analysis, ZOLA-3D, Gpufit, PERPL |

[candidates.md](candidates.md) lists every candidate from all the files in one table.

Each file has the same sections: the software (licence, port route), worth
porting (ranked), better than SMAPpy's own, user experience, licence-restricted
(re-implement only), not worth it, references.  Where two files cover the
same feature (CBC, the PC-PALM fit, the accumulation check, sCMOS maps) they
reached the same conclusion independently.

**Port routes.**  MIT / BSD / Apache: port with attribution in the module
docstring.  GPL / LGPL, no licence, closed: re-implement from the paper only.
All of Picasso, Locan, saSPT, ExTrack, trackpy, laptrack, DME/photonpy and the
TU Delft fusion code are permissive; GDSC SMLM, PYME, TrackMate, u-track,
Spot-On, NanoJ, ImDecorr and ZOLA-3D are GPL; AutoStepFinder and storm-analysis
(at its root) have no licence file -- storm-analysis's `spliner` and `L1H` are
academic non-commercial, its background and clustering modules MIT.
LocMoFit and fit3Dcspline are GPL but the lab's own; relicensing LocMoFit may
need its co-authors' agreement.

## Where SMAPpy already stands well

Every agent compared against SMAPpy's code and found these as good as or
better than what it read: RCC and COMET drift, fiducials, FRC, NeNA, the
fitter's likelihood (Gpufit's MLE has no read-noise term), split-chip channel
registration, Ripley/pair correlation (exact set-covariance edge correction
for any window, Monte Carlo envelopes, the *keep blinks* null -- no open tool
has better), MSD with the optimal number of lags, the sparse-data linkers
(as good as LAP with gap closing there), the Picasso HDF5 reader and writer,
and the 3D view.  AIM and NanoJ drift add nothing COMET and RCC lack.

## Fixes to SMAPpy's own code

Things the survey found wrong or weaker in SMAPpy itself, each checked in the
code:

1. **Motion blur in MSD and jump distances** (`tracking/analysis.py:27`, its
   docstring says it is not modelled): the offset from the precisions is the
   static error; the blurred model is 4DΔt(n − 1/3) + 4σ² (GDSC SMLM;
   Michalet 2010).  D of fast molecules comes out low.  S.
2. **Jump distances from one lag, and no out-of-focus correction**
   (`fit_jumps`): long tracks dominate, the fast fraction is underestimated
   in 2D.  Spot-On's multi-lag model with saSPT's MIT `f_remain_rbm`.  M.
3. **Repeated blinks of one molecule after a gap** still pair with
   themselves in pair correlation and clustering (only consecutive frames
   merge).  Count only pairs more than T frames apart (DDC, Bohrer 2021) as a
   frame filter in `Index.pair_histogram`; the expectation stays exact.  S–M.
4. **COMET's kernel is isotropic** (`csrc/drift.hpp`, one `inv_sigma` for x,
   y and z), why `docs/drift.md` needs a hard z cut; DME weights each pair by
   the two localizations' own precisions per axis, and gives a split-half
   error bar for the drift, which nothing in SMAPpy reports yet.  M.

## A shortlist across the areas

Ranked by value for the effort; sizes S ≤ a day, M a few days, L a week+.

| # | what | from | size | route | fits in SMAPpy as |
| --- | --- | --- | --- | --- | --- |
| 1 | **Counting fit (B4)**: oligomer fractions by ML on bursts per region, blinking from monomers or the NPC fit | papers; `smappy/npc.py` already has the pgf | M | own | a Summarize plugin over *Blinking* |
| 2 | **qPAINT**: sites = 1/(influx · τ_dark), influx calibrated on known picks; an ML dark-time fit | Picasso | S | MIT | a value of *Blinking*, calibration in Summarize |
| 3 | fixes 1–3 above | | S–M each | | |
| 4 | **Colocalisation per localization** (CBC / DoC; Coloc-Tesseler) | Locan; PoCA paper | S | BSD | a *Neighbourhood* quantity (Locan's weighting looks wrong: follow the paper) |
| 5 | **Single-molecule test per cluster** (hull area against its expectation) and a temporal accumulation check | Locan (Ebert 2022) | S | BSD | values of *Cluster shape*, so *Filter ROIs* drops them |
| 6 | **Sticky / partial-event filters**: clusters whose frames bunch in time; binding-event cores | Picasso (RESI clusterer) | S | MIT | an option shared by *Find clusters* |
| 7 | **G5M**: clusters to molecules by a constrained Gaussian mixture, BIC, Sparrow test | Picasso | M | MIT | a `per_region` measurement or a *Find clusters* stage |
| 8 | **Per-track diffusive states**: state arrays (posterior per track as a column); HMM with transitions | saSPT; ExTrack | M; M–L | MIT | *Tracking analysis* approaches |
| 9 | **Step finding** | AutoStepFinder (paper); BNP-Step | S–M | re-implement | a `per_region` *Find steps*; prerequisite for StepsMINFLUX |
| 10 | **LocMoFit core** | SMAP | L | own | a `per_region` *Model fit* (ring, dual ring, cap, arc, cluster) |
| 11 | **Template-free particle fusion and classification** | smlm_datafusion2d/3d | L | Apache | data-driven alignment for *Average ROIs* |
| 12 | **PC-PALM fit of g(r)** (Sengupta) | GDSC SMLM, papers | M | re-implement | a model in *Spatial statistics* |
| 13 | **SPINNA**: oligomer proportions from k-NN distances against masked simulations | Picasso | M–L | MIT | `per_region` with `smappy.simulate` |
| 14 | **Depth-dependent z correction** from beads in gel | fit3Dcspline | M | own | a calibration plus a table operation |
| 15 | smaller: GLRT p-value per localization (photonpy, S), sCMOS maps (Picasso, M), local FRC map (S), *pick similar* (S), drift from tracks (trackpy, S), ToMATo (S–M, re-implement), automatic merge dark time (S), G/F functions (S), AnDi test data (S) | | | | |

## User experience and automation

The ideas the agents ranked highest, in [platforms.md](platforms.md) and the
UX section of each file:

* **History to chain**: the logged runs written as a re-runnable
  `.chain.yaml` -- in PYME the session is its recipe.
* **Batch**: several named inputs per job zipped by position, files run in
  parallel (both on SMAPpy's own "not yet" list), an HTML report per batch,
  and a folder watcher with a per-file QC table (Picasso Server).
* **Live**: run a chain on a file when the acquisition finishes it (PYME's
  cluster).
* **Trying settings**: test a clustering on one ROI before the whole field;
  a parameter scan; ToMATo's persistence diagram as the Preview.
* **Browsing**: a slider through the ROIs, pick curation by time trace,
  linked selection between a plot and the image, per-track features as
  columns so the layer filter is the track filter (TrackMate).
* **Scripting**: notebook display of chains and results, an embedded console.
* **Plugins**: a registry of external plugins with pinned hashes, loaded only
  when enabled (Picasso).
* **Exchange**: TrackMate XML/CSV import and export.

## Licence-restricted, worth considering for re-implementation

Collected from the files' licence sections: Spot-On's jump model (GPL; saSPT
has the correction under MIT), u-track's lag-aware search and likelihood cost
and MSS (GPL), vbSPT (code not found), AutoStepFinder (no licence), GDSC
SMLM's blink estimator, residence times, OPTICS with re-cutting, TC-PALM and
PC-PALM fit (GPL), PYME's jittered-triangulation renderer and shell fits
(GPL), ToMATo (RSMLM, GPL), Bayesian cluster analysis (BaClAva claims MIT but
adapts Griffié's code), NanoJ-SQUIRREL error maps (GPL), decorrelation
resolution (GPL), ZOLA-3D and 3D-DAOSTORM multi-emitter fitting (GPL / no
licence), and StormGraph, CAML, SuperResNET (no public code).

## Caveats

* Some DOIs could not be re-checked online and are marked in each file's
  references; the commercial tools rest on search snippets only.
* Sizes are the agents' estimates, including page and tests.
* Two apparent bugs in other tools were noted (Locan's CBC weighting,
  BaClAva's `plabel`); a port should follow the paper there.
