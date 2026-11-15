# DNA-PAINT: Picasso's tools in SMAPpy

The plan for bringing the DNA-PAINT analysis of Picasso (Jungmann lab,
`jungmannlab/picasso` 0.11.3, MIT) into SMAPpy as a few plugins, written
2026-11-14.  The survey behind it, with Picasso's code references, is
[survey/picasso.md](survey/picasso.md); this file decides what is built, in
which plugins, and what is taken from Picasso as it is.

## Picasso's shape and what SMAPpy has already

Picasso is separate programs (Localize, Render, Filter, Average, Design,
Simulate, SPINNA, Nanotron, Server) over one package.  Its DNA-PAINT
analysis sits in Render's *Postprocess* menu, mostly as
`postprocess.py`, with the newer methods in modules of their own:
`clusterer.py` (the SMLM clusterer, RESI), `g5m.py`, `spinna.py`, `aim.py`.

| Picasso | in SMAPpy |
| --- | --- |
| link, dark times (0.11.3's convention) | `group.py`, *Blinking* -- the same convention |
| pick kinetics, qPAINT, influx calibration | *Binding kinetics*, *Summarize/qPAINT* (ML dark time, censored) |
| frame analysis (sticky sites) | values of *Binding kinetics* for *Filter ROIs*; not yet a cluster filter |
| pick similar | *Segment regions*' *like examples* |
| show trace, select picks by trace | *Binding kinetics*' trace in the ROI manager |
| RCC, NeNA, DBSCAN, HDBSCAN | *RCC*, *Localization Precision*, *Find clusters* -- equal or better |
| undrift from picked | *Fiducials* with *markers* = structures, from ROIs or an id column |
| AIM | not ported: decided *skip* ([survey/candidates.md](survey/candidates.md)), COMET and RCC cover it |
| sCMOS maps | *Camera noise calibration* |
| HDF5 + yaml files | the *Picasso* reader and writer, `io/picasso.py` |
| Server | `smappy-batch watch` and the summary page |

## Ported or re-written: one rule

**Nothing depends on the `picassosr` package**: it brings PyQt5, numba and
numba-CUDA, and a reader of Picasso's files needs none of them.  Every
algorithm is written again in SMAPpy's way -- module-level functions on
arrays, the cell search of `csrc/neighbours.cpp` where it counts
neighbours -- and **a port keeps Picasso's constants and its decisions**
(G5M's width bounds, the frame-analysis thresholds), naming the function it
follows.  Where SMAPpy departs from Picasso (a likelihood where Picasso fits
a histogram, a joint spline where Picasso averages frame by frame) the module
docstring and the page's *Differences* say so, as *Binding kinetics* does.

**Acknowledgement**: every ported module's docstring names Picasso, its
licence and the functions; every page cites the method's paper and Picasso
(Schnitzbauer 2017); `THIRD_PARTY_NOTICES.md` lists what was ported, and
Picasso's licence ships in the wheel (`data/licenses/picasso.txt`) as MIT
asks; *Help > About* names the work SMAPpy includes.  A file SMAPpy writes in
Picasso's format names SMAPpy and its steps in the metadata list, as each of
Picasso's own steps does.

## The plugins

### 1. *Find clusters*: local maxima and frame analysis  (S)

* **A fourth *algorithm*, *local maxima*** -- Picasso's SMLM clusterer
  (`clusterer._cluster`; Schlichthaerle 2021, Reinhardt 2023): each
  localization's neighbours within *radius* counted (itself included); a
  local maximum is a point whose count is the largest in its neighbourhood
  and at least *min points*; the maxima, in order, take their unlabelled
  neighbours; clusters below *min points* dropped.  One cluster per binding
  site, where DBSCAN chains neighbouring sites.  On `Index.pairs`.
* **A *frame analysis* tick for every algorithm** (`clusterer._frame_analysis`):
  a cluster is noise when its mean frame is outside 20-80 % of the
  acquisition or when one of twenty equal time bins holds more than 80 % of
  it -- sticky imager, a site found late.  `binding_kinetics.frame_spread`
  already computes the numbers.
* **`follow_roi`** with a Preview: Picasso's *test clustering* on one pick.

### 2. *Molecular mapping*  (M) -- new, `Analysis/Regions/`

From clusters to molecules: the step after *Find clusters* for DNA-PAINT
and RESI.

* *method*: **centres** (RESI's: the mean, with `std / sqrt(n)` as its
  precision, `clusterer.find_cluster_centers`) or **G5M** (`g5m.py`;
  Kowalewski 2026): a Gaussian mixture in each cluster whose number of
  components is chosen by BIC, each component's width held to 0.8-1.5 times
  the localization precision, models with components that cannot be told
  apart refused (the Sparrow test), a p-value per molecule; the default
  post-filter on `std_frame`, `p_val` and `n_events` as Picasso's.
* runs per cluster (`per_region`, a block over `cluster_id`, or the whole
  table at once over its ids); writes `molecule_id` onto the localizations
  and adds **a table of molecules as a file of its own** (`Result.files`,
  as *Average ROIs*) with `x_nm`, `y_nm`, (`z_nm`), `xy_err_nm`,
  `n_locs`, `n_events`, `std_frame`, `p_val`, `cluster_id`.  The molecules
  are then localizations like any others: rendered, measured by *Spatial
  statistics*, counted by *Binding kinetics*, registered by *Register
  layers*.
* **RESI is this plugin** on each round (a file or a channel), the centres
  keeping the round as their channel; no RESI code of its own.

### 3. *SPINNA*  (M-L) -- new, `Analysis/Measure/`

Oligomeric states from nearest-neighbour distances (`spinna.py`; Masullo,
Kowalewski 2025): structures (monomer, dimer at d, ... of one or several
targets) placed at random in the observed window, rotated, labelled with an
efficiency, blurred by the precision; the k-th nearest-neighbour distances of
simulation and data compared by the mean two-sample KS statistic; the
proportions found by a grid search, then coarse to fine; bootstrap errors;
the labelling-efficiency fit of Hellmeier 2024 as an option.  The structures
are YAML presets; the window is the ROI or the occupied area, as *Spatial
statistics* takes it, not Picasso's mask generator; the targets are layers.
Picasso's Bayesian search later, if the grid is too slow.

### 4. Drift from DNA origami: *Fiducials* with structures  (S-M)

Picasso's *Undrift from picked* (`postprocess.undrift_from_fiducials`,
`_undrift_from_picked_coordinate`) takes picks of DNA origami (found by
*pick similar*) as markers: per frame, the mean offset of each pick's
localizations from its centre of mass, averaged over the picks with weights
1 / (each pick's mean squared deviation from the average), frames without a
localization interpolated.  Picasso's documentation recommends two rounds:
whole origami first, then the single binding sites.

SMAPpy's *Fiducials* already fits a marker's offset jointly with one spline
through every marker (`beads.fit_offsets_and_spline`), which is the better
model for sparse, blinking markers -- a frame needs no localization, and a
structure that appears late lands where it belongs.  What is missing is only
how markers are taken: `fiducials.beads_in_regions` keeps the brightest
localization per frame above *min photons* (a bead), which throws away a
structure's many dim ones.  So:

* *Fiducials* gains a *markers* choice, **beads** (as now) or **structures**:
  every localization of a region is a point of that marker, weighted by its
  precision (`xy_err_nm`), no photon cut;
* *beads from* (now *markers from*) gains **an id column** beside *find them* and *ROI manager*:
  the binding sites of *Find clusters* (local maxima) as markers -- the
  second round; the ROI manager's ROIs, from *Segment regions*' *like
  examples*, are the first;
* the per-marker rejection (`beads.select_beads`) and the bead window work
  unchanged; Picasso's 1/MSD weighting is what that rejection already does
  better (a marker out, or in, by its noise against the others').

### 5. A shipped chain: `dna_paint.chain.yaml`

Merge blinks, drift (RCC, then *Fiducials* on the binding sites), *Find
clusters* (local maxima, frame analysis), *Molecular mapping*, then *Binding
kinetics* per molecule and *qPAINT* -- the workflow as one step, and a batch.

### 6. Smaller pieces  (S each)

* **A DNA-PAINT preset in *Simulate***: `simulate.simulate_binding_sites`
  exists but no preset reaches it; origami-like grids and dimers at known
  spacing give the tests and the pages' figures of 1-4.
* **The Picasso writer**: a linked table written with `len` and `n`
  (Picasso's names for the event's frames and localizations); the original
  file's metadata list kept and SMAPpy's steps appended to it, so a file that
  goes to Picasso and back keeps its history; `molecule_id` / `cluster_id`
  offered as Picasso's `group`.
* **Binding-event cores** (Steen 2024, `postprocess.select_binding_event_cores`):
  a region of *Remove Localizations*, *binding-event edges* (remove, or hide
  as a flag), that drops each event's first and last frame,
  where the imager bound or left during the exposure.
* **Temporal median background** for detection (`localize.TemporalMedianMovie`):
  built and removed again (2026-11-15, commit 9471059f reverted).  The
  difference of Gaussians already removes background smoother than a few
  spots, which free imager is; what it leaves -- static structures the size
  of a spot, hot pixels -- is filtered after the fit.

## Status, 2026-11-14: all built

| step | where | measured on simulations |
| --- | --- | --- |
| 1 | *Find clusters*: *local maxima*, *frame analysis*; *Simulate*'s origami preset | Picasso's clusters, independent of the table's order; 816 of 816 sites |
| 2 | *Molecular mapping*, `smappy/molecules.py` | G5M splits an origami cluster into its 12 sites, each within 2 nm; two molecules resolved from 7 nm at 3 nm precision |
| 3 | *SPINNA*, `smappy/spinna.py` | 30 % dimers fitted 30-40 % (bootstrap error 4 %); labelling efficiency to 0.01 |
| 4 | *Fiducials*, *markers* = structures, *markers from* an id column | origami round 1.5 nm rms, binding-site round 0.2 nm |
| 5 | `dna_paint.chain.yaml`, *Workflow/DNA-PAINT* | drifting origami to 818 of 823 molecules |
| 6 | Picasso writer (`len`, `n`, the metadata list); *Remove Localizations*' *binding-event edges* (the fitters' *temporal median* was built and removed, above) | cores' photons within 0.5 % |

Departures from Picasso are in each module's docstring and each page's *In
detail*; the details of each step in its commit.  Not ported from SPINNA:
the Bayesian search, the label-uncertainty fit, model comparison and the
density masks.

## Not ported

AIM (decided *skip*), Design (origami staples), Nanotron (an MLP on rendered
picks), Average and Average3 (reference-free rotational averaging; behind
particle fusion in the candidates), Picasso's own localization (nothing
SMAPpy's fitters lack), the Server's Streamlit pages.

## Order

1 and the DNA-PAINT preset first (the test data of everything after), then
2 (centres, then G5M), 4, 3, and 5 and the rest of 6.  Each lands with its
page and tests and its line in `candidates.md`.

## References

* Schnitzbauer J, et al. Super-resolution microscopy with DNA-PAINT. Nat Protoc 2017 -- Picasso. [10.1038/nprot.2017.024](https://doi.org/10.1038/nprot.2017.024)
* Schlichthaerle T, et al. Nat Commun 2021 -- the local-maximum clusterer. [10.1038/s41467-021-22606-1](https://doi.org/10.1038/s41467-021-22606-1)
* Reinhardt SCM, et al. Ångström-resolution fluorescence microscopy. Nature 2023 -- RESI. [10.1038/s41586-023-05925-9](https://doi.org/10.1038/s41586-023-05925-9)
* Kowalewski R, et al. Molecular mapping in DNA-PAINT via modified Gaussian Mixture Modeling. Nat Commun 2026 -- G5M. [10.1038/s41467-026-70198-5](https://doi.org/10.1038/s41467-026-70198-5)
* Masullo LA, Kowalewski R, et al. Nat Commun 2025;16:4202 -- SPINNA. [10.1038/s41467-025-59500-z](https://doi.org/10.1038/s41467-025-59500-z)
* Hellmeier J, et al. Nat Methods 2024;21:1702 -- labelling efficiency. [10.1038/s41592-024-02242-5](https://doi.org/10.1038/s41592-024-02242-5)
* Steen PR, et al. Nat Methods 2024 -- binding-event cores. [10.1038/s41592-024-02374-8](https://doi.org/10.1038/s41592-024-02374-8)
