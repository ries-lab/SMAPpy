# MINFLUX: what SMAPpy has, what is asked, and a plan

Written 2026-11-15 to restart the MINFLUX work.  It brings together three
sources:

* the survey of other software, [minflux.md](minflux.md) (2026-10);
* the SMAP plugins the lab uses: *Vibrations*, *DCoffset_Beads_ROIs*,
  *Localizationnumber*, *MotorPAINT*, *Segment_cotracks*,
  *Segment_singletracks* and *Tails_analyze*; from the ROI manager
  *Cluster_MINFLUX_Roi*, *StepsMINFLUX* and *StepsMINFLUX_dc*; and the
  script *MINFLUXStepsAnalyze* (read in `../SMAP` at b87902f2);
* the wishlist of N. Sergeev and T. Engelhardt, compared with their scripts,
  [Nuka-Kola/MFX_data_analysis](https://github.com/Nuka-Kola/MFX_data_analysis)
  (private, no licence; it is the lab's own group, so its ideas and
  definitions may be used; read at bf16b3c, 2026-11-11).

The details and file references are in the survey.  This page is the
overview: one table, then work packages in a proposed order.  The live list,
with states, priorities and comments, is the artifact page
[MINFLUX Feature List](https://claude.ai/artifact/7Nxbs5LRQ1adzasivPdBTu).

**Packaging comes last.**  The packages below (A-I) group the work, not the
plugins.  Once the list says what is needed, the functions are re-thought as
plugins: which extend existing ones (*Id statistics*, *Tracking analysis*,
*Stability*, *Segment regions*), which are new (sequence QC, trace starts,
*Find steps*, two-colour pairing), and whether they form a MINFLUX group of
their own.

## What is done (2026-10)

Reading `.msr`, `.npy`, `.json` and `.zip` in both record layouts.  MBM
beads as a file of their own.  *MBM drift correction* (redone from `lnc`,
beads judged against each other) and *MBM registration*.  `xy_err_nm` is
the beam pattern's CRLB.  `trace_index` is each localization's number in its
trace, so a filter can drop the creeping start.  *Id statistics* per `tid`
gives a region table and works with the region filter.  *Assign colours*
reads `dcr`, though it has not been tried on real two-colour data.
*Stability* (vibrations, Allan deviation) works on beads or traces, and
*Tracking analysis* (MSD, jump distances) on `tid`.

## What was built on 2026-11-15/16

The plan was regrouped by plugin with the user's decisions in the live
table ([MINFLUX Plan](https://claude.ai/artifact/PMD2ex78Sj8puWDYPZivpV));
built since, each with tests, a page and a `--slow` run:

* **Reader**: *every iteration* on request (`final` marks the
  localizations, the layers start on them, `group_use` = final), each
  thread's own sequence and pattern size, L scaled by the excitation
  wavelength (L = patGeoFactor x 360 nm x lambda/642, abberior's wiki).
* *Analysis/MINFLUX/Sequence QC* -- ParaFlux's plots from every iteration.
* *Analysis/MINFLUX/Sequence table* -- the sequences side by side.
* *Analysis/MINFLUX/Pair colours* -- `pair_id` for two threads, the offset
  between the colours (per pair, time, field; a transformation).
* *Analysis/MINFLUX/Quality filter* -- cfr and efo cuts from the data.
* *Analysis/Tracking/Trace statistics* -- displacement, velocity, axes,
  consecutive-difference precision, trend, tails, the rotated track.
* *Analysis/Tracking/Find steps* -- AutoStepFinder, split/merge, pooled
  steps and dwells, steps as a file, two-colour pairs.
* *Analysis/Tracking/Trace inspector* -- one trace, or a gallery.
* *Analysis/Regions/Summarize/Compare groups* and *Analysis/Process/File
  conditions* -- region tables compared between files and conditions.
* *Stability*'s random scatter split into eta x precision and a floor.
* Id columns: -1 is "in none" everywhere, so trace 0 is a region; a file's
  ids kept apart from the table's (`tid_file`).

Open for the user: a filter range drops NaN, so a cfr bound cannot be set
where cfr is measured once a trace -- should a bound be able to let NaN
through?  bnp-step's licence.  Real two-thread data to test the pairing
and the per-thread precision on.  Not done: the pooled consecutive-pair
mode in *Localization precision*, MINFLUXStepsAnalyze's hypoexponential
fits, manual step editing, a MINFLUX tutorial, MINFLUX plugins in the
shipped GUI's tabs.

## The functionality, side by side

In the table, ✅ means SMAPpy has it, ◐ means partly, and ✗ means not yet.
"MFX scripts" is Sergeev and Engelhardt's repository.  "Others" covers
ParaFlux (Abberior), pyMINFLUX, minflux-viewer and PYME.

| area | SMAP | MFX scripts | others | SMAPpy | package |
| --- | --- | --- | --- | --- | --- |
| **Import** of `.msr` and its measurements | Loader_minflux | specpy, every iteration row | all | ✅ final iterations only | – |
| Several files at once, unique trace ids | – | `utid` = date + acq + tid | – | ✅ a file's `tid`s moved apart as it is added, its own kept in `tid_file` | A |
| Conditions per file ("independent variables") | – | from file names, by hand | – | ✗ | I |
| **Sequence** parameters kept | – | joined onto every row, every thread | PYME | ◐ thread 0, in the metadata | H |
| Sequence as a table, files side by side | – | PDF table | – | ◐ text in the import dialog | H |
| Sequence maker, creator, condenser | – | from reference sequences ("Toolkit") | ParaFlux calculator | ✗ | H? |
| **All iterations**: per iteration plots, abort reasons (`sta`), iteration efficiency, scouting times | MINFLUX_timings_scouting | `param_by_iteration`, abort reason | ParaFlux, PYME Paraflux | ✗ (the reader keeps final rows) | G |
| Theoretical precision | – | L/(2√(2N)), L scaled by λ/640 | minflux-viewer CRLB | ✅ with background, L scaled by λ/642 | A |
| **Per-trace metrics**: n, centre, spread, duration, medians of efo / cfr | ClusterStatistics_MINFLUX, Cluster_MINFLUX | `compute_trace_metrics` | PYME, pyMINFLUX | ✅ *Id statistics* | – |
| Trace end-to-end displacement, velocity, photon budget (sum) | StepsMINFLUX stats | ✓ | – | ✗ | B |
| Trace precision std(diff)/√2, robust std, dt | Cluster_MINFLUX_Roi | `trace_precision_x/y` | PYME (tracking) | ✗ (*Stability* has the Allan deviation) | B |
| Principal axes: eccentricity, angle, trend across the track | ClusterStatistics_MINFLUX | `trace_eccentricity`, `trace_trend_yrot_nm` | – | ◐ *Cluster shape* through a block (slow) | B |
| Binned std against photons, compared with L/√(8N); scatter against CRB with a noise floor | ClusterStatistics, Cluster_MINFLUX_Roi | experiments C3 and C3b | – | ◐ *Stability*'s scatter | B |
| **Track rotation** (long axis on x, forward in time) | MotorPAINT, StepsMINFLUX | `rotate_all_tracks` | – | ✗ | C |
| **Per-localization steps**: displacement, rolling std, robust z | – | `loc_displacement_*`, `rolling_disp_std` | – | ✗ | D |
| **Tails** (a trace's start converging) | Tails_analyze: d(n), length, duration, d against photons | adaptive labels (rolling std ≤ 7 nm; static: 3 σ about the centroid of locs 50–75) | ParaFlux maps | ◐ fixed cut by `trace_index` | D |
| CFR cut after the first violation | – | `cfr_posthoc_filter` (unused) | – | ✗ | D |
| **Step finding**: steps, dwells, step sizes | StepsMINFLUX (AutoStepFinder, split/merge, refine, manual editing, movie) | bnp-step | – | ✗ | E |
| Pooled step analysis: step-size fit, dwell times (Erlang, hypoexponential), velocity | MINFLUXStepsAnalyze | – | – | ✗ | E |
| Steps as a file, track angle and length as columns | MotorPAINT | – | – | ✗ | E |
| An ROI per long trace | Segment_singletracks | – | – | ◐ *Segment regions* from an id column; no start-to-end criterion | E |
| **Two colours by `thi`**: traces paired by time overlap | Segment_cotracks, StepsMINFLUX_dc | `get_co_traces` | – | ✗ (`thi` is never read) | F |
| Offset between colours: fast, over time, over the field | DCoffset_Beads_ROIs (+ bead-scan script) | – | – | ✗ | F |
| Two colours by DCR | – | – | PYME, pyMINFLUX, minflux-viewer | ✅ untried on real data | – |
| **Vibrations**, bead spectra | Vibrations | MBM FFT | ParaFlux MBM | ✅ *Stability* | – |
| Localization number in trace | Localizationnumber | `loc_idx`, `loc_idx_vld` | – | ✅ `trace_index` | – |
| MBM drift, registration | – | MBM_analysis | minflux-viewer, PYME | ✅ | – |
| MSD per trace, bounded or wandering | StepsMINFLUX (optional) | experiments C1 and C2 | minflux-viewer | ✅ *Tracking analysis* | – |
| FRC on trace means | – | – | pyMINFLUX | ✗ (small) | B |
| **Plot grids**: facets by any column; gallery of the longest traces | – | `plot_trace_stats`, `plot_traces_and_params` | – | ◐ overlays by layer, summaries by keyword | I |

## Work packages, in a proposed order

**A. Fixes (S).**
* *Id statistics*, and every region table of an id column
  (`ColumnRegions`, `split_ids`), should keep files apart: as it stands two
  files shown in one layer merge their `tid`s.  Merging by `tid`, *Collapse
  by id* and *Combine files* already key on file and channel.  The ids that
  SMAPpy writes (`cluster_id`, `track_id`) are made unique across files
  (`region_runs.merge_ids`); the ids a file brings (`tid`) were not.
  *Done 2026-11-15*: a file added has its overlapping ids moved past the
  table's (`file_ids.keep_apart`), its own numbers kept in `tid_file`.
  Open: region tables skip id 0, and whether Imspector's `tid` can be 0.
* The pattern size L.  *Done 2026-11-15*: abberior's wiki defines it as
  `patGeoFactor × 360 × λ/642` (MFX took 640); `minflux.pattern_sizes`
  takes λ from the iteration's excitation, else the thread's MINFLUX laser,
  else 642 nm.  At 561 nm L, and with it `xy_err_nm`, is 13 % smaller.

**B. Trace metrics in *Id statistics* (S–M).**  All are vectorised and come
out as region-table columns:
* end-to-end displacement and velocity;
* a *sum* statistic, which gives the photon budget;
* std(diff)/√2 per axis;
* the principal axes: long and short std, eccentricity and angle;
* the trend across the track, √(std_short² − precision²);
* the median dt.

The skip is the existing `trace_index` filter.  The SMAP numbers that go
with these are the binned std against photons, with L/√(8N) drawn beside
it, as a figure.

**C. Track rotation (S).**  A per-id step that writes `x_rot_nm` and
`y_rot_nm`: the long axis on x, forward in time, about the trace's centre.
With two colours, the partner trace is rotated in the reference trace's
frame (after F).  B's axis values and E both use it.

**D. Trace starts and per-localization steps (M).**  A plugin under
MINFLUX:
* per localization: the displacement from the trace's previous one, its
  rolling std and its robust z;
* an adaptive `in_tail` label, by the rolling-std rule (moving molecules)
  or the centroid rule (still ones);
* *Tails_analyze*'s figures: d(n), tail length and duration, and d against
  cumulative photons beside 120/√N;
* optionally, a cut after the first `cfr` violation.

This replaces the fixed `trace_index` cut where a fixed cut is wrong.

**E. Steps (M–L).**  This was "skip, project code" in the porting list
(2026-10); it is now asked for.
* *Find steps*, `per_region` over traces (as ROIs, or a block over `tid`):
  rotation from C, AutoStepFinder re-implemented from the paper (no
  licence), SMAP's split/merge and refine, and optionally bnp-step if its
  licence allows.  The per-step tables (level, time, size, dwell, std) are
  `Result.arrays`; the per-track values go in the region table.
* A Summarize plugin that is *MINFLUXStepsAnalyze*: a fit of the step
  sizes, dwell-time models (exponential, Erlang n, hypoexponential), and
  the velocity per model.
* *MotorPAINT*'s steps as a file, plus `track_angle` and `track_length`.
* A start-to-end criterion for *Segment regions*' id-column ROIs.
* Manual step editing and the movie come later.
* Leave SMAP's bugs behind: StepsMINFLUX smooths for the levels and for the
  boundaries the opposite way, and MINFLUXStepsAnalyze filters dwells but
  not sizes.

**F. Two colours by `thi` (M).**
* A shared helper that pairs the traces of the two channels by time
  overlap, then pairs their localizations by nearest time.
* On top of it: the offset between colours (per site, over time, over the
  field, as a named transformation for *Transform*), co-trace ROIs
  (*Segment_cotracks*), and the partner trace in E's figures
  (*StepsMINFLUX_dc*).

**G. Every iteration, and sequence QC (M–L).**
* A reader option to keep every iteration row: a second, raw table beside
  the localizations, as the beads are.
* A *Sequence QC* plugin on that table: ParaFlux's iteration efficiency and
  background per iteration, the parameters per iteration (`eco` and `efo`
  against the localization's number in its trace, the head-start passes
  marked), the abort reason per trace from `sta`, the time per iteration
  (scouting), and the true photon budget.
* Its numbers become values a batch compares across files.
* It waits on the meaning of the `sta` codes per Imspector version (MFX
  has a list).

**H. Sequences (S, then ?).**
* Keep every thread's sequence.
* Show a table of the sequence's parameters per iteration for the open
  files side by side (*File properties*, or a small File plugin), and
  export it.
* The maker, creator and condenser write sequences for the instrument,
  from Abberior's reference sequences.  *To discuss*: combined with
  SimuFLUX, a sequence could be simulated as it is made, and both could
  later come into SMAPpy together.

**I. Conditions and plot grids (M).**
* File attributes, `key = value` typed or taken from the file name by a
  pattern, turned into a column, so that summaries group by them.
* A faceted plot of region-table values: rows, columns and hue each any
  attribute; histogram or violin, annotated with N and the median.
* A gallery of the longest traces (rotated xy, x against t with the steps).

## Open questions

* **Order.**  The proposal goes A → B → C → D → E, then F, G, H, I.
  Motor stepping (E) and the colleagues' tracking needs (B, D, I) come
  first; QC from every iteration (G) is the largest change to the reader.
* **The sequence maker and creator**, with SimuFLUX: to discuss.
* **The pattern size's wavelength scaling**: to discuss (A).
* **Step finders**: both, AutoStepFinder (SMAP's, re-implemented from the
  paper) and bnp-step (licence to check).
* **Test data** (asked for, 2026-11-15): a two-colour `.msr` (`thi` and
  DCR), a kinesin stepping measurement, and an m2410 file with every
  iteration.
