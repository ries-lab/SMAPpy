# Wave 4 (spatial statistics, counting, tracking): notes for the planning

Collected 2026-10-14, before the planning; not a plan.

## What Jonas said (2026-10-14)

> All of these plugins are quite powerful plugins with a lot of
> functionality, we have to carefully plan how to wrap them, what to
> combine...  Tracking might even be a separate package, as it would be nice
> to have a track renderer in a layer that draws the tracks... then the
> statistical analyses...

To be done as an interview and a plan before any code.

## smappy, as it is (origin/main c9f650a)

* **Layers** are of two kinds, `locs` and `image`, and both are rasters
  (`session.Layer`, `layer_state.LayerSnapshot`, `render_panels` sums
  RGB).  No per-layer vector drawing; the only vector items in the view
  are the scale bar, axis bars and the ROI.  `render.MODES` and the GUI's
  `RENDER_MODES` are closed tuples.  A **track layer** is a core change
  either way: (a) a raster *tracks* render mode drawing segments by
  `track_id` then frame (exports and overlays work; cull per track, not
  per point), or (b) a vector layer kind (`pg.PlotCurveItem`), redrawn on
  changes, missed by exports unless rasterised as well, and touching every
  `is_image` branch.
* **Linking** (`group.py`, `csrc/group.hpp`): a greedy walk, first match
  in a box (or a precision gate), `dt` dark frames, a running mean
  position, lateral only.  `connect()` could hand out track ids, but SPT
  needs global assignment (LAP), gap closing, a motion model.
* **Tracks as regions**: an integer `track_id` is an id column, so every
  `per_region` plugin runs per track (results in `Session.id_regions`).
  For 10^5 short tracks a vectorised whole-table analysis is faster than
  the per-region machinery.
* **A separate package**: plugins can live in one (the user adds its
  folder to the plugin roots; or a small core change: an entry-point
  group).  A new layer kind or render mode cannot come from outside
  without core changes.
* **Time**: only MINFLUX has `time_s`; camera tables have `frame` and
  `metadata["camera"]["exposure_ms"]`; no frame interval (cycle time) is
  stored.  A diffusion coefficient needs a frame-time setting.

## SMAP's tracking

* Two id conventions: `track_id` (+ `track_length`) from SimpleTracking,
  `tid` from MINFLUX.  No track renderer in SMAP: tracks are coloured
  points, or lines in a plugin's own figure.
* **SimpleTracking**: Crocker-Grier-Weeks linker (`shared/myfunctions/
  mytrack.m`), max displacement 1000 nm, memory 2 frames, min length 2,
  2D or 3D.  The core linker.
* **AnalyzeSPT**: MSD from lags 2-4, D = slope/4, dt hard-coded 30 ms; D
  histogram with lognormal fit, immobile fraction, interactive track view
  coloured by log D, diffusion maps on a grid.  Partly broken; rewrite.
* **DisplaySingleTrack**: one track in the ROI: x(t), running filters,
  AutoStepFinder (Loeff/Kerssemakers 2021) step detection, step-size
  histogram.
* **Single_tracking_analysis** (active, 2025): processive motors: per
  track covariance shape, directed-motion filter, robust linear velocity
  fit (Huber, `analyse_processive_track.m`), v, run length, run time.
* **Dual_tracking_analysis**: two-colour co-movement of processive tracks
  (pairs within 200 nm, angle < 30°, ...).
* **Tracks_to_movie**: tracks drawn onto the raw frames, exported.
* **uTrack**: a stub; skip.
* **MINFLUX**: Segment_singletracks / Segment_cotracks make ROI-manager
  sites of tracks; MotorPAINT rotates tracks and finds steps;
  **StepsMINFLUX(_dc)** (most maintained, 2026): per site, the track's
  steps (AutoStepFinder or split-merge), dwell times, step sizes, track
  statistics, optional MSD, interactive step editing.

## SMAP's spatial statistics

* **paircrosscorrelationfunction** (shared/myfunctions): FFT pair
  (cross-)correlation with an ROI mask, Veatch 2012 normalisation; the one
  to port.  (SpatialPointPattern's own cross term drops the phase: a bug.)
* **Ripley K/L** (`myRipleysK`): pairwise distances, minus-sampling edge
  correction; L-r plotted.  No H, **no CSR envelopes**, and no
  Sengupta-style blinking correction anywhere (only the first r bins are
  dropped) -- worth adding new.
* **correlation.m**: 1D along a line (profile, autocorrelation, FFT,
  period near a guess, averaged over ROIs) and 2D pair correlation per
  layer and pair of layers.
* **ClusterStatistics**: despite its citation, per-cluster shape only
  (covariance eigen stds, angle, extent; hull area/volume).
* ClusterVoronoi (SharpViSu density), density_calculator (neighbour count,
  circle/cylinder or Gaussian), DBSCAN, dyeblinking (on/off times vs the
  grouping dark time) -- the last a useful diagnostic.
* ROI evaluators: clusterAnalysis (two-cluster distance), shiftCross-
  Correlation (shift along a line between layers), correlationAnalysis
  (kinetochore-specific).

## SMAP's counting

* Clusters from a density image and local maxima (`cluster_counting_2`),
  or per ROI (`countingStatistics`: locs, gaps, dark-time-merged counts).
* Histogram of locs / blinks / dark-time-merged counts per cluster.
* Model (`brightnessdistribution.m`): N0 proteins, binomial maturation
  p_mature, a negative-binomial number of events per molecule (blink
  probability), optional overlapping clusters (zero-truncated Poisson);
  least squares on the cumulative histogram, N0 scanned as an integer.
  Close to Lee et al. 2012 / Puchner et al. 2013 (an inference; the
  plugin cites none).  `cfithist.m` has a bug (`p_ref` undefined).
