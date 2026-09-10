# Split-frame dual-color bead calibration

The first version generates two spline PSFs with one common axial reference,
a strictly two-dimensional projective transformation, and pair diagnostics.
Simultaneous dual-channel localization is a separate future feature.

## Use

Open **Tools → bead calibration…** in the localization viewer,
or run `smappy-calibrate`. Set **Calibration mode → Dual color**. Choose a layout, the main channel,
and optionally a split position. Supported layouts are right-left and up-down,
each with or without reflection of the secondary channel along the split axis.
The main channel selector names the actual half: left/right or upper/lower.

The split is the first zero-based pixel index in the second half. A 512-row
image defaults to 256: rows 0–255 and 256–511. Unequal half sizes are supported.
Coordinates in saved transformations are `(x,y)` pixel centers; image arrays use
`(z,y,x)`. The transformation maps native secondary coordinates to native main
coordinates and includes the reflection and split displacement.

```sh
smappy-calibrate /path/to/beads --layout 'up-down mirrored' --main-channel lower
smappy-calibrate /path/to/beads --layout 'up-down mirrored' --main-channel lower \
  --out dual_calibration.h5
```

Additional CLI options include `--split-position`, `--min-pairs` (default 8,
minimum 4, for transformation support), `--reprojection-threshold` (initial RANSAC
radius, default 2 pixels), and `--transform-axis-limit` (round-two absolute dx/dy
limit, default 0.15 pixels on each axis). Geometry and
numerical settings are available through the Python API as well.

```python
from smappy.calibrate import (
    DualColorSettings, collect_dual_beads, build_dual_calibration,
    load_dual_color_calibration,
)

settings = DualColorSettings(layout='up-down mirrored', main_channel='lower')
beads = collect_dual_beads('/path/to/beads', settings)
result = build_dual_calibration(beads)
# Stable pair IDs: exclusion rebuilds the projective transform AND both PSFs.
result = build_dual_calibration(beads, excluded=[3, 7])
result.save('dual_calibration.h5')
cal = load_dual_color_calibration('dual_calibration.h5')
main_psf, secondary_psf = cal.main, cal.secondary
main_xy = cal.transform(secondary_xy)
```

The same window handles both modes. Switching mode retains acquisitions and
common parameters, hides or reveals geometry controls, and clears the old result
so recalculation is required. The Overview tab shows the original split frame
and the main-channel FoV with PSF + transformation pairs (green), transformation-only
pairs (orange), rejected pairs (red), and unmatched beads. Selecting a bead
changes the displayed source image. Aggregate diagnostics always pool all
acquisitions and highlight selected bead IDs; there is no acquisition filter.
Boundary-excluded candidates are omitted, as in detection itself.

The Transformation tab shows density-colored transformation-pair `(dx,dy)` residuals,
outlined rejected pairs, zero lines, per-axis standard deviations, and the
first-round residuals and screening box. The Field diagnostics tab shows a
main/transformed-secondary overlay (including unmatched detections), and
shape mismatch across the FoV with PSF acceptance marked. Pooled convex-hull
areas quantify transformation and PSF support; they do not measure uncertainty.
Picking a pair selects its table row and highlights it in the plots.
The Bead diagnostics tab restores per-channel brightness versus shared XY shift,
shape residual versus z shift, and shape residual versus correlation. Colors
distinguish PSF, transformation-only, and rejected pairs, with gold selection
outlines and clickable points. Select rows and toggle exclusion, then use
**Rebuild from beads**. Settings or exclusions must be applied before saving.
The paired average browser displays synchronized XY/XZ/YZ panels, with shared
slice position, linked contrast by default, and optional independent contrast.
The upper-limit factor multiplies the current slice maximum (or stack maximum
when auto contrast is off): 0.5 reveals dim structure by saturating bright pixels.
Display defaults to the mirror-corrected joint-registration orientation; native
camera orientation is optional. These controls never change saved PSFs.

Fit quality shows both channels' fitted-z curves, centered z errors, and lateral
and axial profiles, including individual beads, unsmoothed averages, and final
spline knots. Profile values remain on their existing common calibration scale;
each panel independently autoscales its axes to the displayed curves. The weaker
channel fills its panel without changing data or relative intensities; read the
numerical intensity ticks when comparing channels. **Calculate fit quality** runs
the diagnostics once per calibration and is disabled when results are available;
tabs handle navigation. Rebuilding clears diagnostics and re-enables calculation.
Refits use the existing
single-channel production fitter independently;
they are diagnostics, not a simultaneous dual-channel fit.

Saving defaults to `<dataset-folder>_calibration.h5` in the parent of the first
stack directory. For example, input `260709_MM_BeadCal_2C/Pos0/stack.ome.tif`
defaults to `260709_MM_BeadCal_2C/260709_MM_BeadCal_2C_calibration.h5`. The dialog
receives an extension-free default name; repeated `.h5` suffixes are removed
before checking for an existing output and asking about replacement.

## Coordinate provenance

Micro-Manager `ROI` metadata supplies the camera-chip origin. All planes within
one stack must agree; ROI dimensions must match the recorded image. When every
acquisition has this information, the transform uses full camera-chip pixels.
Each source's ROI, axes, shape, objective z coordinates, and path are saved.

If ROI information is missing, calibration continues with a warning and uses
internal image coordinates. This fallback is recorded explicitly. Known differing
ROIs cannot be pooled with missing ROI information. Before future application,
`cal.validate_image(shape, roi)` enforces the coordinate contract: ROI-local
calibrations require identical dimensions and warn that equal dimensions cannot
prove an identical camera location. A differing known ROI is rejected. Full-chip
application requires the target ROI to locate the pixels; missing metadata warns
and raises an error rather than silently applying an unknown origin.

## Numerical behavior

Detection, scalar border background subtraction, channel-specific brightness
limits, and saturation detection reuse the current single-channel implementation.
Each bead's lateral coordinate is refined by an elliptical Gaussian fit to the
mean of eleven central objective planes. There is no independent channel z fit
or channel z alignment. Failed lateral fits cannot contribute to calibration.

The declared split/reflection initializes matching. A two-pixel-bin translation
vote pools candidate differences **within** each acquisition. Mutual nearest
neighbors within 12 pixels form candidate pairs; pairs never cross acquisitions.
A two-round projective fit pools all brightness-qualified, reliably localized,
non-manually-excluded pairs, independently of PSF shape:

1. Normalized DLT with reproducibly seeded RANSAC and coarse radial inlier
   refinement initializes the transformation. Nonlinear soft-L1 refinement
   then minimizes x/y reprojection error in main-channel pixels.
2. Keep coarse inliers with **both** `abs(dx)` and `abs(dy)` at or below the
   configured axis limit, evaluated against round one. Refit geometrically with
   soft-L1 loss on that fixed subset. There is no repeated PSF-driven pruning.

This follows SMAP's componentwise residual screening, using two rounds rather
than its current sequence of progressively tighter cutoffs. In SMAP's calibration
workflow, `sepscale=5` and the final factor `0.03` give 0.15 pixels. The Python
soft-L1 scale is half the configured axis limit. Saved x/y influence weights
describe geometric robustness, **not** localization precision or photon CRLBs.
The round-two training mask is based on round-one errors; final residuals can
move slightly across the cutoff after refitting. No automatic cutoff relaxation
occurs if too few pairs remain.

Four transformation pairs are the hard minimum; the default requires eight.
The PSF average has its own `min_beads` minimum (default one, with a low-count
warning). Rank/conditioning and forward/inverse denominator checks
reject degenerate coverage or a projective singularity in the image. Point
coverage and residuals remain visible for inspection; low residuals alone do
not demonstrate reliable extrapolation outside bead coverage.

The projective transform predicts the secondary ROI center from the main ROI
center. Fractional extraction offsets and reflection are accounted for explicitly;
the PSF images are not warped projectively. A joint channel correlation estimates
exactly one `(z,y,x)` shift per physical bead. Correlation sums corresponding
channel numerators/energies, without correlating across the channel boundary.
Relative channel intensities are preserved; this is SMAP-style intensity-weighted
correlation, not an optimal Poisson-noise registration objective.

Each pair is divided by one shared brightness scalar (the sum of its channels'
maximum plane integrals), followed by shared robust amplitude matching to the
reference. A bright channel therefore carries more registration weight, but bright
beads do not dominate simply because of their absolute brightness. One failed
channel brightness/saturation check rejects the whole pair. Joint shape and shift
rejection also acts on whole pairs. No separate channel recentering or axial shift
is introduced. Intrinsic relative channel PSF displacements remain in the models.

The current single-channel registration improvements are retained: linear overlap
correlation, coarse full-stack search followed by central refinement, shift limits
as acceptance limits, a leave-one-out shape reference, and common valid axial
support. Final paired volumes are resampled from original extracted data once
with the accumulated extraction and registration shifts. A pair rejected by
shape or shift screening is **not** removed from the transformation. A transformed
ROI correction exceeding extraction padding excludes the pair from PSF averaging
only. Manual exclusion removes the pair from both uses. `result.accepted` is the
PSF selection; `result.transform_accepted` is the geometric training selection.
The measured channel-brightness ratio still uses the PSF-accepted subset.

Shape rejection remains the existing empirical mean-plus-sigma heuristic, not a
photon-noise test. Noise-aware shape scores, soft PSF weights, and spatially varying
PSF models are not implemented in this step. The new FoV diagnostics help identify
smooth field-dependent mismatch before changing those algorithms.

After Gaussian smoothing, a separate constant offset makes each channel's cubic
PSF strictly positive. A Bernstein coefficient lower bound over every cubic cell
accounts for interpolation undershoot between positive knots. The relative floor
is 1e-6 of the channel peak (absolute minimum 1e-12 before common normalization).
Offsets and normalization are saved. Both channels then use **one main-channel
normalization**, preserving their relative amplitudes apart from the explicitly
recorded positivity offsets. Unmodified unsmoothed averages are retained for
review. The measured median secondary/main bead brightness ratio is saved too.
Intensities are in camera ADU; photon counts are not inferred without gain data.
Each model records its peak-plane integral in the common normalization. A
separately fitted secondary-channel amplitude therefore follows this common
scale and should not be interpreted directly as that channel's photon total.

## Storage and validation

One versioned native HDF5 contains native-orientation `.main` and `.secondary`
spline models, the homography, acquisition/geometry metadata, shared shifts,
fractional extraction corrections, paired coordinates, acceptance reasons,
unmatched detections, independent PSF/transformation masks, first-round inliers,
first/final signed dx/dy, per-axis robust influence weights, raw joint averages,
and optional per-channel refits. Single-channel calibration files continue to use
their existing loader. Loading a dual file with the single-channel loader gives
an explicit instruction to use `load_dual_color_calibration()`.

Focused tests cover all four layouts, selectable reference halves, unequal regions,
shared z shifts, retention of relative channel axial offsets and intensity ratios,
continuous spline positivity, projective outliers/degeneracy, coordinate fallback,
exclusion/rebuild, and save/load compatibility. To reproduce the real-data check:

```sh
python scripts/check_dual_calibration.py /path/to/beads --out /path/to/new-results
```

The script fits both channels, excludes the last acquisition, rebuilds, refits its
held-out beads, and writes HDF5 files, JSON metrics, and a review plot. It also
refits both transformation rounds while holding out each acquisition and each
spatial quadrant. These geometric checks retain detected coordinates and
correspondences but do not use held-out points in either transformation round.
Unavailable folds are reported, not silently omitted from the summary. Centered z
errors remove each bead's unknown constant height and measure curve distortion;
they do not establish absolute axial accuracy. Reported ADU-based refits do not
claim calibrated Poisson noise or physical CRLBs. The first implementation reads
the selected split-frame acquisitions into memory during collection.

### Two-round validation: 260709_MM_BeadCal_2C

With up-down mirrored, lower main channel, and midpoint split 256, the new method
finds 44 pairs and retains all 41 brightness-qualified pairs for transformation;
34 remain in the joint PSF average. The geometric hull retains the entire eligible
area (26,695 px²), while the PSF subset covers 22,060 px². Training transformation
residuals have median 0.0258 px and maximum 0.0799 px. Both PSFs retain 81 × 27 × 27
knots and the secondary/main brightness ratio is 0.2076.

All nine acquisition holdouts and all four spatial-quadrant holdouts were
available, evaluating all 41 eligible pairs in each scheme:

| Geometric prediction check | Median | 90th percentile | Maximum |
|---|---:|---:|---:|
| Leave one acquisition out | 0.0351 px | 0.0689 px | 0.0898 px |
| Leave one spatial quadrant out | 0.0390 px | 0.1084 px | 0.1814 px |

Holding out Pos8 leaves 33 transformation pairs and 24 PSF pairs. The held-out
single-channel z diagnostics have median absolute centered error 4.84 nm (main)
and 6.82 nm (secondary), with 90th percentiles 15.75 and 17.45 nm. Neither channel
has failed or boundary fits in these 120-plane diagnostics. These z results are
similar to, not uniformly better than, the historical baseline; geometric
coverage is the intended improvement. Different PSF subsets can change the valid
refit-plane selection, so the z summaries are not strictly identical samples.

### Historical baseline: 260709_MM_BeadCal_2C (PSF-coupled transformation)

The following numbers describe the earlier, PSF-coupled algorithm, retained for
comparison; they are not results of the current two-round implementation.

Using up-down mirrored, lower main channel, and midpoint split 256, all nine
101-plane acquisitions (20 nm spacing) yielded 44 automatic pairs. Brightness
checks retained 41; iterative joint shape rejection retained 20 in the final
calibration. The final PSFs have 81 × 27 × 27 knots with shared ±800 nm support.
Accepted projective residuals have median 0.0227 pixels and maximum 0.0384 pixels.
The measured secondary/main brightness ratio is 0.2075.

Holding out all eight pairs from Pos8 left 17 training pairs. The held-out
projective residuals ranged from 0.0192 to 0.0701 pixels. Each channel supplied
122 held-out bead planes, with no failed fits or z-boundary solutions:

| Diagnostic | Main/lower | Secondary/upper |
|---|---:|---:|
| In-sample median absolute centered z error | 3.24 nm | 4.86 nm |
| Held-out median absolute centered z error | 4.77 nm | 6.51 nm |
| Held-out 90th percentile absolute centered z error | 15.30 nm | 18.68 nm |

The final exported float32 coefficient arrays have strictly positive continuous
Bernstein lower bounds of 5.56e-8 and 1.13e-8 in their common normalized scale.
These measurements are a single-acquisition holdout, not a general accuracy claim.
Automatic shape rejection is conservative: inspect the rejected pairs and the
field coverage before relying on extrapolation across the entire camera.

The TIFF acquisition metadata reports ROI `[192, 0, 231, 512]`; the supplied
historical SMAP calibration instead reports x origin 91 and split position 255.
The new calibration uses the TIFF metadata and requested midpoint convention.
Consequently the transform matrices should not be compared numerically without
first reconciling those coordinate conventions. Native spline xz sections were
also inspected qualitatively against the saved SMAP models; identical PSFs or
identical supported z ranges are not required.
The SMAP file's historical EM x-mirror flag is accounted for when comparing its
coefficients with native-camera Python PSFs.
