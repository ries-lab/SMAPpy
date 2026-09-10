# Bead spline calibration

Build one single-channel PSF from all selected bead stacks. Lateral coordinates
are pixels; axial coordinates are objective nanometres. Camera pixel size is
not required. No EM mirroring, spatial subdivision, or field ROI is applied.

Generated calibrations explicitly store `em_mirror=False`; the fitter therefore
uses unmirrored image ROIs too. Imported MATLAB calibrations retain their mirror
flag for compatibility. Identical acquisition settings for beads and data are
assumed for the native workflow.

## Interactive use

Install the existing viewer extra and use a Python installation with Tk:

```sh
pip install -e '.[viewer]'
smappy-calibrate /path/to/bead_acquisitions
```

The window also opens with no arguments. **Add files** supports multiple files;
**Add directory** can be used repeatedly to accumulate directories from different
locations. Recursive discovery is on by default. The list contains acquisitions,
not TIFF chunks: selecting the same series or NDTiff directory twice has no effect.
Remove rows to exclude entire acquisitions. Everything remaining contributes to
one calibration.

**Detect + calibrate** runs in a background thread. Select bead rows to inspect
their source projection. **Toggle exclusion**, then **Rebuild from beads** to
remove beads from the average. Change detection/alignment/smoothing settings and
use **Detect + calibrate** to re-extract. Changes to settings or exclusions must
be applied before saving. Brightness range and shape rejection remain visible
because they directly decide which beads enter the average. Less frequently
changed detection, alignment, smoothing, and minimum-bead controls are collapsed
under **Show optional parameters**. Alignment padding is the temporary border around the
requested PSF ROI: it permits subpixel xy shifts without losing edge samples and
is discarded after alignment. A 27-pixel ROI with 5-pixel padding is therefore
extracted as 37 by 37 pixels and returned to 27 by 27 pixels after alignment.

The Bead diagnostics tab shows brightness versus applied xy-shift magnitude
and shape residual versus applied z shift. Rows selected in the bead table are
highlighted in both plots and in the source projection. **Browse average stack**
opens a small viewer for the unsmoothed average. Choose XY, XZ, or YZ, then walk
through the remaining axis with its slider, the mouse wheel, or arrow keys.
**Auto contrast to slice maximum** scales each displayed plane from zero to its
own maximum; turn it off to retain one fixed contrast scale while scrolling.
The upper-limit factor adjusts display contrast without modifying data: a factor
of 0.5 saturates brighter pixels to reveal dimmer structure. The same calibration
window now offers a Single channel / Dual color mode selector; switching retains
the acquisitions and common settings and requires recalculation. Dual-color
diagnostics add Transformation and Field diagnostics tabs; see
[dual-color calibration](dual_color_calibration.md).

**Calculate fit quality** runs the production fitter on bead planes within the model's
supported z interval, using five z starts. Every bead is a separate thin, uniquely
colored line; the independently fitted unsmoothed average bead stack is a thick
black line. Two additional panels compare lateral-x and axial-z midline profiles
through every aligned accepted bead and the average. Profile values are left
unchanged; each panel autoscales its axes to the displayed curves. After fitting,
the calculation button is disabled until recalibration; Overview and Fit quality
tabs handle navigation without recalculating. Aggregate plots always pool all
acquisitions, while selecting a bead changes the source image and highlights that
bead in the plots. Failed fits and fits at
the z boundary remain included.
These in-sample refits measure consistency, not independent accuracy.

**Save calibration** writes a native HDF5 file, defaulting to
`<dataset-folder>_calibration.h5` in the parent of the first stack directory.
The filename is guaranteed to have only one trailing `.h5` extension. The normal fitting API and
`smappy-fit --cal` accept this file through `load_spline_calibration`.
Existing MATLAB calibration loading is retained.

## Python and headless use

```python
from smappy.calibrate import CalibrationSettings, calibrate

result = calibrate(
    ['/data/beads1', '/data/beads2'],
    CalibrationSettings(roi_size=27, smooth_z_nm=20),
    progress=print,
)
result.save('beads_calibration.h5')

from smappy.io.calibration import load_spline_calibration
from smappy.psf import SplinePSF
model = SplinePSF(load_spline_calibration('beads_calibration.h5'))
```

```sh
smappy-calibrate /data/beads --out beads_calibration.h5 --roi-size 27 --smooth-z 20
```

`--dz` overrides spacing when metadata is incorrect or unavailable. It does not
reinterpret time frames as z planes. Without `--out`, numerical command-line
options prefill the corresponding fields in the window.

For in-memory data, pass `BeadStack(images, z_nm)` objects, with images shaped
`(z, y, x)` and increasing, uniform objective z coordinates. For independent
validation, call `collect_beads`, exclude a whole acquisition's bead IDs with
`build_calibration`, then pass those IDs to `fit_bead_diagnostics` from
`smappy.calibrate.validation`. Excluded beads are aligned against the completed
reference for diagnostics only and never contribute to the model.

## Numerical choices

- Input assembly groups NDTiff index axes or OME-TIFF Micro-Manager plane tags by
  channel, position, time, and other non-z coordinates. Z planes are sorted by
  their coordinates; descending objective scans are reversed. Missing interior
  planes, duplicate planes, irregular spacing, incompatible stack spacing, and
  pooled channels fail explicitly. Different stack lengths are center-cropped.
  Plain TIFFs need an explicit dz. Arbitrary time-encoded z scans are not inferred.
- Detection uses a Gaussian-filtered maximum projection with a median/MAD
  threshold. Local maximum plateaus yield one candidate. A spatial tree rejects
  both members of close pairs, and beads without sufficient edge padding are
  excluded. Detection sigma, threshold, separation, and padding are explicit.
- One scalar background per bead z-stack is the median of its lateral border
  pixels pooled across all z planes. It is subtracted uniformly from every plane
  and stored as `background_adu`; diagnostic refits restore original camera values.
  Axial variation in the background and PSF tails is preserved. Negative
  noise is retained through averaging; only the final smoothed PSF is clipped
  at zero. Beads are initially normalized by their maximum plane sum and then
  matched in amplitude to the reference, so brightness does not dominate shape
  rejection. This is a practical background model, not a camera-noise calibration.
- Beads containing a pixel at the integer camera dtype maximum are marked
  saturated and excluded. `saturation_adu` can specify the actual clipping value
  for cameras such as 12-bit detectors stored in 16-bit TIFFs, or for floating-point
  input; leaving it blank uses automatic integer full scale where available. The
  default brightness filter accepts a full factor-of-six interval centered on
  the median unsaturated bead in log space (median/sqrt(6) through
  median*sqrt(6)). Set
  `brightness_range` blank/`None` to disable only the relative-brightness filter;
  saturation rejection remains active. Rejected beads remain visible in the table
  and plots as `saturated`, `too dim`, or `too bright`.
- Registration follows SMAP's two-stage design. The least-deviant half of the
  unaligned, brightness-normalized beads seeds the reference (at least five when
  available). A full-stack coarse 3D correlation is followed by one or more
  central-window refinements; the default two passes and 500 nm central range
  match the structure and scale of SMAP's full-stack plus 50-frame refinement at
  10 nm spacing. Correlation uses the central 13 by 13 pixels. Unlike SMAP's
  circular FFT, overlap-normalized linear correlation cannot wrap signal from one
  edge to the other. Subpixel refinement is performed locally around its peak.
  The common shift origin is the median coarse shift.
- Maximum shifts are acceptance limits, not search bounds. After the broad search,
  beads beyond +/-250 nm in z or a radial 3-pixel xy displacement are excluded
  before any template is made. Amplitudes are matched iteratively from the top
  intensity quartile. Shape rejection compares every bead against an average
  that excludes that bead, preventing self-inclusion from making an outlier look
  more typical. A robust top-quartile amplitude match is followed by full-volume
  normalized RMS error; division by the independent correlation penalizes broad
  disagreement. The rejection threshold is a robust mean plus 1.5 sample standard
  deviations and is repeated after central refinement. Each accepted volume is
  resampled from its original only once for a given accumulated shift. Coarse/final
  shifts, correlations, residuals, and explicit rejection reasons are saved.
- The final model is cropped symmetrically to complete axial overlap, with a
  two-plane interpolation margin. This deliberately reduces the usable z range
  when bead heights differ substantially. Lateral padding is discarded.
  Specifically, `ceil(max(abs(z_shift))) + 2` planes are removed from each end,
  using accepted beads only. The initial
  run removed 46 planes per end from 201, leaving 109 knots: 1.08 micrometres.
  This is a port policy, not a change in dz or a mirroring effect. MATLAB's
  `getstackcal_g` normally removes only the first and last plane; its
  `registerPSF3D_g` uses zero extrapolation for shifted volumes, preserving a
  wider nominal range even where not all beads contribute measured samples.
  A future alternative is to average only valid samples at each z and report
  support counts, rather than require full overlap or include extrapolated zeros.
- Smoothing is Gaussian with separate sigma in **nm for z** and **pixels for xy**.
  The default z sigma is 20 nm; xy sigma is zero. These are not numerically
  equivalent to MATLAB's dimensionless regularization parameter.
- Tensor-product not-a-knot cubic interpolation creates coefficients using three
  batched 1D transforms. It avoids the MATLAB per-voxel 64-by-64 solve and stores
  the exact coefficient order required by the existing C++ fitter.
- One common normalization makes the maximum PSF plane sum one; plane-by-plane
  renormalization is not applied, preserving z-dependent collection within the
  chosen ROI. The unsmoothed average is stored too.

Native coefficient layout is `(64, nz, ny, nx)`, with index
`16*pz + 4*py + px`. The PSF dataset contains the knots, one sample larger than
the coefficient grid in each spatial dimension. `z0` is zero-based and refers to
the aligned stack center. SMAPpy's established convention is retained:
`emitter_z_nm = -(spline_index - z0) * dz`. Thus increasing objective z has the
opposite sign to fitted emitter z. Spline fitting ROIs should be smaller than
the calibration PSF to leave lateral interpolation margins.

The versioned HDF5 stores coefficients, PSF knots, settings, conventions, original
source paths/axes/z coordinates, bead positions and IDs, brightness, acceptance,
rejection reasons, shifts, residuals, raw average, and optional refit diagnostics.
Writes use a temporary sibling file and refuse overwrite unless explicitly
requested. No Python object serialization is used in the output format.

## MATLAB audit and intentional differences

Source: `SMAP/fit3Dcspline/calibrate3D_GUI_g.m`, `calibrate3D_g.m`, and
`private/{images2beads_globalfit,getstackcal_g,registerPSF3D_g,readbeadimages}.m`.
No MATLAB source files were changed.

| Finding | Port decision |
|---|---|
| Circular ROI uses `length(xypos<2)` and an incorrect circle equation | Confirmed bugs; ROI selection deferred |
| Tile y boundaries end at `imsize(1)` | Confirmed non-square-image bug; spatial calibration deferred |
| Simple reader falls back to pixel size 100 while metadata path supplies micrometres | Remove calibration's dependence on lateral physical pixel size |
| Global-minimum subtraction treats a noise extreme as background | Use one border median over the whole bead z-stack; inspect structured background in future work |
| Threshold relies on six brightest peaks and fixed constants | Use explicit robust-noise threshold; no forced bead when threshold fails |
| Pairwise bead-distance loop is quadratic | Use a spatial neighbor tree |
| GUI fixes xy smoothing to zero; z lambda hides factors of dz | Expose smoothing in stated units and save all settings |
| Registration/rejection writes plotting markers into rejected stack data | Keep diagnostic state separate from pixel arrays |
| Registration uses circular correlation | Retain its coarse/full plus central-refinement structure, but use overlap-normalized linear correlation to prevent wraparound |
| 64-by-64 solve is repeated for every spline cell | Use batched separable cubic transforms |
| GUI objects/global variables are passed into numerical routines | Pure computation with a progress callback; optional Tk/matplotlib frontend |

Scientific improvements to consider after this version: robust background surfaces,
noise-aware registration/averaging, saturation detection with camera-specific limits,
validation-guided smoothing, reporting regional PSF differences, and model-supported
z limits based on held-out bead errors. Global two-channel calibration needs explicit
bead pairing and channel transforms; it is not implemented by averaging channels.

## Validation on the supplied example

The two-stage registration implementation was introduced as
`smappy_bead_spline_v3`. It ports SMAP's alignment structure and changes shift
limits into post-search rejection criteria. Version 4 retains that registration
and changes only shape rejection to use a leave-one-out reference. The historical
real-data measurements below predate those revisions and must not be interpreted
as v3/v4 performance numbers.

The measurements below describe the initial per-plane-background implementation
(`smappy_bead_spline_v1`). The revised implementation (`smappy_bead_spline_v2`)
uses one background per bead stack; its results are recorded separately below.

The 10 acquisitions in `250520_AA_beadsCal_638z_TIRF_M2` each contain 201 planes
at 10 nm spacing. With defaults, the initial run detected 40 isolated beads and
accepted 38, producing 109 x 27 x 27 PSF knots (about +/-540 nm). Registration
and coefficient generation took about 38 seconds on this machine, excluding I/O.

Refits of 787 supported planes had a median absolute error of about 31 nm after
removing each bead's constant z offset; 42 fits reached a model z boundary. Holding
out acquisition `_9` (last in lexical order), including all six of its beads,
left 32 accepted training beads. Its 122 supported held-out planes had a median
absolute centered error of about 28 nm (90th percentile 67 nm), with 7 boundary
fits. The constant offset removal measures curve distortion, not absolute z
accuracy; bead height is unknown. These figures include boundary fits.

On the same 787 planes, the supplied historical MATLAB calibration gave about
24 nm median absolute centered error across 783 finite fits (four failed). Its
model has a wider z range. That comparison uses the historical calibration's
required x mirror only for evaluating that reference; the Python calibration
and its normal fitting path apply no mirror. The first port is therefore not
claimed to match MATLAB calibration quality yet. Smoothing, rejection, and
registration should be reviewed on additional acquisitions before replacing an
established calibration workflow.

Focused checks cover coefficient values off-grid, known fractional 3D shifts,
synthetic held-out bead fits, production C++ fitting, native HDF5 round trips,
acquisition deduplication, and z/channel/time grouping in both TIFF formats.
The provided real dataset is OME-TIFF; NDTiff is covered by constructed files,
not yet by a real bead acquisition.

With the supplied MATLAB file enabled, the focused run has 27 passing checks and
one existing test failure: `test_coefficients_match_bead_stack` demands that the
correct orientation beat the x-mirrored orientation by more than 0.1 correlation.
This file gives 0.999947 versus 0.904763 (a margin of 0.095185). The unchanged
HEAD reader produces exactly the same coefficients and PSF, hence the same
failure. The direct coefficient/PSF agreement is excellent; the data-dependent
mirror-separation assertion is not met by this sample. The test was not relaxed.
Without that optional real-file fixture, the focused run is 26 passed, 2 skipped.

### Single-background revision

The revised run still detects 40 beads and accepts 38. It produces 105 x 27 x 27
knots at 10 nm spacing (1.04 micrometres total, +/-520 nm): the changed background
slightly changes the alignment shifts, and the unchanged overlap rule now removes
48 planes per end. Runtime including I/O was about 39 seconds.

In-sample refits: 760 supported planes, no failed fits, 43 z-boundary fits, median
absolute centered error 27.75 nm. Holding out the same six beads in acquisition
`_9`: 32 accepted training beads, 117 supported test planes, no failed fits,
7 z-boundary fits, median absolute centered error 25.46 nm and 90th percentile
59.50 nm. Refits now use restored original camera values. Since the supported
planes and diagnostic background treatment changed too, these are current-run
metrics rather than a controlled estimate of the background change alone.

All ten calibration tests pass, including preservation of axial background
variation after one scalar subtraction. The saved native model and a constructed
localization engine both report mirroring disabled.

### SMAP-inspired registration revision

The `smappy_bead_spline_v3` defaults were run on the same ten acquisitions after
the SMAP registration audit. The current 25-pixel separation found 46 candidates.
The requested intensity checks retained 19: 8 candidates contained saturated
pixels, 8 were more than two-fold below the median brightness, and 11 were more
than two-fold above it. Full-range registration then rejected three beads with
axial shifts of -369, -399, and -471 nm. Sixteen beads entered the final average.
Their accepted axial shifts span -112 to +127 nm. The spline coefficient grid is
170 by 26 by 26 cells (the stored PSF has one additional knot per dimension).

On the normal in-sample diagnostic, v3 fitted 565 bead/average planes with no
failed fits and 15 boundary fits. The median absolute per-bead-centered z error
was 10.23 nm and the 90th percentile was 30.38 nm. This is substantially better
than the recorded v2 result above, but it is not an isolated registration
comparison: candidate separation, saturation filtering, brightness filtering,
maximum shift, rejection threshold, and supported planes also changed.

For a more controlled check, the pre-v3 bounded-search algorithm was reconstructed
and run on exactly the same extracted, brightness-qualified bead ROIs. All three
models below fitted the same 551 raw planes from the same 19 beads over -700 to
+700 nm. Five z starts were used, errors were centered independently per bead,
and the historical MATLAB calibration's required x mirror was applied.

| Calibration | Median absolute error | 90th percentile | Boundary fits |
|---|---:|---:|---:|
| Reconstructed pre-v3 Python | 10.75 nm | 40.17 nm | 38 |
| New Python v3 | 12.09 nm | 35.77 nm | 3 |
| Supplied SMAP MATLAB | 25.37 nm | 51.72 nm | 0 |

Thus v3 does not improve the median in this controlled in-sample comparison, but
it improves the error tail and nearly eliminates boundary solutions. More
importantly for template integrity, it identifies and excludes the three large-z
outliers before averaging. The reconstructed old algorithm accepted 18 beads and
allowed one recentered shift to reach -251.1 nm even though its search had been
bounded at 250 nm. The MATLAB file stores a 198 by 26 by 26 coefficient grid at
10 nm spacing and was generated with `mindistance=25`, `zcorrframes=50`,
`smoothz=1`, `smoothxy=0`, and mirroring enabled. It does not retain its individual
bead shifts, so a direct bead-by-bead shift comparison with SMAP is unavailable.

These are still in-sample comparisons: the supplied SMAP calibration and both
Python calibrations saw the evaluated acquisitions. The next useful algorithmic
step is acquisition-level cross-validation, especially for choosing the central
alignment range and rejection threshold. If axial profile variability remains,
a promising extension is a consensus registration that solves pairwise bead
offsets globally and weights axial gradients/high-information planes, rather than
correlating every bead only to one evolving intensity template. Support-weighted
z averaging would also preserve the wider range without mixing extrapolated zeros
into the template.

### Leave-one-out shape metric revision

`smappy_bead_spline_v4` replaces the self-including shape score with the
leave-one-out metric described above. With the new defaults (brightness range 6
and rejection multiplier 1.5), the old and new metrics both accepted 14 beads on
the full dataset. Their in-sample refits were effectively unchanged: 8.00 versus
8.00 nm median absolute centered error and 29.36 versus 29.37 nm at the 90th
percentile.

For an acquisition-level check, acquisition `_4` and all three of its qualified
beads were excluded from template construction. The old metric retained 10
training beads and achieved 6.34 nm median / 16.79 nm 90th-percentile error on
90 held-out planes. The leave-one-out metric retained 14 training beads and
achieved 5.91 nm median / 15.54 nm 90th-percentile error on 91 held-out planes.
Both had zero failed fits; boundary fits changed from five to six. This single
three-bead holdout supports the new metric without proving generalization across
acquisitions; leave-one-acquisition-out testing over more datasets remains the
appropriate next validation.
