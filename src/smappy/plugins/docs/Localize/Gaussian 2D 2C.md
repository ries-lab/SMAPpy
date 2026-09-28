---
version: "3"
covers: [smappy.dualfit.combine_peaks, smappy.dualfit.build_link, smappy.dualfit.cut_paired_rois, smappy.dualfit.paired_to_localizations, smappy.dualfit.DualChannelEngine, smappy.psf.GlobalGaussianPSF, smappy.detect.find_candidates, smappy.plugins.fit.DualGaussianFit, smappy.plugins.fit.calibration_blocks, smappy.plugins.fit.finish_localizations, smappy.calibrate.transform.register_channels]
---

## What it does

This is the fitter for **two colours on one camera in 2D**.  A dichroic
mirror in the detection path splits each molecule's light between two halves
of the camera chip, so every molecule appears twice in the same frame: once
in the *main* (reference) half and once in the *secondary* half, a little
shifted, turned and magnified.  Both dyes appear in both halves, but in
different proportions, and that proportion -- the **photon ratio** -- is what
tells the dyes apart.  This is *ratiometric* multicolour imaging
([Bossi et al. 2008](https://doi.org/10.1021/nl801471d)).

The plugin fits the two spots of a molecule **together, as one emitter**, the
global fit of [Li et al. 2022](https://doi.org/10.1038/s41467-022-30719-4): one
position for both halves, and a photon number for each.  The position is then
as precise as all the photons allow, and the photon split comes out of the
same fit.  At the end of the run it assigns the colours from that split
([Assign colours](plugin:Analysis/Dual-Color/AssignColors)) and, if asked,
corrects the drift ([RCC](plugin:Analysis/Drift/RCC) or
[COMET](plugin:Analysis/Drift/COMET)).  The detection, the camera settings and
the Gaussian PSF are those of [Gaussian 2D](plugin:Localize/Gaussian 2D),
which explains them.

What it needs:

* **A split-frame movie**, both halves in every frame, and the camera's
  conversion and offset, exactly as for the single-channel fitter.
* **The transformation between the halves**: where the secondary half sees
  what the main half sees.  There are three ways to get it.  Tick *calibrate
  from this movie* and it is measured from the movie itself, which is the
  usual route, because in a ratiometric experiment every molecule is already
  imaged in both halves.  Or choose a file: one written by an earlier run of
  this plugin or by *Analysis/Register/Calibrate transform* (`*_2ct.h5`), or a
  dual-colour bead calibration (**Tools > Dual-colour calibration**), whose
  transformation is used and whose PSF models are not.  With neither, the run
  asks before it starts.

For 3D data with a bead calibration, use
[Spline 3D 2C](plugin:Localize/Spline 3D 2C); for one colour,
[Gaussian 2D](plugin:Localize/Gaussian 2D).

## How it works

```figure-setup
from dataclasses import replace
from scipy.special import erf
from scipy.spatial import cKDTree
from smappy.calibrate.dual import ChannelTransform
from smappy.dualfit import DualChannelEngine, combine_peaks
from smappy.metadata import CameraMetadata
from smappy.camera import to_photons
from smappy.pipeline import FitSettings
from smappy.plugins.fit import DetectionSettings, DualGaussianModelSettings
from smappy.psf import GaussianPSF
from smappy.simulate import dual_camera_frames, dual_transformation
from smappy.simulate.settings import LabellingSettings
# a split camera, 100 x 100 px per half, main half on top: two dyes that send
# 25 % and 75 % of their light into the lower half, behind a known map
frames, truth = dual_camera_frames(150, seed=3, labelling=LabellingSettings(efficiency=0.1))
geometry = {"layout": "up-down", "main_channel": "upper", "split_position": 100,
            "image_shape": [200, 100], "coordinate_system": "camera-chip"}
transform = ChannelTransform(dual_transformation(), geometry, {})
camera = CameraMetadata(conversion=0.5, offset=100.0, pixelsize_um=0.1,
                        em_on=False, roi=(0, 0, 100, 200))
finder = replace(DetectionSettings().finder(), split=(0, 100.0))
engine = DualChannelEngine(camera, finder, DualGaussianModelSettings().model(),
                           transform, FitSettings(roisize=13, output_unit="nm"))
engine.push(frames, 0)
locs = engine.flush()
# each localization's dye: the nearest true molecule in its frame
dye = np.zeros(len(locs), int)
for f in np.unique(locs["frame"]):
    m, t = locs["frame"] == f, truth["frame"] == f
    d, i = cKDTree(np.c_[truth["x_nm"][t], truth["y_nm"][t]]).query(np.c_[locs["x_nm"][m], locs["y_nm"][m]])
    dye[np.flatnonzero(m)] = np.where(d < 50, truth["dye"][t][i], 0)
```

**1. The two halves, and the map between them.**  Which half is which, and
where the seam is, comes with the transformation (its *geometry*: up-down or
right-left, mirrored or not, which half is the main one).  The
transformation itself maps a position on the secondary half to the position
on the main half where the same molecule appears.  It is a projective map --
a shift, a rotation, a magnification and a slight tilt of the image plane --
or, when it was measured with the *polynomial* model, a third-order
polynomial that also follows a field distortion.  Positions are in pixels of
the camera chip, so the same file serves a movie taken with another camera
ROI on the same chip.

```figure Left: one simulated frame.  Each molecule appears in both halves; the lines join the two spots the fit treats as one emitter, each line starting at the main-half spot.  Right: why a shift is not enough -- where each point of the secondary half lands, less where the map's shift at the centre of the field would put it.  What is left is the small turn and magnification between the halves, half a pixel at the edges of this 100-pixel field -- as much as the precision many times over.
shown = 60
here = combine_peaks(finder(to_photons(frames[shown:shown + 1], camera), first_frame=shown)[0],
                     transform, frames.shape[1:], 13)
ref, sec = here[0], here[1]
fig.set_size_inches(7.2, 4.4)
left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.5]})
image = to_photons(frames[shown], camera)
left.imshow(image, cmap="gray", vmax=np.percentile(image, 99.7))
for a, b, c, d in zip(ref.x, ref.y, sec.x, sec.y):
    left.plot([a, c], [b, d], color="#34c3ff", lw=0.8, alpha=0.8)
left.plot(ref.x, ref.y, "o", mfc="none", mec="#ffb000", ms=6, mew=1)
left.axhline(99.5, color="w", lw=0.8, ls="--")
left.text(2, 8, "main", color="w", fontsize=8)
left.text(2, 108, "secondary", color="w", fontsize=8)
left.set_xticks([]); left.set_yticks([])
gx, gy = np.meshgrid(np.linspace(5, 95, 10), np.linspace(105, 195, 10))
points = np.c_[gx.ravel(), gy.ravel()]
off = transform.to_reference(points) - points
off -= transform.to_reference(np.array([[50.0, 150.0]])) - [50.0, 150.0]
size = np.hypot(off[:, 0], off[:, 1])
arrows = right.quiver(points[:, 0], points[:, 1] - 100, off[:, 0], off[:, 1], size,
                      cmap="viridis", angles="xy", scale_units="xy", scale=0.05)
right.set_xlim(0, 100); right.set_ylim(100, 0); right.set_aspect("equal")
right.set_xlabel("x (px)"); right.set_ylabel("y within the half (px)")
right.set_title("the map, less its shift at the centre (arrows x20)", fontsize=9)
fig.colorbar(arrows, ax=right, fraction=0.046, label="px")
```

**2. Candidates in both halves.**  Detection is that of
[Gaussian 2D](plugin:Localize/Gaussian 2D) -- a difference-of-Gaussians
filter and local maxima above the *cutoff* -- with one change: the dynamic
cutoff is set for **each half on its own**.  The two halves are two detection
channels that happen to share a chip; a cutoff set over both would be set by
the brighter one, and the fainter partner of every pair in the dim half would
not be found.  (An absolute cutoff is the same number in both.)

**3. Pairing.**  The peaks of the secondary half are mapped onto the main
half with the transformation.  A main peak and a mapped secondary peak within
4 pixels of each other, each the other's nearest, are one molecule: they
become one candidate, at their mean position weighted by the square root of
each peak's height.  Peaks without a partner are **kept**: a molecule of the
dye that puts little light into one half may be found only in the other, and
it is still fitted.  The candidate is rounded to a pixel in the main half,
that pixel is mapped into the secondary half and rounded again, and a ROI
(*ROI size*) is cut at each.  A candidate whose two ROIs do not both fit
inside the frame, each on its own half, is dropped.

**4. One fit for both ROIs.**  Each ROI has its own Gaussian model, as in
the single-channel fit, and the two are fitted at once by maximum likelihood:
the Poisson likelihood of every pixel of *both* ROIs.  What makes it one
emitter is that some parameters are **linked** -- one number for both halves
-- and the rest are free.  By default:

* **x and y are linked** (*link x, y*).  One position for the molecule, seen
  in the secondary ROI through the transformation.  This is what pins the two
  spots to one emitter, and what lets all the photons of both spots inform
  the position.
* **The photons are free** (*link photons* off).  Each half gets its own
  photon number, and their split is the colour.
* **The background is free** (*link background* off): the two halves see
  different filters and need not have the same background.
* **The width is free** (*link width* off).  The halves see different
  wavelengths and are rarely in the same focus, so their spots have different
  widths; forcing one width on both would push the misfit into the photon
  numbers, which are what carries the colour.

Linking x and y gains precision because the two spots are two measurements of
one position.  With the photons split evenly between halves of equal PSF
width, the position from both is $\sqrt{2}$ times more precise than from one
half alone -- a little less here, because the secondary spot in the
simulation is wider.

```figure The gain from fitting both halves as one emitter.  Spot pairs simulated at known positions, the photons split evenly, 10 background photons per pixel in each half, a 1.3 px PSF in the main half and 1.45 px in the secondary: the scatter of x about the truth (dots) and the precision the fit reports (lines), fitting the main half alone (orange) and both halves linked (blue).
rng = np.random.default_rng(5)
k = np.arange(13)
def edge(c, s):
    return 0.5 * (erf((k[None] - c[:, None] + 0.5) / (np.sqrt(2) * s))
                  - erf((k[None] - c[:, None] - 0.5) / (np.sqrt(2) * s)))
levels = np.array([250, 500, 1000, 2000, 4000, 8000])
alone, linked = [], []
for total in levels:
    x0, y0 = 6 + rng.uniform(-0.5, 0.5, (2, 600))
    pairs = np.stack([rng.poisson(10 + 0.5 * total * edge(y0, s)[:, :, None] * edge(x0, s)[:, None, :])
                      for s in (1.3, 1.45)], axis=1).astype(np.float32)
    link = np.zeros((len(x0), 2, 2, 5), np.float32)
    link[:, 1] = 1.0                   # both ROIs cut on the spot: offsets 0, factors 1
    one = GaussianPSF(sigma=1.2)
    a = one.unpack(one.fit(pairs[:, 0]))
    both = DualGaussianModelSettings().model()
    b = both.unpack(both.fit(pairs, link))
    alone.append((np.nanstd(a["x_roi"] - x0), np.nanmedian(a["x_err_pix"])))
    linked.append((np.nanstd(b["x_roi"] - x0), np.nanmedian(b["x_err_pix"])))
alone, linked = np.array(alone) * 100, np.array(linked) * 100
fig.set_size_inches(5, 3.1)
ax = fig.subplots()
for values, colour, name in ((alone, "#ff7f0e", "main half alone"),
                             (linked, "#1f77b4", "both halves, x, y linked")):
    ax.loglog(levels, values[:, 1], color=colour, label=name)
    ax.loglog(levels, values[:, 0], "o", ms=4, color=colour)
from matplotlib.ticker import FixedLocator, NullLocator
ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20]))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_yticklabels(["2", "5", "10", "20"])
ax.xaxis.set_major_locator(FixedLocator(levels))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels([str(v) for v in levels])
ax.set_xlabel("photons of the molecule, both halves")
ax.set_ylabel("precision in x (nm, 100 nm pixels)")
ax.legend(frameon=False, fontsize=8)
```

**5. The photon ratio.**  The fit gives each molecule `photons_ch0` in the
main half and `photons_ch1` in the secondary half.  Their total is
`photons`, and the fraction in the secondary half is

$$\mathrm{ratio} = \frac{N_1}{N_0 + N_1} ,$$

between 0 (everything in the main half) and 1.  Each dye has its own ratio,
set by its emission spectrum and the dichroic, so a histogram of `ratio` over
a two-dye sample has two peaks.  How far apart they are, against how wide they
are, decides how well the colours can be told apart; the width shrinks with
the photons, as $1/\sqrt{N}$ for a pure photon-counting spread.

```figure The colour is in the photon split.  The simulated molecules of the two dyes (true ratio 0.25 and 0.75, dashed) fitted from the frames above.  Left: the histogram of `ratio` of each dye.  Right: the ratio of every localization against its photons -- the dim ones spread widest, and are the ones colour assignment is least sure of.
fig.set_size_inches(7.2, 3)
left, right = fig.subplots(1, 2)
colours = {1: "#d62728", 2: "#2ca02c"}
bins = np.linspace(0, 1, 61)
for d in (1, 2):
    m = dye == d
    left.hist(locs["ratio"][m], bins, color=colours[d], alpha=0.6, label=f"dye {d}")
    right.plot(locs["photons"][m], locs["ratio"][m], ".", ms=2, alpha=0.4, color=colours[d])
for r in (0.25, 0.75):
    left.axvline(r, color="k", ls="--", lw=0.8)
    right.axhline(r, color="k", ls="--", lw=0.8)
left.set_xlabel("ratio"); left.set_ylabel("localizations")
left.legend(frameon=False, fontsize=8)
right.set_xscale("log"); right.set_ylim(0, 1)
right.set_xlabel("photons"); right.set_ylabel("ratio")
```

**6. Finishing.**  When the last frame is fitted, two plugins run over the
whole table, in this order (*after the fit*):

* **Drift correction**, off by default: [RCC](plugin:Analysis/Drift/RCC) or
  [COMET](plugin:Analysis/Drift/COMET), with their own settings under *RCC
  drift* and *COMET drift*.  It is off because it takes minutes on a dataset
  whose size the form cannot know; COMET's question before a long run is not
  asked here, since choosing it is the agreement.
* **Colour assignment**, on by default:
  [Assign colours](plugin:Analysis/Dual-Color/AssignColors) reads
  `photons_ch0` and `photons_ch1`, finds the peaks of their ratio and writes
  a `channel` column -- 1 and 2 for the two colours, 0 for what it will not
  assign.  Its settings are under *colour assignment*.

They are the same plugins as in the Analysis tab, so a fit that finishes
itself and one finished by hand afterwards give the same numbers; running
Assign colours again later is how to change the rule.  A step that fails is
a line in the output and nothing more -- the table is saved regardless.  The
localizations are written to the file as they are fitted; when a finishing
step changed them, the file is written once more with the finished table,
and both steps go into its history.

**Calibrating from the movie.**  With *calibrate from this movie* ticked, a
first pass runs before the fit.  It fits frames of the movie with a plain
[Gaussian 2D](plugin:Localize/Gaussian 2D) fit over the whole frame, with the
same camera and detection settings as the real fit (each half thresholded on
its own about a provisional seam down the middle of the chip's longer side).
Every molecule then shows up as two localizations in the same frame, and
*Analysis/Register/Calibrate transform*'s method finds the map from those
pairs: the vector between every two localizations of a frame is collected, the
offset between the halves is the one that turns up over and over, the pairs
are matched through it and a projective map is fitted, then matched again,
more tightly, through that map and fitted again.  The frames are chosen
carefully:

* the first *skip the first* frames (500) are left out -- at the start of a
  movie every fluorophore is on at once, and a fit of those frames is a mat
  of spurious positions;
* 100 frames are fitted first to measure how many localizations a frame gives
  in the dimmer half, and from that as many frames are taken as reach
  *localizations wanted* (at least 100, at most *at most*);
* those frames are taken as *spread over* evenly spaced blocks across the
  whole movie, first at the start and last at the end, because a map fitted
  on the molecules of one minute describes the part of the field that was
  blinking in that minute.

The measured transformation is used for the fit and, with *save it*, written
beside the output, as `<acquisition>_locs_2ct.h5` by default, ready to be chosen as a file next
time.  The output box reports how many pairs it was registered on and the
residual in x and y.

## In detail

**The model.**  For pixel $i$ of the ROI of half $c$ ($c = 0$ the main half,
$c = 1$ the secondary),

$$\mu_{c,i} = b_c + N_c\, E_x(x_i - x_c; \sigma_c)\, E_y(y_i - y_c; \sigma_c) ,$$

with $E$ the pixel-integrated Gaussian of
[Gaussian 2D](plugin:Localize/Gaussian 2D).  Each half has the five
parameters $(x_c, y_c, N_c, b_c, \sigma_c)$, and the fitted vector
$\theta$ holds one entry for each linked parameter and one per half for each
free one -- by default $(x, y, N_0, N_1, b_0, b_1, \sigma_0, \sigma_1)$.  A
linked parameter reaches half $c$ through a factor and an offset,

$$\theta_c = f_c\, \theta + o_c ,$$

and a free one as $\theta_c + o_c$.  The main half has $f = 1$, $o = 0$.  For
the secondary half, $f$ for x and y is the local scale of the transformation
at the candidate (its derivative, by a one-pixel finite difference), and the
offset is the sub-pixel remainder left when the partner ROI was rounded to a
pixel, plus $(1 - f)\,h$ with $h$ half the ROI size, because the scale acts
about the ROI's centre and not its corner.  The link is diagonal: a rotation
between the halves has no term of its own, and at the fraction of a degree of
a splitter its effect within one ROI is far below the precision.  The images
are never resampled -- warping one half onto the other would correlate the
noise of neighbouring pixels and break the Poisson likelihood.

**The fit.**  The deviance of [Gaussian 2D](plugin:Localize/Gaussian 2D),
summed over the pixels of both ROIs, is minimised with the same
Levenberg-Marquardt steps, with the gradient taken through the link:

$$\frac{\partial \mu_{c,i}}{\partial \theta} = f_c\, \frac{\partial \mu_{c,i}}{\partial \theta_c} .$$

Each half starts as the single-channel fitter would from its own ROI (centre
of mass, border background, brightest pixel); a linked parameter starts from
the main half's value.  The limits ($N \geq 1$, $b \geq 0.01$, the width
inside the ROI) apply per half, and a linked parameter is limited once,
through the main half.  A pair one of whose ROIs has no light at all is
returned as not-a-number and dropped: the two halves are one measurement.

**The precision.**  The Cramér-Rao bound comes from the Fisher information
summed over both ROIs,

$$I_{jk} = \sum_{c} \sum_i \frac{1}{\mu_{c,i}}\, \frac{\partial\mu_{c,i}}{\partial\theta_j}\, \frac{\partial\mu_{c,i}}{\partial\theta_k} ,$$

so a linked x collects the information of both spots.  For two halves with
$N/2$ photons each and the same width, without background, that is
$\sigma_x = \sigma_a / \sqrt{N}$ against $\sigma_a / \sqrt{N/2}$ for one half:
the factor $\sqrt{2}$ of the figure.

**What is written.**  `x_nm`, `y_nm` are the linked position, in the main
half's coordinates.  With the photons free, `photons` is
$N_0 + N_1$ and `photons_err` the two errors added in quadrature; `ratio` is
$N_1 / (N_0 + N_1)$.  With *link photons* on, one number $\hat{N}$ is fitted
and the halves see $\hat{N}$ and $r\,\hat{N}$ for the *photon ratio* $r$;
`photons` is their total, $(1 + r)\,\hat{N}$, and `ratio` is 0: there is no
split left to measure.  (Before version 2 it was the main half's $\hat{N}$
alone.)  `logl_rel` is the log-likelihood per pixel of
the two ROIs together.  EM gain and the camera's read noise are handled as in the
single-channel fit ([Gaussian 2D](plugin:Localize/Gaussian 2D)), the read
noise in both ROIs.
The first frames are checked against the transformation's geometry: a
registration made on another camera ROI is refused, and a movie without a
camera ROI in its metadata is taken to start at the chip's corner, with a
warning.

**Compared with the paper.**  The global fit is that of
[Li et al. 2022](https://doi.org/10.1038/s41467-022-30719-4), whose software
offers a Gaussian PSF beside its spline.  Where the code departs from the publication:

* **The link.**  The paper links a shared parameter through a scale and a
  translation, the translation being the sub-pixel remainder left when the
  partner ROI is rounded to whole pixels.  Here the scale for x and y is the
  transformation's local derivative, and it acts about the ROI's centre (the
  $(1 - f)\,h$ above), so a mirrored splitter, $f = -1$, needs no flipped
  image.
* **No bead calibration is needed.**  The paper builds its transformation from
  beads and may re-measure it on the single molecules; here the movie alone
  can give it (*calibrate from this movie*), by voting on the vectors between
  localizations of the same frame.
* **Unpaired peaks are fitted**, which is the paper's argument for a global
  fit (a molecule too dim in one channel to be detected there): the candidate
  list is every peak of either half, not only the pairs.
* **Colour from the free photon split only.**  The paper also fits each
  molecule with the photon ratio fixed at each dye's value and keeps the most
  likely; that is not offered here.  With *link photons* on, one ratio is
  used, 1 or the bead calibration's, where the paper takes each dye's from
  the single-molecule data.

## Parameters

### model.link_xy
Unlinked, each half fits its own position.  The table's position is the
main half's, and `dx_nm_ch1`, `dy_nm_ch1` say how far the secondary half put
the molecule from where the transformation expects it, in the main half's
coordinates.  Their median over the field should be close to 0 and their
scatter about the precision; a trend across the field means the
transformation does not hold there.  Relink for the real fit: linked, both
halves pin one position and the precision is better.

### model.link_photons
On, one photon number is fitted and the secondary half is expected to hold
*photon ratio* times as many as the main: more precise, and no colour, and
`photons` is the total over both halves.  That suits two halves that see the
same dye, not two dyes.

### model.photon_ratio
Only used with *link photons* on; with the photons free it has no effect.
Left on auto it is 1 for a transformation measured from localizations, and
the beads' measured ratio when the transformation file is a dual-colour bead
calibration.

### model.link_sigma
Worth ticking only to test whether the two halves really differ in width.

### transform.path
Ignored when *calibrate from this movie* is ticked.

### transform.calibrate
The calibration pass fits at most *at most* frames with a plain Gaussian;
then the real fit runs over the whole movie.

### transform.calibrate_locs
10,000 in the dim half is usually plenty; a sparse movie runs out of frames
first, and the output says so.

### transform.calibrate_skip
Set it to 0 for a movie that does not start with everything on, such as a
simulation, or one started after a bleaching phase.

### transform.registration.layout
Auto tries plain, x-mirrored and y-mirrored and keeps the sharpest vote.  Set
it only when the automatic answer is wrong.

### transform.registration.model
*polynomial* is refused, with a message, when the pairs are too few for its
twenty coefficients or cover less than half of the field; the projective map
is then kept.

### transform.registration.coarse_tolerance_px
It has to cover what the voted offset alone cannot: a 3 degree rotation over
512 px is 27 px at the edge.  Looser costs nothing measurable.

### transform.registration.fine_tolerance_px
About seven times the residual of a healthy registration (0.13 px); tighter
strips the edges of the field.

### fit.roisize
Both ROIs of a pair have this size, and both must lie inside the frame, each
on its own half.

### finish.drift
Their pages, [RCC](plugin:Analysis/Drift/RCC) and
[COMET](plugin:Analysis/Drift/COMET), say which to choose; the settings under
*RCC drift* and *COMET drift* are theirs.

## Output

The table, one row per molecule and frame:

| column | meaning |
| --- | --- |
| `x_nm`, `y_nm` | the position, in the main half's coordinates |
| `x_err_nm`, `y_err_nm`, `xy_err_nm` | its precision (CRLB), from both halves |
| `photons`, `photons_err` | the photons of both halves together |
| `photons_ch0`, `photons_ch1` | the photons in the main and the secondary half, and `photons_err_ch0`, `photons_err_ch1` |
| `ratio` | the fraction of the photons in the secondary half |
| `background_ch0`, `background_ch1` | the background per pixel of each half (`background` is the main half's) |
| `sigma_nm_ch0`, `sigma_nm_ch1` | the fitted PSF width of each half (`sigma_nm` is the main half's) |
| `channel` | the colour Assign colours gave it: 1, 2, or 0 for none |
| `logl`, `logl_rel` | the log-likelihood, and per pixel of both ROIs |
| `peak_x_nm`, `peak_y_nm`, `iterations` | the candidate in the main half, and the steps taken |

The finishing steps' figures -- the ratio histogram of Assign colours, the
drift curve -- open with the fit's result.

What to check:

* **The preview.**  It draws the peaks found in both halves, and the
  secondary half's peaks mapped back onto the main half with the
  transformation (cyan crosses).  A cross that sits on its partner's circle is
  a good registration; a cross beside it shows the error directly.
* **The ratio histogram** should show one clear peak per dye.  Peaks that
  merge mean too few photons or dyes too alike; a pile-up at 0 or 1 is
  molecules whose partner ROI held only background -- a dye that sends almost
  nothing into one half, or a partner ROI placed where the molecule is not.
* **The two widths** (`sigma_nm_ch0`, `sigma_nm_ch1`) should each be steady
  across the field; a width that grows towards one side is a half out of
  focus there.

## Differences from SMAP

SMAP does ratiometric 2D in two ways, and this plugin replaces both with one
run.

* **SMAP's ratiometric workflow** (`fit_wavelet_dualcolorratiometric`) fits
  the whole frame single-channel, registers the halves on those localizations
  (`RegisterLocs2`), and then goes back to the movie to *measure* the intensity
  of the partner spot at the transformed position (`Get2CIntImagesWF`,
  `Intensity2Channel`).  Here both spots are fitted in one linked maximum
  likelihood fit, and the photon split comes out of it.
* **SMAP's global fit** (`fit_global_dualchannel`: `PeakFinder`,
  `PeakCombiner`, `RoiAdder`, `RoiCutterWF`, `MLE_global_spline`, whose
  *PSF free* mode is a global Gaussian) needs the transformation from an
  earlier registration.  Here it can be measured from the movie in the same
  run, and the registration differs from `RegisterLocs2`: it votes on the
  vectors between localization pairs rather than cross-correlating rendered
  images, so no split position, initial shift or magnification has to be
  known; it weights pairs by their precision; and it takes its frames spread
  over the movie, skipping the start.
* **Detection per half.**  The dynamic cutoff is set for each half on its
  own; SMAP's workflows run one peak finder over the whole frame.
* **Pairing.**  A matched pair is merged at a mean weighted by the square
  root of the peak heights (as `PeakCombiner` does for 4Pi); its two-channel
  branch weights by their square.  The 4 px matching distance is SMAP's.
* **Fewer options.**  x and y are always the main half's (SMAP can also take
  channel 2's or the mean); no channel weights and no sCMOS variance map.
  SMAP's photon ratio for linked photons is typed in; here it defaults to the
  bead calibration's, or 1.  The
  width is free per half by default.
* **Finishing.**  Drift correction and colour assignment run as part of the
  fit, with the Analysis tab's plugins, instead of as separate steps of the
  workflow.

## References

* Bossi M, Fölling J, Belov VN, et al. Multicolor far-field fluorescence
  nanoscopy through isolated detection of distinct molecular species. *Nano
  Lett* 8, 2463 (2008).
  [doi:10.1021/nl801471d](https://doi.org/10.1021/nl801471d) -- ratiometric
  multicolour localization with two detection channels.
* Li Y, Shi W, Liu S, et al. Global fitting for high-accuracy multi-channel
  single-molecule localization. *Nat Commun* 13, 3133 (2022).
  [doi:10.1038/s41467-022-30719-4](https://doi.org/10.1038/s41467-022-30719-4)
  -- the linked global fit this one follows, and its gain for ratiometric
  colour assignment.
* Smith CS, Joseph N, Rieger B, Lidke KA. Fast, single-molecule localization
  that achieves theoretically minimum uncertainty. *Nat Methods* 7, 373
  (2010). [doi:10.1038/nmeth.1449](https://doi.org/10.1038/nmeth.1449) --
  the maximum likelihood Gaussian fit and its CRLB.
