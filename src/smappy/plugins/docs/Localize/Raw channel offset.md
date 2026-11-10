---
version: "1"
covers: [smappy.plugins.raw_offset.RawChannelOffset, smappy.plugins.raw_offset.find_raw, smappy.plugins.raw_offset.chip_map, smappy.plugins.raw_offset.chip_positions, smappy.plugins.raw_offset.channel_rects, smappy.plugins.raw_offset.roi_corner, smappy.plugins.raw_offset.save_corrected, smappy.raw_offset.PatchSum, smappy.raw_offset.fit_spot, smappy.raw_offset.exclude_seen, smappy.raw_offset.regions_of, smappy.raw_offset.sum_patches, smappy.raw_offset.measure, smappy.raw_offset.to_main, smappy.raw_offset.fit_correction, smappy.raw_offset.correction_nm]
---

## What it does

On a split camera with one dye in each half, the two halves are put on top of
each other by a map measured on fluorescent beads -- the
[registration](plugin:Panels/Glossary), made by
[Overlay channels](plugin:Analysis/After fit/Overlay channels) or
[Calibrate transform](plugin:Analysis/Register/Calibrate transform).  The
beads were imaged at another time than the sample, and in another colour than
the dyes.  So in the sample the map can be off by some nanometres, and
nothing in the localizations shows it: each molecule is found in its own
half only.

But a few percent of each molecule's light reaches the other half too --
[crosstalk](plugin:Panels/Glossary).  For one molecule it is far too dim to
detect, a dozen photons over the background.  Summed over a hundred thousand
molecules it is a bright, sharp spot.  Where that spot lies, compared with
where the map puts the molecule's partner, is how far the map is off --
measured in the sample, with the dye's own light.

This plugin makes that sum from the raw frames.  It reports the offset for
each direction (main half into the other, and back), the fraction of light
that leaks, and the offset in regions of the field.  From the regions it fits
a correction, a shift or an affine map, which
[Transform](plugin:Analysis/Process/Transform) applies to the overlaid table
on request, and which can be folded into a corrected copy of the map for the
next sample.  It changes nothing by itself.

It needs:

* a table fitted as *two channels* (Gaussian 2D or Spline 3D, *channels* >
  *read*: *two channels*), overlaid by *Overlay channels* or not;
* the map: the one *Overlay channels* moved the table with, or a `_2ct.h5`;
* the raw movie the table was fitted from.  The plugin uses the path the
  table records; if the movie is not there any more, it looks for a file of
  that name beside the table, one folder up and one folder down, and the form
  shows what it found.

A direction in which no light leaks -- one dye may not reach the other half
at all -- is reported as such.  The other direction is then enough.

## How it works

```figure-setup
import tempfile
from pathlib import Path
import tifffile
from scipy.special import erf
from smappy.calibrate.dual import ChannelTransform, split_geometry
from smappy.calibrate.transform import save_channel_transform
from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.raw_offset import RawChannelOffset, RawOffsetSettings, _field_plot
from smappy.raw_offset import PatchSum, fit_spot
folder = Path(tempfile.mkdtemp())
SHAPE, SPLIT, PIXEL, SIGMA = (96, 64), 48, 100.0, 1.2
forward = np.eye(3)
forward[:2, 2] = [0.8, SPLIT - 0.5]
t = ChannelTransform(np.linalg.inv(forward),
                     split_geometry("up-down", "upper", SHAPE, split_position=SPLIT), {})
save_channel_transform(folder / "beads_2ct.h5", t)
def spot(frame, x, y, photons):
    cx, cy = int(round(x)), int(round(y))
    xs = np.arange(max(cx - 6, 0), min(cx + 7, frame.shape[1]))
    ys = np.arange(max(cy - 6, 0), min(cy + 7, frame.shape[0]))
    s = np.sqrt(2) * SIGMA
    ex = 0.5 * (erf((xs + 0.5 - x) / s) - erf((xs - 0.5 - x) / s))
    ey = 0.5 * (erf((ys + 0.5 - y) / s) - erf((ys - 0.5 - y) / s))
    frame[np.ix_(ys, xs)] += photons * np.outer(ey, ex)
# the map is off by 12 nm in x and -8 nm in y; the upper dye leaks 8 %, the lower none
rng = np.random.default_rng(1)
offset = np.array([0.12, -0.08])
frames, rows = [], []
for f in range(400):
    img = np.full(SHAPE, 10.0)
    for _ in range(6):
        p = np.array([rng.uniform(8, 56), rng.uniform(8, 40)])
        spot(img, p[0], p[1], 2760)
        q = t.to_secondary((p - offset)[None, :])[0]
        spot(img, q[0], q[1], 240)
        rows.append((f, p[0], p[1], 0))
    for _ in range(3):
        q = np.array([rng.uniform(8, 56), rng.uniform(56, 88)])
        spot(img, q[0], q[1], 3000)
        rows.append((f, q[0], q[1], 1))
    frames.append(rng.poisson(img) + 100)
tifffile.imwrite(folder / "sample.tif", np.stack(frames).astype(np.uint16))
a = np.array(rows)
locs = Localizations({"frame": a[:, 0].astype(np.int64), "x_nm": a[:, 1] * PIXEL,
                      "y_nm": a[:, 2] * PIXEL, "photons": np.full(len(a), 3000.0),
                      "sigma_nm": np.full(len(a), SIGMA * PIXEL),
                      "camera_channel": a[:, 3].astype(np.int8)},
                     {"pixelsize_nm": PIXEL, "source": str(folder / "sample.tif"),
                      "channels": {"regions": [[0, 0, 64, 48], [0, 48, 64, 48]]}})
result = RawChannelOffset().run(Context(locs=locs), RawOffsetSettings(
    registration=str(folder / "beads_2ct.h5"), frames=400, lag=5, grid=2))
```

**1. The molecules.**  Every well-fitted localization of one half -- photons
above *min photons*, a width within 0.6 to 1.6 times the half's median --
whose partner, where the map puts it, lies inside the other half.  A
molecule is left out when a localization of the other half lies in its
patch, in its frame or in the control frame: a molecule of the other dye is
on there, and its light would swamp the crosstalk.

**2. The patches.**  Around where the map puts the partner, the raw frame of
the other half is cut: a square of *patch* pixels on each side of the nearest
whole pixel.  The same square of the frame *control* frames later is cut too.
What does not blink -- autofluorescence, haze seen in both halves -- is in
both and cancels; a molecule is on for a few frames, not fifty.

**3. The sum, finer than a pixel.**  Each molecule's partner lies somewhere
within its pixel.  Every pixel of its patch is added, unchanged, to a grid
four times finer than a camera pixel, at its exact distance from the
predicted position.  The molecules lie at all sub-pixel positions, so
together they fill the fine grid: an image of the crosstalk sampled at a
quarter pixel, without resampling any frame.

**4. The spot.**  A Gaussian fitted to the sum (less the control) gives the
spot's centre.  Its distance from the centre of the grid is how far the map
is off.  The width says whether it is a spot at all; the spot's brightness,
divided by the same sum made at the molecules' own positions, is the fraction
of light that leaks.

```figure The summed crosstalk of the simulated sample, each direction a panel: the cross is the fitted centre, the circles one and two widths, the white plus where the map puts the partner.  The upper dye leaks 8 % of its light, and the spot is clearly off the plus; the lower dye leaks none, and the panel is noise.
fig.set_size_inches(8.4, 4)
result.plot.draw_into(fig)
```

**5. Regions and the correction.**  The same is done in each of
*regions* x *regions* parts of the field.  The two directions are combined in
each region, weighted by their errors.  A *shift* or an *affine* map
(shift, rotation, scale and shear) is fitted to the regions' offsets, and kept
under *name* as a transformation that Transform applies.  With *save
corrected map* on, the map with the correction folded in is also written
beside the map, as `<map>_raw_2ct.h5`, for *Overlay channels* to use.

```figure The offset in each region of the simulated field (blue arrows, greatly enlarged; the numbers in nm with their errors) and the shift fitted to them (grey).  The simulated map is off by the same 12 and -8 nm everywhere, and every region finds that, within its error.
fig.set_size_inches(5.5, 5)
result.plots["field"].draw_into(fig)
```

## In detail

**The sum.**  For molecule $i$ the map predicts the partner at $\mathbf{q}_i$;
the patch is cut around the nearest whole pixel $\mathbf{c}_i$, and pixel
$\mathbf{k}$ of it lies at $\mathbf{u} = \mathbf{c}_i + \mathbf{k} -
\mathbf{q}_i$ from the prediction.  With $I_f$ the frame and $L$ the
*control* lag, bin $b$ of the fine grid (a quarter pixel) holds

$$S_b = \sum_{(i,\mathbf{k}) \to b} \left( I_{f_i}(\mathbf{c}_i + \mathbf{k}) - I_{f_i + L}(\mathbf{c}_i + \mathbf{k}) \right), \quad n_b, \quad \bar{\mathbf{u}}_b = \frac{1}{n_b} \sum_{(i,\mathbf{k}) \to b} \mathbf{u}$$

the sum, the number of pixels that landed in it and their mean offset.  The
fit puts the value $S_b / n_b$ at $\bar{\mathbf{u}}_b$, not at the bin's
centre.  When the molecules sit at every sub-pixel position the two are the
same.  When they all sit alike -- beads, or a sample on a regular grid --
every pixel is rounded to its bin's centre in the same direction, and the
whole spot moves by up to half a bin (16 nm on a 127 nm pixel).  The mean
offset carries no such error.

```figure A spot 10 nm off its prediction, summed from 40,000 simulated molecules that all lie 0.2 px right of a pixel's centre, fitted with each bin at its centre (orange) and at the mean offset of its pixels (blue); the line is the truth.  Rounded to the centres, every pixel moves the same way and the spot with them.
rng = np.random.default_rng(2)
n, true, r = 40000, np.array([0.1, -0.04]), 5
delta = np.tile([0.2, 0.1], (n, 1))
k = np.arange(-r, r + 1)
s = np.sqrt(2) * SIGMA
cx, cy = delta[:, 0] + true[0], delta[:, 1] + true[1]
ex = 0.5 * (erf((k[None, :] + 0.5 - cx[:, None]) / s) - erf((k[None, :] - 0.5 - cx[:, None]) / s))
ey = 0.5 * (erf((k[None, :] + 0.5 - cy[:, None]) / s) - erf((k[None, :] - 0.5 - cy[:, None]) / s))
signal = rng.poisson(300 * ey[:, :, None] * ex[:, None, :] + 20.0) - 20.0
summed = PatchSum(r)
summed.add(signal, np.zeros_like(signal), delta[:, 0], delta[:, 1])
mean_fit = fit_spot(summed).x
centres = (np.arange(summed.n.shape[0]) + 0.5) / summed.fine - r - 0.5
u, v = np.meshgrid(centres, centres)
summed.su, summed.sv = u * summed.n, v * summed.n
centre_fit = fit_spot(summed).x
fig.set_size_inches(5, 2.6)
ax = fig.subplots()
ax.barh([1, 0], [mean_fit * 100, centre_fit * 100], color=["C0", "C1"])
ax.axvline(true[0] * 100, color="k", lw=1)
ax.set_yticks([1, 0], ["mean offsets", "bin centres"])
ax.set_xlabel("fitted offset in x (nm, 100 nm pixels)")
```

**The spot.**  A circular Gaussian plus a constant,

$$S_b / n_b \approx \frac{A}{2 \pi s^2} \exp\left( -\frac{|\bar{\mathbf{u}}_b - \mathbf{r}|^2}{2 s^2} \right) + B$$

fitted by least squares with each bin weighted by $\sqrt{n_b}$.  The centre
$\mathbf{r}$ is the offset in the other half; $A$ is the leaked light per
molecule, in the camera's raw units.  The same fit at the molecules' own
positions gives their own $A_0$, and the fraction that leaks is $A / A_0$:
the camera's offset and gain cancel, so no camera calibration is needed.

**The errors.**  The frames are cut into 8 blocks, each summed and fitted on
its own, and the error of every number is the spread over the blocks divided
by $\sqrt{8}$.  On simulations it came out about a third smaller than the
scatter between repeated simulations: treat a difference of three errors as
borderline.  A direction counts as measured when its leaked fraction is more
than five times its error.

**Into the main half.**  An offset $\mathbf{r}$ measured in the secondary
half (main dye leaking into it) is turned into what the map is off by in the
main half, at the molecules' place $\mathbf{p}$:
$\mathbf{p} - T(T^{-1}(\mathbf{p}) + \mathbf{r})$, with $T$ the map from the
secondary to the main half.  One measured in the main half (secondary dye
leaking into it) is already that.  Both are "where the molecule is, less
where the map puts its partner": the amount the map should add.

**The correction.**  The regions' offsets $\mathbf{d}_j$ at their places
$\mathbf{p}_j$ are fitted with $\mathbf{d}(\mathbf{p}) = \mathbf{b}$ (*shift*)
or $\mathbf{d}(\mathbf{p}) = M(\mathbf{p} - \bar{\mathbf{p}}) + \mathbf{b}$
(*affine*), each axis weighted by the inverse variance of each region's
offset.  It is kept as the affine map $\mathbf{x} \to \mathbf{x} +
\mathbf{d}(\mathbf{x})$ in nm.  Transform applies it to the overlaid table's
channel 1, whose positions are the map's.  The corrected map is that
correction after the map, $C \circ T$, and is written only for a projective
map; a polynomial one cannot take it, and the plugin says so.

**What it measures.**  The crosstalk is one dye's light in the other half:
the offset is the map's error at that dye's colour.  The other dye's colour
in its own half is not measured, and on a microscope with lateral chromatic
error the two differ.  With both directions measurable, the combination
averages the two colours.

## Parameters

### raw
Leave it on *auto* while the movie is where the table says or near the
table.  The plugin reads it frame by frame; over a network share, 10,000
frames of 435 x 213 pixels took about a minute.

### registration
*auto* takes the map from Overlay channels' record in the table.  Choose a
`_2ct.h5` for a table that was not overlaid, or to test another map against
the same sample.

### start
*auto* takes the frames from the middle of the fitted range.  The first
frames are often too dense for single molecules, and the crosstalk sum wants
frames where molecules blink apart.

### frames
The precision grows with the square root of the molecules.  On a movie with
about 15 well-fitted molecules a frame, 10,000 frames measured the offset to
about 1.5 nm.

### grid
1 gives one offset for the field; 2 or 3 show whether it changes across the
field, and an *affine* correction needs at least 2.  Each region needs
thousands of molecules.

### model
*shift* when the regions agree within their errors; *affine* when they show
a gradient -- a small difference in magnification between the bead colour
and the dye's is a common cause.

### name
Running again with the same name replaces the kept transformation.

### save
The corrected map is written beside the map file (`<map>_raw_2ct.h5`), so
choose a map in a folder you can write to.  It needs the map's file: auto
finds it from the table when Overlay channels recorded it.

### min_photons
Dim molecules leak too little light to help, and add their patches' noise.

### lag
Far enough that a molecule blinking now is off again, near enough that the
background has not changed: 50 frames suits typical blinking.  Too short and
the molecule's own light is subtracted; too long and slow background
changes stay in.

### radius
At least three PSF widths.  Larger patches hold more of the other dye's
molecules, which leave a molecule out of the sum.

## Output

The text gives, for each direction, how many molecules were summed, the
fraction of their light that leaks, the offset of the map in nm with its
error, and the spot's width.  Then the correction fitted to the regions,
how far the regions scatter about it compared with their errors, and the
name it is kept under.  A scatter much larger than the errors means the
model is too simple: try *affine*, or more regions.

The figures: the summed crosstalk of each direction with its fitted spot, and
the offsets of the regions with the correction.  A sharp spot of about the
PSF's width is a measurement; a flat or noisy panel is a direction in which
nothing leaks.

The form's notice offers **Apply with Transform** once the correction is
kept, set to move channel 1 of the overlaid table.
