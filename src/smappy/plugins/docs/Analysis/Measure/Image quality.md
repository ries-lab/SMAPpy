---
version: "1"
covers: [smappy.squirrel.error_map, smappy.squirrel.ErrorMap.at, smappy.squirrel.ErrorMap.image, smappy.squirrel.compared_pixels, smappy.squirrel.fine_histogram, smappy.squirrel.scale_to, smappy.squirrel.fit_sigma, smappy.squirrel.reference_plane, smappy.squirrel.normalised_resolution, smappy.squirrel.qnr, smappy.frc.local_frc, smappy.frc.FRCMap.at, smappy.plugins.image_quality.measured_layers, smappy.plugins.image_quality.with_local_frc, smappy.plugins.image_quality.local_frc_image, smappy.plugins.image_quality.draw_error_map, smappy.plugins.image_quality.draw_frc_map, smappy.images.save_tiff]
---

## What it does

A super-resolved picture can look sharp and still be wrong.  A structure can
be missing because it was not labelled, not detected or filtered away.  Two
filaments close together can be drawn as one.  A molecule localized
hundreds of times can make a bright clump that is not in the sample.  None of
this shows in a resolution number.

This plugin draws two **maps** of where the picture can be trusted, one of
each per visible layer of localizations:

* **The error map** (SQUIRREL,
  [Culley et al. 2018](https://doi.org/10.1038/nmeth.4605)) compares the
  localizations with a **widefield image** of the same field: a picture
  taken at the diffraction limit, which shows every labelled molecule,
  blurred.  The localizations are blurred the same way and scaled to the
  widefield image's brightness.  Where the two still disagree, the map is
  bright.  It finds what is missing and what is added.  It cannot judge
  anything finer than the widefield image can show.
* **The local FRC map** measures the [FRC](plugin:Panels/Glossary)
  resolution (Fourier ring correlation: up to which detail two halves of the
  data agree) in each square tile of the field on its own.  One number for
  the whole field hides where the picture is good and where it is not: a
  region that drifted, one sparsely labelled, one out of focus.  This map is
  SQUIRREL's block-wise FRC, measured from the localizations.

The two answer different questions.  The error map asks *is it there?*, the
FRC map *how finely is it resolved?*.  A region can be well resolved and
wrong, or right and blurred.

With two or more layers -- two filter settings, two fitters, before and after
a correction -- *QnR* ranks them by both at once, as eSRRF does
([Laine et al. 2023](https://doi.org/10.1038/s41592-023-02057-w)).

**What it needs.**  For the error map, a widefield image of the same field in
the session: a TIFF opened with File > Open, or the average of the camera
frames a fit keeps with its localizations (chosen by *reference*).  It must
lie where the localizations lie, give or take *max shift*.  For the FRC map, a
`frame` attribute and at least *per tile at least* localizations in a tile.

It measures each visible layer as it is shown: its
[filter](plugin:Panels/Glossary) and the drawn
[ROI](plugin:Panels/Glossary).  It reads single localizations, never merged
blinks.  The FRC needs that, because it splits the acquisition in time.  The
error map counts what was localized, which is what the widefield image saw.

## How it works

```figure-setup
from scipy.ndimage import gaussian_filter
from smappy.simulate import simulate
from smappy.squirrel import error_map
from smappy.plugins.image_quality import draw_error_map
locs = simulate(n_frames=3000, seed=1)
x, y = np.asarray(locs["x_nm"], float), np.asarray(locs["y_nm"], float)
# the widefield image of every localization: blurred by 130 nm, on 100 nm
# camera pixels, with a background and shot noise
fine, pixel, n = 10, 100.0, 104
counts, _, _ = np.histogram2d(y, x, bins=n * fine, range=[[0, n * pixel]] * 2)
camera = gaussian_filter(counts, 130.0 / (pixel / fine)).reshape(n, fine, n, fine).sum(axis=(1, 3))
widefield = np.random.default_rng(0).poisson(4 * camera + 30).astype(float)
# the super-resolved picture is missing a patch, and has a clump the sample has not
gone = (x > 6500) & (x < 8500) & (y > 4000) & (y < 6000)
clump = np.random.default_rng(1).normal([2500, 7500], 15, (400, 2))
sx = np.concatenate([x[~gone], clump[:, 0]])
sy = np.concatenate([y[~gone], clump[:, 1]])
found = error_map(widefield, sx, sy, pixelsize=pixel)
```

**1. The localizations on the widefield image's pixels.**  The localizations
are counted on a grid *finer by* times finer than the reference's pixels.  The
reference itself is left as it is: nothing is interpolated.

**2. Blurred and scaled.**  The counts are blurred by a Gaussian of width
$\sigma$, the *blur*.  Each block of fine pixels is then summed into the
reference pixel it lies in, which is what a camera does with light.  The
result is scaled, $\alpha \cdot \mathrm{blurred} + \beta$: $\alpha$ turns
counts into the camera's brightness, and $\beta$ is the background.
$\alpha$, $\beta$ and $\sigma$ are those that make the two pictures agree
best.

**3. The shift.**  The widefield image and the localizations rarely lie
exactly on each other.  With *find shift* the offset between them is found by
cross-correlation, the localizations are moved by it, and the search repeats
until no shift is left.

**4. The map.**  The error map is the difference, pixel by pixel.  It is
bright where the widefield image shows something the localizations do not,
and where the localizations show something the widefield image does not.
*signed* keeps the sign: negative is missing, positive is added.

```figure Left: the widefield image.  Middle: the localizations blurred and scaled to it.  Right: the error map.  A patch of the structure was taken out of the localizations (bright, right), and a clump of 400 localizations of one point was added (bright, lower left); the rest is noise.
fig.set_size_inches(8.4, 2.9)
draw_error_map(fig.subplots(1, 3), found)
```

**5. Two numbers.**  The *RSE* (resolution-scaled error) is the root mean
square of the map, in the widefield image's units.  The *RSP*
(resolution-scaled Pearson) is the correlation of the two pictures, from -1
to 1.  Read the RSP to compare datasets: it does not depend on how bright the
image is.  Read the map to find where they disagree.

**6. The local FRC map.**  With *local FRC map* the field is cut into squares
of *tile size*, and the FRC of
[Localization Precision](plugin:Analysis/Measure/Localization Precision)
is measured in each square on its own: the same split in time, the same
rendering, the same 1/7 threshold.  A tile needs *per tile at least*
localizations for a value.  A tile with fewer, or whose curve never falls
through 1/7, has none and is drawn grey.  The spread over the *FRC splits* is
each tile's error bar.

```figure A field whose left half was localized with a precision of 6 nm and its right half with 24 nm.  Left: the map, tiles of 2 µm; the two halves are told apart at once.  Right: each tile's resolution against the FRC of its own half measured as one field (dashed).
from smappy.frc import local_frc, frc_resolution
from smappy.plugins.image_quality import draw_frc_map
rng = np.random.default_rng(0)
ex, ey = rng.uniform(0, 8000, 6000), rng.uniform(0, 4000, 6000)
k = np.repeat(np.arange(6000), 12)
s = np.where(ex[k] < 4000, 6.0, 24.0)
mx = np.clip(ex[k] + rng.normal(0, 1, len(k)) * s, 0, 8000)
my = np.clip(ey[k] + rng.normal(0, 1, len(k)) * s, 0, 4000)
mf = rng.integers(0, 4000, len(k))
local = local_frc(mx, my, mf, pixelsize=3.0, tile_nm=2000.0, repeats=3)
fig.set_size_inches(7.4, 2.6)
a, b = fig.subplots(1, 2, gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.6})
draw_frc_map(a, local, "two halves")
centres = (np.arange(local.resolution.shape[1]) + 0.5) * local.tile_nm + local.x0
for i, (half, colour) in enumerate(((mx < 4000, "C0"), (mx >= 4000, "C3"))):
    whole = frc_resolution(mx[half], my[half], mf[half], pixelsize=3.0, repeats=1)
    tiles = local.resolution[:, (centres < 4000) if i == 0 else (centres > 4000)]
    b.plot(np.full(tiles.size, i) + rng.uniform(-0.1, 0.1, tiles.size),
           tiles.ravel(), "o", color=colour)
    b.hlines(whole.resolution, i - 0.3, i + 0.3, colors=colour, linestyles="--")
b.set_xticks([0, 1], ["6 nm", "24 nm"])
b.set_xlim(-0.6, 1.6)
b.set_ylabel("tile (nm)")
b.set_title("tiles against each half", fontsize=7.5, color="0.25")
```

**7. QnR.**  With *QnR of the layers*, each layer's FRC resolution is
measured over its whole field.  It is turned into a rank between 0 and 1
within the layers compared: the best becomes 0.925, the worst 0.075.  QnR is
the harmonic mean of that rank and the RSP.  A layer scores high only if it
is both faithful and sharp.

## In detail

**The fit.**  For a given $\sigma$, the best $\alpha$ and $\beta$ are a
straight-line fit of the reference $R$ against the blurred counts $B_\sigma$,
solved exactly:

$$\alpha = \frac{\sum (B_\sigma - \bar{B}_\sigma)(R - \bar{R})}{\sum (B_\sigma - \bar{B}_\sigma)^2}, \quad \beta = \bar{R} - \alpha \bar{B}_\sigma$$

Only $\sigma$ is searched: by Brent's method between 0 and 5 reference
pixels, minimising

$$\mathrm{RSE} = \sqrt{\frac{1}{N} \sum \left( \alpha B_\sigma + \beta - R \right)^2}$$

over the $N$ compared pixels.  A $\sigma$ that ends on the bound of 5 pixels
is reported: the two pictures then agree best when everything is blurred
away, which usually means they are not of the same field.  The blur is done
in Fourier space, once per shift, which makes each $\sigma$ tried one
multiplication.

**Which pixels.**  The compared pixels are those of the reference that the
localizations cover: their bounding box, cut to the image.  A widefield image
larger than the localizations' field would otherwise count everything
outside it as missed structure.  The fine grid reaches 21 reference pixels
beyond them, and the shift's reach beyond that, so that localizations just
outside still blur into the edge.  Nothing is cropped.

**The shift** is the peak of the cross-correlation of the blurred picture
with the reference, refined below a pixel by a quadratic fit (as in
[Register layers](plugin:Analysis/Register/Register layers)).  The
localizations are moved by it and pictured again, up to 4 times, until the
correlation finds less than a hundredth of a pixel.  The quadratic fit pulls
towards whole pixels; at a residual near zero that pull is gone.

**The localizations, not the rendering.**  The picture compared is a
histogram of the positions, never what the window renders.  The window's
contrast and saturation would break the straight line between counts and
brightness.  Any blur a renderer adds is a Gaussian the fitted $\sigma$ takes
up anyway.

**Drift.**  The localizations are compared as the table holds them.  A
widefield image taken before the acquisition matches a drift curve that is
zero at the first frame; one taken after it, the last frame.  A constant
offset is the shift, found anyway.  The average of the raw frames that a fit
keeps is smeared by whatever drifted during the acquisition.  The fitted
$\sigma$ takes that up only in part, so the map then shows drift too.

**The FRC map's tiles.**  A tile is a square of the whole field's FRC pixel (a
fifth of its resolution), rounded up to a size the Fourier transform is fast
at, so the tile is a little larger than *tile size* asks; the run says how
large.  The grid starts at the lowest x and y of the layer's localizations.
Each tile is tapered at its edges as the whole field is (a Tukey window over
the outer eighth of each side), so the map is a map of regions, not of single
structures.  With *smooth* the value written to a localization is
interpolated linearly between the centres of the measured tiles around it; a
localization whose own tile has no value stays NaN.

**QnR's rank** is eSRRF's logistic: with $r_b$ and $r_w$ the best and worst
resolution of the layers,

$$n(r) = \frac{1}{1 + e^{-k (r - r_0)}}$$

with $k$ and $r_0$ such that $n(r_b) = 0.925$ and $n(r_w) = 0.075$, and

$$\mathrm{QnR} = \frac{2\, \mathrm{RSP}\, n}{\mathrm{RSP} + n}$$

It is a rank, not a measure: the same layer scores differently against other
layers.  With layers of equal resolution every rank is 0.5.

**Differences from the publications.**  SQUIRREL (and its Python port in
NanoPyx) magnifies the widefield image to the super-resolved pixels by
interpolation and compares there.  Interpolation smooths the reference's
noise and so shrinks the error; here the comparison is on the reference's
own pixels.  SQUIRREL searched $\alpha$, $\beta$ and $\sigma$ together by a
particle swarm; here $\alpha$ and $\beta$ are solved exactly and only
$\sigma$ is searched.  SQUIRREL registers by shifting images and crops the
edges; here the localizations are moved, so nothing is cropped.  SQUIRREL's
FRC map splits an image stack into odd and even frames; here the halves are
blocks of frames, so a blink stays in one half.  eSRRF's QnR ranks a sweep of
reconstruction parameters; here it ranks layers.

## Parameters

### squirrel.reference
The widefield image must be of the same field and the same focus.  The
average of the raw frames is the convenient choice for a fit: it is always
there, on the same pixels, but it is smeared by any drift during the
acquisition.  A separate widefield image taken just before the acquisition
shows the sample unbleached and at one place.

### squirrel.sigma_nm
Leave it on auto.  Set it to the PSF's width (from a bead calibration) to
compare datasets at the same blur.  A fitted blur much larger than the PSF
means the localizations are spread more than the optics would: drift, a
reference out of focus, or a different field.

### squirrel.register
Off only when the two are known to lie on each other to a few nanometres --
a widefield image computed from the same camera frames, say.

### squirrel.signed
The absolute map shows where; the signed one shows which way.  Negative is
structure the localizations miss, positive is structure they add.

### squirrel.magnification
4 is enough: the counts' own pixel then adds a blur of about 7 nm on 100 nm
camera pixels, which the fitted $\sigma$ takes up.  More costs time and
memory and changes nothing visible.

### frc_local.tile_um
Choose it so that most tiles hold a few thousand localizations: smaller tiles
show more detail, but more of them are grey and the rest noisier.  Between 1
and 5 µm suits most dSTORM and PAINT data.

### frc_local.min_locs
Below about 1000 a curve crosses 1/7 by chance; raise it if the map is
speckled by single tiles far from their neighbours.

### frc_local.smooth
For a picture coloured by the attribute; leave it off to keep the measured
value of each tile.

### qnr
Compare layers of the same field only: QnR ranks them against each other.

### write_column
Off, the table is left alone and the run only measures.

### save_tiff
Each map is a 32-bit TIFF beside the localization file, named after it, the
map and the layer; ImageJ reads its pixel size, and File > Open puts it back
where it was.  [Save image layer](plugin:File/Export/Image layer) saves any
image layer the same way.

## Output

The **text** has a line per map and layer:

* *error map*: RSE and RSP; the blur $\sigma$ (fitted, set, or at its bound);
  the shift; the scale $\alpha$ and the offset $\beta$; how many
  localizations on how many pixels.
* *local FRC*: the median of the tiles, how many were measured, the 10th to
  90th percentile, and how many had too few localizations or never crossed
  1/7.
* *QnR*: each layer's score, and the best.

Besides the text:

* **Figures**: a tab per layer with the widefield image, the blurred
  localizations, the error map and the FRC map; with *QnR of the layers*, a
  tab comparing RSP, the resolution's rank and QnR.
* With *write to localizations*: `error_map` (and `error_map_signed` with
  *signed*), the map's value at each localization's pixel, in the widefield
  image's units; `frc_local_nm` and `frc_local_err_nm`, its tile's
  resolution and error.  NaN on localizations that were not measured and
  outside the maps; averaged per blink on a layer that merges blinks.  A
  localization shown by two layers keeps the first layer's value.
* With *add image layers*: each map as an image layer, placed over the
  localizations (the shift added back).  The FRC map has a pixel per tile, 0
  where there is no value.
* With *save as TIFF*: each map as a file.

**Reading the error map.**  A good map is noise: a speckle without
structure, and an RSP close to 1.  Structure in the map is structure the two
pictures disagree on.  A line or patch of negative error is missing from the
localizations: look at the filter, the detection threshold, the labelling.
A bright spot of positive error is added: repeated localizations of one
molecule, or a fluorescent speck that blinks.  An error that follows every
edge means the blur does not fit: a reference out of focus, or one taken
through different optics.  The map cannot see errors below the widefield
image's resolution; two filaments 50 nm apart drawn as one look the same as
two drawn as two.

**Reading the FRC map.**  A good map is even over the structure, with values
close to the whole-field FRC.  A patch much worse than its neighbours is a
region the picture does not resolve as well: look there for drift that was
not corrected, a focus that wandered, or labelling that is sparser.  A
tile's curve is noisier than the whole field's, so read differences between
regions larger than the tiles' error bars, not single tiles.

## References

* Culley S, Albrecht D, Jacobs C, Pereira PM, Leterrier C, Mercer J,
  Henriques R. Quantitative mapping and minimization of super-resolution
  optical imaging artifacts. *Nat Methods* 15, 263 (2018).
  [doi:10.1038/nmeth.4605](https://doi.org/10.1038/nmeth.4605) --
  NanoJ-SQUIRREL: the error map, RSE, RSP, the block-wise FRC map.
* Laine RF, Heil HS, Coelho S, et al. High-fidelity 3D live-cell nanoscopy
  through data-driven enhanced super-resolution radial fluctuation.
  *Nat Methods* 20, 1949 (2023).
  [doi:10.1038/s41592-023-02057-w](https://doi.org/10.1038/s41592-023-02057-w)
  -- QnR.
* Nieuwenhuizen RPJ, Lidke KA, Bates M, et al. Measuring image resolution in
  optical nanoscopy. *Nat Methods* 10, 557 (2013).
  [doi:10.1038/nmeth.2448](https://doi.org/10.1038/nmeth.2448) -- FRC, the
  1/7 threshold.
