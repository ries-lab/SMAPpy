---
version: "1"
covers: [smappy.layer_registration.register, smappy.layer_registration.lateral_shift, smappy.layer_registration.axial_shift, smappy.layer_registration.common_grid, smappy.layer_registration.Registration.transformation, smappy.correlation._xcorr, smappy.rcc._peak, smappy.rcc._axial_shifts, smappy.plugins.register_layers.transform_target]
---

## What it does

Measures how far one layer is shifted against another: the *target*
against the *reference*.  Two cameras, two halves of a chip, two colours
imaged one after the other, or a widefield image and the localizations
taken with it rarely sit exactly on each other, and the offset between them
is easiest to read off the data itself.

Each layer can be a layer of localizations, as it shows (its filter, its
files), or an image layer.  The shift is found in x and y, and optionally in
z (*z too*) for two layers of 3D localizations.

It measures and changes nothing.  What it measured is kept as a **named
transformation** of the session, under *name*, and the form then offers
**Apply with Transform**: [Transform](plugin:Analysis/Process/Transform)
opens, set to that transformation and to what the target layer shows (its
file, its channel, or the image).  Look at the figure first; if the peak is
clear, apply.

Only a shift is measured -- no rotation, no change of scale.  For the full
map between two cameras or halves of a chip, use
[Calibrate transform](plugin:Analysis/Register/Calibrate transform).

## How it works

```figure-setup
from smappy.layer_registration import register
rng = np.random.default_rng(0)
sites = np.column_stack([rng.uniform(0, 4000, 150), rng.uniform(0, 4000, 150)])
a = sites[rng.integers(0, 150, 20000)] + rng.normal(0, 10, (20000, 2))
b = sites[rng.integers(0, 150, 20000)] + rng.normal(0, 10, (20000, 2)) + (180.0, -95.0)
found = register({"x": a[:, 0], "y": a[:, 1]}, {"x": b[:, 0], "y": b[:, 1]})
```

**1. Two pictures on one grid.**  A grid of square bins (*bin*, 20 nm by
default) is laid over both layers.  Localizations are counted in its bins; an
image is looked up on it, each bin taking the value of the image pixel under
its centre.  Each picture has its mean taken off, so a widefield image's
background does not count as structure.

**2. Cross-correlation.**  The correlation of the two pictures is computed
for every shift within *max shift* (2500 nm by default) with Fourier
transforms.  It is largest at the shift that lays the target best onto the
reference.

**3. The peak.**  The highest point of the correlation, after a light
smoothing, is refined below a bin by fitting a smooth curved surface
(a quadratic) to the bins around it.  That fit pulls towards whole bins, so
the found shift is taken out of the localizations and the correlation
repeated: the second peak sits near zero, where the fit is exact, and adds
the small rest.

```figure A simulated layer and a copy shifted by (180, -95) nm.  Left: the two over each other.  Right: their correlation, with the peak (+) at the shift.
fig.set_size_inches(7.5, 3.4)
left, right = fig.subplots(1, 2)
left.plot(a[:, 0], a[:, 1], ",", color="#d62728")
left.plot(b[:, 0], b[:, 1], ",", color="#17becf")
left.set_aspect("equal"); left.invert_yaxis()
left.set_title("reference (red), target (cyan)", fontsize=9)
c = found.correlation
lags = c.shape[0] // 2
reach = 400.0
k = int(reach / found.bin_nm)
patch = c[lags - k:lags + k + 1, lags - k:lags + k + 1]
right.imshow(patch, extent=(-reach, reach, reach, -reach), cmap="viridis")
right.plot(found.dx_nm, found.dy_nm, "+", color="#d62728", ms=14, mew=1.5)
right.set_xlabel("shift x (nm)"); right.set_ylabel("shift y (nm)")
right.set_title(f"peak at ({found.dx_nm:.1f}, {found.dy_nm:.1f}) nm", fontsize=9)
for ax in (left, right):
    ax.tick_params(labelsize=7)
```

**4. z, if asked.**  With the lateral shift taken out of the target, the
field is cut into small square tiles (200 nm), the z positions in each tile
are counted in bins (*z bin*), and each tile's z profile of the target is
correlated with the reference's.  The correlations of all tiles are added
and the peak, within *max z shift*, is the shift in z.  A z profile of the
whole field would mix hundreds of structures at different heights into a
featureless slab; the tiles keep each profile local.

**5. Kept, for Transform.**  The transformation that undoes the shift -- the
target moved by minus what was found -- is kept under *name*.  Transform
applies it (*Apply with Transform*), to the target layer's localizations or
to the image.

## In detail

**Which table.**  Each layer of localizations as it shows: merged blinks on
a layer that merges them, only what its filter and its files let through.
An image layer is taken as it is placed (an image moved by Transform is
looked up where it now is), the plane the layer shows.

**The grid** spans both layers: the box around each layer's localizations,
or the image's area.  The search for the peak needs room around the
pictures, so a grid whose bins and that room come to more than 8 million is
refused, with the bin that would fit.  On auto, *max shift* searches up to
half the grid.

**The correlation** is that of Spatial statistics' correlation image,
$c(\mathbf{k}) = \sum_{\mathbf{x}} a(\mathbf{x})\, b(\mathbf{x} + \mathbf{k})$
with $a$ the reference and $b$ the target, each minus its mean, padded so
that it does not wrap.  Unlike there, it is not divided by the window's own
correlation: only the position of the peak is wanted, not its height.

**The peak** is RCC's: the correlation within *max shift* smoothed by a
$5 \times 5$ box to pick the bin, then a quadratic surface fitted to the
unsmoothed correlation over $7 \times 7$ bins around it, whose stationary
point is the sub-bin shift.  It is then refined up to three times, each time
with the shift so far taken out of whichever layer is localizations and the
peak looked for within 6 bins; it stops when a round moves it by less than a
hundredth of a bin.  On an exact shifted copy, at 20 nm bins, the first
peak was up to 4 nm off and the refined one within 1 nm.  Two image layers
are not refined: an image is not binned again.

**z** is RCC's axial pass, with the reference as one time window and the
target as the other: z histograms in 200 nm tiles, each minus its mean,
correlated and summed, the peak fitted by a parabola over 7 bins.  The zero
shift is a measurement like any other here (RCC leaves it out, because in
one acquisition it carries molecules on in both windows).  An image has no
z, so z needs two layers of localizations with `z_nm`.

## Parameters

### bin_nm
About the localization precision, or the image's pixel when one layer is an
image.  Much finer makes the pictures sparse and the correlation noisy;
much coarser blurs the structure the peak is found from.

### max_shift_nm
Above the largest offset expected.  A larger range costs a larger grid and
gives a false peak more room; auto searches everywhere.

### use_z
Off by default: most registrations are lateral, and the z profiles need a
structure with some depth.

### max_z_shift_nm
Smaller than the lateral range, because a z profile is far less structured
than an image and a wide range lets its maximum wander.

### name
Running it again under the same name replaces what was kept.

## Output

* **The text**: the shift of the target, dx, dy (and dz) in nm, and the name
  it is kept as.
* **The figure**: the two pictures over each other (reference red, target
  cyan), and their correlation with the peak marked.  A good registration has
  one sharp peak; a smear, or several peaks of similar height, means the
  layers share little structure, or the bin or *max shift* does not suit
  them.
* **A named transformation** in the session, which Transform applies, and
  which is saved with the localization file.

## Differences from SMAP

Ported from SMAP's (Ries 2020) XYZdisplacement, with the maximum shift of its
RegisterLocs2.

* **Applied, not copied.**  SMAP put the numbers on the clipboard; here they
  become a named transformation that Transform applies.
* **z in square tiles.**  SMAP's XYZdisplacement correlated z histograms in
  100 nm slices along x; here, as in RCC, in 200 nm square tiles, because
  slices spanning the whole field mixed many structures into each profile.
* **Images too**, on the same grid and by the same correlation, and the peak
  refined until it sits near zero.

## References

* Ries J (2020). SMAP: a modular super-resolution microscopy analysis
  platform for SMLM data. *Nature Methods* 17, 870-872.
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
