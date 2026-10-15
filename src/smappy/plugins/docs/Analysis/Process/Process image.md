---
version: "1"
covers: [smappy.image_processing.subtract_background, smappy.image_processing.gaussian_background, smappy.image_processing.wavelet_background, smappy.image_processing.rolling_ball_background, smappy.image_processing.local_contrast, smappy.image_processing.extend, smappy.plugins.process_image.process_image, smappy.plugins.process_image.ProcessImage.run]
---

## What it does

Cleans up an image layer -- a wide-field picture, a transmitted-light image,
a rendered TIFF, the camera frames a fit kept -- so that what is in it can be
seen and compared with the localizations.  Two operations, ticked on or off:

* **subtract background** removes a smooth background: out-of-focus light,
  uneven illumination, autofluorescence of the cell, a camera offset.  What
  is left are the structures narrower than the background varies over, on
  a floor at zero.
* **normalise contrast** divides each part of the image by how much it
  varies locally, so that a dim and a bright part of a cell can be looked
  at with one contrast setting.

The result is a **new image layer**, added after the others and placed
exactly over the original -- the same pixel size, the same corner, the same
placement if [Transform](plugin:Analysis/Process/Transform) moved it -- so it
lies over the localizations as the original does.  The original is never
changed: show the two side by side to judge the settings, and remove the new
layer in the Render tab if they were wrong.  *Preview* draws the result
without adding anything.

Use it to see a wide-field channel under the localizations, to find cells
or structures by eye, or to prepare an image before
[Register layers](plugin:Analysis/Register/Register layers) correlates it
with a rendered layer: a background shared by nothing in the other image
only blurs the correlation peak.  Do not use the result to measure
intensities: the background subtracted is an estimate, and the contrast
normalisation changes the scale from place to place.

## How it works

```figure-setup
from smappy.simulate.camera import camera_frames
from smappy.image_processing import subtract_background, local_contrast
frames, _ = camera_frames(n_frames=200, size_px=64, background=0.0, seed=1)
signal = 3.0 * (frames.mean(axis=0) - 100.0)       # the structure, offset removed
yy, xx = np.mgrid[0:64, 0:64].astype(float)
true_bg = 300.0 + 4.0 * xx + 1500.0 * np.exp(-((xx - 20) ** 2 + (yy - 44) ** 2) / (2 * 22.0 ** 2))
observed = np.random.default_rng(0).poisson(np.clip(signal, 0, None) + true_bg).astype(float)
results = {m: subtract_background(observed, m, sigma_px=5, radius_px=5, level=3)
           for m in ("gaussian", "ball", "wavelet")}
row = int(np.argmax(signal.max(axis=1)))
```

**1. A background is estimated**, by one of three *methods*:

* **Gaussian**: the image is blurred with a Gaussian of SD *width*.  The
  blur keeps what varies slowly and averages away what is narrow -- but on
  its own it takes a share of every bright structure with it, and a dense
  structure lifts the background under itself.  So the estimate is refined
  a few times (*rounds*): the pixels well above the background (more than
  three times the noise) are taken for structure, replaced by the
  background under them, and the image blurred again.
* **rolling ball**: a ball of *ball radius* is rolled along the underside
  of the image, seen as a landscape whose height is the brightness.  The
  background is how high the top of the ball gets.  It cannot climb into
  anything narrower than itself, so it needs no rounds; it is the method
  of ImageJ's *Subtract Background*.
* **wavelet (SMAP)**: the image is smoothed by a wavelet transform, keeping
  only its coarsest part at *level* -- a smoothing over about 2^level
  pixels -- and refined in rounds as the Gaussian is.  This is what SMAP's
  *Tiff_remove_BG* did.

**2. It is subtracted.**  The result is what stands above the background,
around zero where there is nothing.

```figure An image with a sloping, uneven background (left), the background as it really was and as the Gaussian method found it, and what is left after subtracting it.
fig.set_size_inches(8.0, 2.4)
out, bg = results["gaussian"]
lo, hi = observed.min(), observed.max()
for ax, (im, title, lim) in zip(fig.subplots(1, 4), (
        (observed, "observed", (lo, hi)), (true_bg, "true background", (lo, hi)),
        (bg, "estimated background", (lo, hi)), (out, "result", (None, None)))):
    ax.imshow(im, cmap="gray", vmin=lim[0], vmax=lim[1])
    ax.set_title(title, fontsize=8)
    ax.axis("off")
```

```figure One row through the image: the structure as simulated (black) and what each method leaves of the image -- the background gone, the structure close to its height.
fig.set_size_inches(6.0, 2.6)
ax = fig.subplots()
ax.plot(observed[row], color="0.7", label="observed")
ax.plot(signal[row], "k", lw=2, label="truth")
for (method, (out, _)), colour in zip(results.items(), ("C0", "C1", "C2")):
    ax.plot(out[row], color=colour, label=method)
ax.axhline(0, color="0.5", lw=0.5)
ax.set(xlabel="pixel", ylabel="photons")
ax.legend(fontsize=7, frameon=False)
```

**3. Local contrast**, if ticked, after the background.  Around every
pixel the plugin computes the mean and the spread (the standard deviation)
of the image over a Gaussian *window*, subtracts the mean and divides by the
spread.  A faint filament in a dim corner then comes out as strong as a
bright one in the middle.  Where there is nothing but noise the spread is
small, and dividing by it would turn the noise into texture; so the spread
used is never below a *floor*, a fraction of its typical (median) value
over the image.

**4. The new layer** is the processed image, named after the original with
what was done to it -- "wf (background subtracted)" -- and placed where the
original is.  For a stack, only the plane the layer shows is processed, or
every plane, each on its own (*planes*).

## In detail

**Units.**  The widths are set in nm and converted to pixels with the
image's pixel size (the geometric mean of x and y for non-square pixels);
the line under the form says what they are in pixels.  The wavelet's
*level* stays a level, as in SMAP.

**The refinement** of the Gaussian and the wavelet.  With $B_0$ the
smoothed image $f$, each round computes the robust SD of the residual,
$s = 1.4826 \cdot \mathrm{median}\,|r - \mathrm{median}\, r|$ with
$r = f - B_k$, replaces every pixel with $f > B_k + 3s$ by $B_k$, and
smooths that to $B_{k+1}$; 5 rounds by default, 0 for a plain high-pass
filter.  Replacing the bright pixels rather than clipping them at the
threshold matters for dense structures: clipped, each of them still lifted
the background by three SDs, and on the simulation above the background
came out about twice as far off.

**The wavelet** is the *à trous* ("with holes") transform with the
B3-spline kernel $(1, 4, 6, 4, 1)/16$, dilated by $2^j$ at step $j$: the
isotropic wavelet of the SMLM literature (Izeddin et al. 2012).  Its
approximation after $J$ steps is a smoothing with an SD of
$\sqrt{\sum_{j<J} 4^j}$ pixels -- 4.6 at level 3.

**The rolling ball** is the grey opening of the image with a ball:
$B = (f \ominus b) \oplus b$, erosion then dilation, with $b$ the ball's
height profile $\sqrt{R^2 - d^2} - R$ over the disc of radius $R$ pixels.
As in ImageJ the height is in the image's own units, so for an image of
hundreds of photons the ball is nearly flat over its width.  Also as in
ImageJ, the image is first averaged over 3 × 3 pixels for the estimate --
otherwise the ball rests on the noise's lowest values and the result sits
about two noise SDs above zero -- and a large ball is rolled on a coarser
image: the minimum over blocks of 2 pixels above a radius of 10, 4 above
30, 8 above 100, the background interpolated back.  On that coarser image a
slope is followed only to within the slope over a block.

**The edges.**  Every method needs the image beyond its edges.  It is
continued by point reflection, $f(-d) = 2 f(0) - f(d)$, which continues a
slope as a slope, so a gradient across the field is removed right to the
edge; the mirror reflection usual in filters bends it back and leaves a
rim.  The price: a bright spot right at the edge is continued as a dark
one, which pulls the background down next to it.

**Local contrast**: with $m$ and $v$ the local mean and variance over the
Gaussian window,
$$g = \frac{f - m}{\sqrt{v + c^2\, \mathrm{median}(v)}}$$
with $c$ the *floor*.  The result has no unit: about 1 is a typical local
variation.

## Parameters

### layer
Any image layer; *the first image layer* follows whichever is first.

### planes
Only for a stack.  Every plane takes as many times as long as one.

### bg.method
The rolling ball is the safest choice for images with sharp structures on
a smooth background; the Gaussian follows a background that curves within
a few of its widths better; the wavelet is there to do what SMAP did.

### bg.sigma_nm
A few times wider than the widest structure to keep: about 1 µm for
diffraction-limited spots and filaments, more for nuclei or whole cells.
Too narrow and structures are eaten into; too wide and a background
varying over a few µm is left.

### bg.radius_nm
Larger than the half-width of the widest structure to keep, as for ImageJ.

### bg.level
2 to 4 for wide-field images at 100 nm pixels, as in SMAP.

### bg.rounds
More rounds help when bright structures cover much of the image; beyond 5
little changes.

### lcn.window_nm
About the size of the region over which brightness should be evened out:
a few µm to even out a cell, less to bring out fine detail.

### lcn.floor
Lower brings out faint structure and noise with it; higher leaves dim
regions dim.

## Output

A new image layer, its name saying what was done, and a figure of the
plane shown: the original, the background taken away, and the result.  A
good background subtraction leaves the empty parts of the image around
zero and the structures at their height; dark halos around bright
structures mean the width or ball is too small, and a residual gradient
that it is too large.  The new layer is kept in the session's file when it
is saved.

## Differences from SMAP

Based on SMAP's `Tiff_remove_BG` ([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).
The main changes:

* SMAP had one method, the wavelet; here the Gaussian and the rolling ball
  are added, and the local contrast normalisation.
* SMAP's wavelet is the low-pass of a dual-tree complex wavelet
  transform; here it is the à trous transform, which SMAP has too
  (`mywaveletfilteratrous`).  Its refinement -- an estimate at twice the level, the residual
  clipped once at three SDs, smoothed again -- is replaced by the rounds
  described above, so the wavelet and the Gaussian differ only in their
  smoothing.
* SMAP replaced the TIFF or added a copy among the file's TIFFs; here the
  result is always a new image layer at the original's place.

## References

* Izeddin I, Boulanger J, Racine V, et al. Wavelet analysis for single
  molecule localization microscopy. *Opt Express* 20, 2081 (2012).
  [doi:10.1364/OE.20.002081](https://doi.org/10.1364/OE.20.002081) -- the
  à trous wavelet in SMLM.
* Sternberg SR. Biomedical image processing. *Computer* 16, 22 (1983).
  [doi:10.1109/MC.1983.1654163](https://doi.org/10.1109/MC.1983.1654163) --
  the rolling ball.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
