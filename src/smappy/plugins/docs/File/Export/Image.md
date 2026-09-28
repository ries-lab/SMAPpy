---
version: "1"
covers: [smappy.render.save_image, smappy.render.render_locs, smappy.render.FieldOfView.from_range, smappy.render.SigmaSettings.apply, smappy.render.normalize]
---

## What it does

Writes the super-resolved picture to an image file -- PNG, TIFF or JPEG --
at a pixel size of your choice, independent of the window's size or zoom.
It is the picture for a figure, a report or a thumbnail: rendered again
from the localizations, so a 5 nm pixel over a whole cell gives a file
thousands of pixels wide, however small the window is.

It draws what the first localization layer shows -- its filter, and the ROI
if one is drawn -- with that layer's render and colour settings, so the
file looks like the window.  The picture is framed tightly around those
localizations.  They are drawn one per frame, as fitted, even when the
layer shows them grouped, so the picture of a grouped layer is not quite
the one in the window.

It is a picture, not a measurement: 8-bit colour, with the brightness
scaled for display.  To analyse the localizations elsewhere, save the table
instead.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.render import (DisplaySettings, FieldOfView, RenderSettings,
                           SigmaSettings, render_locs)
locs = simulate(n_frames=3000, seed=2)
x, y = np.asarray(locs["x_nm"]), np.asarray(locs["y_nm"])
select = (np.abs(x - 7214) < 500) & (np.abs(y - 6049) < 500)   # an ROI
settings = RenderSettings(sigma_settings=SigmaSettings(factor=0.5))  # the window's
```

**1. The frame.**  The picture covers the localizations that are shown,
from the smallest to the largest x and y, cut into pixels of *pixel*.  The
number of pixels follows: a field of 20 µm at 10 nm is 2000 pixels wide.

**2. The render.**  Each localization is drawn as a small Gaussian blob
whose width is its precision (`xy_err_nm`), scaled by the layer's
settings, and never narrower than 0.7 of a rendered pixel -- or as a plain
count per pixel, or a fixed Gaussian, if the layer is set so.

**3. The colours.**  The layer's colour table (LUT), contrast and gamma
turn the counts into colours.  The contrast sets the brightest pixels to
full scale: at the default, the brightest 0.1 %.

```figure The same ROI of a simulated structure exported at three pixel sizes.  A smaller pixel shows more detail and makes a larger file, down to about the localization precision (here about 5 nm); below it, it adds pixels but no detail.
fig.set_size_inches(7.5, 2.8)
axes = fig.subplots(1, 3)
for ax, pixel in zip(axes, (5.0, 20.0, 50.0)):
    fov = FieldOfView.around(x[select], y[select], pixelsize=pixel)
    rgb = DisplaySettings().apply(render_locs(locs, fov, settings, select=select))
    ax.imshow(rgb, interpolation="nearest", origin="lower")
    ax.set_title(f"pixel {pixel:g} nm: {fov.nx} x {fov.ny}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
```

## In detail

**The size.**  For localizations spanning $x_{\min}$ to $x_{\max}$ and a
pixel $p$ the picture is

$$n_x = \mathrm{round}\left( \frac{x_{\max} - x_{\min}}{p} \right)$$

pixels wide, and likewise high; there is no margin, so the outermost
localizations sit at the edge.

**The blob width.**  In the default *precision* mode a localization with
precision $\sigma_{\mathrm{loc}}$ is drawn with

$$\sigma = \min\left( \max\left( f\,\sigma_{\mathrm{loc}},\ 0.7\,p \right),\ 10\,\tilde\sigma \right),$$

$f$ the layer's factor (0.5 for a layer in the window), $\tilde\sigma$ the
median of $f\,\sigma_{\mathrm{loc}}$ over the localizations, so a handful
with an absurd precision cannot dominate.  The floor of 0.7 pixels keeps a
coarse picture from being a field of single bright pixels; it is why the
50 nm picture above is smoother than its pixels alone would make it.

**The brightness.**  The value set to full scale is the $1 - 10^{-c}$
quantile of the pixels, $c$ the *contrast* (3 by default): the brightest
fraction $10^{-c}$ saturates.  A picture exported at another pixel size is
therefore scaled afresh, and its brightness is not comparable with the
first.

**The file.**  Its format follows the extension (`.png`, `.tif`, `.jpg`).
PNG and TIFF are lossless; JPEG blurs fine structure and is best avoided
for super-resolution pictures.  The pixel size is not written into the
file, so note it, or draw a scale bar, for a figure.

**Without a session** -- from a script -- the table is drawn with the
default settings (precision mode with $f = 1$, the *hot* LUT).

## Parameters

### pixelsize_nm
About half the localization precision is as fine as is useful -- 5 to 10
nm for a typical dSTORM or PAINT table; coarser for an overview.  The
picture's size grows as the square of the inverse: halving the pixel makes
four times as many.  On a picture whose axes are not positions (the
Render tab's *axes*), this is in the units of the x axis.

## Output

The file, and a line in the history saying where it went and with which
settings.

## Differences from SMAP

SMAP's `TifSaver` saves the picture as the render window shows it, all
layers together, optionally with a scale bar, and writes the pixel size
into the TIFF.  Here the picture is rendered again at a chosen pixel size,
from the first localization layer, without a scale bar.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
