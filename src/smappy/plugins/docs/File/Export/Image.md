---
version: "4"
covers: [smappy.plugins.file.export_image, smappy.plugins.file.image_pictures, smappy.picture.render_picture, smappy.picture.frame_fov, smappy.picture.colour_scales, smappy.picture.write_picture, smappy.picture.describe_view, smappy.picture.read_description, smappy.layer_state.render_panels, smappy.render.normalize]
---

## What it does

Saves the super-resolved picture as a file -- PNG, TIFF or JPEG -- as the
render window shows it: every visible [layer](plugin:Panels/Glossary) (one
picture of the localizations with its own filter and colours), the panels
side by side if the window splits them, and what the window draws over the
picture: a scale bar, a colour bar for each layer coloured by a column, and
the panels' names.  Rendered again, at the pixel size the window shows or
one you set, so a 5 nm pixel over a whole cell gives a file thousands of
pixels wide however small the window is.

It is the one way a picture is made: the render window's ⋯ menu and the
3D view's *Save* menu write through it too.  *PNG as displayed...* there is this
plugin at its defaults, on the view as it is; *Save image...* opens it,
set to the window it was asked from.

Use it for a figure, a slide or a record of what was looked at.  Choose
what to save:

* **as the view** -- the panels as the window lays them out, the layers
  added up, or a picture each side by side (the window's *split*), with or
  without the added-up one;
* **composite only** -- the visible layers added up into one picture;
* **each layer** -- a file per visible layer, in its own colours, or as its
  intensity in grey, for a channel to take apart in another program;
* **intensity, 32-bit** -- the summed intensity as floating-point numbers,
  for measuring rather than looking.

The file says how it was made, as SMAP's did: the pixel size in its
resolution tags, and the files, the layers, their render settings and
[filters](plugin:Panels/Glossary) (the lowest and highest values of columns
a localization must lie between), and the merging of blinks, written into
it as text.  It is a picture, not a measurement -- apart from the 32-bit
intensity, the brightness is scaled for display.  To analyse the
localizations elsewhere, save the table instead.

## How it works

```figure-setup
from smappy.layer_state import snapshot_layers
from smappy.picture import render_picture
from smappy.render import FieldOfView
from smappy.session import Session
from smappy.simulate import simulate
session = Session(simulate(n_frames=1500, seed=3))
session.add_layer()
session.layers[1].state.settings.color_field = "z_nm"
session.layers[1].state.settings.color_range = (-400.0, 400.0)
session.layers[1].state.display.lut = "jet"
layers = snapshot_layers(session.layers)
fov = FieldOfView.from_range(*session.full_view(), 45.0)    # the whole field
```

**1. The area.**  The picture covers what the window shows (*the view*),
the box around the drawn [ROI](plugin:Panels/Glossary) (region of
interest), or every localization (*everything*), cut into pixels of
*pixel* -- the window's own pixel when *pixel* is on *auto*.  The number of
pixels follows: a field of 20 µm at 10 nm is 2000 pixels wide.

**2. The render.**  Each visible layer is drawn exactly as the window
draws it -- its filter, its merged blinks, its render mode, colours,
contrast and gamma -- by the window's own render, and the layers are added
up.  Saved at the window's pixel, the composite is the window's picture
pixel for pixel.

**3. What is drawn over it.**  The *scale bar* sits bottom right: a round
length, a fifth of the width or more (the length the window's bar has), or
the length you set; its length in pixels is that length divided by the
pixel size.  The *colour bars* sit top right, one per colour scale, as in
the window: the colour table from the bottom of the *colour range* to its
top, the two ends written in the column's unit, the layers and the column
above.  The *labels* name each panel, or each layer's file, top left.  The
text grows with the picture, so a large export does not carry specks.

```figure The view split into its two layers and the overlay, as the window shows it: the scale bar on the first panel, the colour bar of the layer coloured by z in the top right corner, and each panel's name.
fig.set_size_inches(7.5, 2.7)
ax = fig.add_subplot(111)
picture, = render_picture(layers, fov, what="view", panels=(0, 1, "overlay"))
ax.imshow(picture.image, interpolation="nearest")
ax.set_axis_off()
```

**4. Each layer on its own.**  In its colour, a layer's file is its panel
of the view.  In grey, it is its intensity scaled as its display scales it
(contrast and gamma), without the colours: one channel of a multicolour
picture, for a program that takes channels.

```figure The second layer saved on its own: in its colours (left), coloured by z, and as grey intensity (right), where the colour by z is gone and the brightness is what remains.
fig.set_size_inches(7.5, 3.6)
axes = fig.subplots(1, 2)
colour = render_picture(layers, fov, what="layers")[1]
grey = render_picture(layers, fov, what="layers grey")[1]
axes[0].imshow(colour.image, interpolation="nearest")
axes[1].imshow(grey.image, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
for ax in axes:
    ax.set_axis_off()
```

## In detail

**The size.**  For an area from $x_0$ to $x_1$ and a pixel $p$ the picture
is

$$n_x = \mathrm{round}\left( \frac{x_1 - x_0}{p} \right)$$

pixels wide, and likewise high.  The view at its own pixel is the window's
canvas; the ROI's area is its bounding box, whatever its shape; *everything*
is the data's extent with a margin of 1 % on each side.

**The brightness.**  The value set to full scale is the $1 - 10^{-c}$
quantile of the pixels, $c$ a layer's *contrast*: the brightest fraction
$10^{-c}$ saturates.  It is computed afresh for every render, so a picture
saved at another pixel or over another area is scaled afresh too, and its
brightness is not comparable with the first.  The grey intensity of a layer
is the same normalised value, raised to the layer's gamma, written as 0 to
255.

**The 32-bit intensity** is the sum over the visible layers of what each
accumulates per pixel -- one per localization in *hist* mode, the
Gaussians' values otherwise -- before any contrast, LUT or clipping.  PNG
and JPEG have no floating-point pixels, so it is always written as a TIFF,
whatever the file's extension says; nothing is drawn over it.

**Colours and paper.**  The layers' colours are added up and clipped, and a
layer on *white background* turns the sum over once
([LUT](plugin:Panels/Glossary), look-up table, also called a colour map);
the bars and labels are then drawn black instead of white.

**The file.**  Its format follows the extension (`.png`, `.tif`, `.jpg`).
PNG and TIFF are lossless; JPEG blurs fine structure and is best avoided
for super-resolution pictures.  The pixel size is written as the
resolution: in a TIFF as pixels per centimetre, which ImageJ and Fiji read
as µm; in a PNG as pixels per metre.  JPEG's resolution field cannot hold a
nanometre pixel, so a JPEG has none.  How the picture was made is written
as YAML: the TIFF's *ImageDescription* (Fiji: *Image > Show Info*), the
PNG's *Description* text, the JPEG's comment.  It holds the pixel size and
the area, these settings, the files, the merge settings, and per visible
layer its filter, whether it shows merged blinks, the files it shows, the
number of localizations it draws, and its render and display settings.

**Each layer** goes to the file's name with `_layer1`, `_layer2`, ...
before the extension, numbered as the Render tab numbers the layers.

**The 3D view.**  Asked from the 3D window (*window* is *3D view*), the
picture is its projection: the turn, the zoom and the slab as they are.
*The view* is the window's canvas; *the ROI* and *everything* are the
slab's box on the screen.  A tracks layer is drawn as lines through the
volume and an image layer as a flat plane at the height of the centre of
rotation, as the window draws them.  It is drawn on the processor, which gives the
picture of the *auto*, *cpu* and *gpu* engines; the *points* and *spheres*
looks are the graphics card's own, and its *PNG as displayed...* keeps
the screen's pixels instead.

**Without a window** -- a script or a batch -- there is no view: *the
view* is everything, and *pixel* on *auto* is 10 nm.  Without a session,
the context's table is drawn as one layer with the default settings.

**On other axes** (the Render tab's *axes*: a picture of one column
against another) the pixel is in the x axis's unit and no scale bar is
drawn: one bar cannot be a length on both axes.

## Parameters

### what
*as the view* is what the window shows, and the one to use for a figure.
*composite only* saves one picture whatever the window's layout.  *each
layer* writes as many files as there are visible layers.

### region
*the ROI* frames the picture on the box around the drawn ROI; the
localizations outside the ROI but inside the box are drawn too, as in the
window.

### pixelsize_nm
About half the localization precision is as fine as is useful -- 5 to 10
nm for a typical dSTORM or PAINT table; coarser for an overview.  The
picture's size grows as the square of the inverse: halving the pixel makes
four times as many.

### scale_bar_nm
Set it to make the bars of several pictures the same length.  A bar wider
than the picture is left out.

### colour_bars
Only a layer coloured by a column has one; a layer coloured by intensity
needs none, as its brightness is what the picture shows.

### view
Shown once the 3D window has been opened; *Save image...* in a window sets
it to that window.

## Output

The files, a line in the history saying where they went and with which
settings, and in each file how it was made.  A picture that comes out
black or nearly white was scaled by the layers' *contrast*: the same
setting in the Render tab changes it.

## Differences from SMAP

Ported from SMAP's `TifSaver` (Ries 2020).  SMAP saved the window's picture
at the window's pixel, as "rendered image with scalebar", "rendered image",
"layer 1-3 as RGB" or "layer 1 as grayscale", and wrote its render
settings into the TIFF's description as tab-separated text.  Here:

* the picture is rendered again at any pixel and over the view, the ROI or
  everything, not only the window's;
* the layers' files are every visible layer, each in colour or grey,
  rather than the first three as RGB channels or the first alone;
* the colour bars and the panels of a split view are drawn too, and the
  summed intensity can be saved as floating-point numbers;
* the description is YAML, which a program reads back
  (`smappy.picture.read_description`), and it is written into PNG and JPEG
  files as well as TIFF.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
