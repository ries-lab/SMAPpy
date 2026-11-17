---
version: "1"
covers: [smappy.plugins.file.ExportImageLayer, smappy.plugins.file.image_layer_path, smappy.images.save_tiff, smappy.images.find_image]
---

## What it does

Saves an image as a 32-bit TIFF: a widefield picture, the camera frames a fit
kept, or a map a plugin made -- the error map and local FRC map of
[Image quality](plugin:Analysis/Measure/Image quality), an image from
[Process image](plugin:Analysis/Process/Process image).  The values are
written as they are, as floating-point numbers, never scaled to 8 or 16 bits:
an error map or a camera frame is a measurement, and the numbers are what
another program needs.

It is not [Save image](plugin:File/Export/Image).  That one writes the
picture the window shows -- colours, contrast, scale bar, every layer
together -- for a figure.  This one writes one image's numbers, for
analysis elsewhere.

## How it works

**1. The image.**  *image* names it; on auto, the image the first image
layer shows.

**2. The planes.**  A stack is written whole, or with *planes* on *the plane
shown*, only the plane the image layer shows.

**3. The file.**  A TIFF that ImageJ (Fiji) opens as a hyperstack, with:

* the pixel size, in µm, in the resolution tags ImageJ reads;
* a z-stack's step, or a time series' frame interval, when the image knows
  it;
* where the image lies, `x0` and `y0` in nm, in ImageJ's metadata.  File >
  Open reads them back, so a saved image opened again lies under the same
  localizations.

On auto, the file is beside the localization file -- or the image's own,
when the localizations are in none -- named after both:
`cells_sml_widefield.tif` for an image called *widefield.tif* beside
`cells_sml.h5`.

## Parameters

### path
A name without `.tif` gets it.  The folder is made if it does not exist.

### planes
*the plane shown* is the one the first image layer that shows the image is
at, or the image's own when no layer shows it.

## Output

The file, and a line in the log: which image, how many planes and pixels,
the pixel size, and where it went.
