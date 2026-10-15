---
version: "1"
covers: [smappy.transformation.typed, smappy.transformation.transform_table, smappy.transformation.from_channel_transform, smappy.transformation.check_same_frames, smappy.transformation.save_transformation, smappy.transformation.load_transformation, smappy.transformation.Transformation.planar, smappy.images.ImageData.resample, smappy.drift.Drift.apply, smappy.drift.Drift.referenced, smappy.plugins.transform.Transform.run, smappy.plugins.transform.drift_of]
---

## What it does

Moves one file's localizations -- or one channel of them -- or an image
layer, so that it lies where it should.  Some typical cases:

* a second camera, or the second half of a split chip, sees the sample
  shifted, turned and slightly magnified against the first; a
  [registration](plugin:Panels/Glossary) measured once brings it onto the
  first;
* a file was saved with the wrong pixel size, and its coordinates are off by
  a factor;
* a widefield or transmitted-light image was taken on another camera or with
  the stage moved, and should sit under the localizations;
* two cameras recorded at the same time, and the
  [drift](plugin:Panels/Glossary) measured in one file is the drift of the
  other too.

What moves is the **target**.  For localizations the coordinates themselves
change: `x_nm`, `y_nm` and, if the table has it, `z_nm`.  The change is
logged and can be undone, and everything after it -- measurements, exports,
ROIs -- sees the moved coordinates, as the picture does.  An **image layer**
is moved by where it is placed, not by recalculating its pixels, so moving
it loses nothing and undoing it is exact.

What it is moved by is the **source**: values typed in (shift, rotation,
scale, mirror), a **named transformation** the session keeps, a
**transformation file**, or the **drift of another file**.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.transformation import typed
locs = simulate(n_frames=3000, seed=4)
keep = np.asarray(locs["photons"]) > 200
x, y = np.asarray(locs["x_nm"])[keep], np.asarray(locs["y_nm"])[keep]
middle = (0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max()))
```

**1. Typed values.**  *Shift x*, *shift y* and *shift z* move the target;
*rotate* turns it, counterclockwise as the picture shows it (y points
down, as on the camera); *scale* multiplies x and y; *mirror* turns it over,
left-right or up-down.  Rotation, scale and mirror keep one point still, the
centre chosen under *about*: the middle of the target (the middle of the box
around its localizations), the origin (0, 0), or a typed one.  They are
applied in one order: mirror, then scale, then rotate, all about the centre,
then the shift.

```figure A simulated structure turned by 20 degrees and scaled by 1.1 about its middle (red), over where it was (grey).  The middle stays where it was.
t = typed(angle_deg=20.0, scale=1.1, centre=middle)
nx, ny, _ = t.apply(x, y)
fig.set_size_inches(5, 4)
ax = fig.subplots()
ax.plot(x, y, ",", color="0.6")
ax.plot(nx, ny, ",", color="#d62728")
ax.plot(*middle, "+", color="k", ms=12)
ax.set_aspect("equal"); ax.invert_yaxis()
ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
```

**2. A named transformation.**  The session keeps a list of transformations
under names.  A transformation applied here is added to it with *keep
as*, and is then applied by name.  The list is saved with the workspace (*File >
Workspace > Save*, and the GUI part of a saved localization file) and with
the localization file, so the map between a lab's two cameras is measured
once and applied by name from then on.

**3. A transformation file.**  Either a `_2ct.h5` written by
[Calibrate transform](plugin:Analysis/Register/Calibrate transform) (or a
dual-colour bead calibration), or a `_T.h5` this plugin writes (*write to*)
from any transformation it applies.  A `_2ct.h5` maps the second channel onto
the first in camera pixels; it is converted to nanometres with the pixel size
of the file being moved.  *invert* applies a transformation backwards, the
first channel onto the second.

```figure The two channels of a simulated two-camera pair: the second camera's view (blue) is turned and shifted against the first (grey); the _2ct map brings it onto the first (red, drawn over the grey).
from smappy.simulate.camera import dual_transformation
from smappy.calibrate.dual import ChannelTransform, map_points
from smappy.transformation import from_channel_transform
h = dual_transformation(100)
second = map_points(np.linalg.inv(h), np.column_stack([x, y]) / 100.0) * 100.0
t = from_channel_transform(ChannelTransform(h, {}, {}), (100.0, 100.0))
back_x, back_y, _ = t.apply(second[:, 0], second[:, 1])
fig.set_size_inches(5, 4)
ax = fig.subplots()
ax.plot(x, y, ",", color="0.6")
ax.plot(second[:, 0], second[:, 1], ",", color="#3182bd")
ax.plot(back_x, back_y, ",", color="#d62728")
ax.set_aspect("equal"); ax.invert_yaxis()
ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
```

**4. The drift of another file.**  Two cameras that record at the same time
see the same drift.  When one file has been drift-corrected -- by
[COMET](plugin:Analysis/Drift/COMET) or [RCC](plugin:Analysis/Drift/RCC) --
and saved, its drift curve, kept in the file, is subtracted from the target frame by frame:
each localization is moved by the drift of its own frame.  The two files
must cover the same frames; one of other frames is refused.

**5. Images.**  An image is moved by its placement: the image keeps its
pixels, and the picture looks each of them up through the transformation
when it is drawn.  Only transformations that keep a grid of pixels a grid
can do that -- shift, rotation, scale, mirror and any
[affine transformation](plugin:Panels/Glossary) -- so a
[projective](plugin:Panels/Glossary) or polynomial map is refused for an
image (the image would have to be resampled), as are a shift in z and a
drift.

The **Preview** draws the target before (pale) and after (strong) over the
rest of the table, or the image's outline where it was and where it goes,
without moving anything.

## In detail

**Which table.**  The localizations, whatever the layer shows: every
localization of the chosen *file* (or all files) and *channel* (or all)
moves, whatever the layers' filters show, and merged blinks are merged again
from the moved table.

**The typed transformation.**  With $c$ the centre, $M$ the mirror, $S$ the
scale, $R$ the rotation by the angle $a$ and $t$ the shift,

$$\mathbf{x}' = R\,S\,M\,(\mathbf{x} - c) + c + t ,$$

where the rotation takes $(x, y)$ to $(x \cos a + y \sin a,\; -x \sin a + y \cos a)$,
the rotation of SMAP's TransformLocsManually.  z is only shifted.  The
operations are multiplied into one 4x4 matrix on $(x, y, z, 1)$, which is
what is kept, written and applied.

**A `_2ct.h5`.**  Its map $H$ is from the second channel to the first in
camera pixels.  For a table in nanometres with the camera pixel $p_x, p_y$
(the file's pixel size, else the table's), the map in nanometres is
$S H S^{-1}$ with $S$ the diagonal of $p_x, p_y, 1$.  If its last row is
$(0, 0, 1)$ it is affine, and is kept as one (it can then move an image
too); otherwise it is projective.  A `_2ct.h5` fitted with the third-order
polynomial is applied as that polynomial, in camera pixels.  Its inverse is
the polynomial fitted the other way, not an exact inverse.

**What does not change.**  The precisions and widths (`xy_err_nm`,
`sigma_nm`) stay as they are, under a scale too: a scale corrects where the
localizations are, not how well they were measured.  The detection positions
(`peak_x_nm`, `peak_y_nm`) stay, as a drift correction leaves them: they are the
record of what was found where.  A table saved in pixels has its `x_pix` and
`y_pix` moved with `x_nm` and `y_nm`.

**The drift of another file.**  The curve comes from the drift COMET or
RCC kept in a localization file the session saved after running them, or
from the `/drift` group of a `_driftc.hdf5` that `smappy-drift` writes.  The target's frames must lie
within the curve's, and its last frame must be within 1% of the curve's
frames (at least 10 frames) of the curve's last: the last frames of a movie
may hold no localizations in one channel, but anything more is another
acquisition.

**Images.**  The image keeps its grid -- the corner `x0`, `y0` and the pixel
size the Render tab shows -- and the placement is a 3x3 affine matrix
applied after it.  Each pixel of the picture is drawn from the image pixel
its centre maps back to (nearest neighbour), so nothing is interpolated.
The placement belongs to the image, so every layer that shows it moves, and
it is saved with the localization file.

**What is recorded.**  The table's metadata gets a line in
`transformations` for every move: when, the name, which rows and how many,
and the transformation itself (or, for a drift, the file it came from).  An
image move and a table move are each one step of the undo.

## Parameters

### file
A table of several files (*File > Add*) moves one file at a time by
default: two cameras opened together are two files.

### channel
For a table with a `channel` column, such as the two halves of a split
chip fitted as one: move only the second.

### typed.angle_deg
A few degrees at most between two cameras; 90 or 180 for a camera
mounted turned.

### typed.scale
For a wrong pixel size, the right size divided by the one used, about
the origin (*about*: origin) -- SMAP's ChangeCamPixelsize.

### typed.centre
The middle of the target turns it in place; the origin is right for a
pixel-size correction, which scales the coordinates themselves.

### keep_as
A name already in the list is replaced.

### write_to
The `_T.h5` file holds the transformation as text (JSON) in an HDF5
file: a script reads it with `smappy.transformation.load_transformation`,
or with h5py and json alone.

## Output

* **The table**, with the target's coordinates moved, or **the image**,
  moved by its placement.
* **The text**: which file and channel, how many localizations, and the
  transformation -- its shift, rotation and scale.
* **The figure**: the target before and after, as the Preview shows it.  A
  good registration lays the moved channel onto the other; one that is off
  shows two copies of each structure side by side.

## Differences from SMAP

Replaces SMAP's (Ries 2020) ApplyTransform, TransformLocsManually, Mirror
and ChangeCamPixelsize.

* **An image is moved, not resampled.**  SMAP's ApplyTransform wrote a
  transformed copy of the TIFF (`_T.tif`); here the image's placement
  changes.  Nothing is lost, and undo is exact; the projective map is
  refused for an image rather than resampled.
* **One order and one centre.**  SMAP's TransformLocsManually shifted first
  and then scaled and rotated about the centre of the view; here mirror,
  scale and rotation are about the middle of the target, the origin or a
  typed point, and the shift comes last.
* **Mirror** turns the target over its centre.  SMAP's Mirror turned over
  the middle of a 256-pixel chip, or of half of it (*top*, *left*, ...); here
  *file* and *channel* say what moves.
* **The drift of another file** is new: SMAP corrected each file on its own.
  For consecutive files corrected separately, the drift plugins' *reference*
  sets the curve to zero at the first or the last frame, so the two files
  meet where they join.

## References

* Ries J (2020). SMAP: a modular super-resolution microscopy analysis
  platform for SMLM data. *Nature Methods* 17, 870-872.
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
