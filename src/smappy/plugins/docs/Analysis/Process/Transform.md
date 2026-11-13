---
version: "2"
covers: [smappy.transformation.typed, smappy.transformation.transform_table, smappy.transformation.from_channel_transform, smappy.transformation.check_same_frames, smappy.transformation.save_transformation, smappy.transformation.load_transformation, smappy.transformation.transform_image, smappy.transformation.channel_regions, smappy.transformation.region_nm, smappy.plugins.transform.recorded_moves, smappy.plugins.transform.recorded_transformation, smappy.plugins.transform.channel_rects, smappy.drift.Drift.apply, smappy.drift.Drift.referenced, smappy.plugins.transform.Transform.run, smappy.plugins.transform.drift_of]
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
* a widefield image of a split chip should lie under the localizations of
  a two-channel fit: its second half moved onto the first by the same map
  that moved the localizations;
* a widefield or transmitted-light image was taken on another camera or with
  the stage moved, and should sit under the localizations;
* two cameras recorded at the same time, and the
  [drift](plugin:Panels/Glossary) measured in one file is the drift of the
  other too.

What moves is the **target**.  For localizations the coordinates themselves
change: `x_nm`, `y_nm` and, if the table has it, `z_nm`.  The change is
logged and can be undone, and everything after it -- measurements, exports,
ROIs -- sees the moved coordinates, as the picture does.  An **image layer**
is recalculated into a **new image**, `<name>_T` as SMAP called it; the
original stays as it was, and the layers that showed it show the new one.  Of an image, by default only the **secondary channel** moves,
onto the main one -- the half of the chip a channel map is about.

What it is moved by is the **source**: values typed in (shift, rotation,
scale, mirror), a **named transformation** the session keeps, a **move
recorded in a loaded file** (the overlay of a two-channel fit), a
**transformation file** (such as a bead calibration), or the **drift of
another file**.

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
as*, and is then applied by name.  The list belongs to the data, not to
the GUI: it is saved with the localization file and comes back when the
file is opened, but a workspace (*File > Workspace*) does not carry it.  To
use a transformation with other files, write it to a file (*write to*) and
apply that.

**3. A move recorded in a loaded file.**  Every move of a table is written
into it, with the transformation itself.  A movie fitted as two channels and
overlaid by [Overlay channels](plugin:Analysis/After fit/Overlay channels)
records the map that put its secondary half onto the main one; on *auto*
this source takes the last such overlay, so an image of the same chip
follows its localizations without the map being looked for.  Any other
recorded move can be chosen by name and time.

**4. A transformation file.**  Either a `_2ct.h5` written by
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

**5. The drift of another file.**  Two cameras that record at the same time
see the same drift.  When one file has been drift-corrected -- by
[COMET](plugin:Analysis/Drift/COMET) or [RCC](plugin:Analysis/Drift/RCC) --
and saved, its drift curve, kept in the file, is subtracted from the target frame by frame:
each localization is moved by the drift of its own frame.  The two files
must cover the same frames; one of other frames is refused.

**6. Images.**  An image is resampled: each pixel of the new image is taken
back through the transformation to where it came from, and the original is
read there, interpolated between its four nearest pixels.  Every kind of
transformation can do that, a projective or polynomial bead map too.  The
new image lies on the original's pixel grid, so it lies under the
localizations as the original does.  It joins the session's images as
`<name>_T`, and every image layer that showed the original shows it instead,
at the same plane -- what that layer was for; a layer showing any other image
is left alone.  The original stays among the images, for a layer to show
again.

With *what moves* on *secondary channel onto main* (the default for a map, a
recorded move or a file), the image is read as a split chip: only the pixels
in the secondary channel's rectangle are moved, and the main channel's are
cut out into an image of their own, unmoved, `<name>_main` -- the two
channels as two images, one dye each, as the localizations are; show the
main one in a second layer to see both.  The rectangles come with
the map (a `_2ct.h5` keeps where the channels lay on the chip when it was
measured), or else from what the loaded two-channel fit recorded.  An image
of the secondary half alone gives the moved image only.  Typed values move
the whole image.

```figure A split chip: squares in the main half (left) and their partners in the secondary half (right), turned and shifted as a bead map says.  The moved secondary half (magenta) lands on the main squares (green), which turn white where they meet.
from smappy.transformation import transform_image, Transformation, region_nm
from smappy.calibrate.dual import map_points
a = np.radians(2.0)
h = np.array([[np.cos(a), np.sin(a), -66.0], [-np.sin(a), np.cos(a), 3.0], [3e-6, 0.0, 1.0]])
rng = np.random.default_rng(2)
main = np.column_stack([rng.integers(8, 56, 25), rng.integers(8, 56, 25)])
partner = np.rint(map_points(np.linalg.inv(h), main.astype(float))).astype(int)
plane = np.zeros((64, 128), np.float32)
for c, r in np.r_[main, partner]:
    plane[r - 1:r + 2, c - 1:c + 2] = 1.0
grid = np.array([[100.0, 0, -50.0], [0, 100.0, -50.0], [0, 0, 1.0]])
s = np.diag([100.0, 100.0, 1.0])
t = Transformation("projective", s @ h @ np.linalg.inv(s))
moved = transform_image(plane, grid, t, region_nm((64, 0, 64, 64), (100.0, 100.0)))
kept = transform_image(plane, grid, Transformation(), region_nm((0, 0, 64, 64), (100.0, 100.0)))
fig.set_size_inches(7, 2.4)
left, right = fig.subplots(1, 2)
extent = (-50, 12750, 6350, -50)
left.imshow(plane, cmap="gray", extent=extent)
left.set_title("the image", fontsize=9)
rgb = np.stack([moved, kept, moved], axis=-1).clip(0, 1)
right.imshow(rgb, extent=extent)
right.set_title("main green, secondary moved magenta", fontsize=9)
for ax in (left, right):
    ax.set_xlabel("x (nm)")
left.set_ylabel("y (nm)")
```

The **Preview** draws the target before (pale) and after (strong) over the
rest of the table, or the new image -- the main channel green and the moved
one magenta -- under the localizations, without adding anything.

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
too); otherwise it is projective.  The rectangles of its channels on the
chip, when it keeps them, go with it, into the record of every table it
moves.  A `_2ct.h5` fitted with the third-order
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

**Images.**  The new image has the original's corner `x0`, `y0`, pixel size
and planes.  For the centre $\mathbf{p}$ of each of its pixels the source
point is $T^{-1}(\mathbf{p})$, with $T$ the transformation (for a polynomial
map, the polynomial fitted the other way); the original is read there by
bilinear interpolation, every plane of a stack alike.  A source point
outside the image, or outside the moving channel's rectangle, gives 0.  A
chip rectangle of $w \times h$ pixels starting at pixel $(x, y)$ covers
$(x - \frac{1}{2})\,p_x$ to $(x + w - \frac{1}{2})\,p_x$ in x, pixel $i$
centred on $i\,p_x$ as a localization's `x_nm` is, with $p_x$ the camera
pixel of the table (else of the image).

**What is recorded.**  The table's metadata gets a line in
`transformations` for every move: when, the name, which rows and how many,
and the transformation itself (or, for a drift, the file it came from).  A
table move is one step of the undo; an image move is not -- the original is
still among the images, and a layer is pointed back at it in the Render
tab.  The new image's metadata keeps the transformation and which
part moved.

## Parameters

### file
A table of several files (*File > Open*, *Add*) moves one file at a time by
default: two cameras opened together are two files.

### channel
For a table with a `channel` attribute, such as the two halves of a split
chip fitted as one: move only the second.

### recorded
On *auto* the last overlay of channels; a table with several files lists
each file's moves.  Choose the transformation file instead when the
image's movie was overlaid in another session and the table was not kept.

### part
*the whole image* for an image of one camera, or of a secondary half alone
when the map keeps no chip rectangles; the default needs them, from the map
or from the loaded fit.

### image_to
The TIFF holds the moved image only, with its pixel size; its corner is not
kept, so open it beside a table whose image starts at the chip's corner.

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

* **The table**, with the target's coordinates moved, or **new images**:
  the moved one, `<name>_T`, shown by the layers that showed the original,
  and for a split chip the main channel, `<name>_main`.
* **The text**: which file and channel, how many localizations, and the
  transformation -- its shift, rotation and scale.
* **The figure**: the target before and after, as the Preview shows it.  A
  good registration lays the moved channel onto the other; one that is off
  shows two copies of each structure side by side.  For an image, the moved
  channel in magenta on the main in green: where they agree it is white.

## Differences from SMAP

Replaces SMAP's (Ries 2020) ApplyTransform, TransformLocsManually, Mirror
and ChangeCamPixelsize.

* **An image is resampled**, as SMAP's ApplyTransform wrote a transformed
  `_T.tif`; here it is `<name>_T` among the session's images, shown at once
  by the layers that showed the original, and a file on request (*save
  image*).  By default
  only its secondary channel moves, the main one cut out beside it.  The
  map can also be the move a loaded two-channel fit records.
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
