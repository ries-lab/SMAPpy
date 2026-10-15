---
version: "1"
covers: [smappy.movie.frame_steps, smappy.movie.z_steps, smappy.movie.restrict, smappy.movie.stack_frames, smappy.movie.raw_plane, smappy.movie.raw_panel, smappy.movie.write_stack, smappy.movie.estimate_size, smappy.plugins.file.stack_plan, smappy.picture.render_picture]
---

## What it does

Makes a movie of the localizations over the acquisition, or a stack of
slices through z: the picture of the render window, drawn again and again
while one coordinate moves in steps.  Each frame of the stack is exactly
what [Export Image](plugin:File/Export/Image) saves with the
[layers'](plugin:Panels/Glossary) filters set to that step's frames (or z
range): the visible layers, the panels as the window lays them out, and the
scale bar, colour bars and labels over them.

Use it to show how a structure builds up during an acquisition, how
something moves in a live-cell experiment, how localizations are spread
over time (drift, bleaching, a cell that changed during the measurement),
or to go through a 3D structure slice by slice.

* **Along frames** -- a movie.  Each movie frame shows a *window* of
  camera frames, the next one starts a *step* later; *cumulative* shows
  everything from the first frame on, so the picture fills in.
* **Along z** -- a z-stack.  Slices of a *thickness*, one every *step*,
  from *from* to *to*.  It needs a table with z (`z_nm`).
* **Camera frames beside** -- on the left, the frames the fit kept with
  the localizations; on the right, the localizations of the same frames.
  What a single-molecule movie looks like next to what it gives.

It writes a TIFF stack (lossless: every pixel as rendered, with the pixel
size and the time or z step, which Fiji reads), an MP4 (H.264: plays in
slides and browsers, much smaller) or a GIF (plays everywhere, few colours).
The form says how many frames it will make and about how large the file
will be before it runs.

## How it works

```figure-setup
from smappy.layer_state import snapshot_layers
from smappy.movie import frame_steps, stack_frames, z_steps
from smappy.render import FieldOfView
from smappy.session import Session
from smappy.simulate import simulate
session = Session(simulate(n_frames=4000, seed=3))
layers = snapshot_layers(session.layers)
fov = FieldOfView.from_range(*session.full_view(), 50.0)
```

**1. The steps.**  Along frames, the first window starts at the first
frame shown, and a window follows every *step* frames until one reaches
the last frame: frames 0 to 999, 1000 to 1999, ... at a step and window of
1000.  A window longer than the step makes a running average, each frame
overlapping the next (SMAP's default was a step of 1000 and a window of
5000).  Along z, slices from *from* upwards until one reaches *to*.

**2. Each frame.**  Every layer is narrowed to the step's localizations --
its own filter, and the frame (or z) inside the step -- and the picture is
drawn as *Export Image* draws it.  Each frame's brightness is set by its
own contrast, as in the window, so a cumulative movie is visible from its
first frame rather than starting black.

```figure Four frames of a cumulative movie of a simulated structure: the picture fills in as the acquisition goes on.  Each is scaled by its own brightness.
fig.set_size_inches(7.5, 2.2)
axes = fig.subplots(1, 4)
steps = frame_steps(0, 3999, 50, cumulative=True)
steps = [steps[k] for k in (0, 3, 15, -1)]     # four of the 80
for ax, step, image in zip(axes, steps, stack_frames(layers, fov, steps, what="composite",
                                                     scale_bar=False, labels=False,
                                                     text=lambda s: f"frames 0-{s.hi - 1:.0f}")):
    ax.imshow(image, interpolation="nearest")
    ax.set_axis_off()
```

**3. Through z.**  The same with z: each slice shows the localizations
with `z_nm` in its range.

```figure Four slices 150 nm thick through the simulated structure, from below to above: its parts lie at different depths.
fig.set_size_inches(7.5, 2.2)
axes = fig.subplots(1, 4)
slices = z_steps(-300, 300, 150)
for ax, step, image in zip(axes, slices, stack_frames(layers, fov, slices, column="z_nm",
                                                      what="composite", scale_bar=False,
                                                      labels=False,
                                                      text=lambda s: f"z {s.lo:g} to {s.hi:g} nm")):
    ax.imshow(image, interpolation="nearest")
    ax.set_axis_off()
```

**4. Tracks.**  A layer drawn in the *tracks* render mode joins each
track's localizations with lines.  In a movie, a window would cut every
track to the window's frames; a *track tail* draws it instead over the
frames before "now" (the window's end): each particle is followed by the
path it has just taken.

**5. The file.**  The frames are written one by one as they are drawn, so
a movie longer than the memory can be made.  Stopping the run leaves no
file: half a movie is no movie.

## In detail

**The number of frames.**  Along frames, from the first frame shown $f_0$
to the last $f_1$, with a step $s$ and a window $w$,

$$n = \max\left(1,\ \left\lceil \frac{f_1 + 1 - f_0 - w}{s} \right\rceil + 1\right),$$

frame $k$ showing camera frames $f_0 + k s$ up to (not including)
$f_0 + k s + w$, or from $f_0$ when *cumulative*.  Along z the same, with
$z$ from *from* to *to*, the step and the thickness.  The last window or
slice may reach beyond the data.

**Merged blinks.**  A layer that shows merged blinks (one localization per
blink) places each blink at its first frame.

**The camera frames.**  A fit keeps a few dozen frames, evenly spaced (the
fitter's *raw frames*, 50 by default), not every one.  Each movie frame
shows the kept frame nearest to its window's last frame -- never their
average -- so the camera side changes less often than the localizations.
It is resampled onto the picture's grid and shown in grey from its 1st to
its 99.9th percentile.  A table without kept frames (one not fitted here,
or fitted with *raw frames* 0) is refused.

**Time.**  With *write time / z*, a movie frame says the seconds at its
window's end when the file's frame time is known, the frames otherwise; a
slice says its z range.  A TIFF along frames records the time between
movie frames (*step* times the frame time) for Fiji, and a z-stack its
step as the slice spacing.

**The formats.**  A TIFF holds every frame as rendered, 3 bytes a pixel, as
an ImageJ hyperstack (frames along T, slices along Z), with the pixel size
in µm and how it was made in its *Info* (Fiji: *Image > Show Info*).  An
MP4 is H.264 at a constant quality -- *high*, *medium* and *small file*
are x264's rate factors 18, 23 and 28 -- with both sides made even, as the
format needs; it is written by the `imageio-ffmpeg` package, installed
with smappy, and the form says so if it is missing.  A GIF has 256 colours per frame, which is fine for a
picture on black and coarse for smooth colour scales.  The sizes the form
shows for MP4 and GIF are rates measured on simulated movies; a dense
picture makes a larger file.

**The cost.**  Before a run, one frame is drawn and timed, and the time
for all of them is reported; a stack that would take more than a minute
asks first.

## Parameters

### frames.window
Longer than the step for a running average that smooths a sparse movie;
equal to it for frames that do not overlap.  A window of a few hundred to
a few thousand frames is typical: enough localizations per frame to see
the structure.

### frames.tail
Only layers in the *tracks* mode use it.  A few times a track's typical
length shows where each particle came from; much longer and the picture
fills with old paths.

### z.thickness_nm
Thicker than the step makes overlapping slices.  About the axial
precision (50 to 100 nm) is as thin as is useful.

### fps
10 to 25 frames per second for a smooth movie; slower to look at each
frame.

### quality
*medium* is close to what the eye can tell from the TIFF on a picture of
localizations; *high* for a movie that will be zoomed into.

## Output

The file, a line in the history, and in the file how it was made: the
layers, their settings and the steps.  A movie whose frames flicker in
brightness has frames with few localizations: a longer window, or
*cumulative*, steadies it.

## Differences from SMAP

Ported from SMAP's *MovieRunningWindow* and *BlinkingMoviePresentation*
(Ries 2020).  SMAP's movie took a step and a window and rendered each
frame of the ROI; it was saved as a TIFF scaled to the whole movie's
maximum.  Here:

* each frame keeps its own brightness, as the window shows it, so a
  cumulative movie does not start nearly black;
* z-stacks are made the same way, and tracks have a tail;
* the camera frames beside are the ones the fit kept in the localization
  file, a few dozen, rather than every camera frame;
* MP4 and GIF are written as well as TIFF, with the frame rate and
  quality set here rather than in a separate step.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
