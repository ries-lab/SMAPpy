---
version: "3"
covers: [smappy.metadata.file_frame_time, smappy.plugins.time_course.frame_windows, smappy.plugins.time_course.window_edges, smappy.plugins.time_course.rate, smappy.plugins.time_course.window_statistic, smappy.plugins.time_course.tag_times, smappy.plugins.time_course.tag_names, smappy.frametags.FrameTags, smappy.frametags.flatten, smappy.frametags.acquisition, smappy.plugins.image_tags.names, smappy.plugins.image_tags.summary, smappy.plugins.time_course.draw_panels, smappy.plugins.time_course.draw_one, smappy.plugins.time_course.TimeCourse.run]
---

## What it does

Draws how the data change during the acquisition: the number of
localizations per frame, the median of any attribute -- photons, `background`,
the [localization precision](plugin:Panels/Glossary) `xy_err_nm`, the PSF
width `sigma_nm`, `z_nm` -- and the image tags the microscope recorded (the
piezo position, the activation laser), all against frame, time or file.

An acquisition that degrades shows it here first: the density falls as the
dye bleaches, the background climbs with the activation laser, the photons
drop as the buffer is used up, the precision gets worse when the focus lock
lets go.  Look at it before trusting a number measured over the whole
movie, and before deciding which frames to keep.

Each visible [layer](plugin:Panels/Glossary) is a line of its own, read as
the layer shows it -- its [filter](plugin:Panels/Glossary), the drawn
[ROI](plugin:Panels/Glossary), and [merged blinks](plugin:Panels/Glossary)
if the layer merges them, when the rate is blinks per frame.  The image
tags belong to the acquisition and are drawn once, in grey, at every frame.

**The image tags.**  Micro-Manager writes the state of every device into the
metadata of every frame.  The few values that change are a record of what
happened while the camera ran: where the piezo held the focus, how long the
UV pulse was, what the focus lock's sensor read.  The fit keeps them with
the localizations, and they are offered as series named `tag: <name>`.  The
piezo position is the one to check: a focus lock that drifted, jumped or
ran into the end of its range shows there at once, while in the
localizations it shows only as a slow loss of quality.  The fit can draw it
by itself while it runs (*show image tags* in its output settings).  A
table loaded from another program has no image tags.

This plugin took over *Plot image tags* (formerly *Analysis/Process/Image
Tags*): a saved chain or workspace that names it opens this one, with its
tags as tag series and the localization rate beside them.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.plugins.time_course import (COUNT, Line, frame_windows, rate,
                                        window_statistic, draw_panels)
locs = simulate(n_frames=6000, seed=4)
frame = np.asarray(locs["frame"])
rng = np.random.default_rng(0)
tau = 2500.0                                         # bleaching, in frames
keep = rng.random(len(frame)) < np.exp(-frame / tau)
frame = frame[keep]
photons = np.asarray(locs["photons"], float)[keep] * (1 - 0.4 * frame / 6000)
edges, covered = frame_windows(int(frame.min()), int(frame.max()), 60)
x = 0.5 * (edges[:-1] + edges[1:])
start = rate(frame, edges, covered)[0] / np.exp(-x[0] / tau)
lines = [Line(COUNT, "", "C0", x, rate(frame, edges, covered)),
         Line("photons", "", "C0", x, window_statistic(frame, photons, edges))]
```

**1. Windows.**  The frames are cut into *windows*, each a whole number of
frames: 6000 frames in 60 windows are windows of 100 frames.

**2. A value per window.**  For the rate, the localizations in the window
over the frames it covers.  For an attribute, the median (or mean, with
*statistic*) of the localizations in the window; a window with fewer than
10 gives no point.  An image tag is not put in windows: it is one reading
per frame, with none of the counting noise a window averages away, and a
jump of one frame is what one looks for, so it is drawn at every frame it
was read.  A tag that is text (a laser *On* or *Off*) is drawn as steps
between its states.

**3. Drawing.**  A panel per series, one under the other on the same time
axis -- or, with *draw* on *one graph*, every series in one graph, each
divided by its own largest value, so their shapes can be compared.

```figure A focus lock that holds and then, at frame 4137, jumps by 0.25 µm: the piezo tag is drawn at every frame, so the jump stays a sharp step beside the rate in 60 windows.
from smappy.plugins.time_course import TAG
piezo = 50.0 - 0.3 * np.arange(6000) / 6000 + np.random.default_rng(2).normal(0, 0.004, 6000)
piezo[4137:] += 0.25
fig.set_size_inches(6.0, 3.4)
draw_panels(fig, [COUNT, TAG + "PIZStage-Position"],
            [lines[0], Line(TAG + "PIZStage-Position", "", "0.35",
                            np.arange(6000.0), piezo, dense=True)], "frame")
```

```figure A simulated acquisition whose dye bleaches with a time constant of 2500 frames and whose photons fall by 40 % over the movie: both trends come back (dashed: the truth).
fig.set_size_inches(6.0, 4.0)
draw_panels(fig, [COUNT, "photons"], lines, "frame")
rate_ax, photon_ax = fig.axes
rate_ax.plot(x, start * np.exp(-x / tau), "k--", lw=0.8)
photon_ax.plot(x, np.median(np.asarray(locs["photons"], float)) * (1 - 0.4 * x / 6000),
               "k--", lw=0.8)
```

## In detail

**Windows of frames.**  With $F$ frames from the first to the last and $W$
*windows*, each window is $w = \lceil F/W \rceil$ frames long, so there are
$\lceil F/w \rceil$ of them -- about $W$ -- and the last may be shorter.  A
window of a fractional number of frames (19.99) would hold 19 or 20 frames
and a steady rate would wobble by a twentieth, which is why it is whole
frames.  The rate is $n_k / f_k$, the localizations in window $k$ over the
frames $f_k$ it covers.

**Time.**  With *x* on *time (s)*, the frame number times the *frame time*,
which on *auto* is the file's [frame time](plugin:Panels/Glossary): the time
from one frame to the next where the file records it (Micro-Manager's
`ActualInterval-ms`, a time-lapse interval, SMAP's `timediff`, or the
fitter's *frame time*), else the exposure -- the same on a camera that
streams as fast as it can, and short by the camera's dead time on one that
does not; a file that records neither is given one in *File > File
properties*.  With several files open, the first file's.  The windows are still whole frames; only the
axis is relabelled, and the image tags are put on it the same way, frame
times frame time.  A table with a clock of its own (MINFLUX's `time_s`)
is cut into equal windows of that clock instead, and its rate is per
second.  A tag's frame is then put on that clock through the
localizations: each localized frame's median `time_s`, interpolated
linearly between them.  A tag frame before the first or after the last
localized frame has no time there and is not drawn (the text says so); a
table with a clock but no `frame` column draws no tags.

**Files.**  With *x* on *file number*, one point per file of the session
(`filenumber`), every localization of the file in it; the rate is per file.
The image tags are per frame and are not drawn against files.

**The text** gives, per series and layer, its first and last value and the
change between them in percent: the one number to read when the question is
whether the acquisition held.  For each image tag it gives its value at the
first and last frame and its range, and for a text tag its states in the
order they appeared (more than 4: their number).  The note the acquisition
was saved with (`comments.txt`) comes first, if there is one.

**Recording the tags, during the fit.**  The metadata of a frame is read
together with its pixels and flattened to `Device-Property` keys; user data
Micro-Manager wraps as `{"type": ..., "scalar": value}` is unwrapped, which
is how the camera's hardware clock becomes a tag.  Each frame is compared
with the first: a tag is kept from the first frame its value differs,
filled backwards with the first frame's value, so nothing has to be named
in advance.  Values that change by construction (the image number, a UUID,
the file name, the time the computer received the frame) are left out.  A
tag whose values read as numbers is kept as a number; a text tag is a state
until it has had more than 16 values, when it is an identifier and is
dropped.  At most 64 tags are kept.  They are saved in the localization
file beside the table (the group `/frame_tags`, the acquisition's static
metadata in `/acquisition`) and read back when it is opened.  During a fit
they are handed on every 5 seconds, so this plugin runs on a fit that is
still going.

## Parameters

### series
The localization rate and photons are where to start.  Add `background`
when the activation laser is being raised, `xy_err_nm` or `sigma_nm` when
the focus is in doubt, and the piezo tag beside them to see whether the
focus lock moved.  A tag series matches part of a name too, without regard
to case -- `tag: PIZ` is every tag whose name contains `piz` -- and `tag: *`
is every tag the table has.

### windows
100 suits most movies.  Fewer windows give smoother lines and miss a short
event; more show a short event and more noise.  Each window should hold a
few hundred localizations for a median to be steady.

## Output

The figure: a panel per series, or one graph of the shapes, with a line per
visible layer, and a grey line per image tag.  The text: the windows, per
series and layer the first and last value and the change, and per tag its
range or its states.  For a script, `data["lines"]` holds every line drawn,
its `x` and `y`.  A good acquisition has a rate that falls
slowly and smoothly, photons and precision that stay flat, and a piezo that
moves only as much as the focus lock needs: flat apart from slow, smooth
changes.  Steps, a trace that saturates at a limit, or a burst of noise mark
the frames whose z cannot be trusted.

## Differences from SMAP

Based on SMAP's `Analyze/measure/StatsVsTime`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).  The main changes:

* Any attribute can be a series, and the image tags too; SMAP drew a fixed set
  (photons, lifetime, background, PSF width, precision, the falloff frame).
* An attribute's value in a window is its median; SMAP fitted an exponential to
  the photons and took the maximum of the background, PSF width and
  precision histograms.
* Each visible layer is a line, with its own filter and grouping; SMAP drew
  the layers, or the grouped against the ungrouped table.
* The windows are whole frames.
* The image tags are SMAP's `Analyze/other/DisplayImageTags`, merged in.
  SMAP read only the tags named in the camera's `imagemetadata` setting and
  left out the frames where a tag was zero, which also hid a UV pulse that
  was off; here every tag that changes is kept, and a frame that was not
  read is not in the table, so a real zero is drawn.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
