---
version: "2"
covers: [smappy.metadata.file_frame_time, smappy.plugins.time_course.frame_windows, smappy.plugins.time_course.window_edges, smappy.plugins.time_course.rate, smappy.plugins.time_course.window_statistic, smappy.plugins.time_course.tag_in_windows, smappy.plugins.time_course.draw_panels, smappy.plugins.time_course.draw_one, smappy.plugins.time_course.TimeCourse.run]
---

## What it does

Draws how the data change during the acquisition: the number of
localizations per frame, the median of any column -- photons, `background`,
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
tags belong to the acquisition and are drawn once, in grey.  To look at a
tag in detail, beside the localization rate on a second axis, use
[Image Tags](plugin:Analysis/Process/Image Tags).

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
over the frames it covers.  For a column, the median (or mean, with
*statistic*) of the localizations in the window; a window with fewer than
10 gives no point.  For an image tag, its mean over the window's frames.

**3. Drawing.**  A panel per series, one under the other on the same time
axis -- or, with *draw* on *one graph*, every series in one graph, each
divided by its own largest value, so their shapes can be compared.

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
axis is relabelled.  A table with a clock of its own (MINFLUX's `time_s`)
is cut into equal windows of that clock instead, and its rate is per
second.

**Files.**  With *x* on *file number*, one point per file of the session
(`filenumber`), every localization of the file in it; the rate is per file.
The image tags are per frame and are not drawn against files.

**The text** gives, per series and layer, its first and last value and the
change between them in percent: the one number to read when the question is
whether the acquisition held.

## Parameters

### series
The localization rate and photons are where to start.  Add `background`
when the activation laser is being raised, `xy_err_nm` or `sigma_nm` when
the focus is in doubt, and the piezo tag beside them to see whether the
focus lock moved.

### windows
100 suits most movies.  Fewer windows give smoother lines and miss a short
event; more show a short event and more noise.  Each window should hold a
few hundred localizations for a median to be steady.

## Output

The figure: a panel per series, or one graph of the shapes, with a line per
visible layer.  The text: the windows, and per series and layer the first
and last value and the change.  A good acquisition has a rate that falls
slowly and smoothly, photons and precision that stay flat, and a piezo that
moves only as much as the focus lock needs.

## Differences from SMAP

Based on SMAP's `Analyze/measure/StatsVsTime`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).  The main changes:

* Any column can be a series, and the image tags too; SMAP drew a fixed set
  (photons, lifetime, background, PSF width, precision, the falloff frame).
* A column's value in a window is its median; SMAP fitted an exponential to
  the photons and took the maximum of the background, PSF width and
  precision histograms.
* Each visible layer is a line, with its own filter and grouping; SMAP drew
  the layers, or the grouped against the ungrouped table.
* The windows are whole frames.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
