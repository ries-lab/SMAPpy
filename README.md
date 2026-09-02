# smappy

Python implementation of the SMAP single-molecule fitting pipeline: camera
conversion, filtering, peak finding, ROI cutting, maximum-likelihood fitting
with a Gaussian or experimental (cubic-spline) PSF, and streaming output to
HDF5.  Reads SMAP `_3Dcal.mat` calibration files, Micro-Manager TIFF stacks and
NDTiff datasets (pycro-manager).

See [NOTES.md](NOTES.md) for the design decisions and open questions.

## Install

    pip install smappy-smlm[viewer]

The distribution is `smappy-smlm` because `smappy` on PyPI is an unrelated
package; the import name is `smappy` either way.  From a checkout:

    /usr/bin/python3 -m venv .venv                 # native arm64 on Apple silicon
    .venv/bin/python -m pip install ".[viewer]"

That builds the C++ extensions and installs the `smappy-fit`, `smappy-live`,
`smappy-view` and `smappy-drift` commands.  For work on smappy itself, `-e` and
`pytest` instead; `scripts/*.py` run from a checkout without installing
anything.

## Use

    import smappy

    locs = smappy.fit(data, out="OUT.h5",
                      camera={"conversion": 6.7, "offset": 400,
                              "pixelsize_um": 0.127},
                      calibration="..._3dcal.mat")
    smappy.view("OUT.h5")

`data` is a path to an acquisition, an image source, an array of frames, or any
iterable of `(first_frame, block)` -- so images already in memory need no file.
A path may be a Micro-Manager TIFF series or an NDTiff dataset directory; which
one it is follows from what is there, and nothing above `open_stack` has to
know.
`camera` is a dict of the fields, a `CameraMetadata` or the path of a YAML
config, and overrides whatever the image metadata says; `calibration` is a
`_3dcal.mat`, and without one the fit is Gaussian and there is no z.  The table
is returned whether or not it is also written; `collect=False` streams to the
file alone, for an acquisition too long to hold in memory.

`smappy.fit` only assembles the stages, and they are equally available on their
own -- this is the same fit written out:

    from smappy.io.tiff import open_stack, camera_metadata
    from smappy.io.calibration import load_spline_calibration
    from smappy.detect import DoGFilter, DynamicCutoff, PeakFinder
    from smappy.psf import SplinePSF
    from smappy.pipeline import FitSettings, fit_stack

    source = open_stack("...MMStack_Default.ome.tif")
    camera = camera_metadata(source, overrides={"conversion": 6.7, "offset": 400,
                                                "pixelsize_um": 0.127})
    model = SplinePSF(load_spline_calibration("..._3dcal.mat"))
    finder = PeakFinder(DoGFilter(1.2), DynamicCutoff(1.7))

    locs, engine = fit_stack(source.frames(chunk=200), camera, finder, model,
                             FitSettings(roisize=13, output_unit="nm"))

To render an image from a localization table:

    from smappy.filter import LocFilter
    from smappy.render import (FieldOfView, RenderSettings, DisplaySettings,
                                render_locs)

    keep = LocFilter(locs, loc_precision_nm=(None, 20), logl_rel=(-2, 0))
    fov = FieldOfView.around(locs["x_nm"], locs["y_nm"], pixelsize=10.0)
    image = render_locs(locs, fov, RenderSettings(mode="precision"), select=keep)
    rgb = DisplaySettings(lut="hot", gamma=0.7).apply(image)

`mode` is `"hist"`, `"gauss"` (one sigma for all) or `"precision"` (sigma from
the localization precision, the default in SMAP).  Set `color_field` to colour
by z or any other column instead of by density.  Rendering and display are
separate on purpose: contrast, gamma and the colour map change without
re-rendering.

To merge localizations of the same emitter across consecutive frames:

    from smappy.group import group, GroupSettings

    grouped, group_index = group(locs, GroupSettings(dx=50.0, dt=1))

`grouped` carries the same columns, combined by SMAP's per-column rules
(positions weighted by precision, z by its own error, photons summed and their
errors added in quadrature, precisions added in inverse quadrature), plus
`n_in_group`.

To look at the result:

    smappy.view(locs)                # a table, or the path of a saved file

The image and the controls open as two windows.  The image window holds nothing
but the image, so it can be resized to whatever the screen allows -- any shape,
filled edge to edge: a wide window shows more x rather than putting bands beside
a square image, and pixels stay square throughout.  Since the rendered pixel
size follows the canvas, a larger window is a *finer* image, not a scaled-up
one.  Closing it closes both; closing the controls leaves the image alone.

Scroll or pinch to zoom about the cursor, `+`/`-` to zoom about the centre, drag
to pan, `r` to reset.  Panning and zooming re-render as they go, so a gesture
fills in what it exposes instead of dragging a stale image around.  Type a
minimum and maximum to filter on localization precision, z, PSF size, relative
log-likelihood and frame; an empty box means "no bound", and the data range is
shown beside each row.  A window opens with a precision cut at 25 nm, relative
log-likelihood above -1.5 and z within +-500 nm; each is written into its box,
so what has been filtered out is visible rather than hidden in a default.
The "grouped" box switches to the grouped table, which is built on first use and
keeps its own filter; "additive" switches field colouring to SMAP's composite,
where overlapping colours add (red over cyan saturates to white).

"colour by" selects a plain intensity image or one coded by z, frame,
localization precision or photons, with the range typed into the "colour" row
(empty ends fall back to the data's own).  The range is always explicit, so the
same z means the same colour at every zoom and after every block of a live fit;
the LUT follows the choice -- `hot` for intensity, `turbo` for a coded field.
Needs matplotlib (`pip install matplotlib`).

Or from the command line:

    smappy-fit DATA OUT.h5 \
        --camera camera.yaml --cal CAL_3dcal.mat --units nm

The camera is stated in a YAML config (`examples/camera_evolve512.yaml`) or
directly on the command line -- `--pixelsize 0.127 --conversion 6.7 --offset
400` does the same thing without a file, and either overrides what the image
metadata says.  A lab that keeps a SMAP `*_cameras.mat` can pass it with
`--cameras` for the conversion and the per-camera metadata rules, but nothing
requires one.  In Python the same layers are `camera_metadata(source, presets,
overrides)`, where `overrides` is a `CameraMetadata`, a dict or a YAML path and
wins over everything else.

## NDTiff

pycro-manager writes NDTiff: a directory with an `NDTiff.index` and one or more
`*NDTiffStack*.tif`.  The index is a flat table giving, per image, the file and
the byte offset of its pixels, so images are read by seeking and the TIFF page
chain is never walked: opening costs about 7 us per frame against the ~120 us a
page walk takes, so a 46 k-frame dataset is ready in 0.3 s rather than 5 s.  `open_stack` returns an `NDTiffSource` for such a directory and
an `ImageSource` for a Micro-Manager series; everything downstream is the same.

The reader is a port of SMAP's MATLAB loader (`shared/imageloaders/`), including
the parts that are not in any specification: the index table is zero-padded, the
bytes per pixel are more reliably derived from where the metadata starts than
from the declared pixel type, and an interrupted acquisition leaves records
describing images that were never written -- those are dropped rather than read
as noise.

That last rule is also what makes a growing dataset safe to read: a record is
used only once the bytes it points at are there.  Following an acquisition is
therefore just re-reading the index, with none of the care a growing TIFF page
chain needs, and `live_fit.py` takes an NDTiff directory exactly as it takes a
TIFF.

## Drift correction

Sample drift is estimated with [COMET](https://github.com/gpufit/Comet), which
maximises the overlap of localizations between time windows -- no fiducials, no
reference structure.  COMET is somebody else's published method, MIT licensed;
the parts smappy calls are vendored in `src/smappy/_comet`, so it needs no
separate install, and a corrected file records the method and its version in its
`/drift` group.  **Cite COMET** if you publish work that used it.

COMET's cost function -- the sum over every neighbour pair that is the whole
running time of a drift correction -- is compiled with smappy rather than with
numba, so there is nothing extra to install.  `pip install smappy-smlm[cuda]`
adds numba for COMET's NVIDIA GPU backend, which the CPU path rarely needs.

    smappy-drift OUT.h5 \
        --filter loc_precision_nm - 20 --filter logl_rel -2 - \
        --frames-per-window 500 --max-drift 300 --plot

The drift is estimated from the localizations that pass the `--filter` ranges --
the same limits the viewer takes -- and then subtracted from **all** of them,
including the ones the filter hides: a filter is a view, the correction is a
coordinate change.  The result is written as `OUT_driftc.h5`, an ordinary
localization file the viewer opens unchanged, with the drift curve kept in a
`/drift` group.

From Python:

    from smappy.drift import DriftSettings, correct_drift, save_drift_corrected

    keep = LocFilter(locs, loc_precision_nm=(None, 20), logl_rel=(-2, None))
    corrected, drift = correct_drift(locs, DriftSettings(segmentation_var=500),
                                     select=keep)
    save_drift_corrected("OUT_driftc.h5", corrected, drift)

`drift.drift[f]` is `(dx, dy, dz)` in nm for frame `f`, and `drift.plot()` draws
it.  z drift is estimated whenever the table has `z_nm`; `DriftSettings(use_z=
False)` keeps it lateral.

Nearly all the time is the optimizer, which evaluates a cost over every
neighbour pair a few hundred times: 1:17 for 410 k localizations and 314 M pairs
on the CPU backend (46 k frames, 92 time windows), down from 13 minutes -- see
[NOTES.md](NOTES.md) for the measurements, the noise floor they are judged
against, and what did *not* help.

`--group` estimates from grouped localizations instead, one per blink: 314 M ->
21 M pairs and the whole correction takes **5 s**, agreeing with the full
estimate to about 1 nm (median) while being twice as noisy per window, which
costs a few percent of resolution.

**`--spline --group` is the best estimator measured so far**: the drift is
fitted as a cubic B-spline in time (no time windows, no interpolation
afterwards) from grouped localizations.  4 s on the clathrin dataset against
2:33 for free per-window vectors, 0.6-0.7 nm noise per axis against 1.9-5.2, and
a better image out of sample.  `--knot-frames` sets how finely it can bend; the default (2000) is the better
all-round choice, and finer settings buy a better-resolved transient at the
start of an acquisition at the cost of spurious wiggle where the density has
bleached away -- see [NOTES.md](NOTES.md).

`--rcc` estimates the drift by redundant cross-correlation instead -- an
independent method (ported from SMAP's `finddriftfeature`), useful as a second
opinion.  On the clathrin dataset, with both estimating from grouped
localizations and at matched smoothing, the two agree to 1.2 / 1.7 / 1.6 nm rms
in x / y / z -- the level of their own noise (0.6-0.9 nm each).

`--two-stage` runs the grouped pass first and then an ungrouped one over a 30 nm
radius, which is **~9x faster than the single pass for ~99% of the improvement**
-- and, because the fine pass is bounded by its own radius, it cannot produce
the runaway time window the single pass occasionally does.

Filter before estimating, and **include a z cut**: without one the axial drift
follows the out-of-focus tail (`z_err_nm` has a 95th percentile of 108 nm).
`--filter logl_rel -2 - --filter loc_precision_nm - 15 --filter z_nm -300 300`
is a reasonable set for a 3D dataset.

## Online: fit while the microscope writes

    smappy-live DATA OUT.h5 --camera camera.yaml \
        --cal CAL_3dcal.mat --update 3 --timeout 30

`DATA` is the growing Micro-Manager TIFF, or the directory it is being written
into; it does not have to exist yet.  The window opens as soon as the first
frames appear and takes in new localizations every `--update` seconds; the fit
ends `--timeout` seconds after the last frame is written, which is how an
acquisition stops.  `OUT.h5` is written throughout and is the result.

Everything the offline viewer offers works while this runs -- zoom, pan, the
filter boxes, contrast, grouping -- and **an update changes none of them**: new
localizations appear inside the view being looked at, under the bounds already
typed.  The frame comes from the camera field of view, so the image does not
rescale as data arrives.  Grouping cannot be extended, so the grouped table is
marked stale and rebuilt when it is next asked for.

From Python:

    from smappy.live import LiveSettings, live_view

    live_view(directory, camera, finder, model, FitSettings(output_unit="nm"),
              output="OUT.h5", live=LiveSettings(update_seconds=3.0))

`LiveFit` is the same thing without a window: it runs the pipeline in a thread
and queues finished blocks, for a different front end or a headless run.

### Frames that are never written to a file

A control program may have the images already -- pycro-manager hands each one to
a callback, a camera API returns them from a buffer.  `QueueSource` is the same
`ImageSource` interface for that: the producer pushes, the pipeline reads.

    source = smappy.queue_source(shape=(512, 512))

    source.push(image)               # from the acquisition thread, or a hook
    source.close()                   # the acquisition ended

    smappy.live_view(source, camera, finder, model, settings, output="OUT.h5")

Frames are numbered as they are pushed; `push(image, first_frame=n)` states the
acquisition's own number instead, and a gap in the numbering ends a block rather
than being papered over.  `maxsize` bounds the queue for a producer that can
outrun the fit, which then waits -- the only honest answer when the alternative
is growing until memory runs out.

The lower level is there too: drive `LocalizationEngine` directly and
`push(frames)` returns localizations once enough ROIs have accumulated, `flush()`
forces a partial block.  Nothing asks how many frames there will be.

### Reading the result while it is still being fitted

`on_block(locs)` is called with each finished block as it comes out, which is
what a control loop needs -- the localizations per frame are a measure of the
blinking density, and the density is what the activation laser is there to hold
steady:

    def density(locs):
        frames = locs["frame"]
        per_frame = len(locs) / (frames.max() - frames.min() + 1)
        ...                          # act on it

    smappy.fit(data, out="OUT.h5", camera=camera, calibration=cal,
               chunk=25, on_block=density)

**`chunk` sets how often that happens**, because a block is fitted at the end of
a chunk of frames: 25 frames at 100 ms is a reading every 2.5 s.  Smaller chunks
cost a little throughput and buy a shorter loop.  On a real dSTORM acquisition
this reads 109, 107, 105, 104, 103, 98, 92, 93 localizations per frame over the
first 200 frames -- the density decaying as the dye bleaches, which is the signal
to act on.

`progress(engine)` is the cheaper hook: it gives the running counts, including
`stats["candidates"]`, the *detected* spots.  That number needs no fit at all, so
it is available sooner and is the better control signal when the point is
density rather than positions.

### A window is not the only output

`show` and `live_view` open a matplotlib window and want the main thread, which
a program with its own event loop cannot give them.  Two ways round it:

    smappy.save_image(locs, "image.png", pixelsize=10.0)   # no window at all

`save_image` takes a table or a saved file, and the same `RenderSettings`,
`DisplaySettings` and filter the viewer takes, so what it writes is what the
viewer would show.  It needs Pillow.

For a window, run `smappy-view FILE` as a separate process.  And `LiveFit` is
`live_view` without a window: the fit in a thread, finished blocks on a queue,
for a front end of your own.

## Scripts

| script | what it checks |
|---|---|
| `check_calibration.py` | spline coefficients against the bead stack in the same file |
| `check_stack.py` | what the image metadata provides, and what is missing |
| `check_detection.py` | filtering, peak finding and ROI cutting on real frames |
| `check_fit.py` | spline fits on real data, with and without the mirror flip |
| `fit_dataset.py` | the whole pipeline, to HDF5 (`smappy-fit`) |
| `view_locs.py` | opens the viewer on a saved localization file (`smappy-view`) |
| `drift_correct.py` | drift-corrects a saved file with COMET (`smappy-drift`) |

## Tests

    SMAPPY_TEST_CAL=/path/to/_3dcal.mat PYTHONPATH=src .venv/bin/python -m pytest tests/
