# smappy

Single-molecule localization microscopy in Python, with a Qt GUI: camera
conversion, peak finding, maximum-likelihood fitting with a Gaussian or
experimental (cubic-spline) PSF, rendering, grouping, drift correction and bead
PSF calibration -- the SMAP workflow, ported.  Reads SMAP `_3Dcal.mat`
calibration files, Micro-Manager TIFF stacks and NDTiff datasets
(pycro-manager).

**New to smappy?**  The [tutorials](https://ries-lab.github.io/SMAPpy/) are
guided tours of a few minutes each, with subtitles and a voice, on simulated
data you can follow along with.  They are generated from the program itself
(`python -m smappy.tutorial`), so they show the GUI as it is.

## Install

    pip install smappy-smlm

That is the whole install: the GUI comes with it, and there are binary wheels
for CPython 3.9-3.13 on macOS (Intel and Apple silicon), Linux x86_64 and
Windows, so nothing is compiled.  The distribution is `smappy-smlm` because
`smappy` on PyPI is an unrelated package; the import name is `smappy` either
way.  [`uv`](https://docs.astral.sh/uv/) installs the same wheels from the same
index, several times faster, and can keep the GUI in an environment of its own:

    uv tool install smappy-smlm        # `smappy-gui` on PATH, isolated
    uvx --from smappy-smlm smappy-gui  # run it without installing at all

Qt costs about 400 MB, which a headless fitting machine pays for nothing --
though it imports nothing either, since Qt is loaded inside `smappy-gui` and
not at `import smappy`.  `[gpu]` adds wgpu for the 3D viewer's GPU engine.

From a checkout, which builds the C++ extensions.  On Apple silicon,
`/usr/bin/python3` is the native arm64 one:

    /usr/bin/python3 -m venv .venv
    .venv/bin/python -m pip install -e .

Keep the comment off that first line if you paste it: `venv` takes *several*
target directories, so a trailing `# native arm64 on Apple silicon` makes a
venv called `#`, one called `native`, and four more.  Editable installs need
pip 21.3 or newer -- an older one falls back to `setup.py develop`, which
current setuptools refuses to run -- so upgrade pip inside the venv first if it
came with an old one.

## Start the GUI

    smappy-gui                      # or: smappy-gui FILE.hdf5

That is the command after `pip install smappy-smlm` or `uv tool install`, which
put it on PATH.  A checkout installed into a venv of its own does not: either
activate the venv (`source .venv/bin/activate`), which puts all six
`smappy-*` commands on PATH along with that venv's `python`, or call the one
you want by path and leave the rest of the shell alone:

    .venv/bin/smappy-gui            # or link it: ln -s "$PWD/.venv/bin/smappy-gui" ~/bin/

Equivalently, without the console script:

    python -m smappy.gui.app FILE.hdf5

Two windows open: the render view, and a compact control window with four tabs.

* **Localize** -- fit an acquisition, Gaussian 2D or spline 3D.
* **Render** -- layers, and one field filtered at a time against its own
  histogram (drag the shaded range to the edge for "no bound"); contrast, gamma,
  LUT and colour-by are applied after rendering, so they change without
  re-rendering.
* **Analysis** -- drift correction (COMET or RCC), colour assignment for
  two-channel data, and whatever else is registered.
* **ROI** -- the regions drawn in the render window; they are saved inside the
  localization file.

File → Open / Add file / Save.  View → 3D view (Ctrl+3), Reset view (Ctrl+0).
Tools → ROI manager (Ctrl+R), bead calibration, dual-colour calibration.  The
render window's toolbar has *Save* (PNG as displayed, or a TIFF re-rendered at a
chosen pixel size with the pixel size in its resolution tags) and *ROI*
(left-click draws, right-click picks the kind, the line width, or clears).

Every plugin is callable without the GUI, and the GUI is generated from the
plugin's declaration -- nothing in the plugin or session layer imports Qt.  See
[GUI.md](GUI.md) for that contract and the rest of the architecture,
[NOTES.md](NOTES.md) for the design decisions and measurements.

## Command line

| command | what it does |
|---|---|
| `smappy-gui [FILE.hdf5]` | the GUI |
| `smappy-fit DATA OUT.h5 --camera camera.yaml --cal CAL_3dcal.mat` | fit an acquisition |
| `smappy-live DATA OUT.h5 ...` | fit while the microscope writes |
| `smappy-drift OUT.h5` | drift-correct a saved file |
| `smappy-calibrate /path/to/bead_acquisitions` | bead PSF calibration |
| `smappy-view FILE.h5` | the older matplotlib viewer |

The camera usually needs no flags: smappy recognises it from the file's own
tags (a serial number, a camera ID) and takes the numbers Micro-Manager does
not record -- the e-/ADU conversion, and the baseline an iXon never reports --
from its **camera database**, which also knows that those two depend on the
readout mode and reads the mode out of the metadata.  `src/smappy/data/cameras.json`
ships the Ries lab's cameras, each one a file can be recognised as; a lab's own go in `cameras.json` next to the
config file, or in a SMAP `*_cameras.mat` converted with
`python -m smappy.io.cameras_mat lab_cameras.mat cameras.json`.
`--camera-name` picks an entry for a file whose camera carries no tag to
recognise it by.

Anything can still be stated by hand, and what is stated wins: a YAML config
(`examples/camera_evolve512.yaml`, `examples/camera_andor_ixon897.yaml`) or
flags (`--pixelsize 0.127 --conversion 6.7 --offset 400`).

## Python

    import smappy

    locs = smappy.fit(data, out="OUT.h5",
                      camera={"conversion": 6.7, "offset": 400,
                              "pixelsize_um": 0.127},
                      calibration="..._3dcal.mat")
    smappy.view("OUT.h5")

`data` is a path to an acquisition (a Micro-Manager TIFF series or an NDTiff
directory -- which one it is follows from what is there), an image source, an
array of frames, or any iterable of `(first_frame, block)`, so images already in
memory need no file.  `camera` is a dict, a `CameraMetadata` or a YAML path;
`calibration` is a `_3dcal.mat`, and without one the fit is Gaussian and there
is no z.  The table is returned whether or not it is also written;
`collect=False` streams to the file alone, for an acquisition too long to hold
in memory.

`smappy.fit` only assembles the stages, and each is available on its own:

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

Rendering an image from a table, grouping localizations of the same emitter
across consecutive frames, and writing a picture without opening a window:

    from smappy.filter import LocFilter
    from smappy.render import FieldOfView, RenderSettings, DisplaySettings, render_locs
    from smappy.group import group, GroupSettings

    keep = LocFilter(locs, loc_precision_nm=(None, 20), logl_rel=(-2, 0))
    fov = FieldOfView.around(locs["x_nm"], locs["y_nm"], pixelsize=10.0)
    image = render_locs(locs, fov, RenderSettings(mode="precision"), select=keep)
    rgb = DisplaySettings(lut="hot", gamma=0.7).apply(image)

    grouped, group_index = group(locs, GroupSettings(dx=50.0, dt=1))
    smappy.save_image(locs, "image.png", pixelsize=10.0)      # needs Pillow

`mode` is `"hist"`, `"gauss"` (one sigma for all) or `"precision"` (sigma from
the localization precision, SMAP's default); `color_field` colours by z or any
other column instead of by density.  `grouped` carries the same columns,
combined by SMAP's per-column rules, plus `n_in_group`.

## Drift correction

Drift is estimated with [COMET](https://github.com/gpufit/Comet), which
maximises the overlap of localizations between time windows -- no fiducials, no
reference structure.  COMET is somebody else's published method, MIT licensed;
the parts smappy calls are vendored in `src/smappy/_comet`, including its cost
function, which is compiled with smappy rather than with numba, so a plain
install drift-corrects with nothing else added.  **Cite COMET** if you publish
work that used it.

    smappy-drift OUT.h5 \
        --filter loc_precision_nm - 15 --filter logl_rel -2 - \
        --filter z_nm -300 300

The default is **the best estimator measured so far**: the drift is fitted as a
cubic B-spline in time (no time windows, no interpolation afterwards) from
grouped localizations, one per blink -- 4 s on a 410 k localization dataset
against 2:33 for free per-window vectors, and 0.6-0.7 nm noise per axis against
1.9-5.2.  `--per-window` and `--ungrouped` go back to either older behaviour;
`--knot-frames` sets how finely the spline can bend.  `--rcc` estimates by
redundant cross-correlation instead, an independent second opinion that agrees
to about its own noise.  Filter before estimating, and **include a z cut**:
without one the axial drift follows the out-of-focus tail.

The estimate uses only the localizations that pass `--filter`, and is then
subtracted from **all** of them: a filter is a view, the correction is a
coordinate change.  The result is `OUT_driftc.h5`, an ordinary localization file
with the drift curve in a `/drift` group.  From Python, `correct_drift(locs,
DriftSettings(...), select=keep)`.  See [NOTES.md](NOTES.md) for the
measurements, the noise floor they are judged against, and what did *not* help.

## Bead PSF calibration

Tools → bead calibration in the GUI, or `smappy-calibrate` on its own: add files
or directories, pool their stacks, review the automatically selected beads,
exclude some and rebuild.  The window is Qt (`--tk` still opens the old Tk one).

    smappy-calibrate /path/to/bead_acquisitions
    smappy-calibrate PATHS --out CAL.h5           # headless

The native calibration HDF5 loads through `load_spline_calibration` and
`smappy-fit --cal`.  Calibration uses pixels laterally and nanometres axially;
no camera pixel size or mirroring is required.  Split-frame dual-colour
calibration builds a paired PSF model and a projective transformation, from the
mode selector or `--layout 'up-down mirrored' --main-channel lower`.  See
[bead calibration](docs/bead_calibration.md) and [dual-colour
calibration](docs/dual_color_calibration.md).

A two-colour experiment in 2D needs no PSF model, only the registration between
the halves, and in a ratiometric experiment that can be measured from the data
itself -- every molecule is imaged twice in the same frame.  *Localize →
Gaussian 2D 2C* will do it on the leading frames of the movie it is about to
fit, or take a transformation saved by *Analysis → Register → Calibrate
transform* (or a dual-colour bead calibration, which carries one).  No initial
shift, magnification or split position is needed; see [channel
registration](docs/channel_registration.md).

## ROI manager

Tools → ROI manager (Ctrl+R).  Four quadrants: the file, a zoom and the ROI,
with the files and the ROIs listed beside them.  ROIs belong to files; circles
and squares share a global size while polygons keep their own outline.  The
first analysis workflow finds cluster candidates, lets you include or exclude
them, computes localization count, mean precision and mean photons, and shows
histograms.  What an ROI sees is what the image shows: the layer's filters and
grouping.  ROIs, their flags and their evaluation runs are saved inside the
localization file.  See the [ROI manager guide](docs/roi_manager.md).

## Online: fit while the microscope writes

    smappy-live DATA OUT.h5 --camera camera.yaml \
        --cal CAL_3dcal.mat --update 3 --timeout 30

`DATA` is the growing Micro-Manager TIFF or NDTiff directory; it does not have
to exist yet.  The window opens as soon as the first frames appear and takes in
new localizations every `--update` seconds; the fit ends `--timeout` seconds
after the last frame is written.  Zoom, pan, the filter boxes, contrast and
grouping all work while it runs, and **an update changes none of them**.

For frames that are never written to a file -- pycro-manager handing each one to
a callback, a camera API returning them from a buffer -- `smappy.queue_source()`
is the same `ImageSource` interface: the producer `push`es, the pipeline reads.
`on_block(locs)` is called with each finished block as it comes out, which is
what a control loop needs (the localizations per frame measure the blinking
density); `chunk` sets how often that happens.  `LiveFit` is `live_view` without
a window, for a front end of your own.

## NDTiff

pycro-manager writes NDTiff: a directory with an `NDTiff.index` and one or more
`*NDTiffStack*.tif`.  The index gives, per image, the file and the byte offset of
its pixels, so images are read by seeking and the TIFF page chain is never
walked -- a 46 k-frame dataset is ready in 0.3 s rather than 5 s.  The reader is
a port of SMAP's MATLAB loader, including the parts that are in no
specification; records describing images that were never written are dropped
rather than read as noise, which is also what makes a growing dataset safe to
follow.

## Scripts

`scripts/*.py` run from a checkout without installing anything.  The `check_*`
ones exercise one stage against real data -- the stack metadata, the detection,
the fit, a single-channel or dual-colour calibration; `fit_dataset.py`,
`view_locs.py`, `drift_correct.py` and `live_fit.py` are the scripted forms of
the commands above, and `simulate_blinks.py` makes test data.

## Tests

    SMAPPY_TEST_CAL=/path/to/_3dcal.mat PYTHONPATH=src .venv/bin/python -m pytest tests/
