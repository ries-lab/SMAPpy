---
name: smappy
description: Fit SMLM raw frames to localizations with SMAPpy, write HDF5, and view the super-resolution image.
---
# SMAPpy localization fitting (microclaw)

## What SMAPpy is

SMAPpy (`smappy`) is the Python port of the SMAP fitting pipeline: camera ADU →
photon conversion, difference-of-Gaussians filtering, peak finding, ROI cutting,
maximum-likelihood fitting with a Gaussian or experimental (cubic-spline) PSF,
streaming HDF5 output, drift correction, grouping, and an interactive viewer.
It reads SMAP `_3Dcal.mat` calibration files.

It is what turns a microclaw SMLM acquisition into coordinates. Microclaw
acquires; SMAPpy fits. Nothing in this skill drives hardware.

**SMAPpy is an optional dependency.** Check before promising anything:

```python
import importlib.util
available = importlib.util.find_spec("smappy") is not None
```

If it is missing, say so and fall back to the external-software table in
`load_skill(name="smlm")`. Do not attempt to install or build it mid-session —
it has C++ extension modules and building is a setup step, not a session step.
For the user to do later: `pip install smappy-smlm` (or
`pip install microclaw[analysis]`). The distribution is `smappy-smlm` while the
import name is `smappy` — plain `smappy` on PyPI is an unrelated package, so
never tell a user to install that.

---

## The three ways to call the fitter

All three drive the same object, `LocalizationEngine`. Pick by where the frames
come from.

### 1. A finished dataset on disk

```python
import smappy

locs = smappy.fit(tiff_path, out=out_h5,
                  camera={"conversion": 0.49, "offset": 100.0,
                          "pixelsize_um": 0.106},
                  calibration=cal_3dcal_mat, units="nm")
print(len(locs), locs.metadata["stats"])
```

`out` streams to HDF5 as the fit runs; the table is returned as well, and
`collect=False` streams to the file alone for an acquisition too long to hold in
memory. `calibration=None` gives a Gaussian fit and no z. Every stage is also
available on its own (`open_stack`, `PeakFinder`, `SplinePSF`, `fit_stack`) —
`smappy.fit` only assembles them.

**NDTiff is read directly** — pass the dataset directory `run_timelapse` wrote
and smappy reads it through its `NDTiff.index`, no `export_dataset_as_tiff` pass
and no full copy of the raw data. A Micro-Manager TIFF series works the same
way; `smappy.fit` picks the format from what is there. Export to TIFF only when
something *else* needs it (ImageJ, a classifier), never to feed smappy.

### 2. Images already in memory

`smappy.fit` takes an array of frames `(n, y, x)`, or any iterable of
`(first_frame_index, block)`. A file is not required:

```python
locs = smappy.fit(stack, camera={"conversion": 0.49, "offset": 100.0,
                                 "pixelsize_um": 0.106},
                  calibration=cal_3dcal_mat)
```

Raw images carry no metadata, so coordinates are relative to the image unless
`roi=(x, y, w, h)` says where on the chip it sat — pass it when the localizations
have to line up with anything else.

`camera.require()` raises a `ValueError` naming what is missing rather than
producing silently wrong nm coordinates, so a camera is never half-specified.
Get the pixel size from `get_pixel_size()`; the conversion (e-/ADU) and offset
(ADU) come from the rig profile or the camera datasheet, never from a guess.

**The camera is stated as parameters, not as a file.** `overrides` takes a
`CameraMetadata`, a dict, or the path of a YAML config, and wins over both the
image metadata and any preset. A SMAP `*_cameras.mat` is optional — pass its
path as `camera_metadata(source, presets, overrides)` only where a lab keeps
one. On the command line the same three layers are `--camera CONFIG.yaml`,
`--pixelsize/--conversion/--offset`, and `--cameras CAMERAS.mat`. **Store the
rig's camera parameters in a microclaw rig profile and pass them as a dict**;
do not require the user to produce a MATLAB settings file.

To feed frames as they arrive rather than all at once, push them into a
`QueueSource` (route 3 below), or drive the engine directly:

```python
from smappy.pipeline import LocalizationEngine
engine = LocalizationEngine(camera, finder, model, settings)
locs = engine.push(block, first_frame=i)   # None until enough ROIs accumulate
locs = engine.flush()                      # force a partial block
```

`push` never asks how many frames there will be. Nothing in the pipeline needs
to know whether the acquisition is still running.

### 3. Fitting while the acquisition writes, with the image building up

```python
from smappy.live import LiveSettings, live_view

live_view(directory, camera, finder, model, FitSettings(output_unit="nm"),
          output=out_h5, live=LiveSettings(update_seconds=3.0))
```

`directory` is the NDTiff dataset or Micro-Manager TIFF being written, and it
does not have to exist yet — the call waits for it, so it can be made before the
acquisition starts. For NDTiff this follows the index, which gains a record only
once an image is complete. The HDF5 is written throughout and **is**
the result — the window is a look at it. Zoom, pan, filter bounds and contrast
survive every update untouched.

Two constraints that matter inside microclaw:

- `live_view` opens a matplotlib window and, with `block=True` (the default),
  does not return until it is closed. It must run on the main thread of its own
  process. **Do not call it from the microclaw server process** — launch it as a
  separate process, or use `LiveFit`, which is the same fit with no window: it
  runs in a thread and queues finished blocks for whatever front end wants them.
- Frames that never reach a file — a pycro-manager `analyze_frame` hook, a
  camera buffer — go through a `QueueSource`, which `live_view` and `LiveFit`
  take exactly as they take a path:

  ```python
  source = smappy.queue_source(shape=(512, 512))
  source.push(image)      # from the hook; push(image, first_frame=n) states
  source.close()          # the acquisition's own frame number
  ```

  `maxsize=N` bounds the queue if the acquisition can outrun the fit; pushing
  then waits rather than growing until memory runs out.

---

## The output file

`LocalizationWriter` appends block by block and flushes as it goes, so the file
is readable while the fit is still running and an interrupted run keeps
everything up to the last block. One file holds the localization columns under
`/locs` and the full provenance — camera metadata, every stage's settings, the
calibration used — as JSON in the file attributes.

```python
from smappy.io.hdf5 import save_localizations, load_localizations
save_localizations(path, locs)          # complete table in one go
locs = load_localizations(path)         # symmetric read
```

Columns depend on `FitSettings.output_unit`: `"pixel"` gives `x_pix`, `y_pix`,
…; `"nm"` gives `x_nm`, `y_nm`, `z_nm`, `loc_precision_nm`, `logl_rel`,
`photons`, `background`, `frame`; `"pixel+nm"` gives both. **Use `"nm"`** for
anything a user will look at. The column set must not change between blocks —
the writer refuses a block whose columns differ.

---

## Viewing

```python
smappy.view(out_h5)          # a path or a table; blocks until the window closes
```

From a shell: `smappy-view OUT.h5`. The viewer needs matplotlib and, like
`live_view`, blocks and wants the main thread — launch it as a separate process
from microclaw, never inline.

For an image without a window (a PNG for the session record, a tile in the web
GUI):

```python
from smappy.filter import LocFilter

keep = LocFilter(locs, loc_precision_nm=(None, 20), logl_rel=(-1.5, None))
smappy.save_image(locs, "image.png", pixelsize=10.0, select=keep)
```

It takes a table or a saved `.h5`, and the same render settings, display
settings and filter the viewer takes — what it writes is what the viewer would
show. **This is the right call from inside the microclaw process**; it opens no
window and needs no main thread. `render_locs` underneath it returns the array
if you want to do something else with it.

`mode` is `"hist"`, `"gauss"` (one sigma) or `"precision"` (sigma from each
localization's own precision — SMAP's default, and the right one). Rendering and
display are separate: contrast, gamma and LUT change without re-rendering.

---

## After the fit

Filter, then drift-correct, then group — in that order.

```python
from smappy.drift import DriftSettings, correct_drift, save_drift_corrected

keep = LocFilter(locs, loc_precision_nm=(None, 15), logl_rel=(-2, None),
                 z_nm=(-300, 300))
corrected, drift = correct_drift(locs, DriftSettings(segmentation_var=500),
                                 select=keep)
save_drift_corrected(out_h5, corrected, drift)     # writes OUT_driftc.h5
```

Drift is estimated from the localizations that pass the filter and subtracted
from **all** of them: a filter is a view, the correction is a coordinate change.
Include a z cut before estimating — without one the axial drift follows the
out-of-focus tail. Drift correction needs COMET (`externaltools/Comet`), a
separate optional install; `--rcc` (redundant cross-correlation) is an
independent second opinion that does not.

Starting filter values for AF647 dSTORM are in `load_skill(name="smlm")`
(locprec < 20 nm, LLrel > −1, PSF size < 175 nm for 2D).

---

## What to ask the user before fitting

1. Path to the raw dataset, and whether it still needs a TIFF export.
2. 2D or 3D? For 3D, the path to the `_3Dcal.mat` spline calibration — without
   one the fit falls back to a Gaussian PSF and there is no z.
3. Camera conversion (e-/ADU) and offset (ADU) — from the rig profile if it
   records them, otherwise ask. Not guessable, and wrong values give wrong
   photon counts and therefore wrong precisions. A SMAP `*_cameras.mat` can
   supply them where one exists, but is never required.
4. Effective pixel size in nm, if `get_pixel_size()` does not report a
   calibrated one.
5. Where the HDF5 should go.
6. Whether they want the live view during acquisition or a fit afterwards.

---

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Fewer frames than acquired | acquisition stopped mid-write; the index lists images that were never written | expected — those records are dropped; check `source.n_frames` |
| `ValueError: missing camera metadata` | conversion/offset/pixelsize unset | pass them in `overrides`; never default them |
| Coordinates in pixels, viewer rows empty | `output_unit="pixel"` (the default) | set `FitSettings(output_unit="nm")` |
| Microclaw server hangs | `show`/`live_view` called inline | separate process, or `LiveFit` headless |
| Almost no localizations | cutoff too high, or wrong offset | check `engine.stats["candidates"]` first |
| z looks compressed or mirrored | EM mode mismatch with the calibration | `warn_on_em_mismatch(cal, camera.em_on)` |
