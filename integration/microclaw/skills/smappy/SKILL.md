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
                  calibration=cal_3dcal_mat)
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

`camera.require()` raises a `ValueError` naming a missing conversion or offset
rather than producing silently wrong photon counts; a missing pixel size is
fitted on an assumed 100 nm and flagged in the table, so state it.
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

**With the GUI** -- the image building up while the user watches -- start
`smappy-gui` and run a fitter on the Localize tab with its source set to the
dataset being written and *live* ticked, or drive that same GUI fit from a
program through `smappy.gui.live_session.LiveSession` (what the
`ries-lab/smappy-microclaw` package does: it stops the fit and says when the
source is complete). The GUI is a process of its own. **Do not call it from
the microclaw server process** -- start it as a separate process.

**Headless**, inside the microclaw process, use `LiveFit`: the same fit in a
thread, handing finished blocks over a queue to whatever wants them, with no
window:

```python
from smappy.io.watch import watch_stack
from smappy.live import LiveFit, LiveSettings

frames = watch_stack(directory, idle_blocks=True)   # waits for the first frame
fit = LiveFit(frames, camera, finder, model,
              live=LiveSettings(flush_seconds=5.0)).start()
...
for block in fit.drain():                        # whenever the session looks
    log(len(block))
```

`directory` is the NDTiff dataset or Micro-Manager TIFF being written, and it
does not have to exist yet. For NDTiff this follows the index, which gains a
record only once an image is complete.

- Frames that never reach a file -- a pycro-manager `analyze_frame` hook, a
  camera buffer -- go through a `QueueSource`, which `LiveFit` takes exactly
  as it takes a path:

  ```python
  source = smappy.queue_source(shape=(512, 512))
  source.push(image)      # from the hook; push(image, first_frame=n) states
  source.close()          # the acquisition's own frame number
  ```

  `maxsize=N` bounds the queue if the acquisition can outrun the fit; pushing
  then waits rather than growing until memory runs out.

### Reading the density while the fit runs

`on_block(locs)` is called with each finished block as it comes out, so the
session can act on the result before the acquisition ends:

```python
def density(locs):
    frames = locs["frame"]
    return len(locs) / (frames.max() - frames.min() + 1)   # locs per frame

smappy.fit(data, out=out_h5, camera=camera, calibration=cal,
           chunk=25, on_block=lambda block: log(density(block)))
```

`chunk` sets the cadence: a block is fitted at the end of a chunk of frames, so
25 frames at 100 ms is a reading every 2.5 s. `progress(engine)` is cheaper
still — `engine.stats["candidates"]` counts *detected* spots and needs no fit,
which makes it the better signal when the question is density rather than
position.

**This does not by itself make a feedback loop.** Changing the 405 nm activation
power from it is an image-driven `SetIlluminationPower`, which needs the user's
`illumination_envelope` authorized before the run and stays bounded by its
ceiling and write budget — see `load_skill(name="smlm")` for what is and is not
allowed today. Report the density and propose the change; do not invent a
control path around the envelope.

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

Every fit writes positions in nm (`x_nm`, `y_nm`, `z_nm`, `xy_err_nm`) beside
the camera-pixel ones it measured (`x_pix`, `y_pix`, ...), with `logl_rel`,
`photons`, `background` and `frame`. Without a pixel size the nm columns use
an assumed 100 nm and the table says so (`metadata["pixelsize_assumed"]`), so
state the pixel size. The column set must not change between blocks — the
writer refuses a block whose columns differ.

---

## Viewing

```python
smappy.show(out_h5)          # a path, a table or a session: the GUI, its own process
```

From a shell: `smappy-gui OUT.h5`. `smappy.show` starts the GUI as a process of
its own and returns at once -- it needs no Qt or main thread here -- and hands
an in-memory table over through shared memory.

For an image without a window (a PNG for the session record, a tile in the web
GUI):

```python
from smappy.filter import LocFilter

keep = LocFilter(locs, xy_err_nm=(None, 20), logl_rel=(-1.5, None))
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

keep = LocFilter(locs, xy_err_nm=(None, 15), logl_rel=(-2, None),
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
| `ValueError: missing camera metadata` | conversion/offset unset | pass them in `overrides`; never default them |
| `pixelsize_assumed` in the table, nm off by a factor | no pixel size given | pass `pixelsize_um`; the `x_pix` columns are exact |
| Microclaw server hangs | the GUI started inline | `smappy.show` (its own process), or `LiveFit` headless |
| Almost no localizations | cutoff too high, or wrong offset | check `engine.stats["candidates"]` first |
| z looks compressed or mirrored | EM mode mismatch with the calibration | `warn_on_em_mismatch(cal, camera.em_on)` |
