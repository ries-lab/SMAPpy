---
version: "2"
covers: [smappy.plugins.overlay_channels.OverlayChannels, smappy.plugins.overlay_channels.registration_for, smappy.plugins.overlay_channels.with_fitter, smappy.plugins.overlay_channels.shown, smappy.channel_overlay.overlay_table, smappy.channel_overlay.check_layout, smappy.channel_overlay.fit_bead_movies, smappy.channel_overlay.camera_of, smappy.channel_overlay.bead_files, smappy.channel_overlay.bead_key, smappy.channel_overlay.cached_transform, smappy.channel_overlay.transform_path, smappy.plugins.registration.register_table]
---

## What it does

Some two-colour experiments use a split camera: a dichroic mirror sends one
dye's light to one half of the chip and the other dye's light to the other
half.  When the two dyes are spectrally well separated, each molecule is
seen in one half only.  There is no photon ratio to measure, and nothing to
pair, so the two-colour fitters, which fit every molecule in both halves at
once, are the wrong tool.

Instead the movie is fitted with a one-colour fitter
([Gaussian 2D](plugin:Localize/Gaussian 2D) or
[Spline 3D](plugin:Localize/Spline 3D)) with *channels* > *read* set to
*two channels*.  Each half is then detected and fitted on its own, and every
localization keeps the half it was found in, `camera_channel` (0 the main
half, 1 the other).  The two halves are still two pictures side by side.
This plugin puts the second one on top of the first.

To do that it needs the
[registration](plugin:Panels/Glossary) between the halves: the map that says
where a point seen in the second half is in the first.  That map is measured
on fluorescent beads, which shine in both halves.  It comes from one of
three places:

* a *transformation file* -- a registration saved before, by this plugin or
  by [Calibrate transform](plugin:Analysis/Register/Calibrate transform);
* *fitted beads* -- a bead localization file, already fitted as two
  channels;
* *bead movies* -- a bead movie, or a folder of them, which the plugin fits
  itself.

The last two keep what they measured beside the beads and use it again, so a
day of samples against one bead folder fits the beads once.

The bead map is used as it is.  The beads are imaged before or after the
sample, and in between the splitter, the focus or the temperature may move
one half against the other by some nanometres; the two dyes' colours can
also differ from the beads'.  Where that matters, and the two dyes label
structures that lie together, the shift that is left can be measured on
the overlaid table with
[Register layers](plugin:Analysis/Register/Register layers), one layer per
`channel`, and taken out with
[Transform](plugin:Analysis/Process/Transform).

Every fitter offers this plugin as a step of its
[*after the fit*](plugin:Analysis/After fit/After the fit), *overlay
channels*, so a fit can come out overlaid.  There the beads are fitted with
the fitter's own camera and arrangement.

## How it works

```figure-setup
import tempfile
from pathlib import Path
import tifffile
from scipy.spatial import cKDTree
from smappy.plugins import Context
from smappy.plugins.fit import (CameraSettings, ChannelsSettings, GaussianFit,
                                GaussianFitSettings, OutputSettings, SourceSettings)
from smappy.simulate import ISOLATED_NM, dual_camera_frames
from smappy.calibrate.dual import ArrangementSettings, RegisterSettings
from smappy.plugins.registration import register_table, registration_plot
from smappy.channel_overlay import overlay_table
folder = Path(tempfile.mkdtemp())
split = ChannelsSettings(split="two channels", layout="up-down", main_channel="upper")
def fit(frames, name):
    tifffile.imwrite(folder / name, frames)
    s = GaussianFitSettings(source=SourceSettings(path=str(folder / name)), channels=split,
                            camera=CameraSettings(conversion=0.5, offset=100.0, pixelsize_um=0.1),
                            output=OutputSettings(save=False))
    return GaussianFit().run(Context(), s).locs
# beads: bright, seen equally in both halves
beads = fit(dual_camera_frames(40, seed=1, ratios=(0.5, 0.5), photons=20000.0)[0], "beads.tif")
measured = register_table(beads, ArrangementSettings(layout="up-down", main_channel="upper"),
                          RegisterSettings())
# the sample: two dyes, each 97 % in its own half and 3 % in the other
frames, truth = dual_camera_frames(250, seed=2, ratios=(0.03, 0.97))
sample = fit(frames, "sample.tif")
def error_nm(table):
    # each second-half localization of the second dye against the true molecule in its frame
    # (frames far apart in the search, so that a match is always within one frame; isolated
    # spots only, whose fit no neighbour pulls)
    dye = (np.asarray(truth["component"]) == 1) & (np.asarray(truth["neighbour_nm"]) > ISOLATED_NM)
    true = np.c_[1e6 * truth["frame"][dye], truth["x_nm"][dye], truth["y_nm"][dye]]
    rows = (np.asarray(table["camera_channel"]) == 1) & (np.asarray(table["photons"]) > 500)
    found = np.c_[1e6 * table["frame"][rows], table["x_nm"][rows], table["y_nm"][rows]]
    d, i = cKDTree(true).query(found, distance_upper_bound=100)
    ok = np.isfinite(d)
    return found[ok, 1:] - true[i[ok], 1:]
error = error_nm(overlay_table(sample, measured.transform))
```

**1. Fit the sample as two channels.**  Each half is detected with its own
threshold and fitted on its own.  The positions are where each half saw the
molecule on the camera chip, so before this plugin the second dye's
localizations sit in the other half of the picture.

**2. The registration from beads.**  Beads shine in both halves, so every
bead is a pair of localizations in the same frame.  The pairs are found and
the map is fitted exactly as in
[Calibrate transform](plugin:Analysis/Register/Calibrate transform), whose
page explains it: a vote for the offset between the halves, then pairing and
refitting at a shrinking distance, with a projective or a polynomial map.
For *bead movies* the plugin first fits them with Gaussian 2D on two
channels, in the arrangement and with the camera the sample was read with.
A bead's centre is all a 2D registration needs, whatever model fitted the
sample.

```figure Simulated beads, fitted as two channels and registered: the vote for the offset between the halves, the pairs' residuals after the fit, and where in the main half the pairs are.  A sharp vote, a tight residual and pairs over the whole field are what a usable registration looks like.
fig.set_size_inches(10, 3.4)
registration_plot(measured).draw_into(fig)
```

**3. Kept beside the beads.**  With *keep the registration* on, what was
measured is written as `<name>_2ct.h5` beside the bead movie, the folder or
the bead file, together with which bead files it came from and the settings
it was measured with.  The next run that names the same beads with the same
settings reads it instead of measuring again.  When a bead file changes, or
a setting does, it is measured again and the file is replaced.

**4. Move the second half.**  The localizations with `camera_channel` 1 are
moved through the map onto the main half.  The main half's are not moved.
`channel`, the dye, is then written from `camera_channel`, so the Render
tab's layers can show one dye each.

```figure How far each overlaid localization of the second dye is from the true molecule, for the isolated spots of the simulated sample.  Both distributions are centred on zero: the bead map has put the second half onto the first, and what is left is each localization's own precision.
fig.set_size_inches(7, 3)
axes = fig.subplots(1, 2, sharey=True)
bins = np.linspace(-80, 80, 61)
for ax, k, name in zip(axes, (0, 1), ("x", "y")):
    ax.hist(error[:, k], bins, color="C0")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(f"{name} found - true (nm)")
axes[0].set_ylabel("localizations")
```

## In detail

**Where the halves are.**  The registration lives in camera-chip pixels.
The table's nm columns are put back into chip pixels with the pixel size
the table records, the map is applied, and the result goes back into nm.
The fit records where on the chip it found its two halves, and so does the
registration.  When the two disagree, the beads were fitted in another
arrangement than the sample, and the run says so in a warning.  A table
whose second half has already been moved is refused: moving it twice is
never what is meant, and undo puts it back.

**The bead fit.**  Gaussian 2D, *channels* on *two channels*, the dynamic
cutoff of the *bead fit* part, and at most *at most* frames of each movie.
Everything else is the fitter's default.  The camera is the fitter's own
when this runs as its step.  Run on its own, it is what the table recorded
of its camera (conversion, offset, EM gain, pixel size), and else what the
bead files and the camera database say, as for any fit.  The movies of a
folder are every image stack directly in it; each one's frames are kept
apart from the others', so no bead is paired across two movies.

**Keeping it.**  A registration is used again when the file's record of its
beads matches: each bead file's full path, size and time of last change, and
every setting it was measured with -- the arrangement, the registration's
settings, and for movies the bead fit's settings and the camera.  The file
goes beside the movie or the folder, not inside the folder, which stays
images only.  For one bead movie this is the same name *Calibrate transform*
proposes, `<movie>_2ct.h5`.

**No correction from the sample.**  An earlier version corrected the bead
map with crosstalk pairs in the sample: a molecule and the few percent of
its light in the other half, in the same frame.  On a real two-dye movie its
pairs were mostly chance meetings and badly fitted spots, and in the dense
first frames it applied a 100 nm shift that the real pairs did not support;
the crosstalk of a dim dye, some 15 photons a frame, is not found at all.
So the bead map is used as it is, and a shift left between the dyes is
measured, where the structures allow it, by correlating the two channels'
images with *Register layers*.

## Parameters

### source
*transformation file* for a registration you already trust.  *fitted beads*
when the beads were fitted by hand, for instance to check them first.
*bead movies* to have everything done here: the usual choice in a fitter's
*after the fit*.

### path
A dual-colour bead calibration from the
[PSF calibration](plugin:Panels/PSF calibration) window works too: its
registration is used and its PSFs are not.

### beads_table
Fit the beads with the same arrangement as the sample, *channels* > *read*
on *two channels*.  Several bead positions in one file are fine.

### beads
The browse button picks a folder; a single movie's path can be typed.  Image
stacks the camera wrote, at a few positions over the field, so the pairs
cover it: a registration is an extrapolation where there were no beads.

### bead_fit.cutoff
Beads are bright, and the default finds them.  Raise it if dirt or dim
specks are paired too; lower it for dim beads.

### bead_fit.max_frames
A registration needs a few hundred pairs, not every frame of a long bead
movie.  Leave it on *auto* for short movies; set it to a few hundred frames
for long ones, to save time.

### channels.layout
The microscope's entry says how its chip is split.  Choose it here only for
a microscope without one; in a fitter's step the fitter's own is used.

### channels.main_channel
Which half the other is moved onto, for a layout chosen by hand.

### channels.split_position
Only for a split away from the middle of the chip.

### registration.model
As in [Calibrate transform](plugin:Analysis/Register/Calibrate transform):
*projective* is safe everywhere; *polynomial* also follows the field's
distortion where the bead pairs cover the field.

### save
Leave it on unless the bead folder is read-only.  Turn it off to measure the
registration afresh on every run.

## Output

The table with the second half moved onto the first, and `channel` written:
0 for the main half's dye, 1 for the other's.  The move is recorded in the
table's metadata with the map that was used, and is one step of the undo.

The output says how many localizations were moved, where the registration
came from (a file, measured now, or measured before on the same beads), the
and the bead pairs' residual when it was measured now: a few hundredths of
a pixel is a good registration, a tenth or more one to look at in the figure.

The figures: *beads* -- the vote, the residuals and the pairs' coverage, as in
[Calibrate transform](plugin:Analysis/Register/Calibrate transform), when
the registration was measured in this run.
