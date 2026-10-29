---
version: "1"
covers: [smappy.plugins.fit_4pi.FourPiFit, smappy.plugins.fit_4pi.layout_of, smappy.multichannel.MultiChannelEngine, smappy.multichannel.detection_image, smappy.channels.detection_lookup, smappy.channels.onto_reference, smappy.multichannel.channel_candidates, smappy.multichannel.build_link, smappy.multichannel.cut_channel_rois, smappy.channels.ChannelLayout, smappy.channels.from_corners, smappy.channels.maps_from_uipsf_v1, smappy.psf.Global4PiPSF, smappy.fourpi.FourPiCalibration, smappy.fourpi.from_volumes, smappy.fourpi.normalise, smappy.fourpi.load_uipsf_v1, smappy.fourpi.render_frames, smappy.fourpi.spots, smappy.io.quadrants.open_quadrant_frames, smappy.io.dcimg.DcimgFile, smappy.channels.register_channels, smappy.channels.bead_positions, smappy.channels.bead_projections, smappy.multichannel.measure_channel_drift, smappy.multichannel.ChannelDrift, smappy.plugins.fit_4pi.FourPiFit.channel_drift, smappy.group.combine, smappy.channels.ChannelSource, smappy.channels.ChannelLayout.on_camera, smappy.channels.describe_map, smappy.channels.save_registration, smappy.channels.load_registration, smappy.plugins.fit_4pi.registration_from_beads, smappy.plugins.fit_4pi.microscope_layout, smappy.microscope_db.Microscope.channel_layout, smappy.io.dcimg.DcimgSource.watch, smappy.plugins.fit_4pi.FourPiFit.watch_blocks]
---

## What it does

A 4Pi microscope looks at the sample through two objectives, one on each
side, and lets the two beams interfere.  The camera records four images of
every molecule, the *phase channels*.  In each channel the two beams
interfere at a different *phase*, a quarter turn apart.  How the molecule's
light divides between the four channels says where it is in z within one
*fringe period*, about 260 nm.  That position is several times more precise
than the shape of the spot can give it.

This fitter fits each molecule in all four channels at once.  It uses one
model of the microscope's PSF, the IAB model of Li et al. 2020
([doi:10.1364/OL.397754](https://doi.org/10.1364/OL.397754)).  It finds x
and y, and the photons of all four channels together.  It also finds two
answers for z, kept separate:

- **the envelope's z**, from the shape of the spot -- its astigmatism on the
  microscopes this was written for.  It is unambiguous but good to tens of
  nm.
- **the interference phase**, from how the light divides between the
  channels.  It is a few hundredths of a radian, so a few nm, but it repeats
  every fringe.

*After the fit*, [4Pi z](plugin:Analysis/After fit/4Pi z) puts the two
together.  It follows the slow drift of the cavity, and writes z from the
phase on the fringe the envelope's z points to.

It needs a 4Pi model of the microscope, made from beads: with the
[PSF calibration](plugin:Panels/PSF calibration) window's *4Pi* mode, or
uiPSF v1's `psfmodel_*_4pi.h5`.  The model carries each channel's PSF and
how the channels map onto each other.  It also needs to know where the
channels are on the camera frame (*channels*).

It reads Hamamatsu DCIMG files, which hold the four channels side by side on
one frame, and MATLAB files of `qd1` to `qd4`, the channels already cut.
A DCIMG can be fitted *live*, while the camera is still writing it.  The
frames are counted by their own records (each frame's footer, written after
its pixels, holds its number), not by the file's header, which says how
many there are only once the recording ends.  A live fit cannot measure the
channels' drift, which needs the whole acquisition.  It also reads
[Simulate](plugin:File/Simulate/Simulate)'s 4Pi camera, whose model *Fit
this* fills in.

## How it works

```figure-setup
from smappy import fourpi
from smappy.channels import from_corners
from smappy.multichannel import MultiChannelEngine, detection_image
from smappy.psf import Global4PiPSF
from smappy.metadata import CameraMetadata
from smappy.pipeline import FitSettings
from smappy.plugins.fit import DetectionSettings
from scipy.spatial import cKDTree
cal = fourpi.model_calibration()
layout = from_corners([0, 70, 140, 210], [0] * 4, 64, 64, flip_x=[0, 0, 1, 1])
rng = np.random.default_rng(1)
n_frames = 300
frame = np.repeat(np.arange(n_frames), 2)
x, y = rng.uniform(8, 56, (2, len(frame)))
z = rng.uniform(-400, 400, len(frame))
frames = fourpi.render_frames(cal, layout, frame, x, y, z, 3000, phase0=0.7,
                              n_frames=n_frames, background=10)
camera = CameraMetadata(conversion=1.0, offset=0.0, pixelsize_um=0.1)
engine = MultiChannelEngine(camera, DetectionSettings().finder(), Global4PiPSF(cal),
                            layout, FitSettings(roisize=13))
engine.push(layout.cut(frames), 0)
locs = engine.flush()
lx, ly = np.asarray(locs["x_nm"]) / 100, np.asarray(locs["y_nm"]) / 100
pairs = []
for f in range(n_frames):
    here, there = np.flatnonzero(locs["frame"] == f), np.flatnonzero(frame == f)
    if len(here):
        d, i = cKDTree(np.c_[x[there], y[there]]).query(np.c_[lx[here], ly[here]])
        pairs += [(h, there[j]) for h, j, dd in zip(here, i, d) if dd < 1]
loc, true = map(np.array, zip(*pairs))
```

**1. The channels.**  Right after the file is read, the frame is cut into
its channels, and from then on nothing sees the camera's frame.  Detection,
the fit, the registration and the drift see only the channels and the maps
between them.  Three things describe the channels, each kept where it
belongs:

- **Where each channel is on the camera chip**, in whole pixels, and whether
  it is mirrored.  This is part of the microscope, kept in its entry in the
  microscope database.  It is measured on the chip, not on the frame, so it
  holds for any camera ROI.  A mirrored channel is flipped back when it is
  cut, which moves pixels and changes no value.
- **How each channel maps onto the first**, to a fraction of a pixel: the
  registration.  It belongs to a calibration, the model or a bead
  measurement.  After the flip it is close to doing nothing: a shift of a
  pixel or two, and a scale and a rotation a few parts in a thousand from
  none.  A rectangle placed a pixel off costs nothing, because the
  registration's shift takes it up.
- **How the channels move against each other over time**: the channel drift
  (*In detail*).

**2. Finding the molecules on the sum.**  A molecule can be bright in one
channel and nearly dark in the next: the interference moves its light
between them.  Summed, the four channels show the whole molecule, with no
interference left.  Only for this, each channel is read onto the first
channel's pixels, every pixel from the channel's pixel nearest where the
registration puts it, and the channels are added.  The molecules are found
on that sum, one peak per molecule, with *detection*'s filter and threshold.
Detection only decides where the boxes go, and the filter smooths the sum, so
a pixel's alignment is enough.  On a simulation with a rotated, stretched
registration it found the same molecules as interpolating would, at an
eighth of the cost.

```figure One simulated frame: the four phase channels as the camera sees them, the last two mirrored, and their sum, where the molecules are found (circles).  The same two molecules are bright in some channels and faint in others.
fig.set_size_inches(9, 2.2)
cut = layout.cut(frames[:1])
summed = detection_image(cut, layout)
axes = fig.subplots(1, 5)
for c in range(4):
    axes[c].imshow(frames[0, :, layout.regions[c][0]:layout.regions[c][0] + 64], cmap="gray")
    axes[c].set_title(f"channel {c + 1}", fontsize="small")
axes[4].imshow(summed[0], cmap="gray")
axes[4].plot(x[frame == 0], y[frame == 0], "o", mfc="none", color="C1", ms=10)
axes[4].set_title("sum", fontsize="small")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
```

**3. One fit in four channels.**  Each channel gets its own fit box, cut at
its own pixels where the map puts the molecule.  Nothing is resampled for
the fit: the part of the map smaller than a pixel goes to the fitter, as in
[Spline 3D 2C](plugin:Localize/Spline 3D 2C).  The fitter finds one x, y,
z, photon number and phase for all four boxes, and a background for each.

```figure The fit of the simulation against its truth: x (left), the envelope's z (middle) and the phase (right).  Each panel compares the scatter about the truth (bars) with the precision the fit reports for itself, the CRLB (line).
fig.set_size_inches(9, 2.6)
axes = fig.subplots(1, 3)
dx = lx[loc] - x[true]
dz = np.asarray(locs["z_nm"])[loc] - z[true]
dp = np.angle(np.exp(1j * (np.asarray(locs["phase_rad"])[loc] - (cal.phase_per_nm * z[true] + 0.7))))
for ax, d, err, unit, scale in ((axes[0], dx * 100, locs["x_err_nm"][loc], "x - true x (nm)", 40),
                                (axes[1], dz, locs["z_err_nm"][loc], "z - true z (nm)", 120),
                                (axes[2], dp, locs["phase_err_rad"][loc], "phase − true phase", 0.3)):
    ax.hist(d, np.linspace(-scale, scale, 41), color="0.6", density=True)
    s = np.median(err)
    g = np.linspace(-scale, scale, 200)
    ax.plot(g, np.exp(-0.5 * (g / s) ** 2) / (s * np.sqrt(2 * np.pi)), color="C3")
    ax.set_xlabel(unit, fontsize="small"); ax.set_yticks([])
fourpi.pi_axis(axes[2], "x")
```

## In detail

**The model.**  Channel $c$ sees

$$\mu_c = N \left( I_c + A_c \cos\phi + B_c \sin\phi \right) + b_c$$

Here $I_c$ is the light that does not interfere and $A_c$, $B_c$ the part
that does, each a cubic spline in x, y and z.  $N$ is the photon number,
$\phi$ the phase and $b_c$ the channel's background.  $A_c$ and $B_c$ are
*demodulated*: the fast oscillation along z has been taken out of them and
is carried by $\phi$, so they change as slowly as the spot's shape does.
Each channel's own phase offset, and its share of the light, are in its own
$I_c$, $A_c$ and $B_c$.  An offset that is not exactly a quarter turn, or a
channel with less contrast, is then simply part of the model.  Bates et al.
2022 ([doi:10.1038/s41592-022-01465-8](https://doi.org/10.1038/s41592-022-01465-8))
measured 5-10 nm of z bias from assuming equal channels.

**Photons.**  The channels' models are normalised together: their $I$ sum to
one over the focal plane, so $N$ is the photons in all four channels.
uiPSF v1 normalised to two, so its photon numbers were half; its models are
renormalised when read.  The photon number is one for all channels and
cannot be unlinked.  The phase *is* the way the photons divide between the
channels, and a photon number for each channel would take up the
interference and leave the phase undetermined.

**Starting values.**  For a fixed z the model is linear in $N$,
$N\cos\phi$ and $N\sin\phi$.  So at each z plane of the model a small
weighted least-squares fit over every pixel of every channel gives the
phase and the photons directly, from anywhere on the circle.  The planes
where it fits best (*z starts*, two by default) are where the fit starts,
and the fit with the higher likelihood is kept.  On simulations with the
very model that fits them, every fit found the true phase: of 20,000
molecules at the 4Pi-SMS example's photons and background, 0.01 % ended more
than five errors off, with one start or with five.

On real data a fit can stop in a worse optimum where data and model
disagree: a bright molecule, a model of a few beads.  A second start at
another phase (*phase starts* 2) reaches the better one.  On the 4Pi-SMS
example 5 % of the molecules were such fits, the phase off by about 1.5 rad
and z unchanged; on uiPSF's Nup96 example 2.5 %, mostly dim.  It doubles the
fit's time.  Neither a cheap check of the phase a quarter turn away nor a
second start only for poor fits caught them at less cost.  So it is a
choice: one phase start for speed, two for the last few per cent.  More
*z starts* than two change almost nothing: two and three differed in 0.02 %
of the molecules.

**The fit.**  A Levenberg-Marquardt maximum-likelihood fit for Poisson noise
(with an sCMOS camera's read noise, as the other fitters).  x, y, z, the
photons and the phase are shared by all channels.  Each channel has its own
background.  It gives the same numbers as uiPSF v1's 4Pi fitter.  On the
beads v1 learned its model from, the two agree within 0.0003 pixels,
0.02 z slices and 0.001 rad.  On v1's Nup96 molecules the median difference
is 0.11 slices in z and 0.014 rad in phase.

**The envelope's z at the end of the model.**  A fit that ends on the first
or last plane of the model was held there.  Its true z could be anywhere
beyond, and its error, taken at the edge, says nothing.  Such a fit gets
the model's whole z range as its error, which 4Pi z then gives no weight.

**The maps.**  A model from uiPSF v1 keeps, for each channel after the
first, an affine map about the image centre.  v1 read MATLAB files with
rows and columns swapped.  A model learned from such files is turned the
right way round when it is read, its maps with it.  On the Nup96 example,
fitting each channel's position freely gave back the maps' offsets to
within 0.06 pixels.

**Registration from beads.**  For a model without maps, or a camera moved
since the model was made, *registration* can be measured on beads.  A bead
looks different in every channel at any one z, because the interference
moves its light between them.  So the bead acquisition is summed over all
its frames, every z step and every phase step, which leaves each channel
the bead's envelope.  Its centre is fitted with a Gaussian.  The channels
are put roughly on top of each other by the cross-correlation of these
images, the beads are paired within 3 pixels, and each channel's map onto
the first is the robust projective fit of
[Spline 3D 2C](plugin:Localize/Spline 3D 2C)'s registration.  With fewer than
8 pairs it is a shift: their mean offset.  A folder of bead files is pooled.
On uiPSF's 26 bead scans this gave 111 pairs per channel, 0.03 pixels rms
about the maps.  Where the beads were, the maps were within 0.06 pixels of
the ones uiPSF v1 learned with its model.

**Channel drift.**  The channels can move against each other while the
acquisition runs.  A registration measured on another day can also be off,
and a bead's envelope cannot show where the model's PSF of each channel is
centred.  *channel drift* measures all of these from the data before the
fit.  It fits *drift blocks* blocks of *frames per block* frames, spread
over the acquisition, with x and y free in every channel.  The median offset
of each channel from where the map puts it, in each block, over the better
half of the localizations, is that channel's drift there.  Between blocks
it is interpolated, and beyond the first and last it is held.  The fit then
cuts each channel's box where the map and the drift put the molecule.  On
the Nup96 example, with the registration from another day's beads, the
channels were off by up to 1 pixel, steady to 0.1 pixels over the two
files.  SMAP measured this once, on the first 3000 frames.

**Grouping.**  The phase is an angle: the mean of 3.1 and -3.1 rad is pi,
not 0.  Grouping a blink's localizations averages the phase as unit
vectors, each weighted by its error, and combines `phase_err_rad` like any
precision.

**The phase's direction.**  The phase's slope in z is part of the model and
kept with the table, for 4Pi z.  uiPSF v1 built its bead models and its
in-situ models with opposite signs; each is read with its own.

## Parameters

### model.calibration
A 4Pi model made from beads of the same microscope, with the dye's filter.
uiPSF v1's bead model (`psfmodel_zernike_4pi.h5`) and in-situ model
(`psfmodel_iter*_insitu_zernike_4pi.h5`) both work.

### model.channels.layout
*auto* takes the file's own layout where it has one: MATLAB quadrant files
put the channels side by side.  Otherwise it takes the microscope's: in its
entry of the microscope database, each channel can have a `region`
(`[x, y, width, height]` on the chip, or one per camera) and a `mirror`
(`"x"`, `"y"` or `"xy"`).  Set it there once and every acquisition finds
it, whatever its camera ROI.  The shipped entry *4Pi-SMS example*, with its
camera, is one: the layout of the 4Pi-SMS consortium's example data.  To try a layout first, type the corners, on
the chip: SMAP's 4Pi settings file gives them as `x4pi`, `y4pi`,
`width4pi`, `height4pi` and `mirror4pi`.  *4Pi-SMS* is the Bewersdorf-lab
design: four 168-pixel images on a 2048 x 168 strip, the last two mirrored.

### model.channels.registration
Use the model's: the PSF of each channel was learned centred where its map
puts the molecule, so the two belong together.  *from beads* is for a model
without maps, or after the camera has moved.  Tick *channel drift* with it,
which also corrects the part of the registration that beads cannot see.
*none* is for channels that were registered before they were saved.

### model.channels.beads
A bead z stack taken in the same layout: a DCIMG file or series, a quadrant
`.mat` file, or a folder of either.  Beads all over the field make a
projective map.  A few beads make only a shift.  It is measured once and
kept beside the beads (`<name>_channels.json`, or
`channel_registration.json` in a folder).  It is measured again when a bead
file is newer, or the rectangles differ.

### model.channels.path
A registration that beads left, used for another acquisition.  It is
readable as text: each channel's map is also given as its shift at the
channel's centre, its scale and its rotation.

### model.channel_drift
It costs a second fit of *drift blocks* × *frames per block* frames,
5000 by default.  Worth it for every long acquisition, and whenever the
registration was not measured on the same day.

### model.phase_starts
1 by default.  2 for the final localizations of a measurement, where 2-5 %
of the molecules otherwise keep a phase about 1.5 rad off, 70 nm in z from
the phase.

### model.link_xy
Unlinked, each channel gets its own x and y.  The table then has, for each
channel, how far its position is from where the map puts it
(`dx_nm_ch1`, ...).  Their medians are how far the registration is off.

## Output

The localizations, in the first channel's own pixels: the sample's
coordinates, whatever rectangle of which camera the channel was cut from.
The raw frames kept with the table are the first channel's.  A fit region is
drawn on the Preview's *sum* panel, in the same pixels.  Besides the usual
columns:

- `z_nm`, `z_err_nm`: the envelope's z.  After *4Pi z*, z from the phase.
- `z_env_nm`, `z_env_err_nm`: the envelope's z, kept.
- `phase_rad`, `phase_err_rad`: the interference phase, between -pi and pi.
- `background_ch0` to `background_ch3`: each channel's background.

The table's metadata keeps the phase's slope in z and the channel layout.
A good fit has a phase error of a few hundredths of a radian for a bright
molecule.  The envelope's z then spreads about as far as its error says.

## Differences from SMAP

SMAP's 4Pi fitter is `MLE_4Pi`, which ran uiPSF's closed-source fitter
(SMAP: Ries 2020, [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)).

- **Finding molecules.**  SMAP found the molecules in each channel and
  merged the peaks; a molecule dark in one channel then depended on the
  others.  Here they are found on the channels' sum.
- **The phase start.**  SMAP started the phase at 0, π/3 and 2π/3, a third
  of the circle.  Here the phasor start covers all of it.
- **Photons.**  SMAP left a photon number free in each channel by default,
  which leaves the phase poorly determined.  Here it is linked.
- **The combined columns.**  SMAP combined the channels' positions with
  weights of one over the error but an error formula for one over its
  square, from channels 1 and 2 only.  There is nothing to combine here: the
  position is one fitted number.

## References

- Li Y, Buglakova E, Zhang Y, et al. (2020). Accurate 4Pi single-molecule
  localization using an experimental PSF model. *Opt Lett* 45:3765-3768.
  [doi:10.1364/OL.397754](https://doi.org/10.1364/OL.397754).  The IAB model.
- Li Y, Shi W, Liu S, et al. (2022). Global fitting for high-accuracy
  multi-channel single-molecule localization. *Nat Commun* 13:3133.
  [doi:10.1038/s41467-022-30719-4](https://doi.org/10.1038/s41467-022-30719-4).
  The global fit; linking the photons.
- Liu S, Chen J, Hellgoth J, et al. (2024). Universal inverse modeling of
  point spread functions for SMLM localization and microscope
  characterization. *Nat Methods* 21:1082-1093.
  [doi:10.1038/s41592-024-02282-x](https://doi.org/10.1038/s41592-024-02282-x).
  uiPSF, whose 4Pi models this reads.
- Bates M, Keller-Findeisen J, Przybylski A, et al. (2022). Optimal
  precision and accuracy in 4Pi-STORM using dynamic spline PSF models.
  *Nat Methods* 19:603-612.
  [doi:10.1038/s41592-022-01465-8](https://doi.org/10.1038/s41592-022-01465-8).
- Ries J (2020). SMAP: a modular super-resolution microscopy analysis
  platform for SMLM data. *Nat Methods* 17:870-872.
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1).
