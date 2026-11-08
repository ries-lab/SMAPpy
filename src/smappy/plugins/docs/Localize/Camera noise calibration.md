---
version: "1"
covers: [smappy.scmos.dark_map, smappy.scmos.Moments.add, smappy.scmos.NoiseMap.crop, smappy.scmos.NoiseMap.in_photons, smappy.scmos.for_frames, smappy.scmos.summary, smappy.camera_db.with_noise_map, smappy.pipeline.LocalizationEngine.flush, smappy.scmos.for_layout, smappy.multichannel.MultiChannelEngine.flush, smappy.multichannel.SplitEngine._noise]
---

## What it does

An sCMOS camera reads every pixel through its own amplifier.  So every pixel
has its own *offset* -- the count it reports when no light falls on it -- and
its own *read noise*, the random scatter the electronics add to every
reading.  On a modern sCMOS most pixels read about 1 electron (e-) rms, but a
percent or so are "hot" or *random-telegraph* pixels that jump between
levels, with 5 to 30 e-.  The fitters normally assume one offset and one read
noise for the whole chip.  A hot pixel next to a molecule then pulls the fit
towards itself, and its Cramér-Rao bound (the precision a fit reports,
`xy_err_nm`) claims more than the fit delivers.

This plugin measures each pixel's offset and read noise from a movie taken
with the shutter closed, and keeps them as the camera's *noise map*.  A fit
with *per-pixel noise* on (in the fitter's *fit* part, under *more*)
subtracts each pixel's own offset and weights each pixel by its own noise,
as Huang et al. ([2013](https://doi.org/10.1038/nmeth.2488)) introduced for
sCMOS cameras.  The noisy pixels are then believed less, and the precision
the fit reports is honest again.

**What it needs: a dark movie.**  Close the shutter (or block the light
path), and record 1000 to 2000 frames in the **readout mode and binning of
the data** -- the modes of an sCMOS (standard, ultra-quiet, the 12- and 16-bit
ones) differ in noise as they do in offset.  Record the whole chip, or at
least every pixel the data will use: a map covers the camera ROI it was
recorded with, and a file reaching outside it is refused.  The exposure does
not matter much: at tens of milliseconds the dark current is small beside the
read noise.  The map lasts as long as the camera does not change; record
again after a firmware update or a change of the cooling.

**When not to use it.**  An EMCCD amplifies before reading, so its read noise
is a fraction of an electron and needs no map; the plugin refuses frames
taken with EM gain.  At bright backgrounds (tens of photons per pixel) the
read noise is small beside the background's own noise and the map changes
little.  It matters most at low backgrounds -- a dim TIRF sample, short
exposures -- and in samples with many molecules near hot pixels.

## How it works

```figure-setup
import os, tempfile
from smappy import scmos
from smappy.simulate import scmos_maps, scmos_frames, dark_frames
from smappy.simulate.camera import neighbour_distance, ISOLATED_NM
from smappy.metadata import CameraMetadata
from smappy.pipeline import LocalizationEngine, FitSettings
from smappy.plugins.fit import DetectionSettings
from smappy.plugins.camera_noise import draw_maps, draw_histogram
from smappy.psf import GaussianPSF
from scipy.spatial import cKDTree
conversion = 0.5
offset, noise = scmos_maps((64, 64), seed=1, conversion=conversion, hot_fraction=0.03)
measured = scmos.dark_map([dark_frames(offset, noise, 1000, seed=2)])
path = measured.save(os.path.join(tempfile.mkdtemp(), "noise.h5"))
frames, truth = scmos_frames(offset, noise, seed=3, n_frames=3000, photons=1000,
                             background=5, conversion=conversion)
camera = CameraMetadata(conversion=conversion, offset=100.0, pixelsize_um=0.1,
                        noise_map=str(path))

def fitted(on):
    engine = LocalizationEngine(camera, DetectionSettings().finder(0), GaussianPSF(sigma=1.3),
                                FitSettings(per_pixel_noise=on))
    parts = [engine.push(frames[i:i + 500], i) for i in range(0, len(frames), 500)]
    parts = [p for p in parts + [engine.flush()] if p is not None]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("x_nm", "y_nm", "frame", "photons")}

# each fit's error against the isolated true spot it found, and whether a
# noisy pixel (above 3 e-) is within two pixels of that spot
isolated = neighbour_distance(truth["x_nm"], truth["y_nm"], truth["frame"]) > ISOLATED_NM
noisy = noise * conversion > 3.0
errors = {}
for on in (False, True):
    found = fitted(on)
    err, near = [], []
    for f in np.unique(found["frame"]):
        a, b = found["frame"] == f, truth["frame"] == f
        if not b.any():
            continue
        tx, ty = truth["x_nm"][b], truth["y_nm"][b]
        d, j = cKDTree(np.c_[tx, ty]).query(np.c_[found["x_nm"][a], found["y_nm"][a]])
        keep = (d < 200) & isolated[b][j] & (found["photons"][a] > 300)
        err.append(np.hypot(found["x_nm"][a] - tx[j], found["y_nm"][a] - ty[j])[keep])
        px = np.clip(np.rint(tx[j][keep] / 100).astype(int), 0, 63)
        py = np.clip(np.rint(ty[j][keep] / 100).astype(int), 0, 63)
        near.append(np.array([noisy[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3].any()
                              for x, y in zip(px, py)], bool))
    errors[on] = (np.concatenate(err), np.concatenate(near))
```

**1. Each pixel's mean and variance.**  The dark frames are read in blocks,
and for every pixel the plugin keeps a running mean and a running sum of
squared deviations from it.  The mean is the pixel's offset, in counts (ADU,
the camera's analogue-to-digital units).  The variance -- the squared read
noise -- is that sum divided by the number of frames less one.  One pass over
the movie, however long, holding three images.

**2. How well it is known.**  The variance of N frames is uncertain by a
factor $\sqrt{2/(N-1)}$: 4.5 % at 1000 frames, 8 % at 300.  That is plenty,
because the variance only sets how much each pixel is trusted.  Below 100
frames the plugin refuses, and below 1000 it warns.

```figure The read noise of a simulated sCMOS measured from 1000 dark frames (filled) against the noise its pixels were simulated with (line): the bulk around 1 e- and the tail of hot pixels are both found.
fig.set_size_inches(5.5, 3.2)
draw_histogram(fig.subplots(), measured, conversion, truth=noise * conversion)
```

**3. The map, kept with the camera.**  The map is saved as a file, with
where on the chip it was measured, and entered in the camera database as
the noise map of the movie's camera and readout mode.  It is kept in counts,
not electrons, so that a conversion corrected later corrects the map too.

```figure The measured offset (left) and read noise (right) over the simulated chip: the offsets scatter by a few counts, and the hot pixels stand out of an even field of read noise.
fig.set_size_inches(9, 3.6)
draw_maps(fig, measured, conversion)
```

**4. In the fit.**  With *per-pixel noise* on, a fit looks up the noise map
of the file's camera and readout mode.  It cuts out the part under the
file's camera ROI and subtracts each pixel's own offset before detection.
It then fits every spot with each pixel weighted by its own read noise.  Off,
nothing changes: the fit is the same as without a map.

```figure Of the fits of isolated simulated spots (1000 photons, background 5 photons per pixel) with a pixel above 3 e- within two pixels, the fraction that lands farther than a distance from the truth: with the map (orange) fewer fits are pulled 10 to 25 nm off their spot.
fig.set_size_inches(5.5, 3.2)
ax = fig.subplots()
d = np.linspace(0, 30, 151)
for on, colour, label in ((False, "#1f77b4", "one read noise"), (True, "#c06000", "noise map")):
    err, near = errors[on]
    beyond = np.array([np.mean(err[near] > x) for x in d])
    ax.plot(d, beyond, color=colour, lw=1.5,
            label=f"{label}: {np.mean(err[near] > 20) * 100:.1f} % beyond 20 nm")
ax.set(xlabel="distance from the truth (nm)", ylabel="fraction of fits beyond", yscale="log",
       ylim=(3e-3, 1.2))
ax.legend(fontsize=8)
```

## In detail

**The moments.**  Each block of frames gives every pixel a block mean and a
sum of squared deviations $M_2$.  Two partial results of $n_a$ and $n_b$
frames combine exactly (Chan et al.'s pairwise form of Welford's algorithm),
with $\delta = \bar{x}_b - \bar{x}_a$:

$$\bar{x} = \bar{x}_a + \delta \frac{n_b}{n_a + n_b}, \qquad M_2 = M_{2,a} + M_{2,b} + \delta^2 \frac{n_a n_b}{n_a + n_b}$$

in double precision, so no frame count loses digits to a sum of squares.

**The likelihood.**  A pixel's count, in electrons, is a Poisson number of
photoelectrons $\mu_i$ plus Gaussian read noise of variance $v_i$.  Huang et
al.'s approximation adds $v_i$ to both the data $d_i$ and the model, and
takes the sum as Poisson:

$$\log L = \sum_i \left[ (d_i + v_i) \log(\mu_i + v_i) - (\mu_i + v_i) \right]$$

The variance is converted to the fit's units with the camera's conversion,
$v_i = \sigma_i^2 \, g^2$ for a read noise $\sigma_i$ in counts and $g$
electrons per count.  Without a map the fit uses the same form with one $v$
for the chip (the *read noise* of the camera), which the fitted background
takes up; with a map the background stays the background, and $v_i$ is added
pixel by pixel inside the fitter.  Its information matrix, and so the
Cramér-Rao bound, has $1/(\mu_i + v_i)$ in place of $1/\mu_i$: a noisy pixel
adds less, and the reported precision says so.

**Where the map is cut.**  The map knows the chip pixels it covers, $(x, y,
w, h)$.  A file is cut out at its camera ROI's corner.  A file that records
no ROI is taken to start where the map starts if it has the map's size, and
at the chip's corner otherwise.  A file reaching outside the map is refused
with the pixels it covers, rather than read with another pixel's noise.

**Left out on purpose.**  *Gain per pixel*: the gains of an sCMOS's pixels
differ by 1 to 3 %, and measuring each to 1 % from a photon-transfer curve
takes about 20 000 frames per intensity, so a quick calibration would add
more error than it removes; the camera's one conversion is used.  *Dark
current*: small at the exposures of single-molecule imaging, so one dark
movie stands for all of them.  *Detection*: the offset map is subtracted
before detection, but the candidates are not weighted by the noise, so a very
hot pixel can still make a candidate; its fit is then weighted down.

**Several channels.**  A split chip or a 4Pi camera is cut into its
channels, each rectangle flipped where the optics mirror it, and the map is
cut and flipped the same way, so that every channel's pixel keeps its own
noise.  A channel fitted on its own (*two channels* on Gaussian 2D or
Spline 3D) reads its part of the map; a fit across channels (the 2C fitters,
Spline 4Pi) weights each channel's pixels by their own noise inside the one
fit.  On a simulated split chip with 3 % hot pixels, the two-colour fit's
median distance to the truth fell by about a sixth with the map.

## Parameters

### path
Record it in the readout mode and binning of the data, over at least the
camera ROI the data will use; the whole chip serves every ROI.

### start
For skipping the first frames of a camera that settles after it starts.

### stop
Fewer frames make a less certain map: the variance is known to 4.5 % at 1000
frames and 8 % at 300.

### camera
Set it for a file whose metadata does not identify the camera.  Its readout
mode is still read from the file, and the map is entered for that mode.

### enter
Off, the map is only saved; give its file to the camera in the camera editor
(*noise map*) to use it.  On, the readout mode must already be in the camera
database: add it in the camera editor first.

### output
The default folder is the configuration folder's `noise_maps`; the date in
the name keeps an older map, which older fits name, from being overwritten.

## Output

The text gives the read noise's median, its root mean square and its 99th
percentile, the worst pixel, the share of pixels above three times the
median, and the offsets' median and spread -- in electrons when the camera's
conversion is known, else in counts.  A good sCMOS reads a median of 0.7 to
1.5 e- and a percent or two above three times that.  A median offset well
above the camera's stated offset means the frames saw light, and the plugin
warns.

The figures are the offset and read-noise maps and the read noise's
histogram.  A pattern in the offsets (columns, a gradient) is normal for an
sCMOS; a pattern in the read noise beyond scattered hot pixels -- a bright
corner, stripes -- means the frames were not dark or the camera was not
settled.

The map is saved as an HDF5 file (`offset_adu`, `variance_adu2`, and the
chip region, frame count, camera, readout mode and exposure as attributes).
A fitted table records the map's file in its fit provenance (`noise_map`).

## Differences from SMAP

Ported from SMAP's *calibrateCMOS* (Ries 2020), which computes the same mean
and variance with Welford's algorithm, and its camera converter, which
subtracts the offset map pixel by pixel and gives the variance map to the
fitter.  Two differences change what is fitted:

* SMAP finds the map through a correction file named per camera.  Here the
  map is a value of the camera's readout mode, so a file of another mode has
  none rather than another mode's.
* SMAP, when the map does not cover the image, falls back to the map's corner
  and fits with the wrong pixels' noise.  Here the fit is refused.

## References

* Huang F, Hartwich TMP, Rivera-Molina FE, et al.  Video-rate nanoscopy
  using sCMOS camera-specific single-molecule localization algorithms.  Nat
  Methods 2013;10:653-658 -- the per-pixel likelihood.
  [10.1038/nmeth.2488](https://doi.org/10.1038/nmeth.2488)
* Ries J.  SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data.  Nat Methods 2020;17:870-872.
  [10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
