---
version: "1"
covers: [smappy.stability.grid_step, smappy.stability.resample, smappy.stability.bead_traces, smappy.stability.combine, smappy.stability.longest_run, smappy.stability.highpass, smappy.stability.smooth, smappy.stability.spectrum, smappy.stability.cumulative_rms, smappy.stability.peak_amplitude, smappy.stability.peak_frequency, smappy.stability.peaks, smappy.stability.vibration_rms, smappy.stability.analyse, smappy.stability.pooled, smappy.plugins.stability.Stability.run]
---

## What it does

Says how still the microscope holds the sample, measured on
[fiducials](plugin:Panels/Glossary): bright, still particles on the
coverslip such as fluorescent beads, seen in nearly every frame.  A bead
does not move, so whatever movement its localizations show is either the
[localization precision](plugin:Panels/Glossary) -- how far a fitted
position lies from the true one -- or the instrument moving.

It is a diagnostic, with *approaches* ticked on or off.  The first is
**vibrations**: [vibration](plugin:Panels/Glossary) is fast movement of
the sample or the optics, from the building, a fan or a pump, which
repeats many times a second.  It blurs every localization by its
amplitude, and no drift correction removes it.  The plugin shows a bead's
position against time, how much it jitters beyond its precision, and the
[amplitude spectrum](plugin:Panels/Glossary): the movement taken apart
into frequencies, where a vibration is a peak, so its source can be
found -- a 50 Hz peak is the mains, a peak that moves with a pump's speed
is the pump.  Long-term drift and the drift between two channels are to
follow as approaches of their own.  To *correct* drift from beads, use
[Fiducials](plugin:Analysis/Drift/Fiducials).

It needs the beads' localizations with an id column that says which
belong to one bead: `track_id` from [Link
tracks](plugin:Analysis/Tracking/Link tracks), by default.  It reads the
localizations, not merged blinks, those the layer's filter and the drawn
ROI keep: draw an ROI around one bead, as in SMAP, or let it take every
bead in the view.  It needs a short frame time to see fast vibrations: a
frame of 1 ms sees up to 500 Hz, one of 20 ms only up to 25 Hz.  Per
region it runs too: in each ROI of the ROI manager, or as a block over
`track_id`, once per bead.

## How it works

```figure-setup
from smappy.simulate import simulate_beads
from smappy import stability as st

n, dt = 8000, 0.001
t = np.arange(n) * dt
rng = np.random.default_rng(3)
# a common motion: slow drift, 6 nm at 37 Hz and 2 nm at 120 Hz on x
common = np.zeros((n, 3))
common[:, 0] = (40 * np.sin(2 * np.pi * 0.08 * t) + 6 * np.sin(2 * np.pi * 37 * t)
                + 2 * np.sin(2 * np.pi * 120 * t))
locs = simulate_beads(n, common, n_beads=3, precision_nm=3.0, p_missed=0.03,
                      seed=3)
ids = np.asarray(locs["bead_true"]) + 1
xyz = np.column_stack([locs["x_nm"], locs["y_nm"]]).astype(float)
err = np.column_stack([locs["xy_err_nm"], locs["xy_err_nm"]]).astype(float)
times = np.asarray(locs["frame"]) * dt
# the third bead also shakes on its own, 4 nm at 75 Hz
own = ids == 3
xyz[own, 0] += 4 * np.sin(2 * np.pi * 75 * times[own])
beads = st.bead_traces(ids, times, xyz, dt, err)
one = st.analyse(beads[:1], ("x", "y"), highpass_hz=1.0)[0]
```

**1. A regular trace.**  Each bead's localizations are put in time order,
one point per frame; time is the frame times the file's frame time.  A
bead missed in a frame leaves a hole, which is filled by a straight line
between its neighbours, because the spectrum needs a point at every time
step.  How many points were filled is reported.  A table with its own
clock (`time_s`, MINFLUX) is put on a regular grid first.

**2. Which trace.**  Each bead on its own, the mean over beads -- which
keeps what they share and averages their precision down -- or a
**differential** trace: one bead minus another, or each minus the mean of
the others.  What all beads share, the stage's movement and drift,
cancels in a difference, and what is left is how the beads move against
each other plus their localization noise.  A vibration that survives the
difference is not the sample moving as a whole.

**3. Slow drift removed.**  Over seconds the sample drifts, by far more
than it vibrates.  A high-pass filter removes what is slower than
*high-pass*, and the standard deviation (SD) of what is left is the
jitter.  It is shown beside the precision: their difference, in squares,
is the vibration.

```figure One bead (simulated, a frame per ms, 3 nm precision): drift, a 6 nm vibration at 37 Hz and a 2 nm one at 120 Hz on x. Left: the first 300 ms, high-passed, the localizations (light) and their running mean, where the vibration is seen directly. Right: the whole trace as measured (grey) and with the drift removed (blue).
fig.set_size_inches(9, 3.2)
left, right = fig.subplots(1, 2)
x = one.axes[0].detrended
tt = (beads[0].t_s[one.run] - beads[0].t_s[one.run][0]) * 1000
left.plot(tt[:300], x[:300], lw=0.5, color="C0", alpha=0.45)
left.plot(tt[:300], st.smooth(x, 5)[:300], lw=1.2, color="C0")
left.set_xlabel("time (ms)")
left.set_ylabel("x (nm)")
raw = beads[0].values[one.run, 0]
right.plot(beads[0].t_s[one.run], raw, lw=0.4, color="0.6", label="as measured")
right.plot(beads[0].t_s[one.run], x, lw=0.4, color="C0", label="high-passed")
right.set_xlabel("time (s)")
right.set_title(f"SD {one.axes[0].sd_nm:.2g} nm, precision "
                f"{one.axes[0].precision_nm:.2g} nm", fontsize=9)
right.legend(fontsize=7)
```

**4. The spectrum.**  The high-passed trace is cut into segments of
*segment* points, half overlapping, and each is taken apart into
frequencies by a Fourier transform.  Their power is averaged (Welch's
method): one transform of the whole trace is as noisy as its mean at every
frequency, the average over segments much less, so a small peak stands
out.  The amplitude is scaled so that a vibration of amplitude A shows a
peak of height A, in nm.  The spectrum reaches half the sampling rate (the
Nyquist frequency): a vibration faster than that is not seen, or folds
back as a false lower one.

**5. Peaks and the cumulative RMS.**  The largest peaks above *high-pass*
are listed with their frequency and amplitude.  The cumulative
[RMS](plugin:Panels/Glossary) (root mean square) adds up the spectrum from
the lowest frequency: where it climbs is where the jitter is.  At the top
it is the SD.

```figure The same bead: the amplitude spectrum (left), its peaks marked (×) with the true vibrations (circles), and the cumulative RMS (right), which climbs at each vibration and over the flat floor of the localization noise to the SD.
fig.set_size_inches(9, 3.2)
left, right = fig.subplots(1, 2)
a = one.axes[0]
left.semilogy(a.spectrum.f_hz[1:], a.spectrum.amplitude[1:], lw=0.7, color="C0")
for f, amp in a.peaks[:3]:
    left.plot(f, amp, "x", color="k")
left.plot([37, 120], [6, 2], "o", mfc="none", color="C3", label="truth")
left.set_xlabel("frequency (Hz)")
left.set_ylabel("amplitude (nm)")
left.legend(fontsize=7)
right.plot(a.spectrum.f_hz, st.cumulative_rms(a.spectrum), color="C0")
right.axhline(a.precision_nm, color="0.5", ls="--", lw=0.8, label="precision")
right.set_xlabel("frequency (Hz)")
right.set_ylabel("cumulative RMS (nm)")
right.set_ylim(0, None)
right.legend(fontsize=7)
```

```figure Three beads that share the vibrations at 37 and 120 Hz; the third also shakes on its own at 75 Hz. Left: each bead's spectrum shows the shared peaks. Right: each bead minus the mean of the others -- the shared peaks cancel, and the 75 Hz one stays, on the bead that moves (and, smaller, on the others, whose reference includes it).
fig.set_size_inches(9, 3.2)
left, right = fig.subplots(1, 2, sharey=True)
for ax, traces in ((left, beads), (right, st.combine(beads, "others"))):
    for i, r in enumerate(st.analyse(traces, ("x",), highpass_hz=1.0)):
        s = r.axes[0].spectrum
        ax.semilogy(s.f_hz[1:], s.amplitude[1:], lw=0.6, color=f"C{i}",
                    label=r.trace.name)
    ax.set_xlabel("frequency (Hz)")
    ax.legend(fontsize=7)
left.set_ylabel("amplitude (nm)")
left.set_title("each bead", fontsize=9)
right.set_title("each minus the others", fontsize=9)
```

## In detail

**The grid.**  Frames give the grid directly: one point per frame, from
the first frame any bead of the run is seen in to the last, so every bead
shares it.  A table with `time_s` is put on a grid whose step is the 20th
percentile of the intervals between its times, so most intervals hold one
sample.  Samples that land on one grid point are averaged; a grid point
between two samples with none of its own is interpolated linearly and
counted as filled.  Before a bead's first sample and after its last there
is no trace; a spectrum is taken over the longest stretch without such a
hole.  Each trace is centred on its own mean.

**Combining.**  With $N$ beads, the mean is $\bar{x}(t)$, averaged over
the beads seen at $t$.  The differences are $x_i - x_1$ (*minus the first
bead*) or $x_i - \frac{1}{N-1}\sum_{j \neq i} x_j$ (*minus the others*).
The precisions are combined as independent errors: $\sigma/\sqrt{N}$ for
the mean of $N$ beads of precision $\sigma$, $\sqrt{2}\,\sigma$ for a
difference of two, $\sigma\sqrt{1 + 1/(N-1)}$ for one minus the others.  A
bead's precision is the RMS of its localizations' `xy_err_nm` (`z_err_nm`
for z).

**The high-pass.**  A second-order Butterworth filter at *high-pass*, run
forwards and backwards, so that it shifts nothing in time (a fourth-order
response in all).  At 0 Hz only the mean is removed.  The SD and
everything after it are of the high-passed trace.

**The vibration.**  Localization noise and vibration add in variance, so
the vibration's RMS is

$$\mathrm{RMS}_{\mathrm{vib}} = \sqrt{\mathrm{SD}^2 - \sigma^2}$$

with $\sigma$ the precision (0 where the precision explains all of the
SD; the SD itself when the table has no precision).  A sinusoid of
amplitude $A$ has an RMS of $A/\sqrt{2}$.

**The spectrum.**  Segments of $L$ points (*segment*, or the whole trace
when shorter) start every $L/2$ points; each, minus its mean, is
multiplied by a Hann window $w$ and Fourier transformed, $X(f)$.  The
power spectral density is

$$S(f) = \frac{2\,\langle |X(f)|^2 \rangle}{f_s \sum w^2}$$

averaged over the segments, in nm²/Hz, with $f_s$ the sampling rate; its
sum times the bin width $\Delta f = f_s/L$ is the variance.  The
amplitude shown is $2\sqrt{\langle |X(f)|^2 \rangle} / \sum w$, so that
a sinusoid on a bin reads its amplitude, as SMAP's $2|\mathrm{FFT}|/L$
did.  The cumulative RMS at $f$ is $\sqrt{\sum_{f' \leq f} S(f')\,\Delta f}$.

**A peak.**  A local maximum of $S$ above *high-pass* that is at least 4
times the floor around it (the median of $S$ from 3 to 20 bins either
side), and more than a main lobe (4 bins) from a larger one.  A sinusoid
whose frequency falls between two bins reads up to 15 % low at its
tallest bin, so its amplitude is taken from the power $P$ over the five
bins around it, above the floor: $A = \sqrt{2P}$, and its frequency is
their power-weighted mean.

**Per region.**  The values are pooled over the region's traces: the
SD and the precision as the root of their mean squares, the spectra
averaged in power, and the largest peak of that average.  With one trace
they are its own.  The arrays are the first trace's spectrum and
cumulative RMS.

## Parameters

### id_column
Any integer column whose value is the same on one bead's localizations:
`track_id` from Link tracks (link with a generous *memory*, so a missed
frame does not split a bead), `bead_true` of a simulation, or `tid` from a
MINFLUX file.  Ids of 0 and below are not beads.

### traces
*each bead* to see whether all beads agree; *the mean over beads* for the
sample's own motion with less noise; a differential trace to tell motion
of the whole sample (which cancels) from motion within it, or from the
optics of one part of the field.

### vibration.highpass_hz
Above the slowest drift and below the slowest vibration of interest: 1 Hz
for a frame time of a few ms.  Too high removes real vibrations; 0 leaves
drift in the SD, which then says little about vibration.

### vibration.segment
Sets the frequency resolution, the sampling rate over this, and how many
segments are averaged.  1024 points at 1 kHz resolves 1 Hz; longer
segments resolve close peaks, shorter ones give a smoother spectrum.

### frame_time_ms
SMAP took the frame time from the file and allowed it to be overwritten;
here too.  A wrong frame time scales every frequency.

## Output

Per axis -- x, y, and z when the table has it -- a line of the text: the
SD above *high-pass*, the precision, the vibration's RMS and the largest
peak.  The figures, each ticked on or off: the zoomed trace (the
localizations, light, and their running mean), the whole high-passed
trace with its SD and precision in the title, the amplitude spectrum with
its largest peak marked and *high-pass* dotted, the cumulative RMS with
the precision dashed, and a table of the peaks, also in the text.  Up to
8 traces are drawn over each other.

A stable instrument shows an SD at the precision, a vibration RMS of a
nanometre or less, and a flat spectrum.  A vibration of several nm is
worth finding: it blurs every localization.  If more than a fifth of the
points are filled gaps the plugin warns: above the rate of the gaps the
spectrum is interpolation, not data.

Per region the values are `x_sd_nm`, `x_precision_nm`, `x_vibration_nm`,
`x_peak_hz` and `x_peak_nm` (and y, z), with `n_traces`, `n_points` and
`n_filled`; the arrays `frequency_hz`, `x_amplitude_nm` and
`x_cumulative_rms_nm`, which [Curves](plugin:Analysis/Regions/Summarize/Curves)
summarises over beads.

## Differences from SMAP

Ported from SMAP's *Analyze/measure/Vibrations* (Ries 2020).

* **Gaps.**  SMAP took the localizations as consecutive samples, so a
  missed frame shifted all later ones by a step; here it is filled and
  counted.
* **The spectrum** is averaged over segments (Welch), where SMAP took one
  transform of the whole trace and drew a binned copy beside it; and a
  peak's amplitude comes from its power, which does not read low between
  bins.
* **The SD** is of the trace with the slow drift removed, and the
  vibration's RMS is given beside it; SMAP's SD included the drift.
* **Several beads**: SMAP analysed one bead per layer in the ROI.  Here
  every bead of the id column, their mean, or their differences.
* **Irregular times** are put on a grid at the 20th percentile of the
  intervals; SMAP halved that step, which made the upper half of the
  spectrum interpolation.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870-872 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
  -- the *Vibrations* plugin this is ported from.
* Welch P. The use of fast Fourier transform for the estimation of power
  spectra: a method based on time averaging over short, modified
  periodograms. *IEEE Trans Audio Electroacoust* 15, 70-73 (1967).
  [doi:10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901)
  -- the spectrum averaged over segments.
