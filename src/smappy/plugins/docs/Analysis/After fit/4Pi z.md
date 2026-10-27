---
version: "1"
covers: [smappy.fourpi.cavity_phase, smappy.fourpi.unwrap, smappy.fourpi.refine_period, smappy.fourpi._weights, smappy.fourpi._interpolate, smappy.fourpi.cavity_drift, smappy.fourpi.simulate_rois, smappy.plugins.fourpi_z.FourPiZ, smappy.plugins.fourpi_z.slope_of, smappy.plugins.fourpi_z.envelope_of, smappy.plugins.fourpi_z.draw_fringes, smappy.plugins.fourpi_z.draw_cavity]
---

## What it does

In a 4Pi microscope two objectives, one above and one below the sample,
collect the light of each molecule.  The two beams interfere, and how
bright the molecule is in each of the four camera images depends on the
*interference phase*.  The phase changes by a full turn every time the
molecule moves by one *fringe period* in z, about 260 nm.  A 4Pi fit
measures it to a few hundredths of a radian, which is z to 1-3 nm.  But it
gives z only within the fringe: a phase cannot tell one fringe from the
next.

The fit also measures z from the shape of the spot, the *envelope*: how
wide or how elongated the spot is.  With the astigmatism of these
microscopes this envelope z is unambiguous, but only good to tens of nm.
This plugin puts the two together.  For every localization it takes the z
that the phase allows on the fringe nearest the envelope's z.  It writes
that as `z_nm`, with the phase's precision as `z_err_nm`.  The method is the
one of Huang et al. 2016
([doi:10.1016/j.cell.2016.06.016](https://doi.org/10.1016/j.cell.2016.06.016)),
on the IAB model of the fit (Li et al. 2020,
[doi:10.1364/OL.397754](https://doi.org/10.1364/OL.397754)).

Two things make it more than a rounding:

- **The cavity drifts.**  The path difference between the two arms changes
  slowly, by fractions of a wavelength over minutes.  Every phase in a
  frame then moves by the same amount.  The plugin measures this *cavity
  phase* from the localizations themselves, in windows of time, and takes
  it off.
- **It says how sure it is.**  For each localization it computes the
  probability that the fringe is the right one, `fringe_probability`.  A
  wrong fringe puts the molecule a whole period, about 260 nm, above or
  below where it is: a *ghost*.  Filtering on the probability removes the
  likely ghosts.

It needs a table from the 4Pi fit, with `phase_rad`, `phase_err_rad` and the
envelope's `z_env_nm` and `z_env_err_nm`.  It also needs the phase's slope
in z, which comes from the 4Pi model.

## How it works

```figure-setup
from smappy import fourpi
from smappy.psf import Global4PiPSF
from smappy.plugins.fourpi_z import draw_fringes, draw_cavity
cal = fourpi.model_calibration()
rng = np.random.default_rng(0)
n, frames = 5000, 5000
frame = np.sort(rng.integers(0, frames, n))
drift = fourpi.cavity_drift(frames, rate_rad=0.0015, wander_rad=0.003, start_rad=1.0, seed=1)
photons = rng.lognormal(np.log(1500), 0.5, n)
rois, truth = fourpi.simulate_rois(cal, n, photons=photons, background=10,
                                   z_nm=(-500, 500), phase0=drift[frame], seed=2)
model = Global4PiPSF(cal)
fit = model.unpack(model.fit(rois))
k = cal.phase_per_nm
cav = fourpi.cavity_phase(fit["phase_rad"], fit["phase_err_rad"], fit["z_env_nm"],
                          fit["z_env_err_nm"], frame, k, window_frames=500)
found = fourpi.unwrap(fit["phase_rad"], fit["phase_err_rad"], fit["z_env_nm"],
                      fit["z_env_err_nm"], cav.at, k)
```

**1. The cavity's phase.**  A molecule at height z has the phase
$\phi = k z + \phi_0(t)$.  Here $k$ is $2\pi$ over the fringe period, with
the sign of the model, and $\phi_0$ is the cavity's phase in that frame.
The envelope's z gives a rough $k z$, so $\phi - k z_\mathrm{env}$ is a
rough $\phi_0$ for each localization.  Averaged over the many molecules of a
time window, the roughness averages out, and the cavity's phase is known to
a few hundredths of a radian.  Between windows it is interpolated.

```figure The cavity's phase measured in windows of 500 frames (blue), on a simulation where it drifted steadily with a random walk on top (grey).  Each window's estimate comes from about 500 molecules.
fig.set_size_inches(6, 3)
draw_cavity(fig.subplots(), cav, frame, drift[frame])
```

**2. The fringe.**  With the cavity's phase taken off, each localization's
phase allows a ladder of heights, one per fringe, a period apart.  The
plugin takes the one nearest the envelope's z.  How near, compared with the
envelope's error, says how probable that fringe is against its neighbours.

```figure The phase with the cavity's phase taken off, against the envelope's z, for the same simulation.  The molecules lie on stripes: one turn of phase per fringe period, with the slope of the model (red).  The envelope's z decides which stripe a molecule is on.
fig.set_size_inches(6, 3.2)
draw_fringes(fig.subplots(), fit["phase_rad"], fit["z_env_nm"], cav.at, k)
```

**3. z.**  The height on that fringe is `z_nm`.  Its error, `z_err_nm`, is
the phase's error over $|k|$: a few nm, where the envelope alone gave tens.

```figure z from the phase against the true z (left): on the line, apart from the few on a wrong fringe, whole periods off.  The error of each against the truth (right), against the envelope's alone (grey): the phase's z is several times sharper.
fig.set_size_inches(9, 3.2)
left, right = fig.subplots(1, 2)
left.plot(truth["z_nm"], found.z_nm, ".", ms=1, color="0.3", rasterized=True)
left.set_xlabel("true z (nm)")
left.set_ylabel("z from the phase (nm)")
d = found.z_nm - truth["z_nm"]
bins = np.linspace(-60, 60, 121)
right.hist(fit["z_env_nm"] - truth["z_nm"], bins, color="0.75", label="envelope")
right.hist(d, bins, color="C0", label="phase")
right.set_xlabel("z - true z (nm)")
right.set_ylabel("localizations")
right.legend(frameon=False, fontsize="small")
```

## In detail

**The cavity's phase.**  In each window of *time window* frames,

$$\phi_0 = \arg \sum_i w_i \, e^{i (\phi_i - k z_{\mathrm{env},i})}, \quad w_i = \frac{1}{\delta_{\phi,i}^2 + k^2 \delta_{z,i}^2}$$

with $\delta_\phi$ and $\delta_z$ the fit's errors of the phase and the
envelope's z.  This is the maximum-likelihood mean of angles spread about
one direction (a von Mises spread).  A localization whose envelope z is
unsure by more than a fringe is weighted down to almost nothing.  The error
of each window's phase is $\sqrt{-2 \ln R} / \sqrt{n_\mathrm{eff}}$.  Here
$R$ is the length of the weighted mean of the unit vectors, and
$n_\mathrm{eff} = (\sum w)^2 / \sum w^2$ is the effective count.

Windows with fewer than 20 localizations are skipped.  *smoothing* above 1
averages the sums of that many neighbouring windows; the sums are averaged,
not the angles, so nothing jumps at $\pm\pi$.  The window phases are
unwrapped along time.  They are interpolated linearly between window
centres, and continued with the end windows' slope before the first and
after the last.  Held constant there instead, a drift of 1.5 rad per window
left 0.18 rad of error in the first and last half windows.  Each file of the
table is done on its own, since each has its own cavity.

**The fringe.**  The candidates are
$z_n = (\phi - \phi_0 + 2\pi n) / k$.  The probability of each is
proportional to $\exp(-(z_\mathrm{env} - z_n)^2 / 2 s^2)$, normalised over
the two fringes on either side of the nearest.  Here $s$ is the envelope's
error.  The fit's error is the [CRLB](plugin:Panels/Glossary), the least the
envelope z can scatter.  A model that does not fit the data perfectly
scatters more, and every probability would then be too confident.  So the
scatter of $z_\mathrm{env}$ about the chosen z, in units of its error, is
measured first.  1.4826 times the median absolute deviation is used, which
the few wrong fringes do not move.  Where that is above one, the errors are
scaled up by it.  The result line reports the factor.  On the simulation
above, 1.3 % of the localizations were on a wrong fringe, and the
probabilities predicted 1.1 %.

**The error.**  `z_err_nm` is $|\delta_\phi / k|$.  It does not include the
error of the cavity's phase, 0.04 rad on the simulation, which is
common to a whole window.  On the simulation the scatter of z on the right
fringe was 3.8 nm, where the phase alone predicted 2.9 nm.

**Refining the period.**  The fringe period of the model comes from beads
moved by the stage.  A dye a little redder than the beads, or a sample whose
refractive index differs from the immersion's, has another period.  *refine
period* looks for the slope $k'$, within 10 % of the model's, at which the
phases of each window line up best.  It scores
$\sum_\mathrm{windows} |\sum_i w_i \, e^{i(\phi_i - k' z_{\mathrm{env},i})}|$,
with each term divided by $e^{-k'^2 \delta_{z,i}^2/2}$.  The envelope's own
noise shrinks every term by that factor, more for a steeper slope.  Left
alone, it biased the answer low, to about 0.99 of the true slope on
simulations; with the correction the answer was within 0.3 %.  Only localizations with
$|k| \delta_z < 1$ take part.  The answer is only as good as the envelope's
z scale, since that is what the phases are compared with.  The z it gives is
$\phi / k'$, so a z that is right needs a right envelope.

**Which table.**  The localizations, whatever the layer shows.  The plugin
reads `z_env_nm` and its error.  On a table that has not been through it, it
reads `z_nm` and `z_err_nm`, which the 4Pi fit writes as the envelope's.
The envelope's values are kept as `z_env_nm`, so a second run with other
settings starts from the fit again.  The cavity's phase is written per
localization as `cavity_phase_rad`.

## Parameters

### slope
*as fitted* reads the slope the 4Pi fit kept with the table.  Use *from a
model file* for a table from elsewhere, naming the 4Pi model it was fitted
with.

### path
A 4Pi model of this program's, or a uiPSF v1 `psfmodel_*_4pi.h5`.

### window_frames
Each window needs a few hundred localizations with a usable envelope z: 250
give the cavity's phase to about 0.05 rad.  It also needs to be short enough
that the cavity drifts by much less than a radian within it.  At 0.04 rad/s
and 40 frames per second, 1000 frames are a radian.

### smoothing
Use 3-5 when the windows are short and sparse.  It averages the drift over
that many windows, so a fast change is followed later.

### refine_period
Leave it off when the model's beads were taken with the dye's filter in a
sample of the immersion's index.  Turn it on to check that the period fits
the data: the result line gives both periods.

### z_scale
With an oil objective imaging into water, the stage's z and the sample's
differ by about 0.7-0.8.  The silicone objectives and index-matched buffer
of the Ries-lab 4Pi need 1.

## Output

- `z_nm`, `z_err_nm`: z from the phase, on the most probable fringe.
- `fringe_probability`: the probability that the fringe is the right one.
  Filter it in the Render tab, at 0.9 or 0.99, to remove likely ghosts.
- `z_env_nm`, `z_env_err_nm`: the envelope's z, as the fit gave it.
- `cavity_phase_rad`: the cavity's phase in the localization's frame.

The figure on the left is the stripes of the second figure above.  Clean,
separate stripes mean a good result.  Stripes smeared sideways mean an
envelope z too poor to choose between fringes.  Stripes that bend away from
the red line mean a wrong period: try *refine period*.  On the right is the
cavity's phase in each window.  It should be smooth.  A window that jumps
away from its neighbours had too few localizations.

The result line gives the expected fraction of wrong fringes, the mean of
1 - `fringe_probability`.  It also gives how much more the envelope's z
scattered than its error said.  A factor well above 1.5 means the model
does not describe the data well.

## References

- Huang F, Sirinakis G, Allgeyer ES, et al. (2016). Ultra-high resolution
  3D imaging of whole cells. *Cell* 166:1028-1040.
  [doi:10.1016/j.cell.2016.06.016](https://doi.org/10.1016/j.cell.2016.06.016).
  The phase unwrapped by the astigmatic z.
- Li Y, Buglakova E, Zhang Y, et al. (2020). Accurate 4Pi single-molecule
  localization using an experimental PSF model. *Opt Lett* 45:3765-3768.
  [doi:10.1364/OL.397754](https://doi.org/10.1364/OL.397754).  The IAB
  model, whose fit gives the phase and the envelope's z separately.
- Liu S, Huang F (2020). Enhanced 4Pi single-molecule localization
  microscopy with coherent pupil based localization. *Commun Biol* 3:220.
  [doi:10.1038/s42003-020-0908-2](https://doi.org/10.1038/s42003-020-0908-2).
  The cavity's drift and the ghosts it causes.
- Bates M, Keller-Findeisen J, Przybylski A, et al. (2022). Optimal
  precision and accuracy in 4Pi-STORM using dynamic spline PSF models.
  *Nat Methods* 19:603-612.
  [doi:10.1038/s41592-022-01465-8](https://doi.org/10.1038/s41592-022-01465-8).
  Following the cavity's phase in time windows.
