---
version: "1"
covers: [smappy.plugins.file.SimulateTracks.run, smappy.plugins.file.SimulateTracksSettings.model, smappy.simulate.tracks.simulate_tracks, smappy.simulate.tracks.jump_nm, smappy.simulate.tracks.spacing_nm]
---

## What it does

Simulates a [single-particle tracking](plugin:Panels/Glossary) (SPT)
experiment as localizations: particles that diffuse, blink and bleach, each
localized with a known precision.  The table is added as a new file, so
[Link tracks](plugin:Analysis/Tracking/Link tracks) and the tracking
analyses can be tried, and their settings chosen, on data whose answer is
known.  The truth is kept in two columns: `track_id_true`, which particle a
localization belongs to (0 for a spurious one), and `state_true`, the
diffusive state it was in (-1 for a spurious one).

What is simulated is what a linker is given, not the camera: there is no
motion blur, and two particles that come close are still two
localizations.  Use it to see how the density, the jump and the dark frames
affect linking, not to judge a fitter.

## How it works

```figure-setup
from smappy.simulate import simulate_tracks, jump_nm
locs = simulate_tracks(n_frames=400, field_um=8.0, density_um2=0.08,
                       d_um2_s=(0.8, 0.02), fractions=(0.6, 0.4),
                       switch_probability=0.02, p_dark=0.1, dark_frames=1.5,
                       track_frames=40.0, seed=3)
x, y = np.asarray(locs["x_nm"], float), np.asarray(locs["y_nm"], float)
frame, truth = np.asarray(locs["frame"]), np.asarray(locs["track_id_true"])
state = np.asarray(locs["state_true"])
order = np.lexsort((frame, truth))
step = (truth[order][1:] == truth[order][:-1]) & (np.diff(frame[order]) == 1)
jumps = np.hypot(np.diff(x[order]), np.diff(y[order]))[step]
jump_state = state[order][:-1][step]
```

**1. Particles.**  Particles appear at a steady rate anywhere in the
*field* and each lasts a random number of frames, *track length* on
average, before it bleaches.  The rate is set so that *density* particles
are present in each frame on average; the first frame starts with that many
already there.  Particles may wander out of the field: it is only where
they start.

**2. Diffusion.**  In each frame a particle takes a random step, larger the
larger its diffusion coefficient *D*.  With a *second state*, a particle
spends that fraction of the time diffusing with the second state's
coefficient instead; with *switching* above 0 it changes state during its
track.

**3. Blinking.**  A bright fluorophore goes dark with the probability
*going dark* in each frame and stays dark for *dark frames* frames on
average.  A dark frame gives no localization: these are the gaps a linker's
memory must bridge.

**4. Localizations.**  Each frame in which the particle is bright gives a
localization at its position plus a random error of *precision* in each
direction, recorded as `xy_err_nm` (and `z_err_nm` in 3D).  *Spurious*
localizations, in no track, can be scattered over the field.

```figure Two states, 60% of the time at D = 0.8 µm²/s and 40% at 0.02 µm²/s, switching rarely. Left, the tracks coloured by the state of each localization: the slow parts stay put, the fast ones roam. Right, the jumps between consecutive frames in each state, against the distribution expected from D and the precision (lines): the simulation gives what was asked.
fig.set_size_inches(8, 3.6)
left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.2]})
for k, colour in ((0, "C0"), (1, "C3")):
    pick = state == k
    left.scatter(x[pick] / 1000, y[pick] / 1000, s=1, c=colour, linewidths=0)
left.set_aspect("equal")
left.set_xlabel("x (µm)")
left.set_ylabel("y (µm)")
r = np.linspace(0, 400, 200)
for k, (d, colour) in enumerate(((0.8, "C0"), (0.02, "C3"))):
    right.hist(jumps[jump_state == k], bins=np.linspace(0, 400, 41), density=True,
               color=colour, alpha=0.4, label=f"D = {d} µm²/s")
    s2 = jump_nm(d, 10.0) ** 2 / 2 + 2 * 20.0 ** 2      # per axis
    right.plot(r, r / s2 * np.exp(-r ** 2 / (2 * s2)), color=colour)
right.set_xlabel("jump between consecutive frames (nm)")
right.set_ylabel("density")
right.set_yticks([])
right.legend(fontsize=8)
```

## In detail

**Arrivals.**  With $\rho$ the *density*, $A$ the field's area and $L$ the
*track length*, the frame starts with a Poisson number of mean $\rho A$
particles and each later frame adds a Poisson number of mean $\rho A / L$.
Lifetimes are geometric with mean $L$ (cut at the end of the movie); being
memoryless, the particles already present at the start are drawn the same
way.

**Steps.**  Per axis a step is Gaussian with variance $2 D t$, $t$ the
*frame time*, using the state the particle had in the frame before: the
state holds over the interval.  The root-mean-square jump in one frame is
$\sqrt{2 d D t}$ in $d$ dimensions, which the form shows under the settings
together with the mean distance between nearest neighbours,
$1 / (2 \sqrt{\rho})$.

**States.**  In every frame, with the probability *switching*, the state is
drawn again from the two fractions -- possibly the same one.  So the
fractions are exactly the time spent in each state, and the first state is
left with the probability *switching* times the second state's fraction
per frame.  A particle starts in a state drawn from the fractions.

**Blinking.**  A two-state chain per frame: bright to dark with the
probability *going dark*, dark to bright with the probability one over
*dark frames*, so dark runs are geometric with that mean.  A track starts
bright.

**3D.**  With *3D*, particles start uniformly within *z range* around the
focus, diffuse in z with the same coefficient, and get a z error of *z
precision*.  Nothing leaves the focus: the track length stands for that as
well as bleaching.

## Parameters

### density_um2
SPT from single molecules needs them sparse: keep the root-mean-square
jump (shown under the form) below about a tenth of the distance between
molecules, where linking is nearly perfect.  Raise it to see linking fail.

### track_frames
Set it to what the fluorophore lasts in the experiment being planned.
Tracks of only a few localizations are what a per-track analysis (a mean
squared displacement per track) cannot use, so they show how much of the
data a minimum length in Link tracks would throw away.

### diffusion.slow_fraction
0 gives one state.  A bound and a free population (D around 0.01 and 1
µm²/s) is the usual case to test a jump-distance or MSD analysis on.

### blinking.p_dark
0 gives tracks without gaps.  0.1 to 0.2 with *dark frames* 1.5 to 3 is a
fluorophore that is missed now and then; it is what *memory* in Link
tracks is for.

### localizations.noise_density_um2
Spurious localizations -- background mistaken for a spot -- each start a
track of their own or join a real one.  0.005 per µm² per frame is two
per frame in a 20 µm field.

## Output

A new file, `tracks (simulation)`, with `frame`, `x_nm`, `y_nm`,
`xy_err_nm` (and `z_nm`, `z_err_nm` in 3D), `track_id_true` and
`state_true`, sorted by frame as a fit writes it; and a line with the
number of localizations and tracks.  The settings are kept in the file's
metadata (`track_simulation`), with the frame time.
