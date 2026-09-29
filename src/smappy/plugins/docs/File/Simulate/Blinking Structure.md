---
version: "4"
covers: [smappy.plugins.file.SimulateBlinks.run, smappy.simulate.structure.Structure.sample, smappy.simulate.structure._placements, smappy.simulate.kinetics.label, smappy.simulate.kinetics.blink, smappy.simulate.kinetics.off_time_for, smappy.simulate.kinetics.expected_blinks, smappy.simulate.kinetics._spread, smappy.simulate.kinetics.emission, smappy.simulate.localizations.ground_truth, smappy.simulate.localizations.drift_trace, smappy.simulate.localizations.localizations, smappy.simulate.localizations.mortensen, smappy.simulate.localizations.close_groups, smappy.simulate.camera.render, smappy.simulate.camera.camera_truth, smappy.simulate.camera.astigmatic_sigmas, smappy.simulate.source.write_recipe]
---

## What it does

To test an analysis you need data where the answer is known.  This plugin
makes such data.  It takes a structure -- a ring, filaments, nuclear pores --
puts fluorophores on it, lets them blink the way a dye does, and gives either
the **localizations** a fit would have produced or the raw **camera frames**
for a fitter to fit.

Both outputs come from the same molecules.  With the same settings and
*seed*, the localizations and the frames describe the same fluorophores
blinking in the same frames.  So you can look at a simulation as
localizations first, and then fit it from its frames by changing only
*simulate*.

Use it to:

* try the analysis plugins without a microscope (the tutorials do);
* see what a labelling efficiency, a linkage error or a dye's blinking does to
  a picture or a measurement;
* test a fitter, a filter or a drift correction against the truth, with
  [Ground Truth](plugin:Analysis/Measure/Ground Truth).

It is a model, not a microscope.  The background is flat, the spot is a
Gaussian (or a measured PSF), and the dye has one on-state and one
off-state.  What it tests is whether an analysis gets back what was put in.

## How it works

```figure-setup
from smappy.simulate import (SimulationSettings, BlinkingSettings, LabellingSettings,
                             ground_truth, localizations, load_structure, render,
                             mortensen)
from smappy.simulate.kinetics import blink
from smappy.simulate.localizations import drift_trace
settings = SimulationSettings(n_frames=5000, seed=2,
                              labelling=LabellingSettings(efficiency=0.5, linkage_nm=10))
truth = ground_truth(settings)
locs = localizations(truth)
# the structure's labels, drawn as ground_truth draws them first
labels = load_structure("demo").sample(np.random.default_rng(settings.seed))
x, y = np.asarray(locs["x_nm"]), np.asarray(locs["y_nm"])
```

**1. The structure.**  A structure is a file that says where the labels
are, in nanometres.  It is written in YAML, a plain-text format, as a list of
*elements*: single points (or a table of them), lines, circles, filled
polygons, or a grey-scale image whose brightness sets the density.  Points
are placed exactly where the file says.  The other elements are filled at a
*density* (labels per nm of line, or per nm² of area), with the number of
labels drawn at random around that density, so no two simulations are
alike.  A structure can be repeated as *copies* over the field of view,
placed at random or on a grid, turned, tilted and jittered.

The built-in structures are the files in `smappy/data/structures`, and each
one is also an example of the syntax:

* **demo** -- a slightly tilted ring 5 µm across, two crossing lines and a
  few scattered labels over 10 × 10 µm, in 3D.  The ring is dye 1, the rest
  dye 2.  The tutorials measure it.
* **npc** -- nuclear pore complexes labelled at Nup96: 32 labels per pore in
  two rings 50 nm apart in z, radius 53.7 nm
  ([Thevathasan et al. 2019](https://doi.org/10.1038/s41592-019-0574-9)),
  two pores per µm², each turned at random.
* **filaments** -- 40 straight filaments, 3 µm long, placed and turned at
  random.
* **pie** -- a resolution star after ThunderSTORM's
  ([Ovesný et al. 2014](https://doi.org/10.1093/bioinformatics/btu202)): 32
  wedges, 4 µm long, alternately labelled and empty.  The gap between two
  spokes narrows towards the centre -- a fifth of the radius, so 50 nm at
  255 nm from it -- and where the spokes merge is the resolution.  The
  labelled wedges rise in density by √2 each, from 25 to 4525 labels per
  µm², so the same picture shows where localizations start to overlap.

A new structure is a YAML file chosen as the *structure file*, or dropped
into that folder to become a built-in.

```figure The four built-in structures, as label positions.  The pores and filaments are copies of one element, each placed and turned anew; the pie's spokes rise in density going anticlockwise from the right.
fig.set_size_inches(7.5, 2.3)
axes = fig.subplots(1, 4, gridspec_kw={"wspace": 0.35})
for ax, name in zip(axes, ("demo", "filaments", "npc", "pie")):
    lab = load_structure(name).sample(np.random.default_rng(0))
    xy = lab.xyz
    if name == "npc":                     # a few pores, close up
        cx, cy = lab.poses["x_nm"][0], lab.poses["y_nm"][0]
        xy = xy[(abs(xy[:, 0] - cx) < 600) & (abs(xy[:, 1] - cy) < 600)]
    colour = {"c": lab.dye, "cmap": "coolwarm"} if name == "demo" else {"color": "k"}
    ax.scatter(xy[:, 0] / 1000, xy[:, 1] / 1000, s=0.3 if name != "npc" else 2,
               linewidths=0, rasterized=True, **colour)
    if name == "npc":
        ax.set_xlim((cx - 600) / 1000, (cx + 600) / 1000)
        ax.set_ylim((cy - 600) / 1000, (cy + 600) / 1000)
    ax.set_aspect("equal"); ax.set_title(name, fontsize=9)
    ax.tick_params(labelsize=6)
axes[0].set_xlabel("x (µm)", fontsize=8)
```

**2. Labelling.**  Not every label carries a fluorophore.  Each one does
with the *labelling efficiency*, and then carries one fluorophore -- or
another fixed number, or a random (Poisson) number around it.  A fluorophore
does not sit exactly on its label: an antibody or a linker holds it a few
nanometres away.  This *linkage error* comes in two kinds.  A *fixed* one
moves the fluorophore once, and all its blinks share the offset.  A *free*
one is drawn anew for every blink, as for a dye that turns on a flexible
linker.

**3. Blinking.**  Each fluorophore switches between a dark state and a
bright state.  It stays dark for a random time (on average the *off time*)
and bright for a random time (on average the *on time*), and after each blink
it bleaches for good with the *bleaching probability*.  The same model serves
STORM, PALM and PAINT; PAINT is a long off time.  Switching happens at any
moment, not only at frame boundaries, so the first and last frames of a blink
are dimmer than the ones in between.

Nobody knows the off time of a dye, but one knows roughly how often a
molecule comes back in an experiment.  So the plugin asks for *blinks*, the
mean number of blinks per fluorophore **within the measurement**, and works
out the off time that gives it.  A fluorophore cannot blink more often than
$1/p$ times on average, $p$ the bleaching probability, and a request for more
is refused.

In a real dSTORM experiment one raises the activation as the dye bleaches,
to keep the number of molecules per frame steady.  The default *activation*,
*constant*, does the same: the blinks come out spread evenly over the
measurement.  *Decaying* keeps the rate fixed, so the blinks thin out as the fluorophores
bleach -- the more so, the higher the bleaching probability.
*Every blink until bleached* is SMAP's "Dye" model: every fluorophore shows
all its blinks, however long the measurement, spread evenly.

```figure Left: the blinks of 30 fluorophores over the measurement (default settings, 5000 frames): each row one fluorophore, each tick a blink.  Right: how many of 5000 fluorophores are on per frame, for the three choices of *activation*, with a bleaching probability of 0.3 and 2.5 blinks: *decaying* thins out as the fluorophores bleach, the other two stay flat.
rng = np.random.default_rng(4)
fig.set_size_inches(7.5, 2.6)
left, right = fig.subplots(1, 2)
b = blink(30, 5000, BlinkingSettings(), rng)
left.scatter(b.start, b.owner, marker="|", s=40, color="k", lw=1)
left.set_xlim(0, 5000); left.set_ylim(-1, 30)
left.set_xlabel("frame", fontsize=8); left.set_ylabel("fluorophore", fontsize=8)
bins = np.linspace(0, 5000, 26)
for activation, color in (("constant", "#1f77b4"), ("decay", "#d62728"), ("all", "#2ca02c")):
    b = blink(5000, 5000, BlinkingSettings(activation=activation, bleaching=0.3,
                                           blinks=2.5), rng)
    on = np.histogram(b.start, bins)[0] * BlinkingSettings().on_time / np.diff(bins)
    right.step(bins[:-1], on, where="post", color=color, label=activation)
right.set_xlabel("frame", fontsize=8); right.set_ylabel("on per frame", fontsize=8)
right.legend(fontsize=7, frameon=False); right.set_ylim(0, None)
for ax in (left, right):
    ax.tick_params(labelsize=7)
```

**4. Photons.**  Each blink gets a total number of photons, drawn at random
around *photons per blink* with a spread of *photons std*.  The fluorophore
emits them at a steady rate while it is on, so each frame gets the share of
the time it was on in it.

**5. Drift**, if *add drift* is ticked: the whole sample moves along a slow
random path over the measurement, so every localization and every spot is
off by where the sample was in its frame.

**6a. Localizations.**  For each frame a fluorophore was on in, the number
of photons detected is drawn around what it emitted (Poisson).  Below the
*detection limit* there is no localization.  The rest are placed at the true
position plus a random error whose size is the **localization precision** a
good fit would reach -- how precisely a spot of that many photons, on that
background, can be located.  The same precision is written into the table
as `xy_err_nm`, so the reported error is honest: a plugin that checks
precision against the data should find them equal.  Two fluorophores on in
the same frame closer than the *separation* would be one spot to a fitter;
by default both are removed (*close emitters*).

```figure What a simulation of the demo gives, close up on the ring.  Left: the labels (grey) and the fluorophores (black), at a labelling efficiency of 0.5 and a fixed linkage error of 10 nm: half the labels are empty, and a fluorophore sits near its label rather than on it.  Middle: the localizations, a cloud around each fluorophore.  Right: the error of each localization over the precision it reports, for those above 200 photons; it follows the standard normal (line), as an honest precision must.
fig.set_size_inches(7.5, 2.6)
a, b, c = fig.subplots(1, 3)
box = lambda xy: (xy[:, 0] > 2300) & (xy[:, 0] < 3300) & (xy[:, 1] > 3800) & (xy[:, 1] < 4800)
lab = labels.xyz[box(labels.xyz)]
fl = truth.fluorophores.xyz[box(truth.fluorophores.xyz)]
a.scatter(lab[:, 0], lab[:, 1], s=14, color="0.75", linewidths=0)
a.scatter(fl[:, 0], fl[:, 1], s=2, color="k", linewidths=0)
inside = box(np.column_stack([x, y]))
b.scatter(x[inside], y[inside], s=1, color="#d62728", linewidths=0, alpha=0.6)
for ax, title in ((a, "labels, fluorophores"), (b, "localizations")):
    ax.set_xlim(2300, 3300); ax.set_ylim(3800, 4800); ax.set_aspect("equal")
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
bright = np.asarray(locs["photons"]) > 200
true = truth.fluorophores.xyz[np.asarray(locs["emitter"])]
pull = ((x - true[:, 0]) / np.asarray(locs["xy_err_nm"]))[bright]
c.hist(pull, bins=np.linspace(-4, 4, 41), density=True, color="0.6")
t = np.linspace(-4, 4, 200)
c.plot(t, np.exp(-t ** 2 / 2) / np.sqrt(2 * np.pi), color="k")
c.set_xlabel("error / xy_err_nm, x", fontsize=8); c.tick_params(labelsize=7)
c.set_title(f"std {pull.std():.2f}", fontsize=9)
```

**6b. Camera frames.**  Here every spot is drawn onto a camera: the
fluorophore's photons spread over the pixels as a Gaussian of the *PSF
sigma* (astigmatic, whose shape changes with z, if *astigmatic* is ticked,
or a measured PSF from a *PSF calibration*), on the *background*, with the
camera's shot noise, read noise and gain.  Nothing is removed, however dim
or crowded: sorting that out is the fitter's job.

The frames are not written out as images.  The plugin saves the
*simulation file*, `*.sim.yaml`, which is the recipe -- these settings and
the seed -- a few hundred bytes instead of hundreds of megabytes.  Open it in
a fitter as its file, as you would open a TIFF: the frames are drawn as the
fitter reads them, identically every time, and the fitter takes the camera's
pixel size, conversion and offset from it.  Tick *also write a TIFF* for
other software.

```figure Three frames of the same simulation as camera frames (counts), with the true positions of the spots on in each marked.  Some are dim -- the start or end of a blink -- and some overlap; all are drawn.
fig.set_size_inches(7.5, 2.6)
axes = fig.subplots(1, 3)
em = truth.emission
counts = np.bincount(em.frame, minlength=settings.n_frames)
third = settings.n_frames // 3                # the busiest frame of each third
busy = [k * third + int(np.argmax(counts[k * third:(k + 1) * third])) for k in range(3)]
px = truth.positions() / settings.optics.pixelsize_nm
for ax, f in zip(axes, busy):
    frame = render(truth, int(f), int(f) + 1)[0]
    ax.imshow(frame, cmap="gray", origin="lower")
    on = em.frame == f
    ax.scatter(px[on, 0], px[on, 1], s=60, facecolors="none", edgecolors="#ff7f0e", lw=0.8)
    ax.set_title(f"frame {f}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
```

## In detail

**Sampling a structure.**  A line of length $L$ gets a Poisson number of
labels with mean $\rho L$, uniform along it; a circle of radius $r$ gets
$2\pi r \rho$ on average; a polygon of area $A$ gets $\rho A$, uniform inside
it, spread evenly over its `z` range; an image pixel of brightness $g$
(scaled so the brightest pixel is 1) gets $\rho\, g\, a^2$ for an image pixel
of side $a$.  Every element can be blurred by its `width`, a Gaussian per
axis.  Explicit points are the same in every copy, the sampled elements are
drawn afresh for each.  Copies are placed uniformly in the `field` (with a
`min_distance` between them, if given) or on a grid, turned in the plane
(`rotation: random`) or in 3D (`random_3d`), tilted by up to `tilt` degrees,
and moved by up to `jitter` nm.  Each copy's position and turn are kept with
the table, in `metadata["copies"]`, by its number in the `copy` column, as
angles $\alpha, \beta, \gamma$ with the rotation
$R = R_z(\alpha)\, R_y(\beta)\, R_z(\gamma)$.

**The off time from the number of blinks.**  A fluorophore blinks $G$ times
before it bleaches, geometric with $P(G \geq k) = (1-p)^{k-1}$.  Within a
measurement of $T$ frames an unbleached one would be activated $K$ times,
Poisson with mean $T / (t_\mathrm{off} + t_\mathrm{on})$.  It shows
$\min(G, K)$ blinks, whose mean is

$$E = \sum_{k \geq 1} (1-p)^{k-1}\, P(K \geq k) .$$

$E$ falls as the off time grows, and the plugin finds the $t_\mathrm{off}$
that makes $E$ equal to *blinks* by bisection.  With the bleaching
probability of 0.1 and 3 blinks wanted over 20 000 frames, the off time
comes out at about 5600 frames.  Setting *off time* directly overrides the
calculation.

**Spreading the blinks.**  With *constant* activation the blinks are first
drawn at the fixed rate, then their start times are mapped, in order, onto
as many times drawn uniformly over the measurement.  The mapping keeps each
fluorophore's own order, so its dark times come out longer early and shorter
late -- what raising the activation does.  If a squeezed dark time would put
a blink before the previous one had ended, it waits for the end.  *Every
blink until bleached* draws $G$ blinks with $p = 1/$*blinks* and spreads them
the same way, so the measurement's length plays no part.

**Photons.**  The total of a blink is drawn from a gamma distribution with
mean $\mu$ (*photons per blink*) and standard deviation $s$ (*photons std*):
shape $(\mu/s)^2$, which is never negative and is exactly $\mu$ when $s = 0$.
The fluorophore emits it at the rate total / *on time* while it is on.  So
a blink much longer than average gives more than *photons per blink*, and a
short one fewer.

**The precision.**  The lateral precision per axis is that of a
maximum-likelihood fit of a Gaussian spot
([Mortensen et al. 2010](https://doi.org/10.1038/nmeth.1447), their
eq. 5):

$$\sigma_x^2 = \frac{\sigma_a^2}{N} \left( 1 + \int_0^1 \frac{\ln t}{1 + t/\tau}\, dt \right)^{-1}, \qquad \sigma_a^2 = \sigma^2 + \frac{a^2}{12}, \qquad \tau = \frac{2\pi \sigma_a^2 b}{N a^2} ,$$

with $N$ the detected photons, $\sigma$ the *PSF sigma*, $a$ the *pixel size*
and $b$ the background in photons per pixel.  Without background the integral
is 0 and $\sigma_x^2 = \sigma_a^2/N$; the more the background dominates, the
closer the bracket comes to 0 and the larger the error.  With *EMCCD* ticked
the variance is doubled, the excess noise of the gain.  The z precision is
*z precision* times the lateral one (3 by default).  With *astigmatic*
ticked, $\sigma$ is the spot's width at its z (the geometric mean of the x
and y widths).  The noise added to each position is drawn from exactly these
precisions, and `sigma_nm` is the true width plus noise of the same size.

```figure The Mortensen precision against photons for three backgrounds (PSF sigma 130 nm, pixels 100 nm), and the $1/\sqrt{N}$ of no background at all (dashed).  The background matters most for dim spots.
fig.set_size_inches(5.5, 2.8)
ax = fig.subplots()
n = np.logspace(np.log10(50), 4, 100)
for bg, color in ((1, "#1f77b4"), (20, "#ff7f0e"), (100, "#d62728")):
    ax.loglog(n, mortensen(n, bg, 130.0, 100.0), color=color, label=f"{bg} photons/pixel")
ax.loglog(n, np.sqrt((130.0 ** 2 + 100.0 ** 2 / 12) / n), "k--", lw=0.8)
ax.set_xlabel("photons", fontsize=8); ax.set_ylabel("precision (nm)", fontsize=8)
ax.legend(fontsize=7, frameon=False); ax.tick_params(labelsize=7)
```

**Close emitters.**  Localizations on in the same frame within the
*separation* of each other are joined into groups, including chains of
pairs.  With *remove both*, every member of a group is dropped.  With *one
localization, averaged*, a group becomes one localization at the
photon-weighted mean of its members, with their photons added, carrying the
identity of the brightest and an `n_merged` column: what a single-emitter fit
of an unresolved pair converges to.

**The camera.**  A spot's photons are integrated over each pixel (the
difference of error functions of the pixel's edges), and pixel $k$ is centred
at $k \times$ *pixel size*, as the fitters assume.  Behind a cylindrical lens
each axis widens as a focused beam,
$\sigma_x(z) = \sigma \sqrt{1 + ((z - c)/d)^2}$ and
$\sigma_y(z) = \sigma \sqrt{1 + ((z + c)/d)^2}$, $c$ the *focal offset* and
$d$ the *focal depth*
([Huang et al. 2008](https://doi.org/10.1126/science.1153529)).  The expected
photons of each pixel, background included, are drawn as a Poisson count of
electrons; with an *EM gain* that count is multiplied by a gamma of its size,
which doubles the variance.  The counts are electrons / *conversion* +
*offset* plus Gaussian *read noise*, rounded and clipped to 16 bits.  The
background is flat over a frame, and with *background std* it varies from
frame to frame.  Each frame's noise has its own seed, so a frame is the same
whichever block of the stack it is drawn in.

**Drift.**  A random walk of 0.6 nm per frame and axis, plus a straight creep
of 80, −50 and 30 nm in x, y and z over the whole measurement.  It is added to
the true positions, so localizations and frames drift alike, and the path is
kept in `metadata["drift_truth"]`, one row per frame, for comparing a drift
correction with it.

```figure A drift path as *add drift* makes it, over 20 000 frames: the creep, with the random walk on top.
fig.set_size_inches(5.5, 2.4)
ax = fig.subplots()
d = drift_trace(20000, np.random.default_rng(1))
for k, (name, color) in enumerate(zip("xyz", ("#1f77b4", "#ff7f0e", "#2ca02c"))):
    ax.plot(d[:, k], color=color, lw=1, label=name)
ax.set_xlabel("frame", fontsize=8); ax.set_ylabel("drift (nm)", fontsize=8)
ax.legend(fontsize=7, frameon=False); ax.tick_params(labelsize=7)
```

## Parameters

### output
*localizations* is quick and needs nothing else: the table appears at once,
replacing what is open.  *camera frames* writes the *simulation file* and
opens nothing unless *load the truth* is ticked; fit the file to get
localizations.

### n_frames
The number of blinks per fluorophore is set by *blinks* whatever the length,
so a longer measurement is a sparser one: fewer molecules on per frame.  A short
camera stack is dense -- lower the *labelling efficiency* to thin it.

### labelling.efficiency
Real labelling reaches 0.5-0.7 for good antibodies or tags; 0.6 is a
reasonable value for the nuclear pores.

### labelling.linkage_nm
About 10 nm per axis for a primary and secondary antibody, a few nm for a
nanobody or a tag.

### blinking.blinks
The mean over *all* fluorophores, including those that bleach after one
blink.  It must stay below 1 / *bleaching probability*.

### blinking.off_time
Leave it on auto unless the dye's off time is known; set, it overrides
*blinks*.

### blinking.bleaching
With the geometric number of blinks this gives, many fluorophores blink once
and a few many times, as real dyes do.

### blinking.photons
A typical dSTORM dye (Alexa Fluor 647) gives a few thousand photons per blink;
a fluorescent protein a few hundred.

### optics.calibration
The width and astigmatism settings are then not used for the frames.  The
localizations output still uses the Gaussian precision.

### localizations.min_photons
Keep it low: the dim localizations are part of real data.  Filter them away
afterwards (at 200 photons, say) before anything about precision.

### localizations.min_separation_nm
About twice the *PSF sigma*: closer than that, a fitter sees one spot.

### localizations.emccd
For the localizations output only; the camera frames have their own *EM
gain*.

### camera.path
Ground Truth finds the simulation again by this path, so a fit whose file has
been moved or deleted has no truth to compare with.

### camera.conversion
With these defaults and *EM gain* 0 the camera is an sCMOS.  The fitter reads
conversion, offset and pixel size from the simulation file, so there is
nothing to match by hand.

## Output

**Localizations.**  A table in nanometres, replacing what is open:

* `frame`, `x_nm`, `y_nm`, `z_nm`, `photons`, `background`, `xy_err_nm`,
  `z_err_nm` and `sigma_nm`, as a fitter writes them;
* the **truth**: `emitter` (which fluorophore), `dye` (from the structure),
  `copy` (which copy of the structure), and `n_merged` when close emitters
  are averaged;
* in the metadata: the settings (`simulation_settings`), the numbers of
  labels, fluorophores and blinks, the off time that was used, each copy's
  position and turn (`copies`) and, with drift, the drift path
  (`drift_truth`).

The text says how many localizations came from how many fluorophores and
blinks, the off time, and how many were too close together.

**Camera frames.**  The *simulation file*; with *also write a TIFF*, the
frames beside it; with *load the truth*, a table of every spot drawn: `frame`,
the true `x_nm`, `y_nm`, `z_nm` (drift included), the `photons` it emitted in
that frame, the `background`, `emitter`, `dye`, `copy`, and `neighbour_nm`, the
distance to the nearest other spot on in the same frame.

**Comparing with the truth.**
[Ground Truth](plugin:Analysis/Measure/Ground Truth) scores a table against
its simulation.  It needs no truth table: from a fit of a `*.sim.yaml` (the
fitter writes the file into the table as its `source`) or from a simulated
table (which carries its `simulation_settings`), it simulates the same
molecules again and pairs each localization with its true spot.  Because the
precision is drawn honestly, a table simulated here should show an error
over reported precision close to 1.

## Differences from SMAP

Based on SMAP's `ROIManager/Segment/SimulateSites` (`simulatelocs.m`) and
`WorkflowModules/Loaders/SimulateCameraImages` (`simulatecamera.m`)
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).  The physics within
a blink is the same: an exponential on time starting anywhere in a frame,
the photons shared by the time on, Poisson counts per frame, $\sqrt{2}$ for
an EMCCD and z three times the lateral precision.

* **The precision.**  SMAP uses Mortensen's eq. 6, the error of an
  unweighted least-squares fit, with the PSF fixed at 100 nm.  Here it is
  eq. 5, maximum likelihood, with the *PSF sigma* and *pixel size* set: it
  agrees with SMAPpy's own fitter to 1-2 %, where eq. 6 was 10-23 % too
  pessimistic.
* **Blinking.**  SMAP's default model puts each fluorophore's blinks at
  independent random frames, their number drawn directly.  Here the number
  follows from the off time and bleaching, set from the blinks wanted in the
  measurement.  SMAP's "Dye" model is the option *every blink until
  bleached*; SMAP's version scrambles each fluorophore's order of blinks
  when it spreads them, this one keeps it.
* **Photons per blink** are gamma-distributed rather than normal and clipped
  at zero.
* **Close emitters.**  SMAP keeps them in its localizations.  Here they are
  removed or averaged, since a fitter could not have separated them; the
  camera frames keep them, as SMAP's do.
* **Dim localizations.**  SMAP raises every count below 10 photons to 10 and
  keeps it; here a localization below the *detection limit* is not found.
* **Structures** are YAML files of elements and copies, placed at random or
  on a grid, rather than one site per ROI on a grid.  An image is scaled by
  its brightest pixel, not by 255.
* **Camera frames** are a recipe drawn as they are read.  SMAP Poisson-draws
  the photons for its localizations and then again for its frames, which
  doubles their variance; here they are drawn once.

## References

* Mortensen KI, Churchman LS, Spudich JA, Flyvbjerg H. Optimized
  localization analysis for single-molecule tracking and super-resolution
  microscopy. *Nat Methods* 7, 377 (2010) -- the precision, eq. 5.
  [doi:10.1038/nmeth.1447](https://doi.org/10.1038/nmeth.1447)
* Huang B, Wang W, Bates M, Zhuang X. Three-dimensional super-resolution
  imaging by stochastic optical reconstruction microscopy. *Science* 319, 810
  (2008) -- the astigmatic PSF.
  [doi:10.1126/science.1153529](https://doi.org/10.1126/science.1153529)
* Thevathasan JV, Kahnwald M, Cieśliński K, et al. Nuclear pores as versatile
  reference standards for quantitative superresolution microscopy. *Nat
  Methods* 16, 1045 (2019) -- the geometry of the `npc` structure.
  [doi:10.1038/s41592-019-0574-9](https://doi.org/10.1038/s41592-019-0574-9)
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
