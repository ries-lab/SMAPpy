---
version: "1"
covers: [smappy.pixel_size.measure, smappy.pixel_size.frame_shift, smappy.pixel_size.vote, smappy.pixel_size.chance_votes, smappy.pixel_size.pair_shifts, smappy.pixel_size.frame_pairs, smappy.pixel_size.solve_offsets, smappy.pixel_size.rough_maps, smappy.pixel_size.join_groups, smappy.pixel_size.link_beads, smappy.pixel_size.refine, smappy.pixel_size.approach_directions, smappy.pixel_size.backlash_axes, smappy.pixel_size.fit_stage, smappy.pixel_size.decompose, smappy.pixel_size.jackknife, smappy.pixel_size.backlash_nm, smappy.pixel_size.stage_columns, smappy.microscope_db.with_pixel]
---

## What it does

Every distance SMAPpy reports in nanometres is a distance in camera pixels
times the *pixel size*: how much of the sample one camera pixel sees.  The
nominal value -- the camera's pixel divided by the magnification -- is often
off by a percent or two, because the tube lens and relay optics are not
exactly what the label says; a cylindrical lens for 3D imaging can even make
the pixel different along x and y.  A percent of pixel size is a percent of
every distance, every cluster radius and every drift.

This plugin measures the pixel size with the microscope's own stage as the
ruler.  A slide of fluorescent beads is moved by the stage between frames,
and each frame is one image near focus.  The beads move on the camera by the
stage's step divided by the pixel, so the camera's movement against the
stage's is the pixel size.  It also gives the *angle* of the camera against
the stage, and whether the image is *mirrored* -- what a stage scan or a
mosaic needs to put tiles side by side.

**What it needs: one stack with the stage moving.**  A slide of beads (100 to
200 nm), dense enough that every frame holds ten or more, in focus.  Move the
stage between frames along a raster in **two directions** -- a stage moved
along a line only gives the pixel along that line, and the plugin refuses it.
Steps of a third to two thirds of the field work best: frames next to each
other must share beads (three at least), and the further apart the frames, the
better the pixel is known.  The stage's position in every frame has to be in
the file: Micro-Manager writes it as `XPositionUm` and `YPositionUm` in each
image's metadata, which is what the plugin looks for.

**Afterwards.**  The pixel is kept in the microscope database (the
*Microscope manager*), for the state the microscope was in when the beads were
taken -- with the 3D lens in or out, say -- so every fit of a file taken in
that state reads it.

## How it works

```figure-setup
from smappy import pixel_size
from smappy.simulate.stage_scan import simulate_stage_scan
from smappy.plugins.pixel_size import draw_offsets, draw_residuals, draw_beads
scan = simulate_stage_scan(n_x=6, n_y=5, step_um=(8.0, 8.0), pixel_nm=(117.0, 113.0),
                           angle_deg=1.5, backlash_nm=(300.0, 0.0), density_per_um2=0.1,
                           seed=2)
result = pixel_size.measure(scan.frame, scan.xy_px, scan.stage_um, scan.shape)
truth = scan.stage_true_um @ scan.matrix.T
first = np.flatnonzero(np.isfinite(result.offsets_px[:, 0]))[0]
truth = truth - truth[first] + result.offsets_px[first]
```

**1. The beads in each frame.**  Every frame is fitted with a Gaussian,
in camera pixels, and spots dimmer than *min photons* are dropped.  Nothing
about the pixel size is used, here or in the next three steps.

**2. How far each frame is from the others: a vote.**  Take two frames.
Every pair of a bead in one and a bead in the other gives a candidate shift
between the frames.  The beads both frames saw all give the same shift; every
other pair gives a shift at random.  The shift most pairs agree on, within
the *tolerance*, wins -- if so many agree that chance could not have done it
among that many beads.  Every pair of frames votes, not only neighbours in the
scan, so two frames that each share a bead or two with many others are still
tied together.

```figure Two neighbouring frames of the scan: every pair of their beads gives a shift (grey); the ten beads they share agree on one (orange), which is how far the camera moved between them.
fig.set_size_inches(5, 4)
ax = fig.subplots()
a, b = scan.xy_px[scan.frame == 0], scan.xy_px[scan.frame == 1]
d = (b[None] - a[:, None]).reshape(-1, 2)
shift, votes = pixel_size.frame_shift(a, b)
agree = np.hypot(*(d - shift).T) <= pixel_size.TOLERANCE_PX
ax.plot(d[~agree, 0], d[~agree, 1], ".", color="0.6", ms=3)
ax.plot(d[agree, 0], d[agree, 1], "o", color="#c06000", ms=6, label=f"{votes} agree")
ax.set(xlabel="shift x (px)", ylabel="shift y (px)")
ax.set_aspect("equal")
ax.legend(fontsize=8)
```

**3. All frames placed at once.**  The frames are linked into a network by
the pairs that agreed, and every frame's position on the camera -- its
*offset* -- is the one that fits all of its links best (least squares).  A
link that does not fit the others is a wrong vote and is dropped.  Where the
links leave several groups of frames apart, the frames placed so far already
tell roughly where the stage puts the camera; each group is then looked for
there, all its frames' beads voting together, where a few agreeing beads are
already beyond chance.

**4. Each bead, in every frame that saw it.**  Every bead put back by its
frame's offset lands where the bead is on one map of the slide; beads that
land on the same spot are one bead, however far apart in the scan the frames
were.  Then the offsets and the beads' places are fitted together, so each
frame's offset rests on every bead it saw.

```figure The beads of all frames on one map, each frame's outline where its offset puts it: a bead seen in several frames lands on one spot (larger the more frames saw it); the small grey ones were seen once.
fig.set_size_inches(5.5, 4.5)
draw_beads(fig.subplots(), result, scan.frame, scan.xy_px, scan.shape)
```

**5. The camera against the stage.**  Only now are the stage's positions
used.  The offsets $o$ of the frames, in pixels, are fitted as a linear
function of the stage positions $s$, in micrometres:

$$o = M s + c$$

The 2 x 2 matrix $M$ holds everything: the pixel size along each camera
axis, the angle, and the mirror.  Frames far off the fit -- a stage that
missed a step -- are left out.

```figure Each frame's offset (dots) and where the fit to the stage puts it (circles), and the simulation's truth (crosses), in the order of the scan.
fig.set_size_inches(5.5, 4.5)
draw_offsets(fig.subplots(), result, truth=truth)
```

**6. Backlash.**  A stage's position lags a little behind where it was sent,
by an amount that depends on the direction it last moved in.  A serpentine
raster comes back along every other row, so those rows sit a fixed distance
off the others.  With *fit backlash* on, the fit has one more term per stage
axis, $+1$ or $-1$ by the direction that axis last moved; its size is
reported.  Without it the rows' offset goes into the residuals and into the pixel
size: 116.5 nm in place of 117.0 for the scan drawn here.

```figure What the fit leaves of each frame's offset, for a stage with 300 nm of backlash in x: with the backlash term (left) the frames scatter by the stage's own noise, without it (right) every other row is off.
fig.set_size_inches(9, 3.4)
left, right = fig.subplots(1, 2, sharey=True)
draw_residuals(left, result)
draw_residuals(right, pixel_size.measure(scan.frame, scan.xy_px, scan.stage_um, scan.shape,
                                         backlash=False))
left.set_title("with backlash: " + left.get_title())
right.set_title("without: " + right.get_title())
```

## In detail

**The vote.**  The differences $b_j - a_i$ of every spot $a_i$ in one frame
and $b_j$ in the other are binned at the tolerance $t$, and each bin's count
is summed with its eight neighbours'.  The winner is refined to the median of
the differences near it, and its votes are those within $t$.  Unrelated
spots' differences lie at most $\lambda = n_a n_b \pi t^2 / A$ to a disc of
radius $t$, for frames of $n_a$ and $n_b$ spots and $A$ square pixels, so the
count in any one disc is Poisson.  A vote of $k$ counts only where the
expected number of discs with $k$ or more by chance,

$$P(X \geq k \mid \lambda) \, \frac{A}{\pi t^2} \left( \frac{2}{k + 1} \right)^2 ,$$

is below $0.05$ divided by the number of pairs compared -- one false link in
twenty scans.  The factor $(2/(k+1))^2$ counts that two frames overlap less
the further apart they are.  Three votes are needed at least.  With more than
5000 pairs, or more than $2 \times 10^7$ differences, each frame votes with its
20 nearest by stage position only.

**The network.**  The offsets solve the links $o_j - o_i = d_{ij}$ by least
squares weighted by the square root of the votes, with each group's first
frame held at 0.  A link off the solution by more than $t$ is dropped, the
worst first, and the network solved again.

**Joining the groups.**  The largest group's offsets against the stage give
a map (step 5 with its frames alone).  A group apart has offsets known up to
one shift; the shift the map predicts is searched within a radius of 1 pixel
plus five times the map's uncertainty there -- its rms residual (at least 0.2
pixel) times $\sqrt{1 + h}$, $h$ the leverage of the group's frames in the
map's fit, which grows as the map extrapolates.  All the pairs between the
group's frames and the placed ones vote together, against the summed
$\lambda$ over that small search area.  While the placed frames lie along one
line the second axis is guessed: the same scale turned a quarter either way,
searched within 15 % of the distance.  This repeats until no group joins.

**The beads.**  Spots put back by their offsets and within $t$ of each other
are one bead, if seen in two frames or more and never twice in one frame
(two beads too close to tell apart are left out).  Offsets $o_f$ and bead
positions $p_b$ minimise $\sum | q - p_b - o_f |^2$ over all spots
$q$, by alternating means; a spot more than five robust standard deviations
($1.4826 \times$ the median residual, never below 0.05 pixel) from its bead
is taken out, and the fit made again.

**The stage.**  The fit is $o_f = M s_f + c + \beta_x u_{x,f} + \beta_y
u_{y,f}$, with $u_{x,f} = \pm 1$ the direction stage x last moved in to reach
frame $f$ (a step under 1 % of the median step is no move; the frames before
the first move take its direction).  A backlash term is fitted only for an
axis whose direction changes, and only where it is not in step with the
positions -- on a raster whose rows are each approached from one side only,
the direction is the row, and a backlash could not be told from the row's
step.  The hysteresis reported is $2 M^{-1} \beta$ along that axis, in nm:
positive when the stage ends further forward when it arrives moving forward.
A frame off by more than five robust standard deviations is left out, up to
three rounds.  At least three frames more than unknowns are needed.

**The numbers.**  $P = 1000\, M^{-1}$ (nm per pixel) has as columns the
stage's move for one pixel along camera x and along camera y.  Their lengths
are *pixel x* and *pixel y*; the angle of the first is the camera's angle to
the stage's x axis; the sign of $\det P$ is the mirror; the angle between the
two columns, less 90°, is the skew -- a warning beyond 1°, since a camera's
axes are square and a stage's should be too.  Errors are a delete-one
jackknife over the frames, the stage fit repeated without each in turn.

**Entering it.**  The pixel is a parameter of the microscope's entry that may
follow a device -- usually the 3D lens.  The measured value replaces that
device's state's value only; a pixel that followed nothing starts to follow
the 3D lens, the old value kept for its other states; rounded to 0.1 nm.

## Parameters

### path
A Micro-Manager stack, one frame per stage position.  Several frames at a
position are fine; a z stack is not -- take the beads at one focus.

### backlash
Leave it on for a serpentine raster, where rows are approached from both
sides.  A raster that always approaches from the same side has no backlash
to fit, and the term is left out by itself.

### enter
Off, the run only measures; the notice under the form then shows the entry
with the measured pixel in the microscope manager, where *Save* keeps it.
On, it is saved at once -- for a chain or a script.

### pixel_hint_nm
Searches each pair of frames only for shifts of half to twice the length the
stage's step makes at this pixel, and skips pairs that cannot overlap.  If it
leaves frames apart, the search is made again without it, so a wrong value
costs time, not the answer.

### x_tag
Name the tags when the stage is not Micro-Manager's XY stage; the error a run
gives lists the tags that change from frame to frame.

### stage_unit
Micro-Manager writes micrometres; some stages' own tags are in nanometres.

### tolerance_px
Beads are fitted to a few hundredths of a pixel, so 0.5 leaves room for the
field's distortion and a bead slightly out of focus.  Larger only for blurred
beads: a looser tolerance lets more pairs agree by chance.

### min_photons
Beads are bright; noise and dim debris are not.  Raise it for a slide with
much background.

### sigma_px
For the detection only; the default suits beads near focus at the usual
pixel sizes of 60 to 160 nm.

## Output

The text gives the pixel along camera x and y with their errors, the angle of
camera x to the stage's x axis, whether the image is mirrored, how far the
axes are from square, how many frames and beads were used, the residual in
nm, and the backlash of every axis that reverses.  Where the microscope
database has a pixel for the file's state, it is given beside the measured
one with the difference in percent.

A good calibration has a residual of a few to a few tens of nanometres --
the stage's own repeatability -- errors of the pixel well under 0.1 nm, and
a skew under 0.1°.  A skew of a degree or more, or a residual of hundreds of
nanometres, says that frames were placed wrongly or that the stage does not
move where it says: look at the residuals per frame.

The figure has three panels: each frame's offset and the fit's (as in step
5), the residuals per frame (frames left out as crosses), and the beads of
all frames on one map (step 4).
