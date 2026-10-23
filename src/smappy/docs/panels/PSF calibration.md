---
title: PSF calibration
summary: Makes the PSF calibration the Spline 3D fitters read, by one of three methods -- the bead average or a physical model of the microscope's pupil learnt with uiPSF v2 from bead z-stacks, or that model learnt in situ from the blinking molecules of a measurement -- and shows what each found and how well it fits.
widget: smappy.calibrate.psf_window.PSFCalibrationWindow
covers: [smappy.calibrate.pupil.half_split_uncertainty, smappy.calibrate.pupil.fit_quality, smappy.calibrate.plots.draw_pupil_fit_quality, smappy.torch_install.command, smappy.calibrate.psf_methods.BeadAverage, smappy.calibrate.psf_methods.PupilModel, smappy.calibrate.pupil.run_calibration, smappy.calibrate.pupil.bead_rois, smappy.calibrate.pupil.single_rois, smappy.calibrate.pupil.dual_rois, smappy.calibrate.pupil.spline_calibration, smappy.calibrate.pupil.dual_calibration, smappy.calibrate.pupil.save, smappy.calibrate.pupil.OpticsSettings, smappy.calibrate.pupil.BeadSettings, smappy.calibrate.core.find_stack_beads, smappy.calibrate.psf_window.PSFCalibrationWindow, smappy.calibrate.pupil.estimate_seconds, smappy.calibrate.psf_methods.LearningProgress, smappy.calibrate.model_views.average_views, smappy.calibrate.model_views.pupil_views, smappy.calibrate.model_views.draw_model_vs_data, smappy.calibrate.model_views.planes, smappy.calibrate.stack_browser.StackBrowser, smappy.calibrate.stack_view.StackViewer, smappy.calibrate.pupil.covered_z_nm, smappy.calibrate.psf_methods.InSitu, smappy.calibrate.insitu_run.run_insitu, smappy.calibrate.insitu_run.movie_rois, smappy.calibrate.insitu_run.block_order, smappy.calibrate.insitu_run.enough, smappy.calibrate.insitu_run.start_astigmatism, smappy.calibrate.insitu.quotas, smappy.calibrate.insitu_run.insitu_spline, smappy.calibrate.insitu_run.InSituModelSettings, smappy.calibrate.insitu.InSituSettings, smappy.calibrate.insitu.select, smappy.calibrate.insitu_views.draw_z_bins, smappy.calibrate.insitu_views.draw_rounds, smappy.calibrate.insitu_views.model_vs_data, uipsf.workflows.insitu_rounds.calibrate_insitu, uipsf.workflows.insitu_rounds.coverslip_edge, uipsf.workflows.insitu_rounds.focus_from_data, uipsf.workflows.insitu.twin]
---

## What it does

A 3D fit needs to know what one molecule looks like at every height, the
*point spread function* (PSF).  This window makes that calibration by one of
three methods (*method*):

* **bead average**: the beads are aligned onto one another, averaged and
  smoothed -- the [bead calibration](plugin:Panels/Bead calibration)'s
  method, with its settings, pages and fit quality, explained on its page;
* **pupil model (uiPSF)**: the PSF is learnt as a **model of the optics**,
  explained here;
* **in situ (blinking molecules)**: the same model learnt from the blinking
  molecules of a measurement itself, with no beads (*In situ*, below).

All three write the same kind of file and hand it to the same fitter, so they
can be compared.

The pupil model: the light from a point is collected by the objective, and
everything the microscope does to it -- the astigmatism of a cylindrical lens,
the objective's own aberrations, the refractive indices it passes through --
is described by the objective's **pupil**, a disc whose phase is written as a
sum of Zernike modes (standard shapes such as astigmatism, coma or spherical
aberration).  From the pupil the PSF at any height follows by the physics of
light, so the model is smooth without smoothing and also holds where the beads
were noisy.  The method is uiPSF (Liu et al. 2024,
[doi:10.1038/s41592-024-02282-x](https://doi.org/10.1038/s41592-024-02282-x));
this is its re-implementation in PyTorch (uiPSF v2), run inside SMAPpy.

Its input is the bead calibration's: z-stacks of sub-diffraction fluorescent
beads, imaged while the objective steps through focus.  Its output is the same
kind of [spline calibration](plugin:Panels/Glossary) file, so
[Spline 3D](plugin:Localize/Spline 3D) and
[Spline 3D 2C](plugin:Localize/Spline 3D 2C) read it as they read a bead
average, and beside it the model itself, which can start the next calibration
of the same microscope.

Use it to calibrate for fitting, and to look at the microscope: every run
shows the aberrations with their uncertainties and where the model and the
data disagree.  Untick *save model* to only look.

A calibration goes in two steps.  *Detect beads* finds the beads, rings them
on the *Stacks* page and says how many there are and, for the pupil model,
about how long learning them will take -- so a run of an afternoon is never
started by accident.  *Calibrate* then learns from them; pressed first, it
detects too.  Pressed again after a change, it calibrates with the settings
as they are then.  A bar shows how far the run is and about how long it has left.

The pupil model needs PyTorch, an optional part of SMAPpy.  Without it, or
with one too old, *Install PyTorch* installs the right build for the computer
(for Apple's GPU, an NVIDIA card or the CPU) into the Python running SMAPpy;
the window also shows the command to type instead.  The bead average works
without it.  It uses the
GPU where PyTorch finds one (NVIDIA, or Apple's).

## How it works

This section and the next describe the pupil model from beads; in situ
learning has its own section after them, and the bead average's are on the
[bead calibration's page](plugin:Panels/Bead calibration).

**Looking at the stacks.**  The *Stacks* page shows each added file as it was
recorded, one plane at a time, with a slider for the file and one for the
plane.  After *Detect beads* each bead found is ringed and numbered with its
row in the table; after a calibration the ring's colour says whether it was
used (blue) or rejected (red).  Clicking a ring selects the bead.

**Finding the beads** is the bead calibration's own: the same detection, the
same settings under *beads*, the same verdict on each bead's brightness.  Of
that verdict only *too dim* is applied here, before learning: a spot much
fainter than the typical bead (below it divided by the square root of
*brightness range*; the typical bead is the median weighted by each spot's
light, so a crowd of specks does not pull it down) is a speck, not a bead,
and its fit would be noise.  A pair of two channels is judged by its light
in both together.  Specks do not cost a bead its isolation either, so a lower
*threshold* finds more beads rather than fewer.  A *too bright* bead is kept and judged by its fit, which
shows a doublet.  The camera's numbers come from the camera database, as a fit of
the same files would take them.

**Cutting** one box of raw pixels around each bead, converted to photons, with
saturated pixels left out of the fit rather than the bead dropped.  The first
slice of every stack is left out: on real data it is often brighter than the
rest.

**Estimating the time.**  After detection the model is built for these beads
and its image of one bead is computed a few times and timed, which takes a
few hundredths of a second.  Learning costs about two such images per bead and
per fitted number (the pupil's modes, the blur, each bead's position,
photons and background), measured on simulated beads at Zernike orders 3 to
8, so the estimate is that time × 2 × the beads × the numbers.  It is good to
about a factor of two: the number of iterations a fit needs depends on the
beads.

**Choosing beads by hand.**  The detected beads are listed in the table at
once.  A bead marked there (*Exclude / include*) is left out of the next
calibration -- or, after one, for the pupil model a rejected bead let in.
Press *Calibrate* again to apply the marks.  A bead excluded (or included) by
hand stays so through every later *Calibrate* until its mark is withdrawn, as
long as the beads are not detected again.

**Learning** fits one model to all beads at once: the pupil's Zernike modes
(shared), an extra blur of the image (shared), and each bead's position,
brightness and background.  It runs in two stages, the low orders first, then
all of them.  With *start from*, a previous model of the microscope starts
the first stage: its low orders move with the beads before the higher ones are
freed, which saves a third to half of the time and is safe even from a model
of another camera.  For two channels the transformation starts from where an
elliptical Gaussian fit puts each bead in both channels, slice by slice --
closer than the bead centres the detection pairs, and never from the previous
model's, which a splitter can lose between sessions -- unless too few beads
give a position in both.

**Rejecting beads** by their fit: a bead whose residuals, brightness or
position stand out from the others by more than *rejection (k)* robust
standard deviations is left out, and the model is learnt again without it.
The table says which beads were used and why the others were not.

**Checking**, after every calibration without a button.  The fit quality
fits every slice of every bead with the saved spline by the [Spline 3D](plugin:Localize/Spline 3D) fitter itself, as
the bead average's fit quality and SMAP's bead calibration do: what a
localisation with this calibration will see.  Each bead becomes a curve of
fitted z against frame; a bead whose curve jumps or bends away from the
others is one to exclude, and clicking its curve selects it in the table.
For dual colour each pair is fitted by [Spline 3D 2C](plugin:Localize/Spline 3D 2C)'s
global fit, both channels at once with one position and one z, as the data
will be.

Then the model is set against the data (*Model vs data*): every bead is
moved onto the model -- each slice to the height the fit found for it, the
image shifted by the bead's fitted x and y in each channel -- with its
background removed and divided by its photons.  The model of the same bead
is moved the same way, so the residual shows the misfit and never the
alignment.  The average of the used beads is shown, or one bead alone.  For
the bead average the data is its average before smoothing and the model the
smoothed PSF the spline is made from.

**The ends of the stacks.**  Each bead sits at its own height, so its slices
cover the stage's range shifted by it, and at the top and bottom only some
beads remain.  The pupil model keeps to the range every used bead covers: the
spline on *auto* reaches that far, *Model vs data* shows only those planes,
and the fit quality leaves out the slices of a bead beyond the spline.  The
bead average crops its average the same way.

**Saving.**  With *save model*, the model is turned into a spline: the PSF of a
point on the central pixel is computed on a grid through the z range every
used bead covers,
each pixel as the model saw it when it learnt, and normalised as the bead
average is.
The calibration and the model file go beside the stacks, named as the bead
calibration names them, and the calibration goes to the fitter.

## In detail

The model is vectorial and high-NA: the pupil field carries the dipole
emission's polarisation through the immersion, coverslip and sample media
given under *microscope*, and the image is taken at each camera pixel's
centre: the pixel's width ends up in the learnt blur, which is therefore a
little wider than the optics' own (about 30 nm added in quadrature for 110 nm
pixels).
The likelihood is Poisson with the camera's read noise added as a shifted
Poisson, SMAPpy's own fitters' likelihood; with EM gain, the excess noise
factor 2.  Slices closer than 50 nm are thinned for learning.  The first
stage frees the modes up to radial order 4 and stops early; the second frees
all of them up to *Zernike order* (auto: 6).

The pupil magnitude is 1 plus Zernike modes: all of them up to the order
(*free*), only the rotationally symmetric ones (*symmetric*: Noll 4, 11, 22,
...), or none, so that only the apodisation of the objective remains
(*fixed*).  The blur is a Gaussian, σ for x and y, learnt (*free*) or held at
*blur σ* (*fixed*).  Which of these choices are right for which data is under
investigation; they are offered so that their effect can be seen.

Bead rejection scores each bead from the learning fit itself: its mean
deviance per pixel (about 1 where the model is right), |log| of its photons
against the median, its distance from its box's centre and the fraction of
its pixels that were saturated.  A score more than *rejection (k)* robust
standard deviations (1.4826 × the median absolute deviation) above the median
rejects the bead, with floors so that near-identical beads are not rejected
for tiny differences; *include* and *exclude* in the table always win.

The spline is sampled on the knots of SMAPpy's spline grid, a pixel apart,
each pixel sampled as in learning, at its centre, with z steps of *spline z step*
(auto: the stacks' step) through ± *spline z range* (auto: the stacks'
range).  One channel: clipped at zero and scaled so that the brightest plane
sums to 1.  Two channels: both sampled on one z grid and scaled together, so
that each channel's share of the light is the learnt one.  The pixels are not
integrated again: the learnt blur already holds them, and integrating once more
widened the spline's PSF, which pulled Spline 3D's z in by 1 % and bent it by
up to 7 nm against the model's own localisation of the same beads.

In *Model vs data* the beads' images come from the learning itself: the used
beads' fitted positions, photons and backgrounds, and the other beads
localised on the final pupil.  Each slice goes to the emitter's height in it
(stage drift and z scale included) by linear interpolation onto the slices
learnt from; x and y are shifted by the bead's mean position over its
slices, in each channel where the transform puts it.  Planes outside a bead's
own slices are left blank, and only the planes every used bead reaches are
shown.  The XY planes shown are 100 nm apart, or 150 nm when 100 nm would
need more than nine.  Each image is scaled to its own peak, the residual
symmetric about zero.

The covered range is the highest of the used beads' lowest emitter z and the
lowest of their highest, from the fitted geometry; the spline's *auto* range
is the nearer of its two ends about focus.  A slice of the fit quality is left
out when the nominal z plus the bead's offset (the median of fitted − nominal
over its central slices) lies beyond the spline.

The progress bar's fraction during learning is how far the fall of the
objective per iteration has come from the first iteration's to the solver's
tolerance (10⁻³), on a log scale: a fit's gain falls roughly geometrically,
and how many iterations it will need is not known in advance.  The low orders
are its first 45 %, all orders the next 45 %, the fit quality and the model
views the rest.  The time left is the time so far scaled by what is left.

The fit quality fits every bead, used or not, so a rejected bead's curve
shows why.  The fit box is the bead box less two pixels a side; the camera's
read noise is added to the data as the learning's likelihood does; each slice
is started from five heights and the best likelihood kept, so an ambiguous
PSF shows as jumps.  The *centred error* removes each bead's own offset (its
height on the coverslip is not known), leaving the curve's distortion.  For
dual colour each slice of a pair is one global fit: x, y and z shared, the
secondary channel placed where the learnt transform puts it and the photons
free in each channel, so a wrong transform or channels whose z disagree show
as they would in the data; one z per pair, and the photon ratio beside it.

## In situ

A PSF measured with beads on the coverslip is not quite the PSF of molecules
in a cell: the light from deep in the sample crosses the index mismatch
between the sample and the immersion oil, which adds spherical aberration that
grows with depth, and the sample itself may add more.  In situ learning fits
the pupil model to the molecules of the measurement instead (uiPSF's in situ
learning, Liu et al. 2024).  A molecule's height in the sample is not known,
unlike a bead's in a z-stack: it is learnt with the pupil, molecule by
molecule, and the *focus* -- how far above the coverslip the objective put its
focal plane -- with them.

**Finding the molecules.**  The movies are read in blocks of 200 frames,
coarse to fine over the whole of each -- first ten blocks spread from its
start to its end, then the blocks between them -- until *molecules* spots pass
the selection below, or the movies are read.  How many frames that takes
depends on how dense the molecules are and how large the field, which is why
it is a number of molecules that is set and not of frames.  Reading stops
with some z bins still short once another holds twenty times its share: those
heights are rare in the data (below the coverslip there is nothing), and
reading on would not fill them.  In each block the background is a sliding
median over time.  Spots are looked for at three sizes and kept when they
hold significantly more photons than the background around them, never by
comparison with the brightest spot of the frame: a molecule far from focus is
dim per pixel but has as many photons, and it is these that tell most about
the PSF.  Each spot is fitted with an elliptical Gaussian; a molecule's frames
are linked into one blink.  A molecule with another closer than three widths
of each is rejected with all its frames; a farther neighbour's pixels are
masked, unless it has less than a fifth of the molecule's photons.

**Choosing them.**  The molecules are sorted by an estimate of their height
-- an astigmatic spot's elongation (*z proxy*) -- into 21 z bins, and each bin's
share of *molecules* is drawn at random from a window of brightness below its
brightest: the very brightest are where two unresolved molecules collect.  A
bin with fewer than its share gives all it has, and the others make up the
number.  A tenth of the molecules is held out of learning.  The *z bins*
page shows what was chosen.

**Learning, in rounds.**  The first round learns from the Gaussian's choice,
started from *start from* or the *astigmatism* typed in -- the start sets the
sign of astigmatism and spherical aberration, which the data alone can barely
tell.  Each further round localises every molecule found on the model, chooses
again by the model's height, photons and fit, and learns again from the last
model, until nothing moves beyond its noise or after *rounds*.

**The coverslip.**  The focus and all the heights can move together, changing
the images only a little; the fit alone places the focus to some tens of
nanometres.  With *on the coverslip*, the lowest molecules are taken to lie on
the coverslip: the lower edge of the localised heights must be at 0, and
where it is found moves the focus in the next round.  With molecules on the
coverslip this pins the focus to a few nanometres.  On *auto* the *focus* is
found this way before the first round.

**The sign.**  After the last round the mirror image of the learnt pupil --
astigmatism and spherical aberration of the other sign, the molecules
mirrored about the focus -- is fitted too.  If the data clearly prefer it, the
status bar warns: the start's sign may have been wrong.

## Controls

### bead stacks
The files or folders of bead z-stacks, all learnt from together; for *in situ*
the section is *movies*, the measurements' movies.  Several at once -- the
folders of a few z-stacks, say -- with *Add folder...* (Cmd- or
Ctrl-click), by dropping files or folders from the Finder or Explorer on the
window, or on a Mac with *Add files...*, which takes files and folders
together.  A blinking movie added for a bead method, or a bead z-stack for
*in situ*, is said at once -- told apart by what Micro-Manager wrote, z
slices or time points, else by length -- with the method that reads it
offered, or the file removed.

### Add files...
Adds stacks.  On a Mac the system's open panel takes files and folders
together, several at once (Cmd- or Shift-click), so the folders of a few
z-stacks are added in one go, each searched as *Add folder...* searches it.
On Windows and Linux it takes files only; drop folders on the window instead.

### Add folder...
Adds folders, several at once; with *search subdirectories* ticked, every
acquisition inside them.

### Remove
Removes the selected entries.

### search subdirectories
Whether *Add folder...*, and a folder dropped on the window, looks inside the
folder's subfolders.

### method
*bead average*, *pupil model (uiPSF)* or *in situ (blinking molecules)*.  The
window shows only what the method reads.  Switching keeps the files and
settings and clears the result.

### mode
*Single channel*, or *Dual colour* for a camera split into two halves: a PSF
per half and the transformation between them.  The pupil model learns a pupil
per channel, with the photon share and the focus offset.

### start from
Pupil model only, under *model*.  A previous model of this microscope: its `.uipsf.h5` file, or the calibration
saved beside it.  It starts the learning and is drawn beside the new
Zernikes, so a change in the microscope shows.  Empty, the learning starts
from the beads alone.

### save model
On: the calibration (and for the pupil model, the model file) is saved beside
the stacks, named as the bead calibration names it and never over another
file, and handed to the fitter.  Off: the figures only; *Save as...* still
saves.

### microscope
The microscope control, folded, titled with the microscope found for the
first stack (*M2 · 3DA on · iXon 1 · 127 × 116 nm*), orange when something
the method needs is missing.  Open: the microscope, recognised or chosen by
hand for stacks no entry recognises (for this window's runs, not the entry),
the camera, and only the values the method uses -- the camera and the pixel
for the bead average, the optics too for a model.  Each value comes from the
microscope database for the microscope that took the first stack (recognised
from its metadata, else the last one used) and the state its devices were
in, and from the camera database for its camera; the field shows it in grey,
with where it comes from in its label's tooltip.  The values are locked: the
lock opens them, a value typed over the found one wins and is marked, and ↺
gives it back.  Also the
[Microscope manager](plugin:Panels/Microscope manager).  There is no default for the NA and the wavelength: a
microscope the database does not know needs them typed, because a wrong value
cannot be noticed in the result, only in a model that does not fit.  The
calibration keeps the microscope's state -- which filter, the 3D lens in or
out -- and a fit warns when its data were taken otherwise.

### NA
The objective's numerical aperture, as engraved on it.

### emission
The beads' emission wavelength, the centre of the filter's band.  On *auto*
read from the name of the filter the microscope was set to (676/37: 676 nm);
with a filter whose name is not a band (a long-pass) it has to be typed.

### emission, 2nd
For dual colour, the secondary channel's; auto: the microscope's arm on the
other half of the chip.  An arm whose filter gives no wavelength (an empty
slider, the band set by the splitting dichroic) has to be typed; a microscope
with no second arm takes the same as *emission*.

### immersion index
The immersion oil's refractive index; auto: the microscope's, else 1.518 (the
standard oils of Olympus, Leica and Zeiss).

### model
The choices of the model, explained under *In detail*.

### Zernike order
The highest radial order of the pupil's modes.  Higher orders describe finer
aberrations and cost more time and more beads.

### pupil magnitude
*free*, *symmetric* or *fixed* (the apodisation only).

### blur
Whether the extra blur is learnt or held at *blur σ*.

### blur σ
The blur's start, or its value when held.

### bead diameter
The beads' size, so that it is taken out of the model: a 100 nm bead is not a
point.  0 for beads much smaller than the resolution.

### beads
How beads are found, and for the bead average how they are aligned and
averaged: the bead calibration's settings, which its
[page](plugin:Panels/Bead calibration) explains.  The pupil model reads only
the finding; *bead box* is also the size of the boxes it learns from and of
its spline.

### in situ
In situ learning's own settings, and how many molecules to learn from; the
few settings of how they are found that depend on the data are under *more*.

### focus
How far above the coverslip the objective put its focal plane, as it was moved
from the coverslip's focus.  A rough value is enough: the rounds learn it.
*auto*: estimated from the molecules on the coverslip before learning.

### astigmatism
Under *more*, and rarely typed: on *auto* the microscope database gives it, for
the state the 3D lens was in (the entry's `astigmatism_nm`, read from bead
models of the microscope).  It is the start's vertical astigmatism in nm of
wavefront, and what it really sets is the sign: the same molecules are
explained by the mirror-image pupil with z flipped, and the start decides
which of the two is learnt.  For a microscope the database does not know,
*more* opens with the field in orange and calibrating asks for it: read it
from a bead calibration of the microscope, or give a model under *start
from*, with which the field is not used and not shown.

### on the coverslip
Tick when the sample touches the coverslip, so that its lowest molecules lie
on it: their edge pins the focus.  Untick for a sample that floats above it.

### rounds
At most this many rounds; most stop earlier, when nothing moves any more.

### spline z
What the spline's z means: the height above the plane in focus (z = 0 where
the spot is sharpest, as for beads), or the height above the coverslip, so
that Spline 3D reports where the molecule is in the sample.

### molecules
Each frame of a blink counts as one: it is fitted as one.  About 2000 is
enough for the Zernikes up to the default order; more costs learning time in
proportion.  Reading and finding them took about ten seconds per thousand
frames of 128 × 128 pixels on a laptop.

### region
The square each molecule is fitted in.  Spots 600 nm or more from focus
outgrow 15 pixels; a larger region keeps them whole, at the cost of more
neighbours inside it.

### significance
How far above the background's noise a spot must be.  Lower finds dimmer
molecules, and more noise.

### z proxy
*astigmatism* for an astigmatic PSF; *size* for one without astigmatism,
which sorts by distance from focus only.

### spline
How the model is sampled into the spline the fitter reads.

### spline z step
The spline's z spacing; auto, the stacks' own.

### spline z range
How far above and below focus the spline reaches; auto, the range every used
bead covers (*The ends of the stacks*).

### Detect beads
*Detect molecules* for *in situ*: finds the molecules, rings them on their
frames and counts them by state in the table; the *Stacks* page then steps
through the frames molecules were found on, and before it shows a movie's
first frame only.  For beads: finds the beads, lists them in the table, rings them on the *Stacks* page and
says how many there are and how long calibrating them will take (*How it
works*).  A new detection drops an earlier calibration from the window; its
files stay.  Changing the files,
the *mode*, the *method* or a setting under *beads* drops the detection.

### Calibrate
Makes the calibration from the detected beads, detecting them first if they
are not, with the settings as they are and the beads marked in the table;
with *save model*, saves it.  After a calibration it is pressed again for
anything: a setting under *beads* detects the beads anew (and the marks,
which named the old ones, go); any other setting is used with the same
beads.  When nothing but the marks changed, the pupil model starts from the
last fit, which is quicker, and both methods save over that run's files;
otherwise a new calibration saves to a new file.  The fit quality and the model against
the data follow at once.  The bar under the buttons shows the fraction done
and the time left; the status line counts the iterations.

### Stop
Pupil model: stops the learning after the current iteration.  Nothing is
saved.

### Install PyTorch
Shown with the pupil model while PyTorch is missing or too old: installs it
into the Python running SMAPpy, with pip's progress in the status line;
pressed again, it stops.  The pupil model can be used as soon as it is done.

### Save as...
Saves the calibration under another name, or one made with *save model* off.

### Browse stacks...
Under the table.  The beads, slice by slice in any direction (XY, XZ, YZ), in
their own window, with the model and the residual, data − model, beside them
as ticked (*model*, *residual*), each on its own contrast.  With *average* the
average of the used beads; without, a second slider steps through the beads
one at a time, each moved onto the model as the average is.

### Uncertainty from halves...
Pupil model, after a calibration, with at least 8 used beads.  Asks how many
splits (3 by default, 2 to 10) and says about how long one takes, then
splits the used beads at random into two halves that many times and learns
each half on its own, from the same start as the run -- its *start from*
model, or none -- through the same stages, never from the finished fit.  A
split's two halves are two calibrations from different beads; how far they
disagree is the uncertainty: each half has twice the full fit's variance, so
((A − B) / 2)² estimates the full fit's, and the root of its mean over the
splits is shown.  Inside a half nothing is rejected or held out, so every
half measures the same beads.  A split costs about one calibration (two
fits of half the beads); 3 splits give the error bar to about ±40 %, 5 to
about ±30 %, and more add little when the beads are few, since the splits
then overlap.  *Stop* stops it.  The result goes into the *Pupil and
Zernikes* pages and the *Zernike table*, and, when the calibration was
saved, into its model file.

Against the CRLB it shows what the CRLB leaves out: beads that differ from
one another, and a learning that does not end in one place from one start.
It cannot show an error every half shares -- a wrong NA, index or
wavelength.

### Use in the Spline 3D fitter
Puts the saved calibration into the fitter again (it is put there when it is
saved).  For dual colour, the Spline 3D 2C fitter.  Every saved calibration is
also listed, with the microscope and the device states of its beads, so that
a fitter left on *auto* finds it for data taken the same way, without this
button ([Spline 3D](plugin:Localize/Spline 3D)).

### Add to microscope...
After a pupil model or an in situ run: keeps the learnt pupil -- every
Zernike of phase, in nm of wavefront, and of magnitude, with the blur -- in
the microscope's entry, for the state its 3D lens was in, in your
`microscopes.json` (the entry before is kept as a dated version).  It
describes the objective, and in situ learning starts from it when there is
no *start from*; its vertical astigmatism stands in for the entry's
*astigmatism*.  Asks first, saying what it replaces.  Only for beads taken
on a recognised microscope in its current setup.

### Exclude / include
Marks the selected beads for the next *Calibrate*: a bead to be left out,
or, after a calibration, for the pupil model a rejected one to be included.  Pressed again, the mark is
withdrawn; a choice by hand stays until then.  Space does the same.

## Output

*Stacks* is there from the start (*How it works*); the others appear after a
calibration.  Both methods have:

- **Model vs data**: XY planes 100 or 150 nm apart as columns, and three
  rows almost touching: the data, the model and the residual (data − model),
  a set of three per channel, two channels side by side.  A plane's data and model share one contrast,
  the data's peak in that plane, so a model too bright or too dim shows as
  well as one of the wrong shape; each residual is on its own scale,
  symmetric about 0 (red above, blue below), and under it is how large it is:
  its largest |data − model| as a percentage of the plane's data peak (the
  number the stack browser gives).  A side section through the centre
  (*section*: XZ, YZ) is one picture instead, data, model and residual side
  by side, each on its own contrast, and the second channel beside the first.  With beads selected in the
  table the first of them is shown alone, unless *average* is ticked;
  otherwise the average of the used beads.  A good model leaves a residual of
  noise, with no ring, lobe or tilt that repeats from plane to plane; a
  single bead whose residual has a second spot is a double.

The bead average's other pages are the bead calibration's (*Overview*,
*Transformation* for dual colour, *Fit quality*, *Field diagnostics*, *Bead
diagnostics*), explained on its page.  The pupil model's:

- **Pupil and Zernikes**: on the left the learnt phase and magnitude over the
  pupil, the phase in the *units* chosen above the page: radians, as the
  model is fitted; waves, radians / 2π; or nm of wavefront, waves times the
  *emission* wavelength.  (Maréchal's criterion for a near-perfect PSF is an
  rms wavefront error below 1/14 of a wave.)  The phase of an astigmatic PSF is a saddle; the
  magnitude should be close to 1.  On the right every mode's coefficient with
  its uncertainty as an error bar, beside the *start from* model's when
  there is one.  The error bar is one standard deviation of the
  coefficient's Cramér-Rao lower bound (CRLB; [Glossary](plugin:Panels/Glossary)):
  from the Fisher information of the whole fit, all beads together, inverted
  at the solution, so it already allows for each bead's position, photons and
  background being unknown too.  It is the precision the beads' photon noise
  allows if the model is right; an error of the model itself -- a wrong NA,
  index or wavelength, or a bead that is not a point -- is not in it, so the
  real uncertainty can be larger.  After *Uncertainty from halves...* each
  bar has a second, wider grey whisker, the uncertainty from the halves, and
  a dot beside it for every half's own value: the whisker says how large the
  error is, the dots whether one half went its own way.  A grey whisker much
  longer than the black one says the beads differ more than photon noise
  explains.  The aberrations a calibration is read by have colours
  of their own -- astigmatism, coma 1 and 2 (radial orders 3 and 5),
  spherical 1 and 2 (orders 4 and 6) -- and their phase values are written
  over the bars.  Each bar is
  named by a short abbreviation over its Noll index: *Df* defocus, *As*
  astigmatism, *Co* coma, *Tr* trefoil, *Sp* spherical, *Qu* quadrafoil and
  so on, *x* for the cos term (astigmatism at 0°/90°) and *y* for the sin
  term (±45°), a leading *2* or *3* for the second or third order of the
  family (*2Sp* the secondary spherical); the line under the figure lists
  them, and the legend under the bars names the colours.  A
  difference far beyond the error bars is a change in the microscope.
- **Zernike table**: the same numbers, every channel's phase -- in radians,
  waves and nm -- and magnitude with their uncertainties (and the *start from* model's); after
  *Uncertainty from halves...* also the halves' uncertainty (*± halves*, in
  rad and nm, and for the magnitude) and *halves / CRLB*, above 1 where the
  CRLB is optimistic.  The key
  aberrations' rows in their colours; *Copy* puts the table on the
  clipboard, tab-separated, for a spreadsheet.
- **Residual atlas**: where the model and the data disagree, grouped by
  slice, defocus, distance from the spot's centre and position on the field.
  Each group's mean deviance is about 1 where the model is right; a group
  marked in red differs by more than noise, and the title says what that
  usually points to (a stage scale, the blur, field-dependent aberrations).
- **Fit quality**: each bead's fitted z against
  the frame, and its centred error against the nominal z.  The used beads in
  colour, the others grey and dashed, the ones selected in the table thick.
  Click a curve to select its bead.  On good beads the curves lie on top of
  one another and the error stays within a few tens of nanometres.  For dual
  colour one curve per pair, from the two-channel fit.
- **Beads**: where the beads were on the chip, coloured by whether they were
  used, rejected or excluded, the selected ones ringed; a click selects one.

*Pupil and Zernikes*, *Zernike table* and *Residual atlas* describe all beads
together (one pupil per channel), so they do not follow the table's
selection.

For dual colour, *Pupil and Zernikes* comes once per channel.

The bead average's table is the bead calibration's.  The pupil model's lists each bead with its brightness verdict (as the bead
calibration would give it), its state, the reason for a rejection, and its
deviance, photon score and offset from the box's centre.

For *in situ* the table counts the frames and molecules in each state: used,
held out, not drawn, and rejected with the reason.  Its pages are *Pupil and
Zernikes*, *Zernike table* and *Residual atlas* as above, and:

- **z bins**: the frames by their height -- the Gaussian's estimate before
  calibrating, the model's after -- stacked by state.  A height with nothing
  used is one the model learnt nothing at.
- **Rounds and edge**: the focus each round with its uncertainty and the
  coverslip's prior; how much the model changed from round to round (the rounds
  stop below 1); and the lowest localised heights with the coverslip edge
  found in them.  The twin test's verdict is under the figure.
- **Model vs data**: the used frames in bins of their height, the mean of the
  data beside the mean of the model for the same molecules, and the
  difference; a good model leaves noise.

## References

- Liu S, Chen J, Hellgoth J, *et al.* (2024) Universal inverse modeling of point
  spread functions for SMLM localization and microscope characterization.
  *Nat Methods* 21:1082–1093.
  [doi:10.1038/s41592-024-02282-x](https://doi.org/10.1038/s41592-024-02282-x)
