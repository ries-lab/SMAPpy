---
title: Glossary
summary: The words of localization microscopy as SMAPpy uses them, with the other names they go by.
---

Every page explains a word where it first uses it, in a line; this page has
them all in one place, in the same wording.  The first name is the one
SMAPpy's buttons and pages use; the names in brackets are the ones found in
papers and in other software.

## Localizations

* **localization** (also called a spot, an event or a detection): one
  molecule's position, fitted in one camera frame -- one row of the table.
* **frame**: one camera image of the movie.
* **blink** (also called an on-event or a burst): the consecutive frames in
  which one fluorophore stays on.
* **merge blinks** (also called grouping or linking): combine the
  localizations in consecutive frames that come from one blinking event into
  one localization, with the photons added and a better precision.
* **on-time**: how many frames a blink lasts -- `n_in_group` once blinks are
  merged.
* **photons**: the light a fit attributes to one localization, in photons
  rather than camera counts.
* **background**: the photons per pixel that do not come from the molecule.
* **localization precision** (also called localization uncertainty, or the
  CRLB, Cramér-Rao lower bound; the column `xy_err_nm`): how far a fitted
  position is expected to lie from the true one, as one standard deviation,
  computed by the fitter from the photons, the background and the PSF.
* **NeNA** (nearest-neighbour analysis): the precision measured on the data
  itself, from how far apart the localizations of one molecule in
  consecutive frames lie.
* **FRC** (Fourier ring correlation): a resolution measure -- the finest
  detail at which two pictures, each made from half of the localizations,
  still agree.
* **ground truth**: in a simulation, where the molecules really were.

## Fitting

* **PSF** (point spread function): the image one point-like emitter makes on
  the camera; its shape changes with the emitter's z.
* **fit box** (also called ROI size or fitting window): the square of camera
  pixels cut around each candidate spot and fitted.
* **MLE** (maximum-likelihood estimation): fitting by finding the parameters
  under which the pixel values seen are the most probable, given the camera's
  noise.
* **spline calibration** (also called bead calibration, cspline, or an
  experimental PSF model): a 3D model of the PSF measured from z-stacks of
  fluorescent beads, from which the Spline 3D fitters read z.
* **astigmatism**: a cylindrical lens in the detection path makes a spot
  wider in x on one side of the focus and in y on the other, so its shape
  tells its z.
* **z**: the position along the optical axis, of a 3D localization.

## Camera

* **ADU** (analog-to-digital units, also called counts or grey values): the
  camera's raw pixel values.
* **offset** (also called baseline or bias): what a pixel reads with no
  light, in ADU.
* **conversion** (also called sensitivity, or gain in e-/ADU): photoelectrons
  per ADU -- what turns camera counts into photons.
* **EM gain** (electron multiplication gain): how much an EMCCD amplifies the
  signal before reading it out; the counts are divided by it on the way to
  photons.
* **EMCCD, sCMOS**: the two kinds of camera used for single molecules; an
  EMCCD amplifies the signal before reading it, an sCMOS reads every pixel
  through its own amplifier.
* **camera ROI** (also called the readout region or crop): the part of the
  chip the camera read out.

## The picture

* **layer**: one picture of the localizations with its own filter, colours
  and render mode; layers are drawn on top of each other.
* **filter** (also called bounds): the lowest and highest values of columns
  -- photons, precision, frame -- that a localization must lie between to be
  shown and measured.
* **selection**: what a plugin measures -- the localizations that pass the
  layer's filter, inside the drawn ROI and, when the 3D view's slab is used,
  inside the slab.
* **render mode**: how localizations become a picture -- *histogram* counts
  them per pixel, *Gaussian (fixed)* draws each as a blob of one fixed width,
  *Gaussian (precision)* as a blob as wide as its precision,
  *diffraction-limited* as wide as its
  fitted spot on the camera's pixels, like a widefield image.
* **LUT** (lookup table, also called a colour map): the colours that pixel
  values are drawn in, from low to high.
* **slab**: a slice of the sample of fixed thickness in z, in the 3D view.

## Drift

* **drift**: the slow movement of the sample against the camera during the
  acquisition, which smears the picture.
* **drift correction**: estimating the drift over time and subtracting it
  from every localization.
* **RCC** (redundant cross-correlation): a drift correction that makes a
  picture of each time window and cross-correlates every pair of them.
* **COMET**: a drift correction that finds the drift curve under which the
  localizations from different times overlap best, without making pictures.

## Regions

* **ROI** (region of interest): a shape -- rectangle, circle, polygon or line
  -- drawn in the view, or found or drawn in the ROI manager.
* **region**: a set of localizations measured together -- an ROI, or a
  cluster that an id column such as `cluster_id` names.
* **per-region steps**: plugins run once in every region, what each measures
  kept as that region's row.
* **cluster**: localizations that lie closer to each other than to the rest.
* **convex hull**: the smallest outline without dents that contains a set
  of localizations -- the shape a rubber band stretched around them takes.
* **radius of gyration** (Rg): the typical distance of a cluster's
  localizations from its centre, the root of their mean squared distance.
* **DBSCAN** (density-based spatial clustering of applications with noise): a
  clustering in which a localization with enough neighbours within a radius
  starts or extends a cluster, and the rest are noise.
* **complete spatial randomness** (CSR, also called a Poisson pattern):
  positions placed independently, every place in the window equally likely
  -- what "not clustered" means in a test.
* **pair correlation** (g(r), also called the pair correlation function or
  radial distribution function): the number of pairs of localizations at
  distance r, over what complete spatial randomness gives -- 1 for random
  positions, above 1 at the distances where they cluster.
* **Ripley's K** (and **L(r) - r**, also called H(r)): the mean number of
  neighbours within r over the density, and the same made a straight line,
  0 for random positions.
* **local density**: how many localizations lie within a radius of a
  localization -- a number per localization, not a region.
* **Voronoi tessellation** (also called a Voronoi diagram): the division of
  the plane into one cell per localization, the part of the plane closer to
  it than to any other; a small cell means a dense neighbourhood.
* **NPC** (nuclear pore complex): the eight-cornered ring of proteins in the
  nuclear envelope, a standard structure for testing labelling and
  resolution.
* **labelling efficiency** (also called degree of labelling, or effective
  labelling efficiency): the fraction of the target proteins that carry a
  label which is detected.

## Two colours

* **splitter** (also called a dual-view or image splitter): optics that put
  two copies of the image, filtered differently, side by side on one camera
  chip -- the two halves of a split frame.
* **ratiometric**: telling dyes apart by how their photons divide between
  the two halves, rather than by exciting them one at a time.
* **registration** (also called channel alignment, mapping or chromatic
  correction): the transformation -- shift, rotation, scale and distortion --
  that takes a position in one half to the same place in the other.

## Running things

* **chain** (also called a pipeline or a workflow): plugins run one after
  another as one plugin, their settings saved in a file.
* **batch**: a chain run over many files, one after the other.
