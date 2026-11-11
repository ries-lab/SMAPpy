"""The example data's pixelation study: what every step shares (the paths, the camera, the optics).

DECODE's example package (U2OS microtubules, alpha+beta tubulin, an Evolve 512 EMCCD at EM gain 50,
2020-07-23) with the 11 bead stacks it was calibrated from, at ``$DECODE_SMLM_DATA``
(``~/.cache/decode_smlm``) ``/experimental_data_workflow``.  The beads were read out without EM gain
and the movie with it, so the two are mirror images (plan, (d)).  SMAPpy mirrors neither, and a pupil
model cannot be mirrored, so the movie is flipped once along x (`flip_movie.py`) into the beads'
orientation, and every model -- SMAP's spline, the bead pupil model, the in situ model learnt on the
flipped movie -- and every fitter work there.  Pixelation and z histograms do not care about a
mirror; nothing is flipped twice.

The camera and the pixel size are SMAP's fit of this movie (``frames_sml.mat``: 117 x 127 nm, 5
e-/ADU, offset 398.6, EM gain 50); the optics the uiPSF v1 model of these beads used in 2024 (NA 1.43,
indices 1.516 / 1.335, 600 nm); the beads' slices 10 nm apart, as the files and SMAP's calibration say.
"""
import os
import pathlib

DATA = pathlib.Path(os.environ.get("DECODE_SMLM_DATA", pathlib.Path.home() / ".cache" / "decode_smlm"))
PACKAGE = DATA / "experimental_data_workflow"
MOVIE = PACKAGE / "frames.tif"
FLIPPED = PACKAGE / "frames_beadside.tif"           # the movie flipped along x, into the beads' orientation
BEADS = sorted((PACKAGE / "beads").glob("Pos*/*.ome.tif"))
OUT = DATA / "pixelation"                           # models, fits and figures of this study
SMAP_SPLINE = PACKAGE / "spline_calibration_3dcal.mat"

PIXEL_NM = (117.0, 127.0)                           # x, y: SMAP's fit of the movie
# the movie's camera.  EM on, as it was taken; SMAPpy's calibrations of the beads say EM off, and are
# marked as the movie's orientation where a fitter compares the two (`as_movie`)
MOVIE_CAMERA = dict(conversion=5.0, offset=398.6, em_on=True, emgain=50.0, pixelsize_um=(0.117, 0.127))
# the beads': the gain and offset the 2024 uiPSF model of them used, no EM gain
BEAD_CAMERA = dict(conversion=0.22, offset=400.0, em_on=False, pixelsize_um=(0.117, 0.127))


def optics():
    from smappy.calibrate.pupil import OpticsSettings
    return OpticsSettings(NA=1.43, wavelength_nm=600.0, n_immersion=1.516, n_medium=1.335, n_coverslip=1.516,
                          pixel_x_nm=PIXEL_NM[0], pixel_y_nm=PIXEL_NM[1])


def camera(which):
    from smappy.metadata import CameraMetadata
    return CameraMetadata(**(MOVIE_CAMERA if which == "movie" else BEAD_CAMERA))
