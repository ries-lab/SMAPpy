"""The camera of the Zenodo bead stacks, and reading them as SMAPpy's calibration does (docs/uipsf/data.md).

SMAPpy's camera database identifies it from the stacks' tags as the M2 microscope's Evolve 512 in
its normal-port mode, 5 MHz, gain 1: 4.7 e⁻/ADU, offset 398.6 ADU.  The read noise, 21 e⁻ rms, is
what the residuals show at zero signal (docs/uipsf/architecture.md, M3); the database has none.
The pixels are 116 nm in x and 127 nm in y in the camera's frame (docs/uipsf/data.md).
"""
from smappy.calibrate.input import read_bead_stacks
from smappy.metadata import CameraMetadata

CAMERA = CameraMetadata(conversion=4.7, offset=398.6, read_noise_e=21.0, pixelsize_um=[0.116, 0.127])


def read(files):
    """Every file's z stack, as SMAPpy's bead calibration reads it: ADU, z by increasing objective position."""
    return [st for f in files for st in read_bead_stacks(f)]
