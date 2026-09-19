"""The camera options shared by the command-line entry points.

Most acquisitions need none of these: the camera database recognises the
camera from the file's own tags and supplies what the metadata does not carry.
They are for the rest -- a camera nobody has entered, or one whose file has no
tag to recognise it by.  The layers override in this order, each winning over
the ones before it:

    camera database  <  image metadata  <  --camera config  <  options
"""
from __future__ import annotations

import argparse

from ..io.tiff import camera_metadata
from ..metadata import CameraMetadata


def add_camera_arguments(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("camera")
    g.add_argument("--camera", metavar="CONFIG.yaml", default=None,
                   help="camera parameters as YAML (conversion, offset, "
                        "pixelsize_um, ...)")
    g.add_argument("--pixelsize", type=float, default=None,
                   help="effective pixel size in the sample, um")
    g.add_argument("--conversion", type=float, default=None, help="e- per ADU")
    g.add_argument("--offset", type=float, default=None,
                   help="camera baseline, ADU")
    g.add_argument("--emgain", type=float, default=None)
    g.add_argument("--em", dest="em_on", action=argparse.BooleanOptionalAction,
                   default=None, help="EM amplification was used")
    g.add_argument("--cameras", metavar="CAMERAS", default=None,
                   help="a camera database to use besides the shipped one: a "
                        "JSON file, or a SMAP *_cameras.mat")
    g.add_argument("--camera-name", metavar="NAME", default=None,
                   help="use this camera from the database instead of "
                        "identifying one from the file's tags")


def camera_from_args(source, a, require: bool = True) -> CameraMetadata:
    """Build the camera from the layers the user supplied."""
    overrides = CameraMetadata()
    if a.camera:
        overrides = CameraMetadata.from_yaml(a.camera)
    overrides = overrides.merged_with(CameraMetadata(
        conversion=a.conversion, offset=a.offset, pixelsize_um=a.pixelsize,
        emgain=a.emgain, em_on=a.em_on))
    return camera_metadata(source, a.cameras, overrides, require=require,
                           camera=getattr(a, "camera_name", None) or "")
