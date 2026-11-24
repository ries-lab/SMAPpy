"""(docs/4pi.md §26)  What the tubulin scripts share: Gabin's 2017-11-28 4Pi-SMS data, its layout and camera.

The data are in ``~/.cache/uipsf/4pi_tubulin_2017-11-28`` (docs/4pi.md §25); what these scripts make goes
to ``smappy_p1/`` beside them, never into the repository.  The camera has no record in the files: an
ORCA-Flash 4.0 at offset 100 and 0.5 e-/ADU (the photon transfer of the movie's background, measured in
`tubulin_layout.py`), 128 nm pixels (the authors' ``para.pixelsz``).  The layout is the 4Pi-SMS design's
(`channels.YALE_4PI`): its channel centres are the authors' (235, 474, 1557, 1797), the last two mirrored.
"""
from pathlib import Path

DATA = Path.home() / ".cache/uipsf/4pi_tubulin_2017-11-28"
OUT = DATA / "smappy_p1"
SCANS = [DATA / f"Cell0{k}/Cell0{k}_642_000_000.dcimg" for k in (3, 4, 5)]
FIELDS = [DATA / f"Cell{k:02d}" for k in range(6, 13)]
MOVIE = DATA / "Cell01/Cell01_642_000_000.dcimg"
CONVERSION, OFFSET, PIXEL_UM = 0.5, 100.0, 0.128
REGISTRATION = OUT / "registration_cell06-12.json"


def camera():
    from smappy.metadata import CameraMetadata
    return CameraMetadata(conversion=CONVERSION, offset=OFFSET, pixelsize_um=PIXEL_UM)


def layout():
    from smappy.channels import YALE_4PI
    return YALE_4PI
