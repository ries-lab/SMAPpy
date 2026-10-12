"""Files dropped on a window: which are local, and what each one is.

Shared by the main windows (`app`) and the batch window.  Sorting is by name
and by what a folder holds, never by reading a stack, so a drop is answered
at once; the reading is left to whoever opens the file, as it would be after
*File > Open*.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

LOCS = "locs"              # a localization table: `io.formats.reader_for` takes it
STACK = "stack"            # an acquisition a fitter takes: `io.tiff.is_image_stack`
IMAGE = "image"            # a picture to render with the localizations: an image layer

#: More frames than this and a TIFF is taken for raw data to fit; this many
#: or fewer, for an image to show beside the localizations -- a widefield
#: snapshot, a few colour channels, a z-stack of a structure.  Raw SMLM data
#: runs to thousands of frames, so the line is not a fine one.
MAX_IMAGE_FRAMES = 50


def accepts(mime) -> bool:
    """A drag worth taking: URLs, and every one of them a local file or folder.

    A link from a browser is a URL too, and accepting it would only lead to
    "cannot open" once it is dropped; refusing it shows at once, in the
    cursor, that nothing will happen.
    """
    if not mime.hasUrls():
        return False
    urls = mime.urls()
    return bool(urls) and all(url.isLocalFile() for url in urls)


def local_paths(mime) -> List[Path]:
    """The dropped files and folders that exist, in the order they came."""
    paths = []
    for url in mime.urls():
        if url.isLocalFile():
            path = Path(url.toLocalFile())
            if path.exists():
                paths.append(path)
    return paths


def kind_of(path) -> Optional[str]:
    """`LOCS`, `IMAGE`, `STACK`, or None for something neither File > Open,
    *Open image* nor a fitter could take.

    A file is asked of the localization readers first, by the same
    `reader_for` that *Load/Auto* uses -- the extension, then a look inside
    where two formats share one (`.h5`) -- so a table saved beside an
    acquisition, even inside an NDTiff folder, opens as a table.  A folder is
    only ever an acquisition: no reader takes one.  A single TIFF (or PNG) of
    at most `MAX_IMAGE_FRAMES` frames is an image to show, not data to fit.
    """
    from ..io.formats import reader_for
    from ..io.tiff import is_image_stack

    path = Path(path)
    if not path.is_dir():
        try:
            reader_for(path)
            return LOCS
        except Exception:             # no reader, or one whose look inside failed
            pass
        if path.suffix.lower() == ".png":
            return IMAGE
    if not is_image_stack(path):
        return None
    if path.is_file() and path.suffix.lower() in (".tif", ".tiff"):
        frames = frame_count(path)
        if frames is not None and frames <= MAX_IMAGE_FRAMES:
            return IMAGE
    return STACK


def frame_count(path) -> Optional[int]:
    """The frames in a TIFF, from its first series' shape -- the header and,
    for an OME file, its metadata, not the pixels.  None if it cannot be read."""
    import tifffile
    try:
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            shape, axes = series.shape, series.axes
    except Exception:
        return None
    count = 1
    for size, axis in zip(shape, axes):
        if axis not in "YXS":               # rows, columns, samples (RGB)
            count *= int(size)
    return count
