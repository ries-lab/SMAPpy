"""What a file or a folder holds: the data sets in it, of one kind.

One engine for every place that is handed files and folders -- the batch
(its folder rules, run and watched), the PSF and bead calibration windows --
so they agree on what one data set is.  The batch had a plain
``rglob("*.tif")`` of its own, and fitted a Micro-Manager series that had
grown into ``x.ome.tif`` and ``x_1.ome.tif`` as two acquisitions, an NDTiff
folder as one per file inside it, and never found a DCIMG at all, while the
calibration windows read the same folder correctly.

A data set is named by one path: a file, the first file of a series, or an
NDTiff folder.  `parts` gives the files it is written in, which is what a
watch looks at to tell when it is complete.

There are two kinds, and which is wanted is the reader's to say -- for the
batch, the chain's first step (`chain.ChainPlugin.input_kind`):

* ``"images"``: camera frames to fit -- TIFF (a Micro-Manager series once),
  NDTiff, Hamamatsu DCIMG and 4Pi quadrant ``.mat`` series;
* ``"localizations"``: tables -- smappy HDF5, SMAP's ``_sml.mat``, CSV.

A *filter* narrows what a folder gives, by name: `parse_filter`.
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

KINDS = ("images", "localizations")

# what each kind is, said to a user (the batch window's line above its files)
DESCRIPTIONS = {"images": "raw camera images (TIFF and OME-TIFF series, NDTiff, DCIMG)",
                "localizations": "localization files (.h5, _sml.mat, .csv)"}

_IMAGE_SUFFIXES = (".tif", ".tiff", ".dcimg")
_LOCALIZATION_SUFFIXES = (".h5", ".hdf5", ".csv")


# ------------------------------------------------------------------- one path

def is_ndtiff(folder: Path) -> bool:
    return (Path(folder) / "NDTiff.index").exists()


def kind_of(path) -> Optional[str]:
    """What one path is, as a data set: "images", "localizations" or None.

    Only by its name, except a ``.mat``, which is SMAP's table when it ends in
    ``_sml.mat`` and a 4Pi quadrant file when it holds ``qd`` variables."""
    path = Path(path)
    if path.is_dir():
        return "images" if is_ndtiff(path) else None
    name = path.name.lower()
    if name.endswith(_IMAGE_SUFFIXES):
        return "images"
    if name.endswith("_sml.mat") or name.endswith(_LOCALIZATION_SUFFIXES):
        return "localizations"
    if name.endswith(".mat"):
        from .io.quadrants import is_quadrant_mat
        return "images" if is_quadrant_mat(path) else None
    return None


def canonical(path) -> Path:
    """The path that names the data set ``path`` belongs to: the first file
    of a series, the NDTiff folder of one of its files, else ``path``."""
    path = Path(path)
    if path.is_file() and is_ndtiff(path.parent) and path.suffix.lower() in (".tif", ".tiff"):
        return path.parent
    name = path.name.lower()
    if path.is_file() and name.endswith((".ome.tif", ".ome.tiff")):
        from .io.tiff import _series_files
        return _series_files(path)[0]
    if path.is_file() and kind_of(path) == "images" and name.endswith((".dcimg", ".mat")):
        from .io.quadrants import numbered_series
        return numbered_series(path)[0]
    return path


def parts(path) -> List[Path]:
    """The files a data set is written in: every file of a series or of an
    NDTiff folder, else the file itself."""
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file())
    name = path.name.lower()
    if name.endswith((".ome.tif", ".ome.tiff")):
        from .io.tiff import _series_files
        return _series_files(path)
    if name.endswith(".dcimg") or (name.endswith(".mat") and not name.endswith("_sml.mat")
                                   and kind_of(path) == "images"):
        from .io.quadrants import numbered_series
        return numbered_series(path)
    return [path]


# -------------------------------------------------------------------- folders

def find(paths, kind: str, recursive: bool = True,
         keep: Optional[Callable[[str], bool]] = None,
         skip: Iterable[Path] = ()) -> List[Path]:
    """The data sets of ``kind`` in ``paths`` (files and folders), each once,
    sorted by path.

    A folder is searched (its subfolders too with ``recursive``); a folder
    that is itself a data set (NDTiff) is that one.  A file is its data set
    if it is of ``kind``.  ``keep`` is a filter (`parse_filter`) on a found
    data set's path relative to the folder it was found in; ``skip`` are
    folders never searched (the batch's own output)."""
    if kind not in KINDS:
        raise ValueError(f"{kind!r} is not a kind of data; there are {', '.join(KINDS)}")
    skip = [Path(s).resolve() for s in skip]
    found = {}
    for value in ([paths] if isinstance(paths, (str, Path)) else paths):
        root = Path(value).expanduser()
        if not root.exists():
            raise FileNotFoundError(root)
        if root.is_file() or is_ndtiff(root):
            if kind_of(root) == kind:
                found.setdefault(canonical(root).resolve(), None)
            continue
        for candidate in _walk(root, recursive, skip):
            if kind_of(candidate) != kind:
                continue
            data = canonical(candidate)
            if keep is not None and not keep(relative(data, root)):
                continue
            found.setdefault(data.resolve(), None)
    return sorted(found, key=lambda p: str(p).lower())


def _walk(root: Path, recursive: bool, skip: Sequence[Path]):
    """The files and NDTiff folders under ``root``; an NDTiff folder is not
    entered, nor a folder in ``skip``."""
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.is_dir():
            if is_ndtiff(entry):
                yield entry
            elif recursive and entry.resolve() not in skip:
                yield from _walk(entry, recursive, skip)
        elif entry.is_file():
            yield entry


def relative(path: Path, folder: Path) -> str:
    """``path`` inside ``folder``, with forward slashes, as a filter reads it."""
    try:
        return Path(path).resolve().relative_to(Path(folder).resolve()).as_posix()
    except ValueError:
        return Path(path).name


# --------------------------------------------------------------------- filter

class FilterError(ValueError):
    """A filter that cannot be read; the message says where."""


_KEYWORDS = {"AND": "&", "OR": ",", "NOT": "!"}


def parse_filter(text: str) -> Callable[[str], bool]:
    """A filter, read: a test of a data set's path inside its folder.

    Patterns are shell patterns (``*``, ``?``, ``[...]``), matched without
    regard to case against the path inside the folder and against the name
    alone, so ``*640*`` finds 640 anywhere and ``beads/*`` a subfolder.  They
    are joined by

    * ``,`` or ``|`` (or the word ``OR``): either,
    * ``&`` (or ``AND``): both,
    * ``!`` (or ``NOT``) before one: not,

    and grouped by parentheses; NOT binds tightest, then AND, then OR, as in
    most search boxes: ``(*640* & !*647*), *600*``.  A pattern runs up to the
    next operator, so a name may hold spaces; one holding an operator or a
    parenthesis is quoted, ``"cell (2)*"``.  Empty keeps everything.
    """
    tokens = _tokens(text)
    if not tokens:
        return lambda _path: True
    parser = _Parser(tokens, text)
    test = parser.expression()
    if parser.at < len(tokens):
        kind, value, where = tokens[parser.at]
        raise FilterError(f"{value!r} at {where + 1} follows a complete filter: "
                          "join it with , & or !")
    return test


def _tokens(text: str):
    """(kind, value, position): kind is "op" for , & ! ( ) or "pattern"."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c in ",|&!()":
            out.append(("op", "," if c == "|" else c, i))
            i += 1
            continue
        if c in "\"'":
            end = text.find(c, i + 1)
            if end < 0:
                raise FilterError(f"the quote at {i + 1} is not closed")
            out.append(("pattern", text[i + 1:end], i))
            i = end + 1
            continue
        start = i
        while i < n and text[i] not in ",|&()\"'" and not (text[i] == "!" and i == start):
            if text[i] == "[":          # a set, [!0-9] included, is one piece
                close = text.find("]", i + 2)
                i = close + 1 if close > 0 else i + 1
                continue
            i += 1
        word = text[start:i].strip()
        # the words AND, OR and NOT, standing alone, are operators
        pieces = re.split(r"(?<!\S)(AND|OR|NOT)(?!\S)", word)
        offset = start
        for piece in pieces:
            if piece in _KEYWORDS:
                out.append(("op", _KEYWORDS[piece], offset))
            elif piece.strip():
                out.append(("pattern", piece.strip(), offset))
            offset += len(piece)
    return out


class _Parser:
    def __init__(self, tokens, text):
        self.tokens, self.text, self.at = tokens, text, 0

    def _peek(self, value=None):
        if self.at >= len(self.tokens):
            return False if value is not None else None
        token = self.tokens[self.at]
        return token if value is None else (token[0] == "op" and token[1] == value)

    def expression(self):
        tests = [self.conjunction()]
        while self._peek(","):
            self.at += 1
            tests.append(self.conjunction())
        return tests[0] if len(tests) == 1 else (lambda p, t=tuple(tests): any(f(p) for f in t))

    def conjunction(self):
        tests = [self.negation()]
        while self._peek("&"):
            self.at += 1
            tests.append(self.negation())
        return tests[0] if len(tests) == 1 else (lambda p, t=tuple(tests): all(f(p) for f in t))

    def negation(self):
        if self._peek("!"):
            self.at += 1
            inner = self.negation()
            return lambda p: not inner(p)
        return self.atom()

    def atom(self):
        token = self._peek()
        if token is None:
            raise FilterError("the filter ends where a pattern should follow")
        kind, value, where = token
        if kind == "pattern":
            self.at += 1
            return _pattern(value)
        if value == "(":
            self.at += 1
            inner = self.expression()
            if not self._peek(")"):
                raise FilterError(f"the parenthesis at {where + 1} is not closed")
            self.at += 1
            return inner
        raise FilterError(f"{value!r} at {where + 1} where a pattern should be")


def _pattern(pattern: str) -> Callable[[str], bool]:
    pattern = pattern.lower()

    def test(path: str) -> bool:
        path = path.lower()
        return (fnmatch.fnmatchcase(path, pattern)
                or fnmatch.fnmatchcase(path.rsplit("/", 1)[-1], pattern))
    return test


def filter_error(text: str) -> Optional[str]:
    """Why ``text`` is not a filter, or None if it is."""
    try:
        parse_filter(text)
    except FilterError as error:
        return str(error)
    return None
