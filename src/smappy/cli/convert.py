"""``smappy convert IN [OUT]``: a localization file in another format.

Read as the GUI reads it and written as its menus write it: smappy HDF5 by
`Session.save` (*File > Save*), which keeps the files, their cameras, the
ROIs and what was measured; csv, ThunderSTORM's csv and Picasso's HDF5 by
*Save localizations* (`plugins.file.ExportLocalizations`), every
localization and every column.  Those formats hold a table and little else,
so what they leave behind is said, not lost quietly.

A ``.hdf5`` OUT is smappy's unless ``--format Picasso`` says otherwise: the
point of this command is mostly a SMAP, Picasso or MINFLUX file brought in.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

from . import command_name

FORMATS = ("smappy", "csv", "ThunderSTORM", "Picasso")
# the export plugin's names for them
_EXPORT = {"csv": "csv", "ThunderSTORM": "ThunderSTORM", "Picasso": "Picasso"}
_SUFFIX = {"smappy": ".hdf5", "csv": ".csv", "ThunderSTORM": ".csv", "Picasso": ".hdf5"}


def output_format(out: Optional[Path], wanted: Optional[str]) -> str:
    """The format asked for, else the one OUT's suffix says."""
    if wanted:
        return wanted
    if out is not None and out.suffix.lower() in (".csv", ".tsv", ".txt"):
        return "csv"
    if out is None or out.suffix.lower() in (".h5", ".hdf5"):
        return "smappy"
    raise ValueError(f"{out.name}: no format writes {out.suffix}; "
                     f"use .hdf5, .csv, .tsv or .txt, or --format")


def default_output(source: Path, fmt: str) -> Path:
    """``X_sml.mat`` -> ``X.hdf5`` beside it: the reader's suffix off, the
    format's on; ``X.hdf5`` read as smappy and written as Picasso's ->
    ``X_picasso.hdf5``, never the file itself."""
    from ..io.formats import reader_for
    name = source.name
    suffixes = sorted(reader_for(source).suffixes, key=len, reverse=True)
    stem = next((name[:-len(s)] for s in suffixes if name.lower().endswith(s)), source.stem)
    out = source.with_name(stem + _SUFFIX[fmt])
    if out.resolve() == source.resolve():
        out = source.with_name(f"{stem}_{fmt.lower()}{_SUFFIX[fmt]}")
    return out


def convert(source, out=None, fmt: Optional[str] = None, force: bool = False,
            say=print) -> Path:
    """Read ``source``, write it to ``out`` in ``fmt``; the path written."""
    from ..session import Session
    source = Path(source)
    out = Path(out) if out else None
    fmt = output_format(out, fmt)
    out = out or default_output(source, fmt)
    if out.resolve() == source.resolve():
        raise ValueError(f"{out} is the file being read: name another")
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; --force writes over it")
    session = Session()
    session.load(source)
    n, files = len(session.locs), len(session.files)
    rois = len(session.rois.rois)
    if fmt == "smappy":
        written = session.save(out, gui_state=False)
    else:
        from ..plugins.file import ExportLocalizations, ExportSettings
        settings = ExportSettings(path=str(out), format=_EXPORT[fmt],
                                  localizations="all", grouping="ungrouped")
        result = ExportLocalizations().run(session.context(0, progress=say), settings)
        written = Path(result.written)
    held = f"{n:,} localizations" + (f" in {files} files" if files > 1 else "")
    say(f"{source.name} -> {written} ({fmt}): {held}")
    if fmt != "smappy":
        left = [what for what, there in (
            ("the files' cameras" if files > 1 else "the camera", True),
            (f"the {rois} ROIs" if rois > 1 else "the ROI", rois),
            ("the results", session.results)) if there]
        say(f"  not in a {fmt} file: {', '.join(left)}"
            + ("; `filenumber` says which file a row is from" if files > 1 else ""))
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog=command_name("smappy-convert"),
        description="A localization file in another format.  smappy HDF5 keeps "
                    "everything: the files, their cameras, the ROIs, the results.")
    p.add_argument("source", metavar="IN",
                   help="smappy HDF5, SMAP, Picasso, MINFLUX, csv, ...")
    p.add_argument("out", nargs="?", metavar="OUT",
                   help="where to write (default: IN's name, as smappy HDF5)")
    p.add_argument("--format", choices=FORMATS,
                   help="default: smappy for .hdf5 / .h5, csv for .csv / .tsv / .txt")
    p.add_argument("--force", action="store_true", help="write over OUT if it exists")
    args = p.parse_args(argv)
    if not Path(args.source).exists():
        p.error(f"no file {args.source}")
    try:
        convert(args.source, args.out, args.format, args.force)
    except (ValueError, FileExistsError) as e:
        print(f"{p.prog}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
