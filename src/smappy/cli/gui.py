"""``smappy gui [FILE ...]``: the Qt GUI.  Needs ``pip install smappy-smlm[gui]``.

The options are what the menus do at a start: *View > Workspace > Open*
(``--workspace``), *File > Restore saved session* (``--restore``), a plugin in a
window of its own (``--open``).  They are checked here, before Qt starts, so
a mistyped name is a line in the terminal rather than a window and a box.

``smappy gui --handoff FILE.json`` is how *Open in a new window* starts it,
with the session another instance handed over (`smappy.handoff`).

``smappy gui --desktop-icon`` makes an icon on the desktop that starts it, with
SMAPpy's picture, and exits without opening a window (`smappy.shortcut`).
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import command_name


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=command_name("smappy-gui"),
        description="SMAPpy's windows, on the localization files given (opened "
                    "together, as File > Add would).")
    p.add_argument("files", nargs="*", metavar="FILE",
                   help="localization files: smappy HDF5, SMAP, Picasso, MINFLUX, csv, ...")
    p.add_argument("--workspace", metavar="FILE.yaml",
                   help="start with the tabs and plugins of a saved workspace "
                        "(View > Workspace > Save); like the menu's, it is then the "
                        "one the next start opens")
    p.add_argument("--restore", action="store_true",
                   help="bring back the session the file was saved with: its layers, "
                        "views and windows (File > Restore saved session); one smappy HDF5 file")
    p.add_argument("--open", action="append", default=[], metavar="PLUGIN",
                   help='a plugin in a window of its own, by its path or its name: '
                        '"Analysis/Drift/RCC" or "RCC"; may be given more than once')
    p.add_argument("--desktop-icon", action="store_true",
                   help="make an icon on the desktop that starts SMAPpy, and exit")
    p.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    p.add_argument("--handoff", metavar="FILE.json", help=argparse.SUPPRESS)
    return p


def _version() -> str:
    from .. import __version__
    return __version__


def find_plugin(text: str, refs) -> str:
    """The path of the plugin ``text`` names: its path, the name the GUI shows
    or the last part of its path, in any case.  ValueError, with what it
    might have meant, for none or several."""
    if text in refs:
        return text
    wanted = text.strip().lower()
    hits = sorted(p for p, r in refs.items()
                  if wanted in (p.lower(), r.name.lower(), p.rsplit("/", 1)[-1].lower()))
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise ValueError(f"{text!r} names several plugins: " + ", ".join(hits))
    import difflib
    names = {p.rsplit("/", 1)[-1].lower(): p for p in refs}
    names.update({r.name.lower(): p for p, r in refs.items()})
    close = [names[n] for n in difflib.get_close_matches(wanted, names, n=3)]
    hint = ("; perhaps " + ", ".join(sorted(set(close)))) if close else ""
    raise ValueError(f"no plugin {text!r}{hint}; `smappy batch plugins` lists them")


def checked(args: argparse.Namespace, p: argparse.ArgumentParser) -> List[str]:
    """``args.open`` as plugin paths, once every option has been checked."""
    for name in args.files:
        if not Path(name).exists():
            p.error(f"no file {name}")
    if args.workspace and not Path(args.workspace).is_file():
        p.error(f"no workspace file {args.workspace}")
    if args.restore and (len(args.files) != 1
                         or Path(args.files[0]).suffix.lower() not in (".h5", ".hdf5")):
        p.error("--restore takes one smappy HDF5 file: only those carry a session")
    if args.handoff and (args.files or args.restore):
        p.error("--handoff brings its own session: no files beside it")
    if not args.open:
        return []
    from .. import plugins
    refs = plugins.refs()
    try:
        return [find_plugin(text, refs) for text in args.open]
    except ValueError as e:
        p.error(str(e))


def main(argv: Optional[List[str]] = None) -> None:
    p = parser()
    args = p.parse_args(argv)
    if args.desktop_icon:
        from ..shortcut import make
        print(f"made {make()}")
        raise SystemExit(0)
    open_plugins = checked(args, p)
    from ..gui.app import main as run
    raise SystemExit(run([sys.argv[0]], files=args.files, handoff=args.handoff,
                         workspace=args.workspace, restore=args.restore,
                         open_plugins=open_plugins))


if __name__ == "__main__":
    main()
