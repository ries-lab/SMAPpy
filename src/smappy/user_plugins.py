"""The user's own plugin folder: making it, and starting a plugin in it.

Writing an analysis for one dataset should cost a file, not a contribution.
So there is one folder SMAPpy always scans (`config.user_plugin_dir`), and
this module makes it ready to be written in -- by hand, or by an AI assistant
told what the user wants measured:

* ``AGENTS.md`` -- how a SMAPpy plugin is written, for an assistant (Claude
  Code, Codex and the rest read it from the folder they are started in).
  SMAPpy's, and rewritten whenever it differs from the installed version's,
  so that it describes the API that is installed, not the one of the day the
  folder was made;
* ``CLAUDE.md`` -- reads AGENTS.md (``@AGENTS.md``) and is otherwise the
  user's: written once, never again;
* ``README.md`` -- the same for a person, written once;
* ``_template.py`` -- a working plugin to copy, skipped by the scanner for
  its leading ``_``, and SMAPpy's like AGENTS.md.

Nothing is written until something asks for the folder: an install that
never makes a plugin never gets one.  What is shipped with SMAPpy follows the
repository's rules (a page, tests); what is here follows none but the
plugin contract, which is the point.

    smappy-plugins                      # the folder, made ready; its path printed
    smappy-plugins new "cluster sizes"  # cluster_sizes.py, from the template
    smappy-plugins check                # import each one; say what fails
"""
from __future__ import annotations

import argparse
import keyword
import re
import sys
from importlib import resources
from pathlib import Path
from typing import Optional

from . import config

# (the shipped source in data/user_plugins, the name in the folder, SMAPpy's?)
# SMAPpy's own files follow the installed version; the user's are written
# only when missing.
FILES = (("AGENTS.md.in", "AGENTS.md", True),
         ("template.py.in", "_template.py", True),
         ("CLAUDE.md.in", "CLAUDE.md", False),
         ("README.md.in", "README.md", False))

# the names in the template that a new plugin renames
_TEMPLATE_CLASS = "NearestNeighbours"
_TEMPLATE_TITLE = "Nearest neighbours"
# the template's first lines, which say it is the example; a copy is not
_TEMPLATE_HEADER = re.compile(r"\A(#[^\n]*\n)+")


def folder() -> Path:
    """Where the user's plugins live (not made here; see `ensure`)."""
    return config.user_plugin_dir()


def _shipped(name: str) -> str:
    text = resources.files("smappy").joinpath("data", "user_plugins", name).read_text("utf-8")
    from . import __version__
    package = Path(__file__).resolve().parent
    return text.replace("@@VERSION@@", str(__version__)).replace("@@PACKAGE@@", str(package))


def ensure() -> Path:
    """The folder, made if need be, with SMAPpy's files in it up to date."""
    target = folder()
    target.mkdir(parents=True, exist_ok=True)
    for source, name, ours in FILES:
        path = target / name
        if path.exists() and not ours:
            continue
        text = _shipped(source)
        try:
            if path.exists() and path.read_text(encoding="utf-8") == text:
                continue
        except (OSError, UnicodeDecodeError):
            pass
        path.write_text(text, encoding="utf-8")
    return target


def file_name(title: str) -> str:
    """``"Cluster sizes (2D)"`` -> ``"cluster_sizes_2d"``: a module name."""
    stem = re.sub(r"[^0-9a-zA-Z]+", "_", title).strip("_").lower()
    if not stem:
        raise ValueError("give the plugin a name with letters in it")
    if stem[0].isdigit() or keyword.iskeyword(stem):
        stem = f"plugin_{stem}"
    return stem


def class_name(title: str) -> str:
    """``"cluster sizes"`` -> ``"ClusterSizes"``."""
    words = re.findall(r"[0-9a-zA-Z]+", title)
    name = "".join(w[:1].upper() + w[1:] for w in words)
    return name if name and not name[0].isdigit() else f"Plugin{name}"


def new(title: str, directory: Optional[Path] = None) -> Path:
    """A new plugin called ``title``, copied from the template, and its path.

    Refuses to overwrite: a name that is taken is more likely a plugin someone
    has started than one they want replaced by the example.
    """
    target = Path(directory) if directory is not None else ensure()
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{file_name(title)}.py"
    if path.exists():
        raise FileExistsError(f"{path} exists already; choose another name")
    text = _TEMPLATE_HEADER.sub("", _shipped("template.py.in"), count=1)
    cls = class_name(title)
    text = text.replace(_TEMPLATE_CLASS, cls)
    # the display name is the file's stem title-cased, which is what the tree
    # shows anyway; saying it in the class keeps the user's own capitals
    shown = title.strip()
    shown = shown[:1].upper() + shown[1:]
    text = text.replace(f'class {cls}(Plugin):\n',
                        f'class {cls}(Plugin):\n    name = {shown!r}\n', 1)
    text = text.replace(f'"""{_TEMPLATE_TITLE}: how far', f'"""{shown}: started from '
                        "SMAPpy's example, which measures how far", 1)
    path.write_text(text, encoding="utf-8")
    return path


def check(report=print) -> int:
    """Read and import every plugin in the folder; say what fails.

    Importing, unlike the scan: a plugin class that does not say
    `changes_locs`, or a typo below the imports, is only found that way.
    The number of files that failed is returned.
    """
    from . import plugins
    plugins.discover(force=True)
    home = folder().resolve()
    failed = [str(d) for d in plugins.problems()
              if home in Path(d.origin).resolve().parents]
    found = sorted(p for p, r in plugins.refs().items() if r.root == plugins.USER_ROOT)
    for path in found:
        ref = plugins.refs()[path]
        try:
            ref.load()
        except Exception as error:      # anything: it is the user's code
            failed.append(f"{ref.origin}: {type(error).__name__}: {error}")
    for line in failed:
        report(line)
    report(f"{len(found)} plugin(s) in {home}: " + (", ".join(found) or "none"))
    return len(failed)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="smappy-plugins",
        description="Your own SMAPpy plugins: print the folder (made ready "
                    "for an AI assistant), or start a new plugin in it.")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("folder", help="make the folder ready and print its path (the default)")
    start = sub.add_parser("new", help="a new plugin, copied from the template")
    start.add_argument("name", help='what the GUI calls it, e.g. "cluster sizes"')
    sub.add_parser("check", help="list the plugin files that cannot be read")
    args = parser.parse_args(argv)
    if args.command == "new":
        try:
            path = new(args.name)
        except (FileExistsError, ValueError) as error:
            print(error, file=sys.stderr)
            return 1
        print(path)
        return 0
    if args.command == "check":
        return 1 if check() else 0
    print(ensure())
    return 0


if __name__ == "__main__":
    sys.exit(main())
