"""Command-line entry points.

Each module here is one console script, declared in ``pyproject.toml``.  They
are inside the package rather than in ``scripts/`` so that an installed smappy
has working commands and no ``sys.path`` juggling.
"""
import os
import sys


def command_name(default: str) -> str:
    """The name to put in a command's usage line: the script's, from
    ``sys.argv[0]``, as argparse would take it.

    A module run with ``python -m`` has its file there, which says nothing
    to a user, so the script's name stands in for it.
    """
    name = os.path.basename(sys.argv[0]) if sys.argv else ""
    return name if name.startswith("smappy") and not name.endswith(".py") else default
