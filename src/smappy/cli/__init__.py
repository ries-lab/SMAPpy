"""Command-line entry points.

Each module here is one console script, declared in ``pyproject.toml``.  They
are inside the package rather than in ``scripts/`` so that an installed smappy
has working commands and no ``sys.path`` juggling.

``smappy`` (`main`) is the one command that names them all, ``smappy fit``
for ``smappy-fit``; the ``smappy-xxx`` names stay, because scripts, MCP
clients' settings and the batch window start them by those.
"""
import os
import sys


def command_name(default: str) -> str:
    """The name to put in a command's usage line: ``smappy batch`` when
    `main` started it, ``smappy-batch`` when its own script did.

    `main` sets ``sys.argv[0]`` to the two words; a module run with
    ``python -m`` has its file there, which says nothing to a user, so the
    script's name stands in for it.
    """
    name = os.path.basename(sys.argv[0]) if sys.argv else ""
    return name if name.startswith("smappy") and not name.endswith(".py") else default
