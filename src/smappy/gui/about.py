"""*Help > About SMAPpy*: what the program is, which version, and how to cite it.

The version is the one installed (`smappy.__version__`), which is what a bug
report and a methods section need -- not the one a checkout's
``pyproject.toml`` has moved on to.  The citation is SMAP's until SMAPpy has a
paper of its own.
"""
from __future__ import annotations

import html

from PySide6.QtWidgets import QMessageBox

GITHUB = "https://github.com/ries-lab/SMAPpy"
COPYRIGHT = "Copyright (c) 2026 smappy contributors"
LICENSE = "BSD 3-Clause License"
LICENSE_URL = f"{GITHUB}/blob/main/LICENSE"
CITATION = ("Ries J. SMAP: a modular super-resolution microscopy analysis "
            "platform for SMLM data. Nat Methods 17, 870-872 (2020).")
CITATION_DOI = "https://doi.org/10.1038/s41592-020-0938-1"


def version() -> str:
    try:
        import smappy
        return str(smappy.__version__)
    except Exception:                  # an About box must open whatever happens
        return "unknown"


def about_html() -> str:
    """The box's text; a function of its own so a test can read it."""
    return (f"<h3>SMAPpy {html.escape(version())}</h3>"
            "<p>Single-molecule localization microscopy: from camera frames "
            "to localizations, their analysis and pictures.</p>"
            f'<p><a href="{GITHUB}">{GITHUB}</a></p>'
            f"<p>{html.escape(COPYRIGHT)}<br>"
            f'Distributed under the <a href="{LICENSE_URL}">{LICENSE}</a>.</p>'
            "<p><b>If you use SMAPpy, please cite:</b><br>"
            f'{html.escape(CITATION)} <a href="{CITATION_DOI}">'
            "doi:10.1038/s41592-020-0938-1</a></p>")


def show_about(parent=None) -> None:
    QMessageBox.about(parent, "About SMAPpy", about_html())
