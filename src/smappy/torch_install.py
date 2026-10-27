"""Whether PyTorch is here for uipsf, and the pip command that brings the right build for this machine.

torch is an optional part of SMAPpy (`smappy-smlm[uipsf]`), because on Linux its default wheel brings
NVIDIA's libraries, gigabytes for a fitting machine that may never learn a PSF (`docs/uipsf/plan.md`
§7.4).  Where it is needed -- the pupil model of the PSF calibration -- the window offers to install it,
into the Python running SMAPpy (``sys.executable``), with the build for the machine:

* macOS: PyPI's wheel, which uses Apple's GPU (MPS);
* Linux with an NVIDIA card (``nvidia-smi`` found): PyPI's wheel, which is CUDA's;
* Linux without: PyTorch's CPU index, a fifth of the size;
* Windows with an NVIDIA card: PyTorch's CUDA 12.6 index (PyPI's Windows wheel is CPU only); CUDA 12.6
  rather than 13, for the older drivers it still runs on (both carry torch 2.14.1, checked October 2026);
* Windows without: PyPI's wheel, CPU.

The installer is pip when this Python has it, and otherwise uv's own pip aimed at this Python
(``uv pip install --python``): the install the step-by-step guide recommends is a uv tool, whose
environment has no pip at all, so ``python -m pip`` there fails with "No module named pip" (found on a
Windows machine, October 2026).  uv is looked for on the PATH and where its installer puts it, since a
program started from a desktop icon may not see the PATH a terminal does.

Nothing here imports torch: what is installed is read from its package metadata, so asking costs nothing
and an old torch is not loaded into the process before it is replaced.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import List, Optional

MINIMUM = (2, 4)                     # pyproject.toml's extra: torch>=2.4
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
CUDA_INDEX = "https://download.pytorch.org/whl/cu126"


def _version(text: str) -> tuple:
    parts = []
    for piece in text.split("+")[0].split(".")[:2]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


def problem() -> Optional[str]:
    """None when a usable torch is installed; otherwise what is wrong, in a few words."""
    from importlib import metadata
    try:
        installed = metadata.version("torch")
    except metadata.PackageNotFoundError:
        return "PyTorch is not installed"
    if _version(installed) < MINIMUM:
        return f"PyTorch {installed} is too old (uipsf needs {'.'.join(map(str, MINIMUM))} or later)"
    return None


def has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


def _uv() -> Optional[str]:
    """uv's program: on the PATH, or where its installer puts it (``$UV_INSTALL_DIR``, ``~/.local/bin``)."""
    found = shutil.which("uv")
    if found:
        return found
    name = "uv.exe" if os.name == "nt" else "uv"
    for folder in (os.environ.get("UV_INSTALL_DIR"), Path.home() / ".local" / "bin", Path.home() / ".cargo" / "bin"):
        if folder and (Path(folder) / name).is_file():
            return str(Path(folder) / name)
    return None


def installer(has_pip: Optional[bool] = None, uv: Optional[str] = None) -> List[str]:
    """The start of the install command: this Python's pip, or uv's pip aimed at this Python where there is
    no pip (a uv tool's environment).  With neither, pip's, whose error says what is missing."""
    has_pip = importlib.util.find_spec("pip") is not None if has_pip is None else has_pip
    if not has_pip:
        uv = uv or _uv()
        if uv:
            return [uv, "pip", "install", "--python", sys.executable]
    return [sys.executable, "-m", "pip", "install"]


def command(system: Optional[str] = None, nvidia: Optional[bool] = None,
            start: Optional[List[str]] = None) -> List[str]:
    """The install command, as a list for a process, that installs the right torch here."""
    system = system or platform.system()
    nvidia = has_nvidia() if nvidia is None else nvidia
    pip = (start or installer()) + ["--upgrade", f"torch>={'.'.join(map(str, MINIMUM))}"]
    if system == "Linux" and not nvidia:
        return pip + ["--index-url", CPU_INDEX]
    if system == "Windows" and nvidia:
        return pip + ["--index-url", CUDA_INDEX]
    return pip


def described() -> str:
    """What the install would bring, for the button's tooltip and the notice."""
    system, nvidia = platform.system(), has_nvidia()
    build = {"Darwin": "with Apple's GPU", "Linux": "for the NVIDIA card" if nvidia else "for the CPU",
             "Windows": "for the NVIDIA card (CUDA 12.6)" if nvidia else "for the CPU"}.get(system, "")
    return f"PyTorch {build}".strip()
