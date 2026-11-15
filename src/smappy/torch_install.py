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

**A build for the CPU on a machine with an NVIDIA card** (`cpu_build_on_nvidia`) works, and learns several
times slower without saying so: uipsf takes the card only when torch was built for it (`uipsf.device`).  It
happens when torch came from elsewhere than the window -- a plain ``pip install torch`` on Windows, whose PyPI
wheel is CPU only.  The window then warns and offers the button again (`replacement`); its command replaces
the build rather than finding the installed version good enough.
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


def _installed() -> Optional[str]:
    from importlib import metadata
    try:
        return metadata.version("torch")
    except metadata.PackageNotFoundError:
        return None


def cpu_build_on_nvidia(version: Optional[str] = None, system: Optional[str] = None,
                        nvidia: Optional[bool] = None) -> bool:
    """Whether the installed torch is built for the CPU only on a machine with an NVIDIA card (module docstring),
    read from its version's local label: PyTorch's CPU index marks its wheels ``+cpu``; on Windows a CUDA build
    is always ``+cu…`` (PyPI's wheel, without a label, is CPU only), on Linux PyPI's unlabelled wheel is CUDA's."""
    version = _installed() if version is None else version
    system = system or platform.system()
    nvidia = has_nvidia() if nvidia is None else nvidia
    if not version or not nvidia or system == "Darwin":
        return False
    label = version.partition("+")[2].lower()
    if label == "cpu":
        return True
    return system == "Windows" and not label.startswith("cu")


def replacement() -> Optional[str]:
    """What the window says when torch works but leaves the NVIDIA card unused; None when it does not."""
    if problem() is not None or not cpu_build_on_nvidia():
        return None
    return (f"PyTorch {_installed()} is built for the CPU only, though this computer has an NVIDIA card: the pupil "
            f"model learns on the CPU, several times slower")


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
            start: Optional[List[str]] = None, replace: bool = False) -> List[str]:
    """The install command, as a list for a process, that installs the right torch here.

    ``replace``: a CPU build is there and the card's is wanted (`cpu_build_on_nvidia`).  ``--upgrade`` alone
    can keep it -- PyPI's unlabelled 2.x sorts below a 2.x+cpu, and an installed version newer than the CUDA
    index's is "satisfied" -- so the build is reinstalled from the CUDA index.  On Windows without its
    dependencies, which a CUDA wheel there shares with the CPU one (it carries CUDA inside); on Linux the CUDA
    wheel needs NVIDIA's packages, so only the higher local label (+cu126 above +cpu) moves it."""
    system = system or platform.system()
    nvidia = has_nvidia() if nvidia is None else nvidia
    pip = (start or installer()) + ["--upgrade", f"torch>={'.'.join(map(str, MINIMUM))}"]
    if replace and nvidia and system in ("Windows", "Linux"):
        extra = ["--force-reinstall", "--no-deps"] if system == "Windows" else []
        return pip + extra + ["--index-url", CUDA_INDEX]
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
