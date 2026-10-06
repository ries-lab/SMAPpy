"""Starting uiPSF in its own Python and following it.

uiPSF is not a dependency of smappy and is not imported here.  It pins its own
TensorFlow, and a TensorFlow in the GUI's process would cost seconds of import
and gigabytes of memory for every user who never learns a PSF; it is run
instead as `worker.py` under the interpreter of an environment that has it,
the way the batch window runs `smappy-batch`.  The same arrangement is what
lets the work move to another machine (a GPU) later without the plugin
knowing.

Which interpreter: the plugin's own setting, else ``uipsf_python`` in the
config file (`smappy.config`), else ``$SMAPPY_UIPSF_PYTHON``, else this
Python if it can import ``psflearning``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

TAG = "SMAPPY_UIPSF"
WORKER = Path(__file__).with_name("worker.py")
# uiPSF's progress bars: "3/6: learning: 99/150 [02:48s] ... current loss: 0.53"
_BAR = re.compile(r"^(\d/\d: [^:]+)(?::\s*(\d+)/(\d+))?")
_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# how often a progress bar that keeps moving is passed on, in seconds
REPORT_EVERY_S = 5.0


class UiPSFError(RuntimeError):
    """uiPSF could not be started, or stopped with an error."""


def find_python(setting: str = "") -> str:
    """The interpreter of an environment with uiPSF in it."""
    from .. import config
    for candidate in (setting, config.get("uipsf_python") or "",
                      os.environ.get("SMAPPY_UIPSF_PYTHON", "")):
        if candidate:
            path = shutil.which(candidate) or candidate
            if not Path(path).exists():
                raise UiPSFError(f"no Python at {candidate}: set the uiPSF "
                                 "environment's python in the plugin, or "
                                 "uipsf_python in the smappy config file")
            return path
    import importlib.util
    if importlib.util.find_spec("psflearning") is not None:
        return sys.executable
    raise UiPSFError(
        "uiPSF was not found: install it into an environment of its own "
        "(see the plugin's page) and give that environment's python in the "
        "plugin's 'uiPSF python', or as uipsf_python in the smappy config file")


def write_job(directory: Path, images: np.ndarray, psftype: str, channeltype: str,
              params: dict) -> Path:
    """The job directory the worker reads."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "images.npy", np.ascontiguousarray(images, dtype=np.float32))
    (directory / "job.json").write_text(json.dumps(
        {"psftype": psftype, "channeltype": channeltype, "params": params}, indent=1))
    return directory


def _lines(stream):
    """Lines of a byte stream, a progress bar's carriage returns ending one too."""
    buffer = b""
    while True:
        chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
        if not chunk:
            break
        buffer += chunk
        parts = re.split(rb"[\r\n]", buffer)
        buffer = parts.pop()
        for part in parts:
            text = _ESCAPE.sub("", part.decode(errors="replace")).strip()
            if text:
                yield text
    text = _ESCAPE.sub("", buffer.decode(errors="replace")).strip()
    if text:
        yield text


def run(directory: Path, python: str = "", progress: Optional[Callable[[str], None]] = None,
        log: Optional[Path] = None) -> Path:
    """Run the job in ``directory``; the uiPSF result file it wrote.

    ``progress`` gets the worker's stages and uiPSF's progress bars, the
    latter at most every `REPORT_EVERY_S`; everything the worker printed goes
    to ``log`` (``directory/uipsf.log`` by default), which is where to look
    when it fails.
    """
    report = progress or (lambda text: None)
    python = find_python(python)
    directory = Path(directory)
    log = Path(log) if log else directory / "uipsf.log"
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
               TF_CPP_MIN_LOG_LEVEL="2")
    command = [python, "-u", str(WORKER), str(directory)]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=env)
    except OSError as error:
        raise UiPSFError(f"could not start {python}: {error}") from error
    result = error = None
    last, shown = "", 0.0
    try:
        with open(log, "w") as out:
            for line in _lines(process.stdout):
                out.write(line + "\n")
                if line.startswith(TAG + "\t"):
                    _, kind, text = (line.split("\t", 2) + [""])[:3]
                    if kind == "result":
                        result = Path(text)
                    elif kind == "error":
                        error = text
                    else:
                        report(f"uiPSF: {text}")
                    continue
                bar = _BAR.match(line)
                if bar:
                    stage = bar.group(1)
                    now = time.monotonic()
                    if stage != last or now - shown > REPORT_EVERY_S:
                        report(f"uiPSF {line[:120]}")
                        last, shown = stage, now
        process.wait()
    finally:
        if process.poll() is None:          # interrupted: do not leave it running
            process.kill()
            process.wait()
    if process.returncode != 0 or result is None:
        raise UiPSFError(f"uiPSF failed: {error or 'exit code ' + str(process.returncode)} "
                         f"(the whole output is in {log})")
    return result
