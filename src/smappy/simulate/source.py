"""A camera-frame simulation as an acquisition a fitter can open.

A simulation is saved as its recipe, ``*.sim.yaml`` -- the settings and the
seed -- rather than as frames: 20 000 frames of 100 x 100 pixels are 400 MB of
TIFF and a few hundred bytes of YAML, and the recipe is also what a chain or
a batch records, so a fit of simulated data can be repeated exactly.
`open_stack` recognises the suffix and returns a `SimulatedSource`, the same
interface a TIFF has; the molecules are worked out when it is opened and the
frames drawn as the fitter reads them (`camera.render`, seeded per frame, so
a block is the same however the stack is cut).  The source also knows its
camera, so the fitter's conversion, offset and pixel size come from it, and
it carries the truth for a comparison afterwards.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

from ..io.tiff import ImageSource
from .settings import SimulationSettings

SUFFIXES = (".sim.yaml", ".sim.yml")


def is_simulation(path) -> bool:
    return str(path).lower().endswith(SUFFIXES)


def write_recipe(settings: SimulationSettings, path) -> Path:
    """Save ``settings`` as a simulation file; a structure file's path is made
    absolute so the recipe still finds it from wherever it is opened."""
    import yaml
    path = Path(path)
    if not is_simulation(path):
        path = path.with_name(path.name.split(".")[0] + ".sim.yaml")
    values = asdict(settings)
    if values["structure"]["file"]:
        values["structure"]["file"] = str(Path(values["structure"]["file"]).resolve())
    values["output"] = "camera"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# a smappy simulation: open it in a fitter as its file\n")
        yaml.safe_dump({"smappy": "simulation", "version": 1, "settings": values},
                       fh, sort_keys=False)
    return path


def read_recipe(path) -> SimulationSettings:
    import yaml
    from ..plugins import settings_from
    with open(path, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh) or {}
    if spec.get("smappy") != "simulation":
        raise ValueError(f"{path} is not a smappy simulation (no `smappy: simulation`)")
    return settings_from(SimulationSettings, spec.get("settings") or {})


@dataclass
class SimulatedSource(ImageSource):
    """Frames made as they are read."""
    settings: SimulationSettings = None
    camera: Dict[str, Any] = field(default_factory=dict)
    _truth: Any = field(default=None, repr=False)

    @property
    def ground_truth(self):
        return self._truth

    def truth(self):
        """Every spot drawn, as a localization table (`camera.camera_truth`)."""
        from .camera import camera_truth
        return camera_truth(self._truth)

    def frames(self, chunk: int = 100, start: int = 0,
               stop: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
        from .camera import render
        stop = self.n_frames if stop is None else min(stop, self.n_frames)
        for a in range(start, stop, chunk):
            b = min(a + chunk, stop)
            yield a, render(self._truth, a, b)

    def watch(self, chunk: int = 100, **kwargs):
        # nothing is being written: the whole stack is already "there"
        return self.frames(chunk=chunk, start=kwargs.get("start", 0),
                           stop=kwargs.get("stop"))


def open_simulation(path) -> SimulatedSource:
    from .localizations import ground_truth
    settings = read_recipe(path)
    truth = ground_truth(settings)
    size = settings.camera.size_px
    camera = {"conversion": settings.camera.conversion, "offset": settings.camera.offset,
              "pixelsize_um": settings.optics.pixelsize_nm / 1000.0,
              "em_on": False, "emgain": 1.0, "camera_name": "simulation"}
    return SimulatedSource(
        files=[Path(path)], shape=(size, size), dtype=np.dtype(np.uint16),
        n_frames=settings.n_frames, n_frames_declared=settings.n_frames,
        mm_metadata={}, summary={}, settings=settings, camera=camera, _truth=truth)
