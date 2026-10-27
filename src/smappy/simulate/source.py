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


def recipe_path(path) -> Path:
    """Where `write_recipe` writes when asked for ``path``: the path itself if
    it is a simulation file's, else its stem with ``.sim.yaml``."""
    path = Path(path)
    if is_simulation(path):
        return path
    return path.with_name(path.name.split(".")[0] + ".sim.yaml")


def write_recipe(settings: SimulationSettings, path) -> Path:
    """Save ``settings`` as a simulation file; a structure file's path is made
    absolute so the recipe still finds it from wherever it is opened."""
    import yaml
    path = recipe_path(path)
    values = recipe_values(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# a smappy simulation: open it in a fitter as its file\n")
        yaml.safe_dump({"smappy": "simulation", "version": 1, "settings": values},
                       fh, sort_keys=False)
    if dual_of(settings) and not settings.optics.calibration:
        # (a dual-colour calibration carries its own registration)
        write_transform(settings, transform_path(path))
    if settings.camera.fourpi and not settings.optics.calibration:
        # the closed-form model the frames are drawn with, for the fitter
        from ..fourpi import save_fourpi_calibration
        from .camera import fourpi_model
        save_fourpi_calibration(fourpi_path(path), fourpi_model(settings)[0], overwrite=True)
    return path


def fourpi_path(recipe) -> Path:
    """Where a 4Pi camera's closed-form model is written beside its recipe."""
    recipe = Path(recipe)
    return recipe.with_name(recipe.name.split(".")[0] + "_4pi.h5")


def recipe_values(settings: SimulationSettings) -> dict:
    """The settings as a simulation file holds them."""
    values = asdict(settings)
    for part in values.get("components") or [values]:
        if part["structure"]["file"]:
            part["structure"]["file"] = str(Path(part["structure"]["file"]).resolve())
    if values["optics"]["calibration"]:
        values["optics"]["calibration"] = str(Path(values["optics"]["calibration"]).resolve())
    if values["output"] == "localizations":
        values["output"] = "camera"
    return values


def transform_path(recipe) -> Path:
    """Where a split camera's transformation is written beside its recipe:
    ``run.sim.yaml`` -> ``run_2ct.h5``, the name a two-colour fit's own
    calibration from a movie gets."""
    recipe = Path(recipe)
    stem = next((recipe.name[:-len(x)] for x in SUFFIXES if recipe.name.lower().endswith(x)),
                recipe.stem)
    return recipe.with_name(stem + "_2ct.h5")


def write_transform(settings, path) -> Path:
    """The map between the split camera's halves (`camera.dual_transformation`)
    as a two-colour fitter's *transformation* file: the simulation knows it
    exactly, so its fit need not measure it from the movie."""
    from ..calibrate.dual import ChannelTransform
    from ..calibrate.transform import save_channel_transform
    from .camera import dual_transformation
    size = settings.camera.size_px
    geometry = {"layout": "up-down", "main_channel": "upper", "split_position": size,
                "image_shape": [2 * size, size], "coordinate_system": "camera-chip"}
    transform = ChannelTransform(dual_transformation(size), geometry,
                                 {"source": "smappy simulation"})
    save_channel_transform(path, transform, overwrite=True)
    return Path(path)


def recipe_transform(path) -> Optional[Path]:
    """The transformation written beside a split camera's recipe, if ``path``
    is one and it is there: what a two-colour fitter takes for its file."""
    try:
        if not is_simulation(path):
            return None
        settings = read_recipe(path)
        if not dual_of(settings) or settings.optics.calibration:
            return None          # one camera, or a calibration's own registration
    except (OSError, ValueError):
        return None
    found = transform_path(path)
    return found if found.is_file() else None


def recipe_calibration(path) -> Optional[Tuple[str, str]]:
    """The PSF calibration a recipe draws its spots with, if ``path`` is
    such a recipe: ``("single", path)`` for one camera's bead calibration,
    ``("dual", path)`` for a split camera's dual-colour one, ``("fourpi",
    path)`` for a 4Pi camera's model, written beside the recipe when it is
    the closed-form one -- what a spline fitter of that kind takes for its
    own, so the fit meets the PSF the frames were made with.  None for
    Gaussian spots."""
    try:
        if not is_simulation(path):
            return None
        settings = read_recipe(path)
    except (OSError, ValueError):
        return None
    if getattr(settings.camera, "fourpi", False):
        written = fourpi_path(path)
        model = settings.optics.calibration or (str(written) if written.is_file() else "")
        return ("fourpi", model) if model else None
    if not settings.optics.calibration:
        return None
    return ("dual" if dual_of(settings) else "single"), settings.optics.calibration


def dual_of(settings) -> Optional[dict]:
    """How `camera.render` draws a split camera's two halves, or None for
    one camera: the second half's PSF width, the first's on *auto*."""
    if not getattr(getattr(settings, "camera", None), "split", False):
        return None
    first = settings.optics.sigma_nm
    second = settings.camera.split_sigma_nm or first
    return {"sigma_nm": (first, second)}


def read_recipe(path) -> SimulationSettings:
    import yaml
    with open(path, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh) or {}
    if spec.get("smappy") != "simulation":
        raise ValueError(f"{path} is not a smappy simulation (no `smappy: simulation`)")
    from .scene import settings_of
    return settings_of(spec.get("settings") or {})


@dataclass
class SimulatedSource(ImageSource):
    """Frames made as they are read."""
    settings: SimulationSettings = None
    camera: Dict[str, Any] = field(default_factory=dict)
    #: a 4Pi camera's channels (`ChannelLayout`), as the quadrant files give theirs
    layout: Any = None
    _truth: Any = field(default=None, repr=False)

    @property
    def ground_truth(self):
        return self._truth

    def truth(self):
        """Every spot drawn, as a localization table (`camera.camera_truth`)."""
        from .camera import camera_truth
        return camera_truth(self._truth, dual=dual_of(self.settings))

    def frames(self, chunk: int = 100, start: int = 0,
               stop: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
        from .camera import render
        stop = self.n_frames if stop is None else min(stop, self.n_frames)
        for a in range(start, stop, chunk):
            b = min(a + chunk, stop)
            yield a, render(self._truth, a, b, dual=dual_of(self.settings))

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
              "em_on": settings.camera.em_gain > 0,
              "emgain": settings.camera.em_gain if settings.camera.em_gain > 0 else 1.0,
              "frame_time_ms": settings.frame_time_ms, "camera_name": "simulation"}
    from .camera import fourpi_model
    fourpi = fourpi_model(settings)
    layout = None
    if dual_of(settings):
        # a split camera's two halves, laid out as its calibration's, if any
        from .camera import split_layout
        shape = split_layout(settings)["shape"]
    elif fourpi is not None:
        from ..fourpi import frame_size
        layout = fourpi[1]
        shape = frame_size(layout)
    else:
        shape = (size, size)
    return SimulatedSource(
        files=[Path(path)], shape=shape, dtype=np.dtype(np.uint16),
        n_frames=settings.n_frames, n_frames_declared=settings.n_frames,
        mm_metadata={}, summary={}, settings=settings, camera=camera, _truth=truth,
        layout=layout)
