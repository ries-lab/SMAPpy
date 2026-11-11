"""*DECODE*: localization by a trained network (Speiser et al. 2021), milestone (g).

A fitter on the shared `_FitPlugin` -- the source, the camera, live fits, saving, *after the
fit* -- with no peak finder and no PSF fit: a network trained on simulated frames of this
microscope's PSF finds the molecules and gives their positions, photons, background and
uncertainties at once.  The network is a *model file* (``.safetensors``, `smappy.training.
model_file`), trained by ``smappy-decode train`` on a job folder or exported from a model the
original DECODE trained (`decode_smlm.predict.export_original`).

What this module is: the settings, the checks a model can be refused by, and the hooks into
`_FitPlugin.run`.  The engine is `smappy.window_engine.WindowEngine` (blocks of frames joined
without a seam); the network is `decode_smlm.predict`, imported only when a fit runs, so the
form, the discovery and `describe` never load torch -- the model file's header is read with
numpy alone (`model_file.read_metadata`).

**What a model is checked against** (`model_problems`):

* the calibration -- the model learnt one PSF, and fits data of another as confidently as data
  of its own.  The model file keeps the sha256 of its calibration's coefficients
  (`training.job.coefficients_sha256`); a calibration chosen here, or the one the calibration
  list has for this file's microscope, must have the same, or the run is refused.  With no
  calibration to compare, the form says the model is unchecked;
* the mirror: SMAPpy keeps a calibration in the orientation of the camera its beads were taken
  with, so data taken with the EM register and beads without it (or the reverse) are flipped
  before the network sees them (`predict.frames_mirrored`), from the model's
  ``calibration_em_on`` and the data's camera -- never a setting;
* after the run, the fitted photons and background against what the training drew
  (`range_warnings`): a network has no answer for a molecule brighter than any it was shown.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import Context, FormState, Notice, Plot, Result, param, register
from .fit import (CameraMetadata, CameraSettings, FinishSettings, OutputSettings, SourceSettings,
                  _FitPlugin, _open)

MODEL_FILTER = "DECODE models (*.safetensors);;All files (*)"
CALIBRATION_FILTER = "Bead calibrations (*.h5 *.hdf5 *_3dcal.mat *.mat);;All files (*)"
DEVICES = (("auto", "auto"), ("cpu", "CPU"), ("cuda", "CUDA"), ("mps", "Apple GPU"))
METHOD = "decode-v0.10.2"
#: what fraction of the fit may lie outside what the training drew before the run says so
OUTSIDE_WARN = 0.05


@dataclass
class DecodeModelSettings:
    """The trained network, and the calibration it is checked against."""
    path: str = param("", label="model", kind="open_file", file_filter=MODEL_FILTER,
                      help="a trained DECODE model (.safetensors): a job folder's model.safetensors, "
                           "or an original DECODE model exported")
    calibration: str = param("", label="calibration", kind="open_file", file_filter=CALIBRATION_FILTER,
                             help="the data's bead calibration, which the model must have been trained "
                                  "on; empty: the one listed for this file's microscope, else unchecked")
    threshold: float = param(0.1, label="threshold", min=0.01, max=0.99, advanced=True,
                             help="the detection probability a pixel must reach to start a localization "
                                  "(DECODE's raw_th); every localization keeps its own in prob")
    device: str = param("auto", label="device", choices=DEVICES, advanced=True,
                        help="where the network runs; auto: a GPU if there is one")
    batch: int = param(32, label="frames per batch", min=1, advanced=True,
                       help="frames through the network at once; only memory and speed depend on it")
    where: str = param("here", label="run on", choices=lambda: _where_choices(),
                       help="this computer, or a training host of Preferences > Training hosts: the frames "
                            "are read and converted here and sent there, the network runs there")


@dataclass
class DecodeFitSettings:
    """The other fitters' parts under the same names (`fit._Reading`, `_Writing`, `_Finishing`),
    without detection, the fit region, the channels and the fit: none of them is the network's."""
    source: SourceSettings = field(default_factory=SourceSettings)
    camera: CameraSettings = param(default_factory=CameraSettings, label="microscope",
                                   collapsed=True, kind="microscope")
    model: DecodeModelSettings = field(default_factory=DecodeModelSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    finish: FinishSettings = param(default_factory=FinishSettings, label="after the fit", collapsed=True)


# ---------------------------------------------------------------------- the checks, torch-free
def read_model(path) -> dict:
    """A model file's metadata, from its header alone; refuses a file that is not a DECODE model."""
    from ..training import model_file
    if not path:
        raise ValueError("choose a trained model (.safetensors)")
    if not Path(path).is_file():
        raise ValueError(f"{Path(path).name}: no such model file")
    meta = model_file.read_metadata(path)
    if meta.get("method") != METHOD:
        raise ValueError(f"{Path(path).name} is not a DECODE model"
                         + (f" (it is {meta['method']})" if meta.get("method") else ""))
    return meta


def calibration_sha256(path) -> str:
    """The sha256 of a calibration's coefficients as SMAPpy loads it: what a model file keeps."""
    from ..io.calibration import load_spline_calibration
    from ..training.job import coefficients_sha256
    return coefficients_sha256(load_spline_calibration(path))


def model_problems(meta: dict, calibration_path: Optional[str]) -> list:
    """Why this model must not fit data calibrated with ``calibration_path``, as sentences; empty
    when it may, or when there is nothing to compare."""
    problems = []
    kept = meta.get("calibration_sha256")
    if calibration_path and kept:
        if calibration_sha256(calibration_path) != kept:
            problems.append(f"the model was trained on another calibration than "
                            f"{Path(calibration_path).name}: train one on this calibration, or "
                            f"choose the calibration the data were taken with")
    return problems


def range_warnings(locs, meta: dict) -> list:
    """The fit's photons and background against what the training drew, as sentences."""
    drew = meta.get("trained_on") or {}
    out = []
    n = len(locs)
    if not n:
        return out
    mu_sig = drew.get("photons_mu_sig")
    if mu_sig and "photons" in locs:
        top = mu_sig[0] + 3 * mu_sig[1]
        above = float(np.mean(np.asarray(locs["photons"]) > top))
        if above > OUTSIDE_WARN:
            out.append(f"{above:.0%} of the localizations have more photons than the training drew "
                       f"({top:,.0f}, its mean and three spreads): their photons and positions are "
                       f"extrapolated; train on the data's range")
    bg = drew.get("background")
    if bg and "background" in locs:
        b = np.asarray(locs["background"])
        outside = float(np.mean((b < bg[0]) | (b > bg[1])))
        if outside > OUTSIDE_WARN:
            out.append(f"{outside:.0%} of the localizations have a background outside the training's "
                       f"{bg[0]:g} to {bg[1]:g} photons per pixel")
    return out


@register("Localize/DECODE")
class DecodeFit(_FitPlugin):
    description = ("Localize with a trained DECODE network: no peak finding, positions, photons, "
                   "background and their uncertainties from each frame and its neighbours.")
    Settings = DecodeFitSettings
    version = "1"
    channel_part = "none"
    # a model is trained for these data in the Train window (`plugins/train_decode.py`)
    window_buttons = (("Train a model...", "decode training"),)
    preview_help = ("localize one frame with the network and draw what it found over the frame. "
                    "Nothing is saved and the session is not touched.")

    # ---------------------------------------------------------- the calibration it is checked against
    def checked_calibration(self, settings) -> Optional[str]:
        """The calibration the model is checked against: the one chosen, else the calibration list's
        match for this file (as Spline 3D's *auto* finds it), else None."""
        if settings.model.calibration:
            return settings.model.calibration
        if not settings.source.path:
            return None
        try:
            found = self.found_calibrations(settings)
        except Exception:
            return None
        return found[0][0].path if found and found[0][1] else None

    # ---------------------------------------------------------- the hooks of `_FitPlugin.run`
    def region_finder(self, settings, camera=None):
        return None                       # the network finds the molecules

    def model(self, settings, camera: CameraMetadata):
        meta = read_model(settings.model.path)
        problems = model_problems(meta, self.checked_calibration(settings))
        if problems:
            raise ValueError(problems[0])
        device = settings.model.device
        if settings.model.where != "here":
            return remote_predictor(settings, meta, camera)
        from decode_smlm.predict import load_predictor
        return load_predictor(settings.model.path, None if device == "auto" else device,
                              frames_mirrored=_mirrored(meta, camera), raw_th=settings.model.threshold,
                              batch_size=settings.model.batch)

    def engine(self, settings, camera, finder, model, source=None):
        from ..window_engine import WindowEngine
        return WindowEngine(camera, model)

    def provenance(self, camera, finder, model, settings, source) -> dict:
        from ..pipeline import _version
        meta = dict(model.metadata)
        return {"source": str(source) if source is not None else None, "camera": camera.to_dict(),
                "decode": {"model": str(settings.model.path), "device": model.device,
                           "threshold": model.raw_th, "frames_mirrored": model.frames_mirrored,
                           "window": model.window, "calibration": self.checked_calibration(settings),
                           "model_metadata": meta},
                "smappy_version": _version()}

    def checks(self, ctx: Context, settings, locs) -> list:
        """What the fit found outside what the network was trained on (`range_warnings`): in the
        run's text and in its file."""
        try:
            return range_warnings(locs, read_model(settings.model.path))
        except ValueError:
            return []

    # ---------------------------------------------------------- the form
    def describe(self, ctx: Optional[Context], settings) -> FormState:
        state = super().describe(ctx, settings)
        checked = None
        try:
            checked = self.checked_calibration(settings)
        except Exception:
            pass
        if not settings.model.calibration:
            state.hints["model.calibration"] = (f"auto: {Path(checked).name}" if checked
                                                else "auto: none listed; the model is not checked")
        if settings.model.path:
            # a model the run would refuse says so over anything else; a mere remark only where
            # the camera's form has nothing to say
            try:
                meta = read_model(settings.model.path)
                problems = model_problems(meta, checked)
                if problems:
                    state.notice = Notice(problems[0])
                elif state.notice is None and not meta.get("calibration_sha256"):
                    state.notice = Notice("the model does not say which calibration it was trained on: "
                                          "it cannot be checked against the data's")
            except Exception as error:      # a path still being typed, a file that is not a model
                state.notice = Notice(str(error))
        return state

    def preview(self, ctx: Context, settings, frame: int = 0) -> Result:
        """Localize one frame, with the neighbours the network needs, and draw what it found."""
        from ..camera import to_photons
        from ..window_engine import WindowEngine, _joined
        src = settings.source
        if not src.path:
            raise ValueError("choose a source file")
        source = _open(src, watch=False)
        camera = self._camera(settings)
        if camera is None:
            raise ValueError(f"could not read the camera from {src.path}")
        camera.require("conversion", "offset")
        predictor = self.model(settings, camera)
        index = int(np.clip(frame, 0, max(source.n_frames - 1, 0)))
        k = predictor.window
        start, stop = max(index - k, 0), min(index + k + 1, source.n_frames)
        frames = np.concatenate([block for _, block in source.frames(chunk=stop - start, start=start, stop=stop)])
        engine = WindowEngine(camera, predictor)
        parts = [engine.push(frames, start), engine.end()]
        x = y = np.zeros(0)
        prob = np.zeros(0)
        table = _joined(parts)
        if table is not None:
            from ..locs import in_pixels
            here = np.asarray(table["frame"]) == index
            px, py = in_pixels(table, "x_pix", "y_pix")
            ox, oy = camera.roi_offset
            x, y = np.asarray(px)[here] - ox, np.asarray(py)[here] - oy
            prob = np.asarray(table["prob"])[here]
        photons = to_photons(frames[index - start], camera)

        def plot(ax) -> None:
            ax.imshow(photons, cmap="gray", vmax=np.percentile(photons, 99.8) or None)
            if len(x):
                points = ax.scatter(x, y, c=prob, cmap="autumn", vmin=0, vmax=1, s=30, marker="+")
                ax.figure.colorbar(points, ax=ax, fraction=0.046, label="prob")
            ax.set(title=f"frame {index}: {len(x)} localized")

        return Result(text=f"frame {index}: {len(x)} localizations", settings=settings,
                      plot=Plot(draw=plot, size=(7, 6)), data={"frame": index, "localizations": int(len(x))})


def _where_choices():
    from ..training import jobs
    return [("here", "this computer")] + [(h.name, h.name) for h in jobs.hosts()]


def remote_predictor(settings, meta: dict, camera: CameraMetadata):
    """The network on the training host *run on* names (`smappy.decode_remote`): the frames are
    read, cut and converted here and only the network runs there.  A live fit runs here: its
    frames arrive as they are taken, and a network across the network adds nothing then."""
    from ..decode_remote import RemotePredictor
    from ..training import jobs
    m = settings.model
    host = jobs.host(m.where)
    if host is None:
        raise ValueError(f"no training host {m.where!r}: add it in Preferences > Training hosts, "
                         f"or run the network on this computer")
    if getattr(settings.source, "live", False):
        raise ValueError("a live fit runs its network on this computer: set run on to this computer")
    return RemotePredictor(host, m.path, None if m.device == "auto" else m.device,
                           frames_mirrored=_mirrored(meta, camera), raw_th=m.threshold, batch_size=m.batch)


def _mirrored(meta: dict, camera: CameraMetadata) -> bool:
    """Whether the data's frames are mirrored against the model's calibration.

    SMAPpy keeps a calibration in the orientation of the camera its beads were taken with
    (`smappy.io.calibration`), and an EM register reads the chip out mirrored: beads taken
    without EM gain and data with it (the example package), or the other way, are mirrored
    against each other.  Unknown on either side: not flipped, as Spline 3D does not flip.
    """
    em = meta.get("calibration_em_on")
    if em is None or camera.em_on is None:
        return False
    return bool(em) != bool(camera.em_on)
