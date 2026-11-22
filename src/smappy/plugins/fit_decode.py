"""*DECODE*: localization by a trained network (Speiser et al. 2021), milestone (g).

A fitter on the shared `_FitPlugin` -- the source, the camera, live fits, saving, *after the
fit* -- with no peak finder and no PSF fit: a network trained on simulated frames of this
microscope's PSF finds the molecules and gives their positions, photons, background and
uncertainties at once.  The network is a *model file* (``.safetensors``, `smappy.training.
model_file`), trained by ``smappy-decode train`` on a job folder or exported from a model the
original DECODE trained (`decode_smlm.predict.export_original`).  Two methods train one: the port of
DECODE v0.10.2 (``decode-v0.10.2``) and SMAPpy's own network (``netfit-v1``, *DECODE, optimised*,
`smappy.netfit`; docs/decode/gui-plan.md), loaded alike by `training.predictors.load`.

What this module is: the settings, the checks a model can be refused by, and the hooks into
`_FitPlugin.run`.  The engine is `smappy.window_engine.WindowEngine` (blocks of frames joined
without a seam); the network is `decode_smlm.predict` or `smappy.netfit.predict`, imported only when
a fit runs, so the
form, the discovery and `describe` never load torch -- the model file's header is read with
numpy alone (`model_file.read_metadata`).

**What a model is checked against:**

* the conditions -- the model learnt one PSF, and fits data of another as confidently as data
  of its own.  The model file keeps the microscope its calibration's beads were taken with, the
  states of the devices the PSF follows (the 3D lens, the filters) and the channel
  (``calibration_microscope``, from the calibration list); data the microscope database resolves
  otherwise are warned about in the form, the run's text and its file
  (`calibration_list.differences`).  A warning, not a refusal: a model of the same lens on a
  neighbouring day may serve.  There is no calibration to choose here: the network needs none, and
  checking one the user picks checked only the pick (*Model info...* says what the model was
  trained on);
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
from .fit import (
    CameraMetadata,
    CameraSettings,
    FinishSettings,
    OutputSettings,
    SourceSettings,
    _FitPlugin,
    _open,
)

MODEL_FILTER = "DECODE models (*.safetensors);;All files (*)"
DEVICES = (("auto", "auto"), ("cpu", "CPU"), ("cuda", "CUDA"), ("mps", "Apple GPU"))
METHOD = "decode-v0.10.2"           # the port's; `training.predictors.METHODS` are every one the fitter runs
#: what fraction of the fit may lie outside what the training drew before the run says so
OUTSIDE_WARN = 0.05


@dataclass
class DecodeModelSettings:
    """The trained network, and the calibration it is checked against."""
    path: str = param("", label="trained model", kind="open_file", file_filter=MODEL_FILTER,
                      help="a trained DECODE model (.safetensors): a job folder's model.safetensors, "
                           "or an original DECODE model exported")
    run_on: str = param("here", label="run on", choices=lambda: _where_choices(),
                        help="this computer, or a training host (Training hosts... below): the frames "
                             "are read and converted here and sent there, the network runs there")
    threshold: float = param(0.1, label="threshold", min=0.01, max=0.99, advanced=True,
                             help="the detection probability a pixel must reach to start a localization "
                                  "(DECODE's raw_th); every localization keeps its own in prob")
    device: str = param("auto", label="device", choices=DEVICES, advanced=True,
                        help="where the network runs; auto: a GPU if there is one")
    batch: int = param(0, label="frames per batch", min=0, advanced=True,
                       help="frames through the network at once; only memory and speed depend on it; "
                            "0: as many as half the GPU's free memory holds")


DUAL_FILTER = "Two-channel calibrations (*_2ct.h5 *.h5);;All files (*)"


@dataclass
class DecodeChannelSettings:
    """A model that reads a split camera's two channels (*aligned channels*): where they are, and
    the checks (docs/decode/gui-plan.md, step 3).  Shown for such a model only."""
    calibration: str = param("", label="calibration", kind="open_file", file_filter=DUAL_FILTER,
                             help="the two-channel calibration the model was trained on, for its registration; "
                                  "empty: the one the model names, where it is on this computer")
    roi_column: Optional[int] = param(None, label="camera ROI at column", min=0, advanced=True,
                                      help="where the frames' first column lies on the chip; empty: the "
                                           "file's camera ROI (a file that names none: 0)")
    roi_row: Optional[int] = param(None, label="row", min=0, advanced=True, beside=True,
                                   help="where the frames' first row lies on the chip; empty: the file's")
    moved_rows: float = param(0.0, label="moved since the beads", unit="rows",
                              help="how far channel 1 lies from where the beads' registration puts it, along "
                                   "the rows: a splitter moved between calibration and measurement")
    moved_columns: float = param(0.0, label="and", unit="columns", beside=True,
                                 help="the same along the columns")
    measure_move: bool = param(False, label="measure the move",
                               help="find how far the channels moved since the beads on the first 20 frames -- "
                                    "where the network finds the fewest molecules -- starting from the move "
                                    "typed; about a minute; the move found is in the run's text and file")
    goodness: bool = param(False, label="goodness of fit",
                           help="each localization's neighbourhood against the fit drawn back, a column "
                                "goodness (about 0 explained, a few and more not): finds false positives; "
                                "about 0.1 s per 10 000 localizations on a GPU, 1 s on a CPU")
    bound: bool = param(False, label="bound from the fit", beside=True,
                        help="each localization's CRLB from its own fitted values, crlb_x_nm and so on: a "
                             "reported sigma far above it marks a position outlier; about 0.4 s per 10 000 "
                             "on a GPU, 9 s on a CPU")


@dataclass
class DecodeFitSettings:
    """The other fitters' parts under the same names (`fit._Reading`, `_Writing`, `_Finishing`),
    without detection, the fit region, the channels and the fit: none of them is the network's."""
    source: SourceSettings = field(default_factory=SourceSettings)
    camera: CameraSettings = param(default_factory=CameraSettings, label="microscope",
                                   collapsed=True, kind="microscope")
    model: DecodeModelSettings = field(default_factory=DecodeModelSettings)
    # not ``channels``: `_FitPlugin` reads a part of that name as the spline fitters' split camera
    two_channels: DecodeChannelSettings = param(default_factory=DecodeChannelSettings, label="two channels")
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
    from ..training.predictors import METHODS
    meta = model_file.read_metadata(path)
    if meta.get("method") not in METHODS:
        raise ValueError(f"{Path(path).name} is not a DECODE model"
                         + (f" (it is {meta['method']})" if meta.get("method") else ""))
    return meta


def two_channel_calibration(settings, meta: dict) -> Path:
    """The two-channel calibration a model's registration is read from: the one typed, else the one
    the model names; checked against the model's sha256 of it."""
    import hashlib
    typed = settings.two_channels.calibration
    path = Path(typed) if typed else Path(meta.get("calibration_source") or "")
    if not str(path) or not path.is_file():
        raise ValueError("the model reads two channels: choose its two-channel calibration (the model names "
                         + (f"{Path(meta['calibration_source']).name}, not on this computer)" if meta.get("calibration_source")
                            else "none)"))
    want = meta.get("calibration_sha256")
    if want:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != want:
            raise ValueError(f"{path.name} is not the calibration the model was trained on")
    return path


#: what a run on this computer says without PyTorch: the base install has none, and the fitter is
#: in the shipped GUI; the form and *Model info...* need none, nor a run on a training host
NO_TORCH = ("DECODE's network needs PyTorch, which this Python does not have: install it with "
            "pip install \"smappy-smlm[decode]\", or set run on to a training host")


def torch_here() -> bool:
    """Whether PyTorch can be imported here, without importing it."""
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def summary_line(meta: dict) -> str:
    """The model in a line: what it was trained for and how far."""
    bits = []
    if meta.get("calibration_source"):
        bits.append(f"trained on {Path(meta['calibration_source']).name}")
    scope = meta.get("calibration_microscope") or {}
    if scope.get("name"):
        bits.append(f"microscope {scope['name']}")
    depth = meta.get("calibration_sample_depth") or {}
    if depth.get("focus_nm") is not None:
        bits.append(f"focus {float(depth['focus_nm']):.0f} nm in the sample")
    if meta.get("epoch") is not None:
        bits.append(f"epoch {meta['epoch']}")
    jac = (meta.get("test_metrics") or {}).get("jaccard")
    if jac is not None:
        bits.append(f"test Jaccard {jac:.2f}")
    from ..training.predictors import DECODE, describe
    kind = "" if meta.get("method") == DECODE else f" ({describe(meta)})"     # the original's, as before
    return f"model{kind}: " + (", ".join(bits) if bits else "trained by " + str(meta.get("trained_by") or "DECODE"))


def _drawn_for(meta: dict) -> str:
    """Where in the sample the network's PSF was drawn, and so what its z is (`calibrate.depth`)."""
    from ..calibrate.depth import describe_record
    drawn = describe_record(meta.get("calibration_sample_depth") or {})
    if drawn:
        return drawn
    # a SMAPpy job names its calibration; a model from elsewhere (DECODE's own, exported) says nothing
    return "the beads: z above their focus" if meta.get("calibration_source") else "not recorded"


def model_summary(meta: dict, data: Optional[dict] = None) -> list:
    """``(what, value)`` rows of what a model file says about itself, for *Model info...*: the
    calibration and the conditions it was trained for, the ranges it drew, the training, and how
    well it did on its test set; with ``data`` (the data's conditions) the differences first."""
    from ..calibration_list import differences
    from ..microscope_db import one_line
    rows = []
    for d in differences(meta.get("calibration_microscope"), data):
        rows.append(("differs from the data", d))
    from ..training.predictors import describe
    drew = meta.get("trained_on") or {}
    mu = drew.get("photons_mu_sig")
    span = drew.get("photons_range")
    bg = drew.get("background")
    z = drew.get("z_nm")
    scope = meta.get("calibration_microscope")
    test = meta.get("test_metrics") or {}
    window = meta.get("channels_in") or meta.get("window")
    rows += [
        ("model", f"{describe(meta)} ({meta.get('method')})"),
        ("PSF calibration", Path(meta["calibration_source"]).name if meta.get("calibration_source") else "not recorded"),
        ("calibration file", meta.get("calibration_source") or ""),
        ("microscope", one_line(scope) if scope else "not recorded"),
        ("EM gain of the beads", {True: "on", False: "off"}.get(meta.get("calibration_em_on"), "not recorded")),
        ("PSF drawn for", _drawn_for(meta)),
        ("photons a frame", f"{mu[0]:,.0f} \u00b1 {mu[1]:,.0f}" if mu
         else f"{span[0]:,.0f} to {span[1]:,.0f}" if span else ""),
        ("background", f"{bg[0]:g} to {bg[1]:g} photons a pixel" if bg else ""),
        ("z", f"{z[0]:g} to {z[1]:g} nm" if z else ""),
        ("density", str(meta.get("density") or "")),
        ("frame window", str(window or "")),
        ("trained", f"epoch {meta['epoch']} of at most {meta.get('epochs')}" if meta.get("epoch") is not None else ""),
        ("started from", str(meta.get("base_model") or "scratch")),
        ("test set", ", ".join(f"{k} {v:.3g}" for k, v in (("Jaccard", test.get("jaccard")),
                                                          ("lateral RMSE (nm)", test.get("rmse_lat_nm")),
                                                          ("axial RMSE (nm)", test.get("rmse_ax_nm")))
                               if v is not None)),
        ("job written", str(meta.get("created") or "")),
        ("trained by", str(meta.get("trained_by") or f"SMAPpy {meta.get('smappy_version') or ''}".strip())),
    ]
    return [(k, v) for k, v in rows if v not in ("", None)]


def range_warnings(locs, meta: dict) -> list:
    """The fit's photons and background against what the training drew, as sentences."""
    drew = meta.get("trained_on") or {}
    out = []
    n = len(locs)
    if not n:
        return out
    mu_sig, span = drew.get("photons_mu_sig"), drew.get("photons_range")
    if (mu_sig or span) and "photons" in locs:
        top = mu_sig[0] + 3 * mu_sig[1] if mu_sig else span[1]
        how = "its mean and three spreads" if mu_sig else "the top of its range"
        above = float(np.mean(np.asarray(locs["photons"]) > top))
        if above > OUTSIDE_WARN:
            out.append(f"{above:.0%} of the localizations have more photons than the training drew "
                       f"({top:,.0f}, {how}): their photons and positions are "
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
    description = "Deep-learning fitter for high densities (DECODE)."
    Settings = DecodeFitSettings
    version = "1"
    channel_part = "none"
    # a model is trained for these data in the Train window (`plugins/train_decode.py`), handed
    # the movie, microscope, PSF calibration and *run on*; the hosts are Preferences' (`HostsPage`)
    part_windows = (("model", "Model info...", "decode model"), ("model", "Train a model...", "decode training"),
                    ("model", "Training hosts...", "training hosts"))
    preview_help = ("localize one frame with the network and draw what it found over the frame. "
                    "Nothing is saved and the session is not touched.")

    # ---------------------------------------------------------- the data's conditions against the model's
    def data_conditions(self, settings) -> Optional[dict]:
        """The data's microscope, its devices' states and its channel (`MicroscopeResolution.summary`),
        as the microscope database resolves them for this file; None when unknown."""
        if not settings.source.path:
            return None
        try:
            resolved = self.resolution(settings)
        except Exception:
            return None
        return None if resolved is None else resolved.camera_metadata().microscope

    def condition_warnings(self, settings, meta: dict) -> list:
        """How the data were taken otherwise than the model's beads (`calibration_list.differences`)."""
        from ..calibration_list import differences
        found = differences(meta.get("calibration_microscope"), self.data_conditions(settings))
        return [f"the model was trained for other conditions than the data's: {d}; train a model for "
                f"these data (Train a model...)" for d in found]

    # ---------------------------------------------------------- the hooks of `_FitPlugin.run`
    def region_finder(self, settings, camera=None):
        return None                       # the network finds the molecules

    def model(self, settings, camera: CameraMetadata):
        from ..training.predictors import two_channel
        meta = read_model(settings.model.path)
        device = settings.model.device
        if settings.model.run_on != "here" and two_channel(meta):
            raise ValueError("a two-channel model runs on this computer for now: set run on to this computer")
        if settings.model.run_on != "here":
            return remote_predictor(settings, meta, camera)
        if not torch_here():
            raise ValueError(NO_TORCH)
        from ..training.predictors import load, two_channel
        if not two_channel(meta):
            return load(settings.model.path, None if device == "auto" else device,
                        frames_mirrored=_mirrored(meta, camera), threshold=settings.model.threshold,
                        batch=settings.model.batch)
        return self.two_channel_model(settings, meta, camera)

    def two_channel_model(self, settings, meta: dict, camera: CameraMetadata):
        """A two-channel model with its registration, moved as typed (`netfit.fit_two`)."""
        if _mirrored(meta, camera):
            raise ValueError("the data were taken with the EM register set otherwise than the beads: a "
                             "two-channel model takes frames as its calibration has them")
        from ..calibrate.dual import load_dual_color_calibration
        from ..netfit.fit_two import TwoChannelPredictor
        from ..netfit.predict import Predictor
        c = settings.two_channels
        cal = load_dual_color_calibration(two_channel_calibration(settings, meta))
        device = settings.model.device
        predictor = Predictor(settings.model.path, None if device == "auto" else device,
                              threshold=settings.model.threshold, batch=settings.model.batch)
        sizes, _ = camera.pixelsize_nm_or_default
        ox, oy = camera.roi_offset
        corner = (ox if c.roi_column is None else c.roi_column, oy if c.roi_row is None else c.roi_row)
        model = TwoChannelPredictor(predictor, cal, corner, (c.moved_columns, c.moved_rows),
                                    em=bool(camera.em_on), goodness=c.goodness, bound=c.bound, pixel_nm=sizes,
                                    measure=c.measure_move)
        model.cal_path = two_channel_calibration(settings, meta)
        self._two = model                  # for the run's text: the move used (`checks`)
        return model

    def engine(self, settings, camera, finder, model, source=None):
        from ..window_engine import WindowEngine
        return WindowEngine(camera, model)

    def provenance(self, camera, finder, model, settings, source) -> dict:
        from ..pipeline import _version
        meta = dict(model.metadata)
        return {"source": str(source) if source is not None else None, "camera": camera.to_dict(),
                "decode": {"model": str(settings.model.path), "device": model.device,
                           **({"moved_since_beads": model.moved_record,
                               "two_channel_calibration": str(model.cal_path)} if hasattr(model, "moved") else {}),
                           "threshold": model.raw_th, "frames_mirrored": model.frames_mirrored,
                           "window": model.window, "calibration": meta.get("calibration_source"),
                           "model_metadata": meta},
                "smappy_version": _version()}

    def checks(self, ctx: Context, settings, locs) -> list:
        """What the fit found outside what the network was trained on (`range_warnings`): in the
        run's text and in its file."""
        try:
            meta = read_model(settings.model.path)
        except ValueError:
            return []
        said = []
        two = getattr(self, "_two", None)
        if two is not None and two.measure:
            said.append(f"the channels moved since the beads by {two.moved[1]:.2f} rows and {two.moved[0]:.2f} "
                        f"columns (measured); type them under two channels to fit this microscope's data so")
        return said + self.condition_warnings(settings, meta) + range_warnings(locs, meta)

    # ---------------------------------------------------------- the form
    def describe(self, ctx: Optional[Context], settings) -> FormState:
        state = super().describe(ctx, settings)
        # the two channels' part for a model that reads them, nothing for one that does not
        state.shown["two_channels"] = self._two_channel(settings)
        if settings.model.run_on == "here" and not torch_here():
            state.notice = Notice(NO_TORCH)        # said before the run, not after it
            return state
        if settings.model.path:
            # other conditions than the model's say so over anything else; else the model in a line,
            # where the camera's form has nothing to say
            try:
                meta = read_model(settings.model.path)
                warnings = self.condition_warnings(settings, meta)
                if warnings:
                    more = f" (and {len(warnings) - 1} more: Model info...)" if len(warnings) > 1 else ""
                    state.notice = Notice(warnings[0] + more)
                elif state.notice is None:
                    state.notice = Notice(summary_line(meta))
            except Exception as error:      # a path still being typed, a file that is not a model
                state.notice = Notice(str(error))
        return state

    @staticmethod
    def _two_channel(settings) -> bool:
        from ..training.predictors import two_channel
        try:
            return two_channel(read_model(settings.model.path))
        except Exception:                   # no model yet, a path being typed: nothing to show
            return False

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
    host = jobs.host(m.run_on)
    if host is None:
        raise ValueError(f"no training host {m.run_on!r}: add it in Preferences > Training hosts, "
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
