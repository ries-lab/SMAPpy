"""uiPSF calibration: a PSF model learnt by uiPSF, saved as smappy's calibration.

A separate plugin from the bead calibration window on purpose, even where the
two overlap (reading bead stacks, saving): uiPSF is an optional external
program with its own Python, and keeping it out of the window keeps that
window's code free of it.  Everything is in `smappy.uipsf`; this is the form.

What it adds to the bead calibration: a model of the optics (a pupil, its
Zernike aberrations) rather than an average of beads, so it is smooth without
smoothing and extends past the beads' noise; and in situ learning from the
blinking molecules of an acquisition, for a PSF the sample has changed.
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from . import Context, Plot, Plugin, Result, param, register
from .fit import TIFF_FILTER, CameraSettings

CALIBRATION_FILTER = "smappy calibration (*.h5);;All files (*)"
UIPSF_FILTER = "uiPSF or smappy uiPSF calibration (*.h5);;All files (*)"


def _microscopes():
    from ..uipsf.microscopes import choices
    try:
        return choices()
    except Exception:                          # noqa: BLE001 -- the list must open
        return [("", "uiPSF defaults")]


@dataclass
class BeadPart:
    """Bead z-stacks: the objective stepped through focus."""
    z_step_nm: Optional[float] = param(None, label="z step", unit="nm", min=1,
                                       help="auto: from the stack's z positions")
    whole_folder: bool = param(True, label="all in folder",
                               help="every bead acquisition in the file's folder, "
                                    "not only the one chosen")
    max_beads: int = param(40, label="max beads", min=1, advanced=True,
                           help="at most this many beads are used")
    bead_diameter_nm: float = param(0.0, label="bead size", unit="nm", min=0,
                                    advanced=True,
                                    help="the beads' diameter, deconvolved from "
                                         "the model; 0 for point-like beads")


@dataclass
class BlinkingPart:
    """Single molecules in the sample (in situ)."""
    depth_um: float = param(1.0, label="depth", unit="um", min=0,
                            help="how far the objective's focus was moved into "
                                 "the sample from the coverslip; uiPSF refines it")
    z_range_um: float = param(2.0, label="z range", unit="um", min=0.2,
                              help="the depth of the model, around focus")
    z_step_nm: float = param(50.0, label="z step", unit="nm", min=5,
                             help="the model's plane spacing")
    first_frame: int = param(0, label="first frame", min=0,
                             help="frames before this one are not read")
    last_frame: Optional[int] = param(3000, label="last frame", min=1,
                                      help="auto: to the end; more frames, more "
                                           "molecules, and more memory")
    start_from: str = param("", label="start from", kind="open_file",
                            file_filter=UIPSF_FILTER, advanced=True,
                            help="a uiPSF result (a bead model, say) to start "
                                 "the pupil from")
    min_photon: float = param(0.4, label="photon quantile", min=0, max=1,
                              advanced=True,
                              help="molecules dimmer than this quantile are "
                                   "not used")
    z_bins: int = param(21, label="z bins", min=1, advanced=True,
                        help="molecules are chosen evenly over this many slices "
                             "in z")
    per_bin: int = param(100, label="per bin", min=1, advanced=True,
                         help="at most this many molecules per slice")
    repeat: int = param(2, label="rounds", min=1, advanced=True,
                        help="learning repeated, each round starting from the last")


@dataclass
class LearningPart:
    """What uiPSF is told beyond the microscope."""
    roi_size_px: Optional[int] = param(None, label="model size", unit="px", min=11,
                                       help="auto: the microscope profile's; at "
                                            "least 3 px more than the fit's ROI")
    emission_wavelength_nm: Optional[float] = param(None, label="wavelength",
                                                    unit="nm", min=300, max=1200,
                                                    help="auto: the profile's")
    iterations: int = param(100, min=10, help="uiPSF's optimiser steps per round")
    zernike_order: int = param(8, label="Zernike order", min=2, max=12,
                               advanced=True,
                               help="the highest radial order of the pupil's "
                                    "Zernike terms")
    peak_height: float = param(0.2, label="threshold", min=0, max=1, advanced=True,
                               help="relative brightness an emitter must reach "
                                    "to be used")
    python: str = param("", label="uiPSF python", kind="open_file", advanced=True,
                        help="the python of uiPSF's environment; auto: "
                             "uipsf_python in the config file, or "
                             "$SMAPPY_UIPSF_PYTHON")
    keep_job: bool = param(False, label="keep job folder", advanced=True,
                           help="keep uiPSF's input, log and result in a "
                                "folder beside the calibration")


@dataclass
class UiPSFSettings:
    data: str = param("beads", choices=(("beads", "bead z-stacks"),
                                        ("blinking", "blinking molecules")),
                      help="beads: a calibration as the bead window makes one; "
                           "blinking: the PSF in the sample itself")
    path: str = param("", label="file", kind="open_file", file_filter=TIFF_FILTER,
                      help="a file of the acquisition")
    microscope: str = param("", choices=lambda: _microscopes(),
                            help="the optics: NA, refractive indices, "
                                 "wavelength, the splitter")
    channels: str = param("auto", choices=(("auto", "as the microscope"),
                                           ("single", "one"),
                                           ("dual", "two (split camera)")),
                          help="auto: two if the microscope has a splitter")
    model: str = param("zernike", choices=(("zernike", "Zernike pupil"),
                                           ("voxel", "voxels (beads)"),
                                           ("pupil", "free pupil (blinking)")),
                       help="Zernike: smooth and physical; voxels: any shape, "
                            "needs many beads; free pupil: for unusual PSFs "
                            "such as a tetrapod")
    output: str = param("", label="save as", kind="save_file",
                        file_filter=CALIBRATION_FILTER,
                        help="auto: <acquisition>_uipsf_3dcal.h5 beside the data")
    beads: BeadPart = param(default_factory=BeadPart, label="beads")
    blinking: BlinkingPart = param(default_factory=BlinkingPart, label="blinking",
                                   collapsed=True)
    learning: LearningPart = param(default_factory=LearningPart, label="learning",
                                   collapsed=True)
    camera: CameraSettings = param(default_factory=CameraSettings, label="camera",
                                   collapsed=True)


def output_path(settings: UiPSFSettings) -> Path:
    """Where the calibration goes."""
    if settings.output:
        return Path(settings.output)
    source = Path(settings.path)
    if settings.data == "beads" and settings.beads.whole_folder:
        base = source.parent if source.is_file() else source
        return base / f"{base.name}_uipsf_3dcal.h5"
    stem = source.name.split(".")[0] if source.is_file() else source.name
    return (source.parent if source.is_file() else source) / f"{stem}_uipsf_3dcal.h5"


def check(settings: UiPSFSettings) -> None:
    """Refuse what cannot work, before minutes are spent on it."""
    if not settings.path:
        raise ValueError("choose a file of the acquisition")
    if not Path(settings.path).exists():
        raise ValueError(f"{settings.path} does not exist")
    if settings.data == "beads" and settings.model == "pupil":
        raise ValueError("a free pupil is learnt from blinking molecules; for "
                         "beads choose Zernike or voxels")
    if settings.data == "blinking" and settings.model == "voxel":
        raise ValueError("a voxel model is learnt from beads; for blinking "
                         "molecules choose Zernike or a free pupil")


@register("Localize/uiPSF calibration")
class UiPSFCalibration(Plugin):
    """A PSF model from beads or blinking molecules, learnt by uiPSF, for Spline 3D."""
    description = ("learn a PSF model with uiPSF from bead stacks or from the "
                   "blinking molecules themselves, and save it as a calibration "
                   "for Spline 3D and Spline 3D 2C")
    Settings = UiPSFSettings
    version = "1"
    main = ("data", "path", "microscope", "channels", "model", "output")

    def run(self, ctx: Context, settings: UiPSFSettings) -> Result:
        from ..uipsf import convert, learn, microscopes, prepare, uipsf_parameters
        check(settings)
        profile = microscopes.load(settings.microscope)
        dual = (profile.dual is not None if settings.channels == "auto"
                else settings.channels == "dual")
        if dual and profile.dual is None:
            profile = replace(profile, dual={})       # the bead window's default split
        if settings.data == "beads":
            paths = [Path(settings.path).parent if settings.beads.whole_folder
                     and Path(settings.path).is_file() else settings.path]
            data = prepare.read_beads(paths, settings.camera, profile, dual,
                                      settings.beads.z_step_nm, ctx.report)
            dz, sources = data.dz_nm, data.sources
        else:
            b = settings.blinking
            data = prepare.read_movie(settings.path, settings.camera, profile, dual,
                                      b.first_frame, b.last_frame, ctx.report)
            dz, sources = b.z_step_nm, [data.source]
        ctx.report(f"{data.images.shape} photons, "
                   f"{'two channels' if dual else 'one channel'}")
        lp, b = settings.learning, settings.blinking
        params = uipsf_parameters(
            profile, data=settings.data, pixelsize_xy_um=prepare.pixel_size_xy_um(data.camera),
            dz_nm=dz, roi_size_px=lp.roi_size_px,
            emission_wavelength_nm=lp.emission_wavelength_nm, peak_height=lp.peak_height,
            max_beads=settings.beads.max_beads,
            bead_diameter_nm=settings.beads.bead_diameter_nm, iterations=lp.iterations,
            zernike_order=lp.zernike_order, depth_um=b.depth_um, z_range_um=b.z_range_um,
            min_photon=b.min_photon, z_bins=b.z_bins, per_bin=b.per_bin, repeat=b.repeat,
            start_from=_uipsf_file(b.start_from) if settings.data == "blinking" else "")
        shift = (prepare.channel_shift(data.images)
                 if dual and settings.data == "beads" else None)
        out = output_path(settings)
        job = (out.with_name(out.stem + "_uipsf_job") if lp.keep_job
               else Path(tempfile.mkdtemp(prefix="smappy-uipsf-")))
        result = learn(data.images, data=settings.data, model=settings.model, dual=dual,
                       params=params, workdir=job, python=lp.python, progress=ctx.report,
                       channel_shift=shift)
        em_on = getattr(data.camera, "em_on", None)
        calibration = (convert.dual_calibration(result, data.split, sources, em_on=em_on)
                       if dual else convert.single_calibration(result, em_on=em_on))
        _note_sources(calibration, settings, sources)
        convert.save(out, calibration)
        raw = out.with_name(out.stem + "_uipsf.h5")
        shutil.copyfile(result, raw)
        if not lp.keep_job:
            shutil.rmtree(job, ignore_errors=True)
        from ..uipsf import plots
        learnt = convert.read(raw)
        figures = [Plot(draw=draw, name=name, panels=panels, size=size)
                   for name, (draw, panels, size) in plots.figures(learnt).items()]
        return Result(text=summary(calibration, out, raw, learnt), settings=settings,
                      plot=figures[0], plots={f.name: f for f in figures[1:]},
                      data={"calibration": str(out), "uipsf_result": str(raw),
                            "dual": dual})


def _uipsf_file(path: str) -> str:
    """A smappy uiPSF calibration names the uiPSF file beside it; uiPSF wants that."""
    if not path:
        return ""
    p = Path(path)
    beside = p.with_name(p.stem + "_uipsf.h5")
    if not p.name.endswith("_uipsf.h5") and beside.exists():
        return str(beside)
    return str(p)


def _note_sources(calibration, settings, sources) -> None:
    models = ([calibration.main, calibration.secondary] if hasattr(calibration, "main")
              else [calibration])
    for model in models:
        model.parameters["data"] = settings.data
        model.parameters["microscope"] = settings.microscope
        model.parameters["sources"] = [s.get("path") for s in sources]


def _models(calibration):
    if hasattr(calibration, "main"):
        return [("main", calibration.main), ("secondary", calibration.secondary)]
    return [("", calibration)]


def summary(calibration, out: Path, raw: Path, learnt=None) -> str:
    """What the run made: the file, the model's extent, the main aberrations,
    and -- from uiPSF's result -- how well the beads and the channels fit."""
    lines = [f"saved {out}", f"uiPSF's own result: {raw}"]
    for name, model in _models(calibration):
        nz = model.psf.shape[0]
        lines.append(f"{name + ': ' if name else ''}{nz} planes, {model.dz:g} nm apart, "
                     f"z from {model.z_index_to_nm(nz - 1):.0f} to "
                     f"{model.z_index_to_nm(0):.0f} nm")
        terms = [row for row in model.parameters.get("zernike", []) if row[0] > 4]
        terms = sorted(terms, key=lambda row: -abs(row[3]))[:4]
        if terms:
            lines.append("  largest aberrations: " + ", ".join(
                f"{label or 'Z' + str(j)} {phase:+.0f} nm" for j, label, _, phase in terms))
    if hasattr(calibration, "transformation"):
        lines.append("channel transformation (secondary -> main, chip px): "
                     + np.array2string(calibration.transformation / calibration.transformation[2, 2],
                                       precision=4, suppress_small=True).replace("\n", ""))
    if learnt is not None:
        from ..uipsf import plots
        if plots.has_localization_bias(learnt):
            bias = [np.nanmedian(np.abs(b)) for b in plots.localization_bias(learnt)]
            lines.append("beads refitted with the model, median |bias|: x {:.1f}, "
                         "y {:.1f}, z {:.1f} nm".format(*bias))
        if len(plots.channels(learnt)) > 1 and "T" in learnt["res"]:
            _, residual = plots.transformation_residuals(learnt)
            p = learnt["params"]["pixel_size"]
            rms = np.sqrt(np.mean(residual ** 2, axis=0)) * 1000 * np.array(
                [float(p["y"]), float(p["x"])])
            lines.append(f"transformation residual, rms: x {rms[1]:.1f}, y {rms[0]:.1f} nm")
    return "\n".join(lines)
