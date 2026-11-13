"""Bead calibration: the calibration window, as a section of the Localize tab, and a run without it.

By hand it is a launcher (`Plugin.window`): the calibration is made by looking
at the beads -- a table to exclude them from, a stack to browse, pairs to pick
on the transformation plot -- and that is a window, which holds every setting,
so the panel shows the launcher alone (`form_in_window`).  Being a plugin is
what lets it be pinned to a tab, or taken out of one, like the fitters beside
it.

Without a window -- the assistant's `run_plugin`, a chain, a batch, a script
-- `run` makes what the window's *Detect + calibrate* makes, from these
settings and through the same functions (`calibrate.core`, `calibrate.dual`):
the beads marked doubles left out in single channel, as the window starts
them; then, on request, *Calculate fit quality*; then the calibration saved
where the window's *Save calibration...* offers, or at *save as*.  Until
2026-11 it refused, and an assistant could open the window and do nothing in
it.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

from ..calibrate.dual import DualColorSettings
from . import Context, FormState, Plot, Plugin, Result, param, register

PAGES = ("Overview", "Transformation", "Fit quality", "Field diagnostics", "Bead diagnostics", "CRLB")


@dataclass
class BeadCalibrationSettings:
    """The window's settings, for a run without it: the window asks single or dual colour itself,
    and its form is the one the GUI shows (`form_in_window`)."""
    stacks: str = param("", label="bead stacks", kind="open_file",
                        help="a bead z-stack, or a folder of them: every acquisition found in it is used")
    subdirectories: bool = param(True, label="search subdirectories",
                                 help="a folder's subfolders searched too")
    mode: str = param("Single channel", label="mode", choices=("Single channel", "Dual colour"),
                      help="one channel, or a split camera's two registered onto each other")
    beads: DualColorSettings = param(default_factory=DualColorSettings, label="settings")
    fit_quality: bool = param(False, label="fit quality",
                              help="fit the beads back with the fitter the data will meet, for the "
                                   "Fit quality page; takes as long as the calibration again")
    save_as: str = param("", label="save as", kind="save_file",
                         help="where the calibration is written; empty: beside the stacks, named as "
                              "the window's Save offers")


def bead_paths(stacks: str, subdirectories: bool = True):
    """The acquisitions in ``stacks``, as the window's *Add...* finds them (`psf_calibration.stack_paths`)."""
    from .psf_calibration import stack_paths
    return stack_paths(stacks, subdirectories)


def single(settings: DualColorSettings):
    """The single-channel settings within the dual ones: `DualColorSettings` is `CalibrationSettings`
    with the channels and the registration, and the window keeps one form of each."""
    from ..calibrate.core import CalibrationSettings
    return CalibrationSettings(**{f.name: getattr(settings, f.name) for f in fields(CalibrationSettings)})


def calibrate_beads(paths, settings: DualColorSettings, dual: bool, progress=None):
    """What the window's *Detect + calibrate* runs: the beads collected and averaged into a spline,
    in single channel the beads marked doubles excluded (the window's table lets them in again)."""
    from ..calibrate.core import build_calibration, collect_beads, doubles
    from ..calibrate.dual import build_dual_calibration, collect_dual_beads
    if dual:
        return build_dual_calibration(collect_dual_beads(paths, settings, progress), progress=progress)
    beads = collect_beads(paths, single(settings), progress)
    return build_calibration(beads, excluded=doubles(beads.records), progress=progress)


def fit_quality(result, dual: bool):
    """*Calculate fit quality*: the beads fitted back, and the profiles the Fit quality page draws."""
    from ..calibrate.validation import (aligned_midline_profiles, fit_bead_diagnostics,
                                        fit_paired_bead_diagnostics, paired_profiles)
    if dual:
        fitted = fit_paired_bead_diagnostics(result)
        result.registration.refits = {"global": fitted}
        return fitted, paired_profiles(result)
    return fit_bead_diagnostics(result), aligned_midline_profiles(result)


def save_path(paths, dual: bool, save_as: str = "") -> str:
    """Where the calibration goes: ``save_as``, or beside the stacks as the window's dialog offers --
    one ``.h5``, whatever the name was typed with (`core.calibration_save_path`)."""
    from ..calibrate.core import calibration_save_defaults, calibration_save_path
    if save_as:
        path = str(Path(save_as).expanduser())
    else:
        folder, name = calibration_save_defaults(paths, dual=dual)
        path = str(Path(folder) / (name + ".h5"))
    return path if path.lower().endswith(".hdf5") else str(calibration_save_path(path))


def pages(result, dual: bool, quality=None) -> dict:
    """The window's pages as the result's figures, nothing selected (`calibrate.plots`)."""
    from ..calibrate import plots
    excluded = {i for i, reason in enumerate(result.reasons) if reason == "manual exclusion"}
    out = {"Overview": (plots.draw_pairs if dual else plots.draw_overview, (result, [], excluded))}
    if dual:
        out["Transformation"] = (plots.draw_transformation, (result, [], excluded))
        out["Field diagnostics"] = (plots.draw_field, (result, [], excluded))
    if quality is not None:
        diagnostics, profiles = quality
        out["Fit quality"] = ((plots.draw_paired_fit_quality, (result, diagnostics, profiles, []))
                              if dual else (plots.draw_fit_quality, (diagnostics, profiles, [])))
    out["Bead diagnostics"] = (lambda figure, *a: plots.draw_beads(figure, *a, dual=dual),
                               (result, [], excluded))
    out["CRLB"] = (lambda figure, *a: plots.draw_crlb(figure, *a, dual=dual), (result,))
    return {name: Plot(lambda figure, d=draw, a=args: d(figure, *a), panels=2, size=(9.0, 6.0),
                       summary=False)
            for name, (draw, args) in sorted(out.items(), key=lambda kv: PAGES.index(kv[0]))}


@register("Localize/Bead calibration")
class BeadCalibration(Plugin):
    """An experimental spline PSF from bead z-stacks, in its own window."""
    description = ("build a spline PSF calibration for Spline 3D or Spline 3D 2C "
                   "from bead z-stacks; opens its own window")
    changes_locs = False
    Settings = BeadCalibrationSettings
    # 2: no *mode*: the window's own combo is the one place it is chosen
    # 3: faint specks no longer cost a bead its isolation (`core.select_beads`): more beads are kept
    # 4: a neighbour the detection's smoothing merged into a bead costs it its isolation, and a bead
    #    shaped unlike the others is marked a double and left out (`core.merged_neighbours`, `judge_shapes`)
    # 5: a run without the window, from settings (the mode among them again, for that run only)
    version = "5"
    window = "bead calibration"
    form_in_window = True

    @classmethod
    def specs(cls):
        from ..calibrate.dual import bead_specs
        from . import param_specs
        specs = param_specs(cls.Settings)
        specs["beads"].children = bead_specs(DualColorSettings)
        return specs

    def describe(self, ctx, settings: BeadCalibrationSettings) -> FormState:
        dual = settings.mode == "Dual colour"
        return FormState(shown={"beads.channels": dual, "beads.registration": dual})

    def run(self, ctx: Context, settings: BeadCalibrationSettings) -> Result:
        dual = settings.mode == "Dual colour"
        paths = bead_paths(settings.stacks, settings.subdirectories)
        ctx.report(f"detecting beads in {len(paths)} stack(s)...")
        result = calibrate_beads(paths, settings.beads, dual, ctx.report)
        quality = None
        if settings.fit_quality:
            ctx.report("fitting the beads back...")
            quality = fit_quality(result, dual)
        path = save_path(paths, dual, settings.save_as)
        result.save(path, overwrite=True)
        import numpy as np
        used, n = int(np.sum(result.accepted)), len(result.beads.records)
        fitter = "Spline 3D 2C" if dual else "Spline 3D"
        text = (f"{settings.mode}, {len(paths)} stack(s): {used} of {n} beads used\n"
                f"saved {path}: for the {fitter} fitter's calibration")
        return Result(text=text, settings=settings, plots=pages(result, dual, quality),
                      data={"files": paths, "mode": settings.mode, "beads": n, "used": used,
                            "saved": path, "fitter": f"Localize/{fitter}"})
