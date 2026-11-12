"""PSF calibration: the bead average, a pupil model learnt by uipsf, or that model in situ.

Two ways in.  By hand, the button opens the window (`Plugin.window`), because
a calibration is made by looking at it -- the beads, the pupil, the Zernikes,
the residuals -- and marking beads; the window holds every setting, so the
panel shows the launcher alone (`form_in_window`).  Without a window -- the
assistant's `run_plugin`, a chain, a batch, a script -- `run` makes the same
calibration from these settings: the window's *Detect beads* and *Calibrate*
with nothing marked by hand, through the same code (`calibrate.psf_run`, which
the window calls too), and the method's pages as the result's figures.
Until 2026-11 `run` only refused, and an assistant could open the window and
do nothing in it.

*PSF calibration* rather than a mode of the bead calibration because in situ
learning learns from the blinking molecules of a measurement, not from beads
(`docs/uipsf/plan.md` §7.5).
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import List

from ..calibrate.dual import DualColorSettings
from ..calibrate.elliptical import EllipticalCalibrationSettings
from ..calibrate.fourpi_beads import FourPiBeadSettings
from ..calibrate.insitu import InSituSettings
from ..calibrate.insitu_run import InSituModelSettings
from ..calibrate.molecule_stack import MoleculeStackSettings
from ..calibrate.pupil import BeadSettings, OpticsSettings, RunSettings, SplineSettings
from . import Context, FormState, Plot, Plugin, Result, param, register

#: the methods by key, as `calibrate.psf_methods.METHODS` has them; the sample follows from the method
METHOD_KEYS = ("pupil", "average", "elliptical", "molecules", "molecules elliptical", "insitu")
MODES = ("Single channel", "Dual colour", "4Pi")


@dataclass
class PSFCalibrationSettings:
    """The window's settings, for a run without it.  Each part is the window's section of that name,
    read only where the method reads it (`calibrate.psf_run.inputs`)."""
    stacks: str = param("", label="stacks", kind="open_file",
                        help="a bead z-stack, z sweep or movie, or a folder of them: every acquisition "
                             "found in it is used")
    subdirectories: bool = param(True, label="search subdirectories",
                                 help="a folder's subfolders searched too")
    method: str = param("pupil", label="method", choices=METHOD_KEYS,
                        help="pupil: uiPSF's model from beads; average: the bead average; elliptical: "
                             "astigmatic widths; molecules: from molecules on the coverslip; insitu: "
                             "from a blinking measurement")
    mode: str = param("Single channel", label="mode", choices=MODES,
                      help="one channel, a split camera's two (dual colour), or 4Pi's four")
    save_as: str = param("", label="save as", kind="save_file",
                         help="where the calibration is written; empty: beside the stacks, as the "
                              "window saves it, if save model is on")
    run: RunSettings = param(default_factory=RunSettings, label="run")
    microscope: OpticsSettings = param(default_factory=OpticsSettings, label="microscope", collapsed=True)
    model: BeadSettings = param(default_factory=BeadSettings, label="model", collapsed=True)
    spline: SplineSettings = param(default_factory=SplineSettings, label="spline", collapsed=True)
    beads: DualColorSettings = param(default_factory=DualColorSettings, label="beads", collapsed=True)
    in_situ: InSituModelSettings = param(default_factory=InSituModelSettings, label="in situ",
                                         collapsed=True)
    in_situ_detection: InSituSettings = param(default_factory=InSituSettings,
                                              label="in situ detection", collapsed=True)
    molecules: MoleculeStackSettings = param(default_factory=MoleculeStackSettings, label="molecules",
                                             collapsed=True)
    elliptical: EllipticalCalibrationSettings = param(default_factory=EllipticalCalibrationSettings,
                                                      label="elliptical Gaussian", collapsed=True)
    fourpi: FourPiBeadSettings = param(default_factory=FourPiBeadSettings, label="4Pi", collapsed=True)


#: `psf_run.inputs`' names for the parts, against the settings' fields
PARTS = {"run": "run", "microscope": "microscope", "model": "model", "spline": "spline", "beads": "beads",
         "in situ": "in_situ", "in situ detection": "in_situ_detection", "molecules": "molecules",
         "elliptical": "elliptical", "4Pi": "fourpi"}


def stack_paths(stacks: str, subdirectories: bool = True) -> List[str]:
    """The acquisitions in ``stacks`` -- a file or a folder -- as the window's *Add...* finds them; the
    path itself when nothing is found in it, for the detection to say what is wrong with it."""
    from ..calibrate.input import discover_acquisitions
    if not stacks:
        raise ValueError("give the stacks: a file, or a folder of them")
    path = Path(stacks).expanduser()
    if not path.exists():
        raise ValueError(f"no file or folder {path}")
    return [str(p) for p in discover_acquisitions([str(path)], recursive=subdirectories)] or [str(path)]


def part_of(settings, dual: bool):
    """``part(name)`` for `psf_run.inputs`: the settings' part, the beads' as the single-channel form
    has them outside dual colour (`DualColorSettings` is `CalibrationSettings` with the channels and the
    registration; the window keeps one form of each)."""
    from ..calibrate.core import CalibrationSettings

    def part(name: str):
        value = getattr(settings, PARTS[name])
        if name == "beads" and not dual:
            return CalibrationSettings(**{f.name: getattr(value, f.name) for f in fields(CalibrationSettings)})
        return value
    return part


def page_plots(method, run) -> dict:
    """The method's pages as the result's figures, each drawn as the window first shows it: no bead
    selected, the marks the run left, each page's first choice.  A page that is a table goes to the
    result's data instead (`page_tables`)."""
    marks = method.initial_marks(run) if hasattr(method, "initial_marks") else {}
    excluded = {b for b, wish in (marks or {}).items() if wish == "exclude"}
    out = {}
    for page in method.pages(run):
        if page.table is not None or page.draw is None:
            continue
        chosen = {label: (choices if isinstance(choices, bool) else list(choices)[0])
                  for label, choices in (page.controls or {}).items()}

        def draw(figure, page=page, chosen=chosen):
            if page.controls:
                page.draw(figure, [], excluded, chosen)
            else:
                page.draw(figure, [], excluded)
        out[page.title] = Plot(draw, panels=2, size=(9.0, 6.0), summary=False)
    return out


def page_tables(method, run) -> dict:
    out = {}
    for page in method.pages(run):
        if page.table is None:
            continue
        try:
            made = page.table()
        except Exception as error:                  # a table is a view: never the run's failure
            out[page.title] = f"not made: {error}"
            continue
        columns, rows = made[0], made[1]
        out[page.title] = {"columns": list(columns), "rows": [list(r) for r in rows]}
    return out


@register("Localize/PSF calibration")
class PSFCalibration(Plugin):
    """A PSF calibration: the bead average, or a pupil model learnt by uipsf from beads or in situ."""
    description = ("make the PSF calibration Spline 3D and Spline 3D 2C read: from bead z-stacks, the bead "
                   "average or a physical model of the pupil and its aberrations learnt with uiPSF v2; or that "
                   "model learnt in situ from the blinking molecules of a measurement; opens its own window")
    changes_locs = False
    Settings = PSFCalibrationSettings
    # 2: the bead detection's: merged neighbours cost isolation, doubles by shape left out (`core.judge_shapes`)
    # 3: a run without the window, from settings: detect, calibrate, finish, saved and listed
    version = "3"
    window = "psf calibration"
    form_in_window = True

    @classmethod
    def specs(cls):
        from ..calibrate.dual import bead_specs
        from . import param_specs
        specs = param_specs(cls.Settings)
        specs["beads"].children = bead_specs(DualColorSettings)
        return specs

    def describe(self, ctx, settings: PSFCalibrationSettings) -> FormState:
        """Only the parts the method reads: the window's sections, shown as it shows them."""
        from ..calibrate.psf_run import method_for
        try:
            method, dual = method_for(None, settings.method, settings.mode)
        except ValueError as error:
            return FormState(notice=str(error))
        reads = method.sections
        insitu, fourpi = reads.get("in situ", False), reads.get("4Pi", False)
        molecules = reads.get("molecules", False)
        return FormState(shown={
            "model": reads["model"], "spline": reads["spline"], "in_situ": insitu,
            "in_situ_detection": insitu, "molecules": molecules, "elliptical": reads.get("elliptical", False),
            "fourpi": fourpi, "beads": not insitu and not fourpi and not molecules,
            "run.start_from": reads["start from"], "beads.channels": dual, "beads.registration": dual,
            "molecules.channels": dual, "molecules.registration": dual})

    def run(self, ctx: Context, settings: PSFCalibrationSettings) -> Result:
        from ..calibrate import psf_run
        from ..calibrate.psf_methods import LearningProgress, sample_of
        method, dual = psf_run.method_for(None, settings.method, settings.mode)
        paths = stack_paths(settings.stacks, settings.subdirectories)
        run_settings = settings.run
        if settings.save_as:                       # saved where asked, once finished, not beside the stacks
            run_settings = replace(run_settings, save_model=False)
        inputs = psf_run.inputs(method, dual, paths, part_of(replace(settings, run=run_settings), dual))
        tracker = LearningProgress()

        def progress(entry):
            fraction, left, text = tracker.update(entry)
            ctx.report(text + ("" if fraction is None else f" ({100 * fraction:.0f} %)"))
        ctx.report(f"{method.label}, {settings.mode}: {len(paths)} file(s)")
        run = psf_run.calibrate(method, dual, paths, inputs, progress, ctx.stop)
        saved = run.saved
        if settings.save_as:
            saved = Path(method.save_as(run, settings.save_as))
        camera = inputs.get("camera")
        if saved is not None:
            psf_run.list_calibration(saved, method, dual, paths, camera)
        fitter = psf_run.fitter_of(method, dual)
        summary = method.summary(run)
        text = "\n".join([f"{method.label} ({sample_of(method) if method.key in METHOD_KEYS else '4Pi'}), "
                          f"{settings.mode}, {len(paths)} file(s): {run.extra.get('detected', '')}",
                          summary,
                          (f"saved {saved}: for the {fitter.split('/')[-1]}"
                           + (", listed for its auto" if getattr(method, "listed", True) else "")
                           if saved is not None else "not saved: save model is off and save as is empty")])
        warning = run.extra.get("warning")
        if warning:
            ctx.warn(warning)
        data = {"method": method.label, "mode": settings.mode, "files": paths, "summary": summary,
                "saved": str(saved) if saved is not None else None, "fitter": fitter}
        data.update({f"table: {k}": v for k, v in page_tables(method, run).items()})
        return Result(text=text, data=data, plots=page_plots(method, run), settings=settings)

