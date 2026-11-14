"""Z sweeps of molecules on the coverslip to a PSF calibration: the PSF calibration window's *molecules on the
coverslip in a z stack* sample with the pupil model, for a script too.

smappy reads, uipsf learns, as for beads (`pupil`) and in situ (`insitu_run`): the files are read and their
molecules found and chosen here (`molecule_stack`), uipsf learns the pupil in rounds with a z check after
each (`uipsf.workflows.molecule_stack`), and the result is saved as a bead model's -- the spline the fitter
reads, a PSF at the coverslip against defocus, listed for Spline 3D's *auto* and usable with its focus
model (`depth`), its model inside it.  What it adds to a bead model:

* **The molecules' height**, held at 5 nm (*height*, under *more*) or learnt on *auto*, is the sample's: the pupil saved is
  the microscope's own, and the spline is drawn for a molecule at that height, the one these data show.
* **The z check** -- the bend, the scatter against the CRLB, the molecules that left the plane -- and the
  foci per sweep with the focal plane's slope are kept with the calibration's record.

torch is imported inside the functions, as in `pupil`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .molecule_stack import MoleculeStackROIs, MoleculeStackSettings, molecule_rois

#: the spline's z range on *auto* reaches this share of the molecules' defocus on either side (the 2nd and
#: 98th percentiles of the used ones', the nearer): the last bins hold few molecules and bend away (M5:
#: +900 and +1000 nm)
SPLINE_QUANTILE = 0.02


@dataclass
class MoleculeStackResult:
    rois: MoleculeStackROIs
    run: object                          # uipsf `StackRun`
    model: object                        # the ForwardModel learnt
    start: object = None                 # the LearnedPSF it started from, or None
    calibration: object = None           # the SplineCalibration
    saved: Optional[Path] = None
    messages: list = field(default_factory=list)

    @property
    def cal(self):
        """The last round's uipsf `Calibration`."""
        return self.run.final.calibration

    @property
    def check(self):
        """The last round's z check (`uipsf.workflows.molecule_stack.ZCheck`)."""
        return self.run.final.check

    def used(self) -> np.ndarray:
        """The units the last round learnt from and kept, indices of ``rois``."""
        last = self.run.final
        return np.asarray(self.rois.ids("used"))[last.ids][np.asarray(self.cal.beads.used, int)]

    def height_nm(self) -> float:
        return float(self.cal.shared()["geometry.height"])


def model_for(rois: MoleculeStackROIs, optics, model, stack: MoleculeStackSettings, device=None, dtype=None):
    """The forward model learnt from ``rois``: ``optics`` an `OpticsSettings`, ``model`` a `BeadSettings`."""
    import torch

    from uipsf.workflows.calibrate import molecule_stack_model
    C = rois.n_channels
    o = optics.optics(rois.camera, C, _main_half(rois))
    roi = int(rois.units.regions.shape[-1])
    span = float(np.nanmax(np.abs(rois.defocus_nm))) if len(rois.defocus_nm) else 1000.0
    transforms = None if C == 1 else [None, _unit(rois.transform.transformation)]
    return molecule_stack_model(o if C > 1 else o[0], model.psf, (roi, roi), rois.sweeps, np.zeros(rois.sweeps),
                                rois.field_centre(), height_nm=stack.height_nm, transforms=transforms,
                                z_max_nm=span + 100.0, device=device, dtype=dtype or torch.float32)


def _main_half(rois) -> Optional[str]:
    """A split chip's main half (`pupil.main_half`'s): the registration's."""
    return None if rois.transform is None else (rois.transform.geometry or {}).get("main_channel")


def _unit(h) -> np.ndarray:
    h = np.asarray(h, float)
    return h / h[2, 2]


def run_molecule_stack(paths, optics, model, stack: MoleculeStackSettings | None = None, spline=None, run=None,
                       camera=None, rois: MoleculeStackROIs | None = None, save_to=None, progress=None,
                       cancel=None, device=None, chunk_size: int | None = None,
                       dual: bool = False) -> MoleculeStackResult:
    """Z-sweep files to a learnt model and, with *save model*, its spline on disk (module docstring).
    ``optics`` an `OpticsSettings`, ``model`` a `BeadSettings` (its PSF model and *rejection (k)*; the beads'
    size, drift, field dependence and held-out share are not read: `pupil.model_shown`); ``rois``, the molecules found already (*Detect*), skips finding them;
    ``device`` where it is learnt (default the fastest); ``chunk_size`` molecules at once (default from the
    memory available, `uipsf.workflows.molecule_stack.molecule_chunk`).  ``dual``: a chip split in two colours
    (`molecule_dual`), saved as a dual calibration for Spline 3D 2C."""
    from uipsf.workflows.calibrate import start_values
    from uipsf.workflows.molecule_stack import StackRoundSettings, learn_molecule_stack

    from .pupil import RunSettings, SplineSettings, default_save_path, load_start, save
    stack = stack or MoleculeStackSettings()
    spline = spline or SplineSettings()
    run = run or RunSettings()
    if rois is None and dual:
        from .molecule_dual import dual_molecule_rois
        rois = dual_molecule_rois(paths, stack, camera, progress=progress)
    elif rois is None:
        rois = molecule_rois(paths, stack, camera, progress)
    used = rois.ids("used")
    if not len(used):
        raise ValueError("no molecules chosen: review the detection's settings")
    learnt = model_for(rois, optics, model, stack, device)
    previous = load_start(run.start_from) if run.start_from else None
    start = start_values(learnt, previous) if previous is not None else None
    if start is not None:                  # a previous model's foci are another sample's sweeps
        start = {k: v for k, v in start.items() if not k.startswith("geometry.")}
    batch, noise = rois.batch()
    from uipsf.workflows.rejection import RejectionSettings
    rejection = RejectionSettings(k=model.rejection_k, max_masked=1.0, max_xy_offset_px=2.0)
    learned = learn_molecule_stack(learnt, batch, rois.defocus_nm[used], noise,
                                   StackRoundSettings(max_rounds=stack.max_rounds, rejection=rejection), start=start,
                                   chunk_size=chunk_size, progress=progress, cancel=cancel)
    result = MoleculeStackResult(rois, learned, learnt, previous)
    cal = result.cal
    cal.options = model                    # kept with the model file, as a bead model's
    if rois.transform is not None:         # the split, which the dual calibration and its model file keep
        cal.geometry = dict(rois.transform.geometry)
    check = result.check
    if check is not None:
        h = result.height_nm()
        held = "held" if stack.height_nm is not None else "learnt"
        result.messages.append(f"height {h:.1f} nm ({held}); z bend {check.bend_rms(600):.1f} nm rms over ±600 nm, "
                               f"scatter {check.scatter:.2f} × CRLB, {int(check.bad.sum())} off the plane")
        if stack.height_nm is None and h > 0.9 * 200.0:
            result.messages.append("warning: the height ran to its bound: hold it (more) at what the sample has")
    if run.save_model and cal.result.stop_reason != "cancelled":
        result.calibration = calibration_of(result, spline)
        path = Path(save_to) if save_to else default_save_path(paths, rois.n_channels == 2)
        result.saved = save(path, cal, result.calibration, overwrite=bool(save_to))
    return result


def spline_range(result: MoleculeStackResult, dz_nm: float) -> float:
    """± the spline reaches on *auto*: the nearer of the used molecules' 2nd and 98th percentiles of
    defocus (`SPLINE_QUANTILE`), whole steps."""
    d = result.rois.defocus_nm[result.used()]
    low, high = np.quantile(d, [SPLINE_QUANTILE, 1 - SPLINE_QUANTILE])
    return max(dz_nm, float(np.floor(min(-low, high) / dz_nm) * dz_nm))


def stack_record(result: MoleculeStackResult) -> dict:
    """What the calibration keeps of the sample: the height, the foci and the plane, the z check."""
    shared = result.cal.shared()
    out = {"sample": "molecules on the coverslip in a z stack",
           "height_nm": float(shared["geometry.height"]),
           "sweep_focus_nm": [round(float(v), 1) for v in shared["geometry.focus"].detach().cpu()],
           "focal_plane_nm_per_um": [round(float(v), 3) for v in shared["geometry.plane"].detach().cpu()],
           "file_focus_um": [float(v) for v in result.rois.curve.focus_um]}
    check = result.check
    if check is not None:
        out.update(z_bend_grid_nm=check.grid_nm.tolist(), z_bend_nm=[None if not np.isfinite(v) else round(float(v), 2)
                                                                     for v in check.bend_nm],
                   z_bend_rms_600_nm=check.bend_rms(600), z_scatter_over_crlb=check.scatter,
                   z_crlb_median_nm=float(np.median(check.crlb_nm)), off_plane=int(check.bad.sum()))
    return out


def calibration_of(result: MoleculeStackResult, spline=None):
    """The run's spline: the saved one, else sampled now as saving would -- a molecule at the learnt height,
    at the sweeps' step, over `spline_range`."""
    if result.calibration is None:
        from .pupil import dual_calibration, spline_calibration
        dz = (spline.dz_nm if spline is not None and spline.dz_nm else None) or result.rois.step_nm
        span = (spline.z_range_nm if spline is not None and spline.z_range_nm else None) or spline_range(result, dz)
        roi = int(result.rois.units.regions.shape[-1])
        if result.rois.n_channels == 2:
            from ..io.calibration import keep_microscope
            result.calibration = dual_calibration(result.cal, dz, span, roi, depth_nm=result.height_nm())
            for channel in (result.calibration.main, result.calibration.secondary):
                keep_microscope(channel, result.rois.camera)
                channel.parameters.update(stack_record(result))
        else:
            result.calibration = spline_calibration(result.cal, dz, span, roi, camera=result.rois.camera,
                                                    depth_nm=result.height_nm())
        result.calibration.parameters.update(stack_record(result))
    return result.calibration



#: a run costs about this many forward passes of one molecule per molecule and per parameter, each timed as
#: `estimate_seconds` times one, for the first round; each later round a fraction `LATER_ROUND` of it (M5's
#: AF647, 4000 molecules: the first round 16 min, the second 2.4)
PASSES_PER_PARAMETER = 2.0
LATER_ROUND = 0.15
#: molecules timed at once, and passes timed, the median kept
TIMED_MOLECULES, TIMED_PASSES = 16, 3


def estimate_seconds(rois: MoleculeStackROIs, optics, model, stack: MoleculeStackSettings | None = None) -> float:
    """About how long learning ``rois`` takes, seconds, in under a second: the model's forward pass on a few
    molecules, timed, times the molecules, the parameters and `PASSES_PER_PARAMETER`, and the later rounds'
    share (`LATER_ROUND`).  A figure to decide by, within a factor of two, as the beads' (`pupil.estimate_seconds`)."""
    import time

    import torch
    from uipsf.device import on_model_device
    from uipsf.params import Params
    stack = stack or MoleculeStackSettings()
    used = rois.ids("used")
    if not len(used):
        return 0.0
    learnt = model_for(rois, optics, model, stack)
    batch, _ = rois.batch(ids=used[:TIMED_MOLECULES])
    batch = on_model_device(batch, learnt)
    start = Params(learnt.specs(batch), batch.n_emitters, dtype=batch.data.dtype, device=batch.data.device)
    values = start.physical()
    times = []
    with torch.no_grad():
        for k in range(TIMED_PASSES + 1):            # the first pass warms the kernels up: not counted
            t = time.perf_counter()
            learnt(values, batch).sum().item()
            if k:
                times.append(time.perf_counter() - t)
    shared = sum(int(torch.as_tensor(v).numel()) for n, v in values.items() if start.specs[n].scope == "shared")
    per = sum(int(torch.as_tensor(v).numel()) for n, v in values.items() if start.specs[n].scope != "shared")
    per //= max(1, batch.n_emitters)
    one = float(np.median(times)) / batch.n_emitters
    return one * PASSES_PER_PARAMETER * len(used) * (shared + per) * (1 + LATER_ROUND * (stack.max_rounds - 1))
