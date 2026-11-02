"""Learning from single molecules on the coverslip through repeated z sweeps, in rounds (`docs/uipsf/molecule-stacks.md`).

Each detection is its own emitter, a molecule seen in one frame, with its z known: the stage's, less its
sweep's focus and the focal plane at its place (`CoverslipStackGeometry`).  So this is a bead calibration
in what it learns -- the pupil from emitters at known defocus -- and in situ's only in what it reads:
thousands of single frames, chosen by rules.  It learns as beads learn (`learn_beads`: stage 1 to order 4,
the beads' rejection on its scores, stage 2 with every mode), and needs none of in situ's rounds
(`insitu_rounds`), which exist to find each molecule's unknown height: starts from the widths, the
coverslip's edge, a measured focus-per-height ratio.  Nor a twin test: the stage's z fixes the sign of the
even aberrations, as it does for beads.

**The z check** (`z_check`), after each round: every molecule learnt from is localised again on the
learnt model with its z free (`CoverslipStackGeometry.checking`).  Its offset from where the plane puts it,
against its CRLB, says what the model is worth for 3D: the median offset per 100 nm of defocus is the
**z bend** (what a bead stack measures the same way, here with no bead and no stage-z kernel), and the
offsets' robust spread about it, in CRLBs, the **z scatter** (1.75 on M5's AF647, November 2026).  A
molecule farther off than max(3 CRLB, `Z_REJECT_NM`) is not on the glass -- free dye, or stuck above it
-- or not one molecule, and the next round learns without it.

**Rounds**: the first learns from the selection; each next from what the last one used and its z check
passed, from the last model (its pupil replaces stage 1: every molecule localised on it, judged, then every
mode free).  On M5 the second round moved γ by 0.01, the Zernikes by under a nanometre, and took a sixth of
the first's time; it stops when the pupil moved by less than ``stop_sd`` standard errors (`_moved`) or after
``max_rounds``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..noise import ShiftedPoisson
from .beads import Calibration, learn_beads
from .insitu_rounds import _moved
from .rejection import RejectionSettings
from .scoring import localise

#: the z check's floor (nm): a molecule this far off the plane is not on the glass
Z_REJECT_NM = 100.0
#: the z bend's bins (nm of defocus), and how many molecules a bin needs
BEND_STEP_NM, BEND_MIN = 100.0, 10
#: memory one molecule of a 25-pixel region takes in a chunk on the device: at chunk 256 the learning held
#: about 31 GB on an M-series GPU (M5's AF647, 2 November 2026), into swap; 64 was safe.  It scales with
#: the region's pixels.  The solver's own cap (`solve.problem.HOST_BYTES_PER_VALUE`, 16 kB a value) is a
#: tenth of this model's, whose vector PSF at a fine pupil grid costs ~190 kB per pixel of a region.
BYTES_PER_MOLECULE_25 = 0.12e9
#: the share of the available memory a chunk may take, and its bounds
CHUNK_SHARE, CHUNK_MIN, CHUNK_MAX = 0.5, 8, 256


def molecule_chunk(roi_px: int, available_bytes: int | None = None) -> int:
    """Molecules per chunk: `CHUNK_SHARE` of the memory available at `BYTES_PER_MOLECULE_25` (scaled by the
    region's area), within (`CHUNK_MIN`, `CHUNK_MAX`); 64 when the system does not say."""
    if available_bytes is None:
        from ..device import available_memory_bytes
        available_bytes = available_memory_bytes()
    if available_bytes is None:
        return 64
    per = BYTES_PER_MOLECULE_25 * (roi_px / 25) ** 2
    return int(np.clip(CHUNK_SHARE * available_bytes // per, CHUNK_MIN, CHUNK_MAX))


@dataclass
class StackRoundSettings:
    max_rounds: int = 3
    stop_sd: float = 1.0               # stop when the pupil moved by less than this many standard errors
    z_reject_nm: float = Z_REJECT_NM
    first_order: int = 4
    # the beads' rejection, but a neighbour's masked pixels are masked on purpose (capped when chosen), and a
    # molecule's region is cut about a whole pixel of its detection
    rejection: RejectionSettings = field(default_factory=lambda: RejectionSettings(max_masked=1.0,
                                                                                    max_xy_offset_px=2.0))


@dataclass
class ZCheck:
    dz_nm: np.ndarray                  # [n] each molecule's z offset from the plane (the check's molecules)
    crlb_nm: np.ndarray                # [n] its CRLB
    defocus_nm: np.ndarray             # [n] its defocus, the stage's
    bad: np.ndarray                    # [n] off the plane (module docstring)
    grid_nm: np.ndarray                # the bend's bins
    bend_nm: np.ndarray                # the median offset per bin, NaN with too few
    scatter: float                     # the robust spread about the bend, in CRLBs
    ids: np.ndarray                    # [n] the molecules, indices of the round's batch

    def bend_rms(self, within_nm: float) -> float:
        """The bend's rms about its mean over the bins within ``within_nm`` of focus."""
        v = self.bend_nm[(np.abs(self.grid_nm) <= within_nm) & np.isfinite(self.bend_nm)]
        return float(np.sqrt(np.mean((v - v.mean()) ** 2))) if len(v) else float("nan")

    @property
    def passed(self) -> np.ndarray:
        return self.ids[~self.bad]


def z_check(cal: Calibration, batch, defocus_nm, noise, chunk_size=None, z_reject_nm: float = Z_REJECT_NM,
            span_nm: float = 1000.0) -> ZCheck:
    """The used molecules of ``cal`` localised again with z free (module docstring).  ``batch`` is the
    round's, ``defocus_nm`` [E] each molecule's defocus by the stage (for the bend's bins)."""
    from ..model import ForwardModel
    model = cal.model
    checking = ForwardModel(model.channels, geometry=model.geometry.checking(), intensity=model.intensity,
                            background=model.background)
    used = np.asarray(cal.beads.used)
    sub = batch.subset(torch.tensor(used))
    params = cal.result.params
    start = {n: v.detach() for n, v in params.physical().items() if params.specs[n].scope == "emitter"}
    fit, _ = localise(checking, cal.shared(), sub, noise, start=start, chunk_size=chunk_size)
    dz = fit.params.physical()["emitter.xyz"][:, 2].detach().cpu().double().numpy()
    crlb = fit.std["emitter.xyz"][:, 2].detach().cpu().double().numpy()
    d = np.asarray(defocus_nm, float)[used]
    bad = ~np.isfinite(dz) | ~np.isfinite(crlb) | (np.abs(dz) > np.maximum(3 * crlb, z_reject_nm))
    grid = np.arange(-span_nm, span_nm + 1, BEND_STEP_NM)
    bend = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        m = ~bad & (np.abs(d - g) <= BEND_STEP_NM / 2)
        if m.sum() >= BEND_MIN:
            bend[i] = float(np.median(dz[m]))
    ok = ~bad & (crlb > 0)
    resid = (dz - np.interp(d, grid, np.nan_to_num(bend))) / np.where(crlb > 0, crlb, np.nan)
    scatter = float(1.4826 * np.median(np.abs(resid[ok]))) if ok.any() else float("nan")
    return ZCheck(dz, crlb, d, bad, grid, bend, scatter, used)


@dataclass
class StackRound:
    index: int
    ids: np.ndarray                    # the molecules this round learnt from (indices of the whole batch)
    calibration: Calibration
    check: ZCheck | None
    moved: float = float("inf")


@dataclass
class StackRun:
    rounds: list = field(default_factory=list)

    @property
    def final(self) -> StackRound:
        return self.rounds[-1]


def learn_molecule_stack(model, batch, defocus_nm, noise=None, settings: StackRoundSettings | None = None,
                         start: dict | None = None, chunk_size: int | None = None, progress=None,
                         cancel=None) -> StackRun:
    """Learn from the molecules of ``batch`` (a `CoverslipStackGeometry` model's: each one's region, its
    nominal z, sweep and chip position) in rounds (module docstring).  ``defocus_nm`` [E] each molecule's
    defocus by the stage, for the z check's bins; ``start`` shared values to start stage 1 from (a previous
    model's, `start_values`).  ``progress`` gets the learning's entries, tagged with ``round``, and a
    ``{"message", "round"}`` before each step; ``cancel`` stops after the current iteration."""
    s = settings or StackRoundSettings()
    noise = noise if noise is not None else ShiftedPoisson()
    chunk = chunk_size or molecule_chunk(int(batch.data.shape[-1]))
    defocus = np.asarray(defocus_nm, float)
    tell = progress or (lambda entry: None)
    run = StackRun()
    ids = np.arange(batch.n_emitters)
    previous = None
    for index in range(1, s.max_rounds + 1):
        tell({"message": f"round {index}: learning from {len(ids)} molecules", "round": index})
        sub = batch.subset(torch.tensor(ids))

        def tagged(entry, index=index):
            tell(dict(entry, round=index))
        kwargs = dict(rejection=s.rejection, first_order=s.first_order, chunk_size=chunk, max_dz_nm=None,
                      progress=tagged, cancel=cancel)
        if previous is None:
            cal = learn_beads(model, sub, noise, start=start, start_stage1=start is not None, **kwargs)
        else:
            cal = learn_beads(model, sub, noise, start=previous.calibration.shared(), **kwargs)
        if cal.result.stop_reason == "cancelled":
            run.rounds.append(StackRound(index, ids, cal, None))
            return run
        tell({"message": f"round {index}: the z check", "round": index})
        check = z_check(cal, sub, defocus[ids], noise, chunk, s.z_reject_nm)
        moved = (float("inf") if previous is None else
                 _moved(cal.shared(), previous.calibration.shared(), cal.result.std, previous.calibration.result.std))
        this = StackRound(index, ids, cal, check, moved)
        run.rounds.append(this)
        if moved < s.stop_sd:
            break
        ids, previous = ids[check.passed], this
    return run
