"""The prefit: a short Spline 3D fit of the data, for the training's ranges.

SMAP read its estimates off whatever fit was loaded, grouped as the layer grouped it
(``DECODE_training_estimates``).  Here the prefit is a fit of its own, so the estimates do not
depend on how someone last set a layer.  Three choices in it change the numbers:

* **Blocks spread over the acquisition** (`blocks`): 10 of 50 consecutive frames by default, from
  its start to its end.  An acquisition is usually densest at its start, so its first frames
  alone would describe the training worst; consecutive frames within a block are what grouping
  into blinks needs.  Each block is fitted and grouped on its own, its frame numbers its own, so
  no blink is joined across a gap.
* **The grouping box is 150 nm**, not SMAPpy's 50.  A blink's first and last frames catch only
  part of it and are dim, and their positions scatter by more than 50 nm, so the 50 nm box cut
  them off: on a sparse simulation of 1.5 and 3 frames' on-time it gave blinks of 2.22 and 3.19
  frames for the true 2.48 and 3.95, and 150 nm 2.44 and 3.81.  A box that wide also joins
  neighbours on dense data; those blinks fit worse, and the brightness leaves them out
  (`decode_smlm.prefit.fully_on`).
* **The frames are read by the caller** (``read(start, stop)``, counts, ``(n, rows, columns)``),
  so the area the density is measured on is known and nothing here opens a file.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GROUP_DX_NM = 150.0
BLOCKS, BLOCK_FRAMES = 10, 50


@dataclass
class Prefit:
    localizations: object        # smappy Localizations, one row a frame's spot, with group_id and n_in_group
    blinks: object               # the same grouped across frames
    area_px: float               # the area fitted, in camera pixels
    n_frames: int                # the frames fitted
    max_iterations: int          # the fitter's limit: a fit that reached it did not converge

    def columns(self) -> tuple[dict, dict]:
        """The blinks' and the localizations' columns the estimates read, as arrays."""
        b = {k: np.asarray(self.blinks[k]) for k in ("photons", "n_in_group", "frame") if k in self.blinks}
        if "z_nm" in self.blinks:
            b["z_nm"] = np.asarray(self.blinks["z_nm"])
        names = ("photons", "background", "group_id", "frame", "n_in_group", "logl_rel", "z_nm", "iterations")
        return b, {k: np.asarray(self.localizations[k]) for k in names if k in self.localizations}


def blocks(n_frames: int, n_blocks: int = BLOCKS, length: int = BLOCK_FRAMES) -> list[tuple[int, int]]:
    """``(start, stop)`` of ``n_blocks`` blocks of ``length`` frames spread from the first frame to
    the last; the whole acquisition as one block when it is no longer than they are together."""
    if n_frames <= n_blocks * length:
        return [(0, int(n_frames))]
    starts = np.linspace(0, n_frames - length, n_blocks).round().astype(int)
    return [(int(s), int(s) + length) for s in starts]


def prefit(read, n_frames: int, camera, calibration, *, n_blocks: int = BLOCKS, block_frames: int = BLOCK_FRAMES,
           group_dx_nm: float = GROUP_DX_NM, iterations: int = 50, **fit_options) -> Prefit:
    """Spline 3D (or Gaussian 2D without a calibration) on blocks of frames, each grouped.

    ``read(start, stop)`` gives frames ``start`` to ``stop - 1`` as counts; an array of frames is
    taken too.  ``camera`` is what `smappy.api.fit` takes.
    """
    from ..api import fit
    from ..group import GroupSettings, group
    from ..locs import Localizations
    from ..pipeline import FitSettings

    if not callable(read):
        stack = np.asarray(read)
        n_frames = len(stack)
        read = lambda a, b: stack[a:b]
    settings = FitSettings(iterations=iterations, **{k: fit_options.pop(k) for k in list(fit_options)
                                                     if k in FitSettings.__dataclass_fields__})
    locs, grouped, shape = Localizations(), Localizations(), None
    for start, stop in blocks(n_frames, n_blocks, block_frames):
        frames = np.asarray(read(start, stop))
        if frames.ndim != 3:
            raise ValueError(f"the prefit's frames are (n, rows, columns), not {frames.shape}")
        shape = frames.shape[1:]
        block = fit(frames, camera=camera, calibration=calibration, raw_frames=0, settings=settings, **fit_options)
        if len(block) == 0:
            continue
        block.columns["frame"] = np.asarray(block["frame"]) + start
        blinks, _ = group(block, GroupSettings(dx=group_dx_nm))
        if len(locs):
            # group ids stay apart across blocks
            offset = int(np.asarray(locs["group_id"]).max())
            block.columns["group_id"] = np.asarray(block["group_id"]) + offset
        locs.extend(block)
        grouped.extend(blinks)
    if len(locs) == 0:
        raise ValueError("the prefit found no molecules: check the camera and the frames")
    fitted = sum(b - a for a, b in blocks(n_frames, n_blocks, block_frames))
    return Prefit(localizations=locs, blinks=grouped, area_px=float(shape[0] * shape[1]), n_frames=int(fitted),
                  max_iterations=int(iterations))
