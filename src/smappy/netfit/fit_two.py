"""A two-channel model in the DECODE fitter: the camera's frames cut into the network's two channels,
its localizations put back on the frame (docs/decode/gui-plan.md, step 3).

`TwoChannelPredictor` wraps `predict.Predictor` for the `window_engine.WindowEngine`, which hands it
whole frames as the camera wrote them.  It cuts them by the registration (`channels.TwoChannels`, built
at the first frames, when their size is known: a camera ROI may cut into the channels), runs the
network on the two channels and their offset maps, and gives the engine SMAPpy's pixel columns on
the frame -- positions on the calibration's main half, as the two-colour spline fitters report them,
each channel's photons and background as ``photons_ch0`` / ``photons_ch1`` (what *Assign colours*
reads) and ``background_ch0`` / ``background_ch1``.

**The move since the beads.**  A splitter moves between the bead calibration and the measurement by
whole pixels and more (Fig. 2a: 5.8 rows, plex-plan §16, real data).  ``moved`` (columns, rows) is
added to where the registration puts channel 1's partner: a translation, which leaves the network's
offset maps within what it was trained on, so no new training.  `measure_move` finds it on the first
frames: the move at which the network finds the fewest molecules.  A registration that is off shows
each molecule as two spots that do not pair, and the network reports both: on Fig. 2a 546
localizations a frame at the move, 5.8 rows, against 720 -- 760 three pixels off either way, and the
minimum along the columns at 0 (2026-11-22).  The frames' deviance with every localization drawn
back does not find it: a wrong registration's extra molecules soak up light the model misses
elsewhere, and the deviance came out lower there.

**The checks**, each optional (plex-plan §16): ``goodness``, each localization's neighbourhood against
the fit drawn back (`goodness.fit_goodness`: false positives); ``bound``, the bound from the fit's own
numbers (`bounds.plugin_crlb`: position outliers, where the reported sigma is well above it).
"""
from __future__ import annotations

import numpy as np


def moved_calibration(cal, moved=(0.0, 0.0)):
    """``cal`` with channel 1's partner ``moved`` (columns, rows) further than the beads put it."""
    from dataclasses import replace
    mx, my = (float(v) for v in moved)
    if not (mx or my):
        return cal
    step = np.array([[1, 0, -mx], [0, 1, -my], [0, 0, 1.0]])
    return replace(cal, transformation=np.asarray(cal.transformation, float) @ step)


MOVE_FRAMES = 20          # frames the move is measured on
MOVE_REACH = 8.0          # pixels either way the coarse scan looks
MOVE_STEP = 0.25          # the fine scan's step, about the minimum


def count_at(predictor, cal, corner, photons, moved) -> float:
    """Localizations a frame of ``photons`` (with ``window`` frames of context each side) at ``moved``."""
    from .channels import TwoChannels
    two = TwoChannels.from_calibration(moved_calibration(cal, moved), corner, main=predictor.channels.main,
                                       shape=photons.shape[1:])
    r = predictor(two.cut(photons), 0, two.offsets())
    return len(r["frame"]) / max(len(photons) - 2 * predictor.window, 1)


def measure_move(predictor, cal, corner, photons, start=(0.0, 0.0), reach: float = MOVE_REACH,
                 step: float = MOVE_STEP) -> tuple:
    """(columns, rows) channel 1 moved since the beads, from frames ``photons`` (n + 2 window, H, W): the
    move with the fewest localizations, rows then columns then rows again, each a scan of whole pixels
    within ``reach`` of where it stands and then of ``step`` about the best."""
    moved = [float(start[0]), float(start[1])]
    for axis in (1, 0, 1):
        def at(v, axis=axis):
            m = list(moved)
            m[axis] = v
            return count_at(predictor, cal, corner, photons, m)
        centre = moved[axis]
        coarse = np.arange(np.round(centre - reach), np.round(centre + reach) + 1)
        counts = [at(v) for v in coarse]
        best = float(coarse[int(np.argmin(counts))])
        fine = best + np.arange(-1.0, 1.0 + step / 2, step)
        counts = [at(v) for v in fine]
        moved[axis] = float(fine[int(np.argmin(counts))])
        reach = 2.0                          # the second pass of rows only refines
    return tuple(moved)


def biplane_total(cal) -> float:
    """The light a biplane molecule of one drawn photon puts in both planes: the two PSFs as the network
    draws them (`channels.splines`), summed at focus.  Fig. 1c's: 0.987 + 0.767."""
    from ..io.calibration import light_at_focus
    from .channels import splines
    return float(sum(light_at_focus(s) for s in splines(cal, "biplane")))


class TwoChannelPredictor:
    """A two-channel `Predictor` as the engine drives a one-channel one (the module's docstring)."""

    def __init__(self, predictor, cal, corner=(0, 0), moved=(0.0, 0.0), *, em: bool = False,
                 goodness: bool = False, bound: bool = False, pixel_nm=(100.0, 100.0), measure: bool = False):
        self.predictor = predictor
        self.beads = cal                     # the registration as the beads measured it
        self.measure = bool(measure)         # the move measured on the first frames (`measure_move`)
        self.cal = moved_calibration(cal, moved)
        self.corner = tuple(round(v) for v in corner)
        self.moved = tuple(float(v) for v in moved)
        # what the run's record says of the move (`fit_decode.provenance`): one dict, updated where the move
        # is measured -- the record is made before the first frames and written after the last
        self.moved_record = {"rows": self.moved[1], "columns": self.moved[0], "measured": False}
        self.em, self.goodness, self.bound = bool(em), bool(goodness), bool(bound)
        self.pixel_nm = tuple(float(v) for v in pixel_nm)
        self.kind = predictor.channels.kind
        # a biplane's photons as the molecule's total over both planes (Jonas, 2026-11-22): the network
        # draws each plane's PSF with its count, the planes' shares in the PSFs (`biplane_total`)
        self.total = biplane_total(self.cal) if self.kind == "biplane" else 1.0
        self.two = None
        self.check_seconds = 0.0
        # the goodness draws each molecule over the network's own background, pixel by pixel: over a level
        # a frame, the light an uneven background puts under a spot counted against it (plex-plan §16)
        predictor.keep_background = self.goodness

    # what the engine and the run's record read, the predictor's
    window = property(lambda self: self.predictor.window)
    device = property(lambda self: self.predictor.device)
    raw_th = property(lambda self: self.predictor.raw_th)
    metadata = property(lambda self: self.predictor.metadata)
    frames_mirrored = False

    def channels(self, shape):
        """The two channels on frames of ``shape``, built once."""
        from .channels import TwoChannels
        if self.two is None:
            self.two = TwoChannels.from_calibration(self.cal, self.corner, main=self.predictor.channels.main,
                                                    shape=tuple(shape))
        return self.two

    def __call__(self, photons, first_frame: int = 0) -> dict:
        import time
        photons = np.asarray(photons, np.float32)
        if self.measure and self.two is None:
            k = self.predictor.window
            self.moved = measure_move(self.predictor, self.beads, self.corner, photons[:MOVE_FRAMES + 2 * k],
                                      self.moved)
            self.cal = moved_calibration(self.beads, self.moved)
            self.moved_record.update(rows=self.moved[1], columns=self.moved[0], measured=True)
        two = self.channels(photons.shape[1:])
        k = self.predictor.window
        stack = two.cut(photons)
        r = self.predictor(stack, 0, two.offsets())
        r0, _, c0, _ = two.box
        grid = np.c_[r["x_pix"] + c0, r["y_pix"] + r0]
        xy = two.on_frame(grid)
        frame = np.asarray(r["frame"], np.int64)
        out = {"frame": frame + int(first_frame), "x_pix": xy[:, 0], "y_pix": xy[:, 1], "z_nm": r["z_nm"],
               "prob": r["prob"], "x_err_pix": r["x_err_pix"], "y_err_pix": r["y_err_pix"], "z_err_nm": r["z_err_nm"],
               "photons": r["photons"] * self.total, "photons_err": r["photons_err"] * self.total}
        bg_grid = np.asarray(r["background_c"]) if "background_c" in r else \
            np.repeat(np.asarray(r["background"])[:, None], 2, 1)
        bg = two.in_order(bg_grid)
        out.update(background=bg[:, 0], background_ch0=bg[:, 0], background_ch1=bg[:, 1])
        if "photons_c" in r:                     # colour: each channel its own count
            pc = two.in_order(np.asarray(r["photons_c"]))
            out.update(photons_ch0=pc[:, 0], photons_ch1=pc[:, 1])
            if "photons_c_err" in r:
                pe = two.in_order(np.asarray(r["photons_c_err"]))
                out.update(photons_ch0_err=pe[:, 0], photons_ch1_err=pe[:, 1])
        if len(frame) and (self.goodness or self.bound):
            t0 = time.perf_counter()
            ph = np.asarray(r["photons_c"]) if self.kind == "colour" else np.asarray(r["photons"])
            if self.goodness:
                from .goodness import fit_goodness
                out["goodness"] = fit_goodness(two, self.cal, self.kind, stack[k:len(stack) - k], frame,
                                               np.c_[r["x_pix"], r["y_pix"]], r["z_nm"], ph, None, self.em,
                                               background_map=r.get("background_map"))
            if self.bound:
                from .bounds import plugin_crlb
                b = plugin_crlb(two, self.cal, self.kind, frame, grid, r["z_nm"], ph, bg_grid, self.pixel_nm,
                                self.em, window=2 * k + 1)
                out.update(crlb_x_nm=b[:, 0], crlb_y_nm=b[:, 1], crlb_z_nm=b[:, 2])
            self.check_seconds += time.perf_counter() - t0
        return {name: np.asarray(v) for name, v in out.items()}
