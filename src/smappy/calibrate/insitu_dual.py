"""Blinking molecules on a chip split in two, for in situ learning of a two-channel model (Spline 3D 2C).

The single-channel steps (`insitu`: detection, linking, the overlap rules, the cut) run on each half of
the movie on its own, and the units -- a frame of a molecule, or a blink's frames summed -- are paired
through the registration (the start calibration's map, `dual.ChannelTransform`), as the z-sweep molecules
are (`molecule_dual`): a main unit and a secondary unit of the same frames within `PAIR_PX` of where the
map puts it are one molecule; a unit seen in one half only keeps the region the map puts in the other,
since a dye of a ratiometric pair may send too little light there to be detected, or the other half's
detection failed.  Each channel's region is cut about its own place on the chip, in the camera's
orientation, so that the model's transform (secondary → main, `uipsf.model.channels`) applies exactly, as
in a dual bead model.

What the rounds choose from and judge by (`insitu.select`, the z proxy, the shape) is the main channel's:
its Gaussian is fitted on every molecule's main region, whichever half found it.  A molecule much dimmer
there is judged on a noisy shape and is more often out -- the learning then leans on the molecules the main
channel sees well, as a bead calibration leans on its beads, and both channels are still learnt from all
that are used.

A partner's region not detected has its own half's neighbours masked: every unit of that half on in the
same frames, out to ``overlap_k`` of its width, as `insitu.neighbour_mask` masks one half's.

The frames of a molecule can pair in some frames and not in others; those not paired are units of their
own half, numbered as its blinks: so held-out blinks are whole within each half, not across them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .insitu import (
    InSituROIs,
    InSituSettings,
    Units,
    choose,
    concatenate,
    gaussian,
    insitu_units,
)

#: a secondary unit within this of where the map puts a main one (px), in the same frames, is the same molecule
PAIR_PX = 2.0


@dataclass
class DualInSituROIs(InSituROIs):
    """In situ's container for a split chip: the main channel's units (what selection reads), and both
    channels' regions and masks [U, 2, R, R] about their own centres on the chip [U, 2, 2], the registration."""
    channel_regions: np.ndarray = None
    channel_masks: np.ndarray = None
    roi_xy: np.ndarray = None
    transform: object = None

    @property
    def n_channels(self) -> int:
        return 2

    def batch(self, state: str = "used", ids=None):
        """(`uipsf.Batch`, its noise) of the units in ``state`` (or ``ids``): both channels' regions,
        [E, 2, 1, R, R] photons, masks, the read noise of the K frames each holds, each channel's chip
        position (`Batch.roi_xy`) and the main channel's as the field's."""
        import torch

        from uipsf import Batch, ShiftedPoisson
        ids = self.ids(state) if ids is None else np.asarray(ids, int)
        u = self.units
        K = np.array([len(u.frames[i]) for i in ids], float)
        s = torch.tensor(K * self.read_noise_var, dtype=torch.float32).reshape(-1, 1, 1, 1, 1)
        batch = Batch(data=torch.tensor(self.channel_regions[ids][:, :, None], dtype=torch.float32),
                      z_nm=torch.zeros(1), s=s, mask=torch.tensor(self.channel_masks[ids][:, :, None]),
                      field_xy=torch.tensor(self.roi_xy[ids][:, 0], dtype=torch.float32),
                      roi_xy=torch.tensor(self.roi_xy[ids], dtype=torch.float32))
        return batch, ShiftedPoisson(float(self.excess_noise))

    def masked_pixels(self, ids) -> np.ndarray:
        """[len(ids)] the pixels each unit's fit uses, over both channels (`insitu_run.heldout_deviance`)."""
        return np.asarray(self.channel_masks[np.asarray(ids, int)], bool).reshape(len(ids), -1).sum(1)


def halves(transform, shape, origin=(0, 0)) -> list:
    """The main and the secondary channel as image rectangles (x, y, width, height), cut to the image of
    ``shape`` (Y, X): the registration's channels on the chip less the camera ROI's ``origin``."""
    from .molecule_dual import halves as chip_halves
    Y, X = shape
    out = []
    for x, y, w, h in chip_halves(transform, origin):
        x0, y0 = max(int(x), 0), max(int(y), 0)
        out.append((x0, y0, min(int(x + w), X) - x0, min(int(y + h), Y) - y0))
    return out


def _shifted(u: Units, x0: int, y0: int) -> Units:
    """``u``, found in a half whose corner is (x0, y0), in the image's pixels."""
    u.x_px = u.x_px + x0
    u.y_px = u.y_px + y0
    u.centre = u.centre + np.array([x0, y0], int)
    return u


def _inside(c, rect, h) -> bool:
    x, y, w, hh = rect
    return x + h <= c[0] < x + w - h and y + h <= c[1] < y + hh - h


def _mask_from(c, frames_of, u: Units, skip, settings: InSituSettings) -> np.ndarray:
    """[R, R] True = used: the region about ``c`` less every pixel near a unit of ``u`` (one half's) on in
    ``frames_of``, but ``skip`` (its partner, or -1)."""
    h = settings.roi_px // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    mask = np.ones(yy.shape, bool)
    sigma = np.nan_to_num(np.fmax(u.gauss["sigma_x_pix"], u.gauss["sigma_y_pix"]), nan=1.5)
    for k in range(len(u.reason)):
        if k == skip or not np.intersect1d(u.frames[k], frames_of).size:
            continue
        if abs(u.x_px[k] - c[0]) > h + 3 * settings.overlap_k * sigma[k] or \
                abs(u.y_px[k] - c[1]) > h + 3 * settings.overlap_k * sigma[k]:
            continue
        mask &= np.hypot(xx - (u.x_px[k] - c[0]), yy - (u.y_px[k] - c[1])) >= settings.overlap_k * sigma[k]
    return mask


def dual_units(frames: np.ndarray, transform, origin, settings: InSituSettings, read_noise_var: float = 0.0):
    """One block of frames [T, Y, X] (photons, the whole chip) to paired units (module docstring): (the main
    channel's `Units`, both channels' regions [U, 2, R, R] and masks, their centres [U, 2, 2] in image pixels,
    the detections in both halves)."""
    s = settings
    rects = halves(transform, frames.shape[1:], origin)
    found, n = [], 0
    for x, y, w, h in rects:
        u, k = insitu_units(frames[:, y:y + h, x:x + w], s, read_noise_var)
        found.append(_shifted(u, x, y))
        n += k
    main, second = found
    o = np.asarray(origin, float)
    roi, h = s.roi_px, s.roi_px // 2

    def other_xy(u, c):
        xy = np.c_[u.x_px, u.y_px] + o
        if not len(xy):
            return np.zeros((0, 2))
        return (transform.to_secondary(xy) if c == 0 else transform.to_reference(xy)) - o

    on_second = other_xy(main, 0)
    by_frame: dict = {}
    for j, f in enumerate(second.frames):
        by_frame.setdefault(tuple(int(v) for v in f), []).append(j)
    partner = np.full(len(main.reason), -1)
    taken = set()
    for i, f in enumerate(main.frames):
        js = [j for j in by_frame.get(tuple(int(v) for v in f), []) if j not in taken]
        if not js:
            continue
        d = np.hypot(second.x_px[js] - on_second[i, 0], second.y_px[js] - on_second[i, 1])
        if d.min() < PAIR_PX:
            partner[i] = js[int(np.argmin(d))]
            taken.add(partner[i])
    on_main = other_xy(second, 1)
    blink_offset = int(main.blink.max()) + 1 if len(main.blink) else 0
    rows = []          # (frames, blink, (x, y) per channel, centre per channel, region, mask per channel, reason)

    def cut_at(xy, fs, rect):
        c = (int(np.rint(xy[0])), int(np.rint(xy[1])))
        if not _inside(c, rect, h):
            return c, None
        return c, frames[np.asarray(fs, int), c[1] - h:c[1] + h + 1, c[0] - h:c[0] + h + 1].sum(0)

    for i in range(len(main.reason)):
        fs = main.frames[i]
        j = partner[i]
        if j >= 0:
            c2, r2, m2, why2 = tuple(second.centre[j]), second.regions[j], second.masks[j], second.reason[j]
        else:
            c2, r2 = cut_at(on_second[i], fs, rects[1])
            m2 = None if r2 is None else _mask_from(c2, fs, second, -1, s)
            why2 = None
        reason = main.reason[i] or why2 or (None if r2 is not None else "partner outside")
        if r2 is None:
            r2, m2 = np.zeros((roi, roi), np.float32), np.zeros((roi, roi), bool)
        rows.append((fs, main.blink[i], (main.x_px[i], main.y_px[i]), (tuple(main.centre[i]), c2),
                     (main.regions[i], r2), (main.masks[i], m2), reason))
    for j in np.setdiff1d(np.arange(len(second.reason)), partner[partner >= 0]):
        fs = second.frames[j]
        c1, r1 = cut_at(on_main[j], fs, rects[0])
        m1 = None if r1 is None else _mask_from(c1, fs, main, -1, s)
        reason = second.reason[j] or (None if r1 is not None else "partner outside")
        if r1 is None:
            r1, m1 = np.zeros((roi, roi), np.float32), np.zeros((roi, roi), bool)
        rows.append((fs, second.blink[j] + blink_offset, tuple(on_main[j]), (c1, tuple(second.centre[j])),
                     (r1, second.regions[j]), (m1, second.masks[j]), reason))
    U = len(rows)
    regions = np.asarray([r[4] for r in rows], np.float32).reshape(U, 2, roi, roi)
    masks = np.asarray([r[5] for r in rows], bool).reshape(U, 2, roi, roi)
    centres = np.asarray([r[3] for r in rows], int).reshape(U, 2, 2)
    g = gaussian(regions[:, 0]) if U else gaussian(np.zeros((0, roi, roi), np.float32))
    reasons = [r[6] if r[6] is not None or np.isfinite(g["photons"][k]) else "no fit" for k, r in enumerate(rows)]
    units = Units([np.asarray(r[0]) for r in rows], np.asarray([r[1] for r in rows], int),
                  np.asarray([r[2][0] for r in rows], float), np.asarray([r[2][1] for r in rows], float),
                  centres[:, 0], regions[:, 0], masks[:, 0], g, reasons)
    return units, regions, masks, centres, n


def dual_movie_rois(paths, transform, settings: InSituSettings | None = None, camera=None,
                    progress=None) -> DualInSituROIs:
    """The paired molecules of the split movies at ``paths`` (module docstring), read as `insitu_run.movie_rois`
    reads one channel's: blocks from all movies in turn, each coarse to fine, until the main channel's
    selection has enough.  ``transform`` the registration (`dual.ChannelTransform`, a dual calibration's)."""
    from ..camera import to_photons
    from ..io.tiff import open_stack
    from .input import discover_acquisitions
    from .insitu_run import MIN_BLOCKS, block_order, enough
    from .pupil import camera_for, pixel_size_nm
    s = settings or InSituSettings()
    paths = [str(p) for p in discover_acquisitions(paths)] if paths else []
    if not paths:
        raise ValueError("no movies")
    camera = camera if camera is not None else camera_for(paths[0])
    s = s.resolved(pixel_size_nm(camera))
    read_noise = float(camera.readout_variance)
    origin = tuple(int(v) for v in (getattr(camera, "roi", None) or (0, 0, 0, 0))[:2])
    plan = [(m, path, open_stack(path)) for m, path in enumerate(paths)]
    orders = [block_order(src.n_frames, s.block_frames) for _, _, src in plan]
    blocks = [(m, orders[m][j]) for j in range(max(map(len, orders))) for m in range(len(plan))
              if j < len(orders[m])]
    first = sum(min(MIN_BLOCKS, len(o)) for o in orders)
    parts, extra, candidates, read, passing = [], [], 0, {m: [] for m in range(len(plan))}, 0
    for k, (m, start) in enumerate(blocks):
        if progress is not None:
            progress({"message": f"finding molecules in both channels: block {k + 1}, {passing} of {s.molecules} "
                                 f"so far", "fraction": min(0.99, max(k / len(blocks), passing / s.molecules))})
        src = plan[m][2]
        stop = min(start + s.block_frames, src.n_frames)
        frames = np.concatenate([b for _, b in src.frames(chunk=s.block_frames, start=start, stop=stop)])
        u, regions, masks, centres, n = dual_units(to_photons(frames, camera), transform, origin, s, read_noise)
        parts.append((u, start, m))
        extra.append((regions, masks, centres))
        read[m].append(start)
        candidates += n
        if k + 1 >= first:
            done, passing = enough(concatenate(parts), s)
            if done:
                break
    sources = [{"path": path, "frames": src.n_frames, "read": sorted(read[m]), "block": s.block_frames}
               for m, path, src in plan]
    rois = choose(concatenate(parts), candidates, pixel_size_nm(camera), s, read_noise, float(camera.excess_noise),
                  sources, camera)
    cat = np.concatenate
    rois = DualInSituROIs(**vars(rois), channel_regions=cat([e[0] for e in extra]),
                          channel_masks=cat([e[1] for e in extra]),
                          roi_xy=cat([e[2] for e in extra]).astype(float) + np.asarray(origin, float),
                          transform=transform)
    rois.messages.append(f"{len(parts)} of {len(blocks)} blocks of {s.block_frames} frames read, both channels")
    return rois


# ------------------------------------------------------------------ learning from them

#: what in situ holds of the start model's two pupils for each *learn* (`depth.LEARN_CHOICES`): every aberration
#: of both, γ and the phase above the order, or nothing
DUAL_HOLDS = {"blur": "all", "phase": "start", "apodisation": "nothing"}


def _optics(model, c: int, n_med: float):
    """Channel ``c``'s optics of the start model at the sample's index ``n_med``: the objective's NA, which a
    minimal pupil's grid reaches beyond (`MinimalPupil`)."""
    from dataclasses import replace
    psf = model.channels[c].psf
    o = psf.grid.optics
    return replace(o, NA=float(getattr(psf.pupil, "na", o.NA)), n_med=float(n_med))


def run_insitu_dual(start, paths, camera, learn: str = "blur", n_med=None, focus_nm=None,
                    detection: InSituSettings | None = None, rounds: int = 2, registration: bool = False,
                    share_per_molecule: bool = False, rois: DualInSituROIs | None = None, transform=None,
                    progress=None, cancel=None, device=None, hold_focus: bool = False):
    """Two-channel in situ: the split movies at ``paths`` learnt from, starting from ``start`` (the two-channel
    `LearnedPSF` a dual pupil calibration carries, `depth.model_of(main, channels=2)`), as `insitu_run.run_insitu`
    learns one channel (`InSituRunResult`).

    ``learn`` says what of the start's two pupils is held (`DUAL_HOLDS`): with ``blur`` both whole -- only the
    focus, each channel's blur and the molecules are learnt, and the channels' offset in z (``ch1.dz``) stays
    the beads'; with ``phase`` each channel's γ and phase above the model's order, the offset learnt;
    with ``apodisation`` nothing, the focus then pinned by the coverslip as in one channel.  ``hold_focus``
    holds the focus at ``focus_nm``, typed, whatever is learnt.  ``registration``
    learns the map between the channels (``ch1.transform``) too, else it is the start's.  ``share_per_molecule``
    gives each molecule its own share of the light in the secondary channel (two dyes, the fit's photons not
    linked), else one share is learnt for all.  The molecules are paired by ``transform`` (default the start
    calibration's registration, required then)."""
    from dataclasses import replace

    import torch

    from uipsf.model.build import Held, held_from
    from uipsf.model.channels import view
    from uipsf.optics.zernike import noll_to_nm, zernike_modes
    from uipsf.workflows import insitu_model, start_values
    from uipsf.workflows.insitu_rounds import (
        RoundSettings,
        calibrate_insitu,
        focus_from_data,
    )

    from .depth import learn_key, resolved_medium
    from .insitu_run import InSituRunResult
    from .pupil import BeadSettings
    learn = learn_key(learn)
    if hold_focus and focus_nm is None:
        raise ValueError("a focus held needs the focus: type it")
    if learn not in DUAL_HOLDS:
        raise ValueError(f"learn: one of {', '.join(DUAL_HOLDS)}, not {learn!r}")
    if start.model.n_channels != 2:
        raise ValueError("two-channel in situ starts from a model of two channels: a dual pupil calibration")
    detection = detection or InSituSettings()
    tell = progress or (lambda entry: None)
    if rois is None:
        if transform is None:
            raise ValueError("the molecules are paired by a registration: give the dual calibration's")
        rois = dual_movie_rois(paths, transform, detection, camera, progress)
    if not len(rois.ids("used")):
        raise ValueError("no molecules chosen in the two channels: review the detection's settings")
    detection = detection.resolved(rois.pixel_size_nm)
    n = resolved_medium(start, n_med)
    shared = start.shared() if callable(getattr(start, "shared", None)) else start.shared
    spec = BeadSettings.from_options(start.options).psf
    hold = DUAL_HOLDS[learn]
    held = [Held(), Held()]
    messages = []
    if hold != "nothing":
        every = hold == "all"
        held = []
        for c in range(2):
            psf = start.model.channels[c].psf
            held.append(held_from(psf.describe(view(shared, c)), spec.order, psf.grid.optics.wavelength_nm,
                                  every=every, source="the start model"))
        if every:
            # nothing of the phase left to learn: the order is the held pupils' own, a mode up to it they lack
            # held at 0 (`insitu_run.run_insitu`)
            top = max([noll_to_nm(j)[0] for h in held for j in h.phase] + [2])
            for h in held:
                h.phase = {j: h.phase.get(j, 0.0) for j in sorted(set(zernike_modes(top)) | set(h.phase))}
            spec = replace(spec, zernike_order=top)
        spec = replace(spec, magnitude=held[0].kind)
    else:
        spec = replace(spec, magnitude="minimal")
    focus_fit = "typed" if hold_focus else "learnt" if hold != "nothing" else "coverslip"
    roi = int(rois.units.regions.shape[-1])
    H = np.asarray(rois.transform.transformation, float)
    H = H / H[2, 2]
    dz = float(torch.as_tensor(shared["ch1.dz"])) if "ch1.dz" in shared else None
    share = float(torch.as_tensor(shared["ch1.share"])) if "ch1.share" in shared else None
    options = [{"fit_transform": bool(registration), "fit_dz": hold != "all", "dz_nm": dz, "share": share,
                "share_per_emitter": bool(share_per_molecule)}]
    guess = 1000.0 if focus_nm is None else float(focus_nm)

    def make(focus):
        return insitu_model([_optics(start.model, c, n) for c in range(2)], spec, (roi, roi), focus, device=device,
                            held=held, fit_focus=focus_fit == "learnt", transforms=[None, H],
                            channel_options=options)
    model = make(guess)
    first = start_values(model, start)
    _, noise = rois.batch(ids=rois.ids("used")[:1])
    result_focus = None
    if focus_nm is None:
        tell({"message": "estimating the focus from the molecules on the coverslip"})
        result_focus, _ = focus_from_data(model, rois, first, noise, guess)
        model = make(result_focus)
    first = dict(first, **{"geometry.focus": torch.tensor(model.geometry.focus_nm)})
    learned = calibrate_insitu(model, rois, first, noise,
                               RoundSettings(max_rounds=rounds, coverslip=focus_fit != "typed",
                                             twin_test=hold != "all"),
                               selection=detection, progress=progress, cancel=cancel)
    result = InSituRunResult(rois, learned, learned.model or model, start, result_focus)
    # what its model file keeps, as a dual bead model's (`molecule_stack_run`): the split it was learnt on and the
    # choices, so that a fitted file or the kept copy can start in situ again
    result.cal.geometry = dict(rois.transform.geometry)
    result.cal.options = BeadSettings.from_options(start.options)
    what = {"all": "every aberration of both channels", "start": "each channel's γ and its phase above order "
            f"{spec.order}", "nothing": "nothing of the pupils"}[hold]
    messages.insert(0, f"held from the start model: {what}" + ("" if registration else "; the registration"))
    if focus_fit == "coverslip":
        messages.append("the focus is pinned to the coverslip: with γ learnt it cannot be learnt too")
    result.messages += messages
    if learned.final.twin is not None:
        result.messages += learned.final.warnings
    return result
