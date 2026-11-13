"""Molecules on the coverslip through z sweeps, on a chip split into two colours: the *Dual colour* mode.

The single-channel steps (`molecule_stack`) run on each half -- its own width curve, the filter matched to
each frame's defocus in that half -- and the molecules are paired through the registration (the halves'
map, `dual.ChannelTransform`), as a dual bead calibration pairs its beads (`pupil.dual_rois`): mutually
nearest within `PAIR_PX` once the secondary half's spot is put onto the main one; a spot seen in one half
only keeps the region the map puts in the other, a ratiometric dye sending little light to it.  Each
channel's region is cut about its own position on the chip, in the camera's orientation (the map carries a
mirror), so that what is learnt saves as a dual bead model does (`pupil.dual_calibration`), for Spline 3D 2C.

**The registration** is a file's -- a dual bead calibration or a registration measured from a movie
(`transform.load_transform`) -- or, with none given, measured here from the molecules' own detections
(`transform.register_channels`, the movie registration: thousands of spots at many defocus, which a bead
slide does not have), on the channels the *channels* part cuts the chip into -- the microscope's entry or
chosen by hand (`dual.chip_channels`): which channel a spot is in is the arrangement's, never guessed.  uipsf learns it further either way (``ch1.transform``).

**The selection** judges each channel that carries at least `CHANNEL_SHARE` of the pair's light by the
single-channel rules (centred, its widths those of its half's curve, no bright neighbour), the brightness
window on both channels' photons together, and draws as one channel does.  One focus per sweep is shared by
both channels; the secondary channel's defocus offset is the model's (``ch1.dz``).
"""
from __future__ import annotations

import numpy as np

from .insitu import Units, cut, gaussian
from .molecule_stack import (DRAW_BIN_NM, MoleculeStackROIs, MoleculeStackSettings, _step_um, detect, draw,
                             first_spots, illumination_radius, inside_circle, neighbours, read_file, width_curve)

#: a secondary spot within this of where the map puts a main one (px) is the same molecule
PAIR_PX = 2.0
#: a channel with less of the pair's light than this is not judged by its own shape and centre
CHANNEL_SHARE = 0.25


def halves(transform, origin=(0, 0)) -> list:
    """The main and the secondary channel as image rectangles (x, y, width, height): the registration's
    channels on the chip (`dual.geometry_layout`) less the file's camera-ROI ``origin``."""
    from .dual import geometry_layout
    ox, oy = origin
    return [(x - ox, y - oy, w, h) for x, y, w, h in geometry_layout(transform.geometry).regions]


def registration_from_molecules(paths, camera, settings: MoleculeStackSettings):
    """The channels' registration measured from every file's spots over the whole chip (module docstring):
    the chip split as the *channels* say -- the microscope's entry or chosen (`dual.chip_channels`) -- each spot in the channel whose
    rectangle it lies in, then the movie registration between them."""
    from .dual import chip_channels
    from .transform import register_channels
    x, y, frame, shape, base = [], [], [], None, 0
    for path in paths:
        f = read_file(path, camera, settings)
        shape = f.frames.shape[1:]
        s = first_spots(f.frames)
        h = 7                                            # `first_spots`' 15-pixel regions
        ok = np.isfinite(s["x_roi"])
        x.append(s["x"][ok] + s["x_roi"][ok] - h)
        y.append(s["y"][ok] + s["y_roi"][ok] - h)
        frame.append(s["frame"][ok] + base)
        base += len(f.frames)
    roi = getattr(camera, "roi", None)
    frame_roi = tuple(int(v) for v in roi) if roi is not None else (0, 0, shape[1], shape[0])
    try:
        told, chip = chip_channels(settings.channels, frame_roi, getattr(camera, "microscope", None))
    except ValueError as error:
        raise ValueError(f"{error} -- or, for molecule stacks, give a registration: a dual bead "
                         f"calibration or one measured from a movie") from None
    x = np.concatenate(x) + frame_roi[0]                 # chip pixels, as a two-channel fit's
    y = np.concatenate(y) + frame_roi[1]
    channel = chip.channel_at(x, y)
    keep = channel >= 0
    found = register_channels(x[keep], y[keep], np.concatenate(frame)[keep], channel[keep], chip, shape,
                              roi=roi, arrangement=told)
    return found.transform


def _pairs(main: dict, second: dict, transform, origin, frames: int):
    """(main xy [n, 2], secondary xy [n, 2], frame [n]) of every molecule in either half, image pixels: the
    mutually nearest pairs within `PAIR_PX`, and each lone spot with its partner where the map puts it.  The
    map is the chip's: image pixels plus the camera ROI's ``origin``."""
    from ..dualfit import _match
    o = np.asarray(origin, float)
    out_m, out_s, out_f = [], [], []
    for t in range(frames):
        a = np.c_[main["x"][main["frame"] == t], main["y"][main["frame"] == t]].astype(float)
        b = np.c_[second["x"][second["frame"] == t], second["y"][second["frame"] == t]].astype(float)
        b_on_main = transform.to_reference(b + o) - o if len(b) else np.empty((0, 2))
        i, j = _match(a, b_on_main, PAIR_PX)
        lone_a = np.setdiff1d(np.arange(len(a)), i)
        lone_b = np.setdiff1d(np.arange(len(b)), j)
        m = np.vstack([a[i], a[lone_a], b_on_main[lone_b]])
        s = np.vstack([b[j], (transform.to_secondary(a[lone_a] + o) - o) if len(lone_a) else np.empty((0, 2)),
                       b[lone_b]])
        out_m.append(m)
        out_s.append(s)
        out_f.append(np.full(len(m), t))
    return np.vstack(out_m), np.vstack(out_s), np.concatenate(out_f)


def dual_molecule_rois(paths, settings: MoleculeStackSettings | None = None, camera=None, transform=None,
                       progress=None) -> MoleculeStackROIs:
    """The molecules of the z-sweep files at ``paths`` on a chip split in two, paired (module docstring).
    ``transform`` the registration (`dual.ChannelTransform`), else ``settings.registration``'s file, else
    measured from the molecules."""
    from .input import discover_acquisitions
    from .pupil import camera_for, pixel_size_nm
    s = settings or MoleculeStackSettings()
    paths = [str(p) for p in discover_acquisitions(paths)] if paths else []
    if not paths:
        raise ValueError("no files")
    camera = camera if camera is not None else camera_for(paths[0])
    tell = progress or (lambda entry: None)
    if transform is None and s.registration:
        from .transform import load_transform
        transform = load_transform(s.registration)
    if transform is None:
        tell({"message": "the registration of the channels, from the molecules", "fraction": 0.0})
        transform = registration_from_molecules(paths, camera, s)
    origin = tuple(int(v) for v in (getattr(camera, "roi", None) or (0, 0, 0, 0))[:2])
    rects = halves(transform, origin)
    px = pixel_size_nm(camera)
    roi = s.region(px)
    # the width curve of each half: its own focus, its own widths
    spots, zs = [[], []], []
    for k, path in enumerate(paths):
        tell({"message": f"the width curves: file {k + 1} of {len(paths)}", "fraction": 0.1 + 0.3 * k / len(paths)})
        f = read_file(path, camera, s)
        zs.append(f.z_um)
        for c, (x, y, w, h) in enumerate(rects):
            spots[c].append(first_spots(f.frames[:, y:y + h, x:x + w]))
    radius = [s.illumination_px or illumination_radius((h, w)) for _, _, w, h in rects]
    curves = [width_curve(spots[c], zs, [inside_circle(p["x"], p["y"], (h, w), radius[c]) for p in spots[c]],
                          s.psf_shape) for c, (_, _, w, h) in enumerate(rects)]
    parts, sweep_base, candidates, sources = [], 0, 0, []
    for k, path in enumerate(paths):
        tell({"message": f"finding molecules: file {k + 1} of {len(paths)}", "fraction": 0.4 + 0.6 * k / len(paths)})
        f = read_file(path, camera, s)
        found = []
        for c, (x, y, w, h) in enumerate(rects):
            d = (f.z_um - curves[c].focus_um[k]) * 1000
            cands = detect(f.frames[:, y:y + h, x:x + w], d, curves[c], radius[c], s.significance)
            found.append({"frame": cands["frame"], "x": cands["x"] + x, "y": cands["y"] + y})
        main_xy, sec_xy, frame = _pairs(found[0], found[1], transform, origin, len(f.frames))
        candidates += len(frame)
        d_frames = [(f.z_um - curves[c].focus_um[k]) * 1000 for c in range(2)]
        m = _measure_pairs(f.frames, frame, main_xy, sec_xy, d_frames, curves, roi, rects)
        state, masks = _select_pairs(m, s, roi)
        m.update(state=state, masks=masks, sweep=sweep_base + f.sweep[m["frame"]], file=k)
        sweep_base += int(f.sweep.max()) + 1
        parts.append(m)
        sources.append({"path": path, "frames": len(f.frames), "sweeps": int(f.sweep.max()) + 1})
        del f
    x, y, w, h = rects[0]
    rois = _assemble_pairs(parts, roi, px, camera, candidates, sources, curves, radius[0], sweep_base, origin,
                           transform, (x + origin[0] + (w - 1) / 2, y + origin[1] + (h - 1) / 2))
    rois.state = draw(rois.state, rois.defocus_nm, s.molecules, s.seed)
    rois.step_nm = _step_um(zs) * 1000
    counts = ", ".join(f"{v} {k}" for k, v in sorted(rois.counts().items()))
    rois.messages.append(f"{candidates} molecules in either half, {sweep_base} sweeps of {len(paths)} files: {counts}")
    return rois


def _measure_pairs(frames, frame, main_xy, sec_xy, d_frames, curves, roi, rects) -> dict:
    """Both channels' regions about their own positions (whole pixels), those inside their halves, and each
    channel's Gaussian with the widths its curve expects at its own defocus (``d_frames``, per channel: each
    half's focus is its own).  The molecule's defocus is the main channel's."""
    pos = [np.rint(main_xy).astype(int), np.rint(sec_xy).astype(int)]
    h = roi // 2
    inside = np.ones(len(frame), bool)
    for c, (x, y, w, hh) in enumerate(rects):
        inside &= ((pos[c][:, 0] - h >= x) & (pos[c][:, 0] + h < x + w)
                   & (pos[c][:, 1] - h >= y) & (pos[c][:, 1] + h < y + hh))
    frame = frame[inside]
    pos = [p[inside] for p in pos]
    out = {"frame": frame, "d_nm": d_frames[0][frame], "channels": []}
    for c in range(2):
        regions, _ = cut(frames, frame, pos[c][:, 0], pos[c][:, 1], roi)      # all inside: checked above
        d = d_frames[c][frame]
        sx, sy = curves[c].at(d)
        out["channels"].append({"frame": frame, "x": pos[c][:, 0], "y": pos[c][:, 1], "regions": regions,
                                "gauss": gaussian(regions), "d_nm": d, "sx_curve": sx, "sy_curve": sy})
    return out


def _select_pairs(m: dict, settings: MoleculeStackSettings, roi: int):
    """Each pair's state and both channels' masks [n, 2, R, R] (module docstring)."""
    from .molecule_stack import MAX_OFF_CENTRE_PX, SHAPE_WIDE_NM, brightness_window
    n = len(m["frame"])
    h = roi // 2
    state = ["candidate"] * n

    def out(mask, reason):
        for i in np.flatnonzero(mask):
            if state[i] == "candidate":
                state[i] = "rejected: " + reason
    photons = np.stack([np.nan_to_num(ch["gauss"]["photons"], nan=0.0) for ch in m["channels"]], 1)
    total = photons.sum(1)
    masks = []
    for c, ch in enumerate(m["channels"]):
        g = ch["gauss"]
        judged = photons[:, c] >= CHANNEL_SHARE * np.maximum(total, 1e-9)
        fitted = np.isfinite(g["photons"]) & np.isfinite(g["sigma_x_pix"]) & np.isfinite(g["sigma_y_pix"])
        out(judged & ~fitted, "fit failed")
        # the limits widen with the spot out of focus, as one channel's (`molecule_stack`, 4)
        off_limit = np.maximum(MAX_OFF_CENTRE_PX, np.hypot(ch["sx_curve"], ch["sy_curve"]) / np.sqrt(2))
        tol = np.where(np.abs(ch["d_nm"]) <= SHAPE_WIDE_NM, settings.width_tol, 2 * settings.width_tol)
        with np.errstate(invalid="ignore"):
            out(judged & ~(np.hypot(g["x_roi"] - h, g["y_roi"] - h) < off_limit), "off centre")
            out(judged & ~((np.abs(np.log(g["sigma_x_pix"] / ch["sx_curve"])) < tol)
                           & (np.abs(np.log(g["sigma_y_pix"] / ch["sy_curve"])) < tol)), "shape")
        rejected, mask = neighbours(ch, roi, settings.neighbour_fraction)
        out(judged & (rejected | ((~mask).mean((1, 2)) > settings.max_masked)), "neighbour")
        masks.append(mask)
    alive = np.array([st == "candidate" for st in state])
    # one window for every z, the in-focus pairs' (`molecule_stack.brightness_window`), on both channels' sum
    lo, hi = brightness_window(total, m["d_nm"], alive)
    out(alive & ~((total >= lo) & (total <= hi)), "brightness")
    return ["not drawn" if st == "candidate" else st for st in state], np.stack(masks, 1)


def _assemble_pairs(parts, roi, px, camera, candidates, sources, curves, radius, sweeps, origin, transform,
                    centre) -> MoleculeStackROIs:
    cat = np.concatenate
    h = roi // 2
    main = [p["channels"][0] for p in parts]
    gauss = {k: cat([c["gauss"][k] for c in main]) for k in main[0]["gauss"]}
    x, y = cat([c["x"] for c in main]), cat([c["y"] for c in main])
    state = [st for p in parts for st in p["state"]]
    n = len(state)
    masks = cat([p["masks"] for p in parts])                          # [n, 2, R, R]
    regions = np.stack([cat([p["channels"][c]["regions"] for p in parts]) for c in range(2)], 1)
    chip = np.asarray(origin, float)
    roi_xy = np.stack([np.c_[cat([p["channels"][c]["x"] for p in parts]),
                             cat([p["channels"][c]["y"] for p in parts])] + chip for c in range(2)], 1)
    units = Units(frames=[[int(f)] for p in parts for f in p["frame"]], blink=np.arange(n),
                  x_px=x + np.nan_to_num(gauss["x_roi"] - h), y_px=y + np.nan_to_num(gauss["y_roi"] - h),
                  centre=np.c_[x, y].astype(int), regions=regions[:, 0], masks=masks[:, 0], gauss=gauss,
                  reason=[st.removeprefix("rejected: ") if st.startswith("rejected") else None for st in state],
                  source=cat([np.full(len(p["frame"]), p["file"], int) for p in parts]))
    d = cat([p["d_nm"] for p in parts])
    rois = MoleculeStackROIs(units, state, np.round(d / DRAW_BIN_NM).astype(int), np.array([]), candidates,
                             tuple(px), float(camera.readout_variance), float(camera.excess_noise),
                             sources=sources, camera=camera, z_nm=-d, defocus_nm=d,
                             sweep=cat([p["sweep"] for p in parts]).astype(int), sweeps=int(sweeps),
                             curve=curves[0], radius_px=float(radius), channel_regions=regions,
                             channel_masks=masks, roi_xy=roi_xy, transform=transform, curves=list(curves),
                             centre_px=tuple(float(v) for v in centre))
    return rois
