"""Simulate a sparse blinking SMLM dataset, without and with a little drift.

Emitters sit on a 3D structure (a tilted ring, two crossing lines, a few
scattered points) spread over ~10 um, far apart compared with a PSF.  Each one
blinks a few times; a blink lasts one to a few frames and every frame gives one
localization with noise from its photon count (precision ~ 150 / sqrt(N) nm
laterally, three times that in z).  Two emitters active in the same frame closer
than ``min_separation`` (a PSF width) could not have been fitted apart, so both
are dropped -- the labelling is dense, the activation sparse, as in a real
experiment.  The drifted variant adds a smooth random walk plus a slow linear
creep of ~100 nm; the true drift per frame is in the metadata as ``drift_truth``.

In the package rather than in ``scripts/`` because it is also a plugin
(``File/Simulate/Blinking Structure``): something to try the whole pipeline on
without a microscope.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .locs import Localizations


def structure(rng) -> np.ndarray:
    """Emitter positions (n, 3) in nm."""
    n_ring = 1200
    t = np.linspace(0, 2 * np.pi, n_ring, endpoint=False) + rng.normal(0, 0.01, n_ring)
    ring = np.column_stack([5000 + 2500 * np.cos(t), 5000 + 2500 * np.sin(t),
                            300 * np.sin(2 * t)])                 # a tilted, wavy ring
    s = np.linspace(0, 1, 500)
    jitter = rng.normal(0, 15, (len(s), 3))          # the lines are not razor thin
    line1 = np.column_stack([1000 + 8000 * s, 1500 + 6000 * s, -200 + 400 * s]) + jitter
    line2 = np.column_stack([1500 + 7000 * s, 8500 - 7000 * s, 150 * np.ones_like(s)]) \
        + rng.normal(0, 15, (len(s), 3))
    scatter = np.column_stack([rng.uniform(500, 9500, 120), rng.uniform(500, 9500, 120),
                               rng.uniform(-300, 300, 120)])
    return np.vstack([ring, line1, line2, scatter])


def unresolvable(x, y, frame, min_separation: float) -> np.ndarray:
    """True where another emitter is active in the same frame within a PSF."""
    from scipy.spatial import cKDTree
    drop = np.zeros(len(frame), bool)
    order = np.argsort(frame, kind="stable")
    edges = np.flatnonzero(np.diff(frame[order]) != 0) + 1
    for block in np.split(order, edges):
        if block.size < 2:
            continue
        pairs = cKDTree(np.column_stack([x[block], y[block]])).query_pairs(
            min_separation, output_type="ndarray")
        if pairs.size:
            drop[block[np.unique(pairs)]] = True
    return drop


def simulate(n_frames: int, seed: int, drift: bool, density: float = 1.0,
             min_separation: float = 250.0):
    rng = np.random.default_rng(seed)
    emitters = structure(rng)
    rows = []
    for i, (x, y, z) in enumerate(emitters):
        for _ in range(rng.poisson(12 * density) + 1):      # blinks per emitter
            start = rng.integers(0, n_frames)
            length = rng.geometric(0.35)                     # 1, 2, 3 ... frames on
            for f in range(start, min(start + length, n_frames)):
                rows.append((i, f))
    rows = np.array(rows)
    emitter, frame = rows[:, 0], rows[:, 1]
    order = np.argsort(frame, kind="stable")
    emitter, frame = emitter[order], frame[order]
    n = len(frame)
    photons = rng.gamma(4.0, 500.0, n).astype(np.float32)          # mean 2000
    prec = (150.0 / np.sqrt(photons)).astype(np.float32)             # ~3-10 nm
    prec_z = (3 * prec).astype(np.float32)
    # two emitters within a PSF in one frame are one blob: neither is fitted
    lost = unresolvable(emitters[emitter, 0], emitters[emitter, 1], frame, min_separation)
    emitter, frame, n = emitter[~lost], frame[~lost], int((~lost).sum())
    photons, prec, prec_z = photons[~lost], prec[~lost], prec_z[~lost]
    true = emitters[emitter]
    xyz = true + np.column_stack([rng.normal(0, prec), rng.normal(0, prec), rng.normal(0, prec_z)])
    drift_nm = np.zeros((n_frames, 3))
    if drift:
        walk = np.cumsum(rng.normal(0, 0.6, (n_frames, 3)), axis=0)   # a random walk
        walk -= walk[0]
        creep = np.linspace(0, 1, n_frames)[:, None] * np.array([80.0, -50.0, 30.0])
        drift_nm = walk + creep
        xyz = xyz + drift_nm[frame]
    columns = {
        "frame": frame.astype(np.int64),
        "x_nm": xyz[:, 0].astype(np.float32), "y_nm": xyz[:, 1].astype(np.float32),
        "z_nm": xyz[:, 2].astype(np.float32),
        "photons": photons, "background": rng.gamma(20, 1.0, n).astype(np.float32),
        "loc_precision_nm": prec, "loc_precision_z_nm": prec_z,
        "sigma_nm": rng.normal(120, 8, n).astype(np.float32),
        "logl_rel": rng.normal(-0.5, 0.3, n).astype(np.float32),
        "emitter": emitter.astype(np.int32),                   # ground truth identity
    }
    metadata = {"units": "nm", "simulation": "smappy sparse blinks", "seed": seed,
                "n_emitters": len(emitters), "n_frames": n_frames,
                "min_separation_nm": min_separation, "n_unresolvable_dropped": int(lost.sum()),
                "drift_truth": drift_nm.round(3).tolist() if drift else None}
    return Localizations(columns, metadata)


def camera_frames(n_frames: int = 2000, seed: int = 0, pixelsize_nm: float = 100.0,
                  size_px: int = 100, photons: float = 2000.0,
                  background: float = 20.0, sigma_nm: float = 130.0,
                  conversion: float = 0.5, offset: float = 100.0,
                  read_noise: float = 1.5, on_per_frame: float = 7.0,
                  astigmatism: Optional[Tuple[float, float]] = None):
    """Raw camera frames of the same structure, blinking: ``(frames, truth)``.

    For fitting what `simulate` only pretends to have fitted -- the tutorial
    that goes from camera frames to a picture, and anyone who wants to try the
    Localize tab without a microscope.  ``frames`` is uint16 ADU, (n, y, x);
    ``truth`` is a `Localizations` of every emitter that was on in a frame, at
    its true position, with the photons it emitted there.

    The PSF is a Gaussian integrated over each pixel (the fitter's own model,
    so the fit is expected to find it); photons are Poisson over a flat
    background, then converted at ``conversion`` e-/ADU on an ``offset``, with
    Gaussian read noise in ADU.  With ``astigmatism`` -- ``(focal offset,
    depth)`` in nm, see `astigmatic_sigmas` -- the spot's widths in x and y
    follow the structure's z, as behind a cylindrical lens, and a spline fit
    with a calibration of the same PSF (`bead_stacks`) finds z; without it
    the frames are 2D and z is ignored.  ``on_per_frame`` is how many molecules shine
    in a frame on average, whatever the number of frames -- the density a
    fitter sees -- and the default of seven on 100 x 100 pixels is sparse,
    because the point is a picture that fits cleanly, not a test of the fitter
    under crowding.
    """
    from scipy.special import erf
    rng = np.random.default_rng(seed)
    emitters = structure(rng)
    mean_on = 1 / 0.35                      # frames per blink, geometric(0.35)
    blinks_per_emitter = on_per_frame * n_frames / (len(emitters) * mean_on)
    rows = []
    for i in range(len(emitters)):
        for _ in range(rng.poisson(blinks_per_emitter)):
            start = rng.integers(0, n_frames)
            for f in range(start, min(start + rng.geometric(0.35), n_frames)):
                rows.append((i, f))
    rows = np.array(rows, dtype=np.int64).reshape(-1, 2)
    rows = rows[np.argsort(rows[:, 1], kind="stable")]
    emitter, frame = rows[:, 0], rows[:, 1]
    n = len(frame)
    emitted = rng.gamma(4.0, photons / 4.0, n)
    # pixel k spans k - 1/2 .. k + 1/2, so x_nm = x_pix * pixelsize, which is
    # the fitter's convention: at k + 1/2 every position came out half a pixel off
    xy_px = emitters[emitter, :2] / pixelsize_nm
    z_nm = emitters[emitter, 2]
    if astigmatism is None:
        sx_nm = sy_nm = np.full(n, sigma_nm)
    else:
        sx_nm, sy_nm = astigmatic_sigmas(z_nm, sigma_nm, *astigmatism)
    sx = sx_nm / pixelsize_nm * np.sqrt(2.0)       # erf's scale: sigma * sqrt(2)
    sy = sy_nm / pixelsize_nm * np.sqrt(2.0)

    image = np.full((n_frames, size_px, size_px), background, dtype=np.float64)
    half = int(np.ceil(4 * max(sx_nm.max(), sy_nm.max()) / pixelsize_nm))
    offsets = np.arange(-half, half + 1)
    for k in range(n):
        cx, cy = xy_px[k]
        ix, iy = int(np.rint(cx)) + offsets, int(np.rint(cy)) + offsets
        keep_x = (ix >= 0) & (ix < size_px)
        keep_y = (iy >= 0) & (iy < size_px)
        if not keep_x.any() or not keep_y.any():
            continue
        ix, iy = ix[keep_x], iy[keep_y]
        px = 0.5 * (erf((ix + 0.5 - cx) / sx[k]) - erf((ix - 0.5 - cx) / sx[k]))
        py = 0.5 * (erf((iy + 0.5 - cy) / sy[k]) - erf((iy - 0.5 - cy) / sy[k]))
        image[frame[k], iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1] += emitted[k] * np.outer(py, px)
    electrons = rng.poisson(image)
    adu = electrons / conversion + offset + rng.normal(0, read_noise, image.shape)
    frames = np.clip(np.round(adu), 0, 65535).astype(np.uint16)
    truth = Localizations({
        "frame": frame, "x_nm": emitters[emitter, 0].astype(np.float32),
        "y_nm": emitters[emitter, 1].astype(np.float32),
        "z_nm": z_nm.astype(np.float32),
        "photons": emitted.astype(np.float32), "emitter": emitter.astype(np.int32),
    }, {"units": "nm", "simulation": "SMAPpy camera frames", "seed": seed,
        "pixelsize_nm": pixelsize_nm, "conversion": conversion, "offset": offset,
        "background": background, "sigma_nm": sigma_nm,
        "astigmatism": list(astigmatism) if astigmatism else None})
    return frames, truth


# A cylindrical lens of moderate strength: the two foci 2 x 300 nm apart and a
# depth of 400 nm, which keeps the spot fittable over about +-600 nm -- the
# structure's z range with room to spare.
ASTIGMATISM = (300.0, 400.0)


def astigmatic_sigmas(z_nm, sigma_nm: float, focal_offset_nm: float, depth_nm: float):
    """The spot's widths in x and y at ``z_nm``, behind a cylindrical lens.

    The textbook model (Huang et al., Science 2008): each axis is a Gaussian
    beam focused at +-``focal_offset_nm``, ``sigma(z) = sigma_0 sqrt(1 + ((z -
    c) / d)^2)``, so a spot is wide in x above focus, wide in y below, and
    round in between.
    """
    z = np.asarray(z_nm, dtype=np.float64)
    sx = sigma_nm * np.sqrt(1 + ((z - focal_offset_nm) / depth_nm) ** 2)
    sy = sigma_nm * np.sqrt(1 + ((z + focal_offset_nm) / depth_nm) ** 2)
    return sx, sy


def bead_stacks(n_stacks: int = 3, seed: int = 0, z_range_nm=(-800.0, 800.0),
                dz_nm: float = 20.0, pixelsize_nm: float = 100.0, size_px: int = 96,
                sigma_nm: float = 130.0, astigmatism=ASTIGMATISM,
                photons: float = 20000.0, background: float = 100.0,
                conversion: float = 0.5, offset: float = 100.0):
    """z-stacks of fluorescent beads for a calibration: a list of (z, y, x) ADU.

    What a bead calibration is measured from -- a few fields of view of beads
    on a coverslip, the objective stepped through focus -- with the same PSF
    `camera_frames` uses, so a calibration built from these fits those.  Nine
    beads per stack on a grid 30 pixels apart, each a little off the pixel
    grid, so the calibration has to register them as a real one does.
    """
    from scipy.special import erf
    rng = np.random.default_rng(seed)
    z = np.arange(z_range_nm[0], z_range_nm[1] + dz_nm / 2, dz_nm)
    # z is where the *objective* is, as a calibration records it: raised by
    # dz, it leaves a bead on the coverslip dz below the focus.  Drawn at +z
    # instead, the calibration came out mirrored and every fitted z with it.
    sx_nm, sy_nm = astigmatic_sigmas(-z, sigma_nm, *astigmatism)
    sx = sx_nm / pixelsize_nm * np.sqrt(2.0)
    sy = sy_nm / pixelsize_nm * np.sqrt(2.0)
    pix = np.arange(size_px)
    stacks = []
    for _ in range(n_stacks):
        image = np.full((len(z), size_px, size_px), background, dtype=np.float64)
        for gy in (18, 48, 78):
            for gx in (18, 48, 78):
                cx, cy = gx + rng.uniform(-0.5, 0.5), gy + rng.uniform(-0.5, 0.5)
                for k in range(len(z)):
                    px = 0.5 * (erf((pix + 0.5 - cx) / sx[k]) - erf((pix - 0.5 - cx) / sx[k]))
                    py = 0.5 * (erf((pix + 0.5 - cy) / sy[k]) - erf((pix - 0.5 - cy) / sy[k]))
                    image[k] += photons * np.outer(py, px)
        adu = rng.poisson(image) / conversion + offset
        stacks.append(np.clip(np.round(adu), 0, 65535).astype(np.uint16))
    return stacks, z
