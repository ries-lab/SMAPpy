"""Camera frames of the ground truth, and the bead stacks to calibrate them.

For fitting what `simulate` only pretends to have fitted: the same molecules
(`localizations.ground_truth`), drawn as spots on a camera.  The PSF is a
Gaussian integrated over each pixel -- the fitter's own model, so a fit is
expected to find it -- astigmatic behind a cylindrical lens if asked, over a
flat background; photons are Poisson, converted at ``conversion`` e-/ADU on an
``offset``, with Gaussian read noise in ADU.  Pixel k is centred on k, the
fitter's convention: at k + 1/2 every position came out half a pixel off.

Nothing is dropped.  Two spots close together in one frame are what a fitter
has to cope with, and the frames are the fitter's test, not the simulator's;
the truth table says for each spot how far its nearest neighbour in the same
frame was (``neighbour_nm``), so a test of precision can take the isolated
ones and a test of crowding all of them.  (Crowding pulls astigmatic z towards
focus: see NOTES.md, "Crowding compresses z".)

Frames are drawn a block at a time and their noise is seeded per frame, so
`render` gives the same frame however the stack is cut -- which is what lets
`smappy.simulate.source` hand a fitter a stack that is never held in memory.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence, Tuple

import numpy as np

from ..locs import Localizations
from .settings import (BlinkingSettings, CameraOutputSettings, LabellingSettings,
                       OpticsSettings, SimulationSettings, StructureSettings)

# a spot closer to its neighbour than this has that neighbour's light in the
# fit's 13 px ROI (600 nm either way plus a defocused spot): the distance the
# tests call isolated
ISOLATED_NM = 1000.0


def render(truth, start: int = 0, stop: Optional[int] = None,
           dual: Optional[dict] = None, chunk: int = 200) -> np.ndarray:
    """Frames ``start`` .. ``stop`` of ``truth`` as uint16 ADU, (n, y, x).

    ``dual`` -- ``{"ratios": per dye, "sigma_nm": per half}`` -- draws a split
    camera instead: each molecule in both halves, the fraction ``ratio`` of
    its light in the lower one, placed there by `dual_transformation`.
    """
    s = truth.settings
    stop = s.n_frames if stop is None else min(stop, s.n_frames)
    size = s.camera.size_px
    height = 2 * size if dual else size
    out = np.empty((max(stop - start, 0), height, size), np.uint16)
    background = frame_backgrounds(s)
    em = truth.emission
    xyz = truth.positions()
    for a in range(start, stop, chunk):
        b = min(a + chunk, stop)
        lo, hi = np.searchsorted(em.frame, [a, b])
        image = np.empty((b - a, height, size))
        image[:] = background[a:b, None, None]
        rows = slice(lo, hi)
        frame = em.frame[rows] - a
        if hi > lo:
            px = xyz[rows, :2] / s.optics.pixelsize_nm
            if dual is None:
                _draw(image, frame, px, em.photons[rows], *_widths(xyz[rows, 2], s.optics))
            else:
                ratio = _ratios(truth, dual["ratios"])[em.owner[rows]]
                forward = np.linalg.inv(dual_transformation(size))
                secondary = (np.c_[px, np.ones(len(px))] @ forward.T)[:, :2]
                for half, (at, share) in enumerate(((px, 1 - ratio), (secondary, ratio))):
                    optics = replace(s.optics, sigma_nm=dual["sigma_nm"][half])
                    _draw(image, frame, at, em.photons[rows] * share,
                          *_widths(xyz[rows, 2], optics))
        for k in range(b - a):
            rng = np.random.default_rng([s.seed, 3, a + k])
            out[a - start + k] = _camera(image[k], rng, s.camera.conversion,
                                         s.camera.offset, s.camera.read_noise)
    return out


def frame_backgrounds(s: SimulationSettings) -> np.ndarray:
    """The background of every frame, photons per pixel: flat over a frame."""
    from .kinetics import draw
    return draw(s.background, s.background_std, s.n_frames,
                np.random.default_rng([s.seed, 2]))


def _widths(z_nm, optics: OpticsSettings):
    """The spot's widths in pixels times sqrt(2), erf's scale."""
    if optics.astigmatism:
        sx, sy = astigmatic_sigmas(z_nm, optics.sigma_nm, optics.focal_offset_nm,
                                   optics.depth_nm)
    else:
        sx = sy = np.full(len(z_nm), optics.sigma_nm)
    k = np.sqrt(2.0) / optics.pixelsize_nm
    return sx * k, sy * k


def _ratios(truth, ratios: Sequence[float]) -> np.ndarray:
    dye = truth.fluorophores.dye
    ratios = np.asarray(ratios, float)
    if len(dye) and dye.max() > len(ratios):
        raise ValueError(f"the structure has dye {dye.max()} and only "
                         f"{len(ratios)} ratios were given")
    return ratios[np.clip(dye - 1, 0, None)] if len(dye) else np.zeros(0)


def camera_truth(truth, dual: Optional[dict] = None) -> Localizations:
    """Every spot drawn, at its true position (drift included), with the
    photons it emitted in that frame and its nearest neighbour's distance."""
    from .localizations import _metadata
    s = truth.settings
    em = truth.emission
    xyz = truth.positions()
    fl = truth.fluorophores
    columns = {
        "frame": em.frame.astype(np.int64),
        "x_nm": xyz[:, 0].astype(np.float32), "y_nm": xyz[:, 1].astype(np.float32),
        "z_nm": xyz[:, 2].astype(np.float32),
        "photons": em.photons.astype(np.float32),
        "background": frame_backgrounds(s)[em.frame].astype(np.float32),
        "neighbour_nm": neighbour_distance(xyz[:, 0], xyz[:, 1], em.frame).astype(np.float32),
        "emitter": em.owner.astype(np.int32), "dye": fl.dye[em.owner].astype(np.int32),
        "copy": fl.copy[em.owner].astype(np.int32),
    }
    metadata = _metadata(truth, "smappy camera frames")
    metadata.update({"pixelsize_nm": s.optics.pixelsize_nm,
                     "conversion": s.camera.conversion, "offset": s.camera.offset,
                     "read_noise": s.camera.read_noise, "sigma_nm": s.optics.sigma_nm,
                     "astigmatism": ([s.optics.focal_offset_nm, s.optics.depth_nm]
                                     if s.optics.astigmatism else None)})
    if dual:
        columns["ratio"] = _ratios(truth, dual["ratios"])[em.owner].astype(np.float32)
        size = s.camera.size_px
        metadata.update({"sigma_nm": list(dual["sigma_nm"]),
                         "ratios": list(dual["ratios"]),
                         "transformation": dual_transformation(size).tolist(),
                         "layout": "up-down", "main_channel": "upper",
                         "split_position": size})
    return Localizations(columns, metadata)


def neighbour_distance(x, y, frame) -> np.ndarray:
    """How far the nearest other spot on in the same frame is (inf if none);
    ``frame`` sorted."""
    from scipy.spatial import cKDTree
    out = np.full(len(frame), np.inf)
    edges = np.flatnonzero(np.diff(frame) != 0) + 1
    for block in np.split(np.arange(len(frame)), edges):
        if block.size < 2:
            continue
        d, _ = cKDTree(np.column_stack([x[block], y[block]])).query(
            np.column_stack([x[block], y[block]]), k=2)
        out[block] = d[:, 1]
    return out


def _settings(n_frames, seed, pixelsize_nm, size_px, photons, background, sigma_nm,
              conversion, offset, read_noise, astigmatism, structure, labelling,
              blinking) -> SimulationSettings:
    structure = str(structure)
    shape = (StructureSettings(file=structure) if structure.endswith((".yaml", ".yml"))
             else StructureSettings(preset=structure))
    optics = OpticsSettings(pixelsize_nm=pixelsize_nm, sigma_nm=sigma_nm,
                            astigmatism=astigmatism is not None)
    if astigmatism is not None:
        optics = replace(optics, focal_offset_nm=astigmatism[0], depth_nm=astigmatism[1])
    return SimulationSettings(
        output="camera", n_frames=n_frames, seed=seed, background=background,
        structure=shape, labelling=labelling or LabellingSettings(),
        blinking=replace(blinking or BlinkingSettings(), photons=photons),
        optics=optics,
        camera=CameraOutputSettings(size_px=size_px, conversion=conversion,
                                    offset=offset, read_noise=read_noise))


def camera_frames(n_frames: int = 2000, seed: int = 0, pixelsize_nm: float = 100.0,
                  size_px: int = 100, photons: float = 5000.0,
                  background: float = 20.0, sigma_nm: float = 130.0,
                  conversion: float = 0.5, offset: float = 100.0,
                  read_noise: float = 1.5,
                  astigmatism: Optional[Tuple[float, float]] = None,
                  structure: str = "demo",
                  labelling: Optional[LabellingSettings] = None,
                  blinking: Optional[BlinkingSettings] = None,
                  settings: Optional[SimulationSettings] = None):
    """Raw camera frames of a blinking structure: ``(frames, truth)``.

    ``frames`` is uint16 ADU, (n, y, x); ``truth`` is a `Localizations` of
    every spot drawn (`camera_truth`).  ``photons`` is the mean of a whole
    blink.  ``astigmatism`` -- ``(focal offset, depth)`` in nm, see
    `astigmatic_sigmas` -- makes the widths follow z, and a spline fit with a
    calibration of the same PSF (`bead_stacks`) finds it.  The density a
    fitter sees is the structure's fluorophores times their blinks and
    on-time over the frames -- about eight spots per frame for the demo over
    2000 frames; a short stack wants a lower labelling efficiency.  Everything
    else is `SimulationSettings`, which ``settings`` gives whole.
    """
    from .localizations import ground_truth
    if settings is None:
        settings = _settings(n_frames, seed, pixelsize_nm, size_px, photons, background,
                             sigma_nm, conversion, offset, read_noise, astigmatism,
                             structure, labelling, blinking)
    truth = ground_truth(settings)
    return render(truth), camera_truth(truth)


def _draw(image, frame, xy_px, photons, sx, sy) -> None:
    """Add a pixel-integrated Gaussian per spot into ``image`` (n, y, x), in
    place; ``sx``, ``sy`` are the widths in pixels times sqrt(2), erf's scale."""
    from scipy.special import erf
    if len(frame) == 0:
        return
    height, width = image.shape[1:]
    half = int(np.ceil(4 * max(sx.max(), sy.max()) / np.sqrt(2.0)))
    offsets = np.arange(-half, half + 1)
    for k in range(len(frame)):
        cx, cy = xy_px[k]
        ix, iy = int(np.rint(cx)) + offsets, int(np.rint(cy)) + offsets
        keep_x = (ix >= 0) & (ix < width)
        keep_y = (iy >= 0) & (iy < height)
        if not keep_x.any() or not keep_y.any():
            continue
        ix, iy = ix[keep_x], iy[keep_y]
        px = 0.5 * (erf((ix + 0.5 - cx) / sx[k]) - erf((ix - 0.5 - cx) / sx[k]))
        py = 0.5 * (erf((iy + 0.5 - cy) / sy[k]) - erf((iy - 0.5 - cy) / sy[k]))
        image[frame[k], iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1] += photons[k] * np.outer(py, px)


def _camera(image, rng, conversion: float, offset: float, read_noise: float):
    """Photons to ADU: Poisson, the gain and offset, Gaussian read noise."""
    electrons = rng.poisson(image)
    adu = electrons / conversion + offset + rng.normal(0, read_noise, image.shape)
    return np.clip(np.round(adu), 0, 65535).astype(np.uint16)


# Where the second half of a split camera sees what the first sees: a little
# shifted, turned and magnified, as a dichroic and two light paths leave it.
# Small, but pixels off at the corners -- the reason a transformation is fitted
# rather than a shift assumed.
DUAL_SHIFT_PX, DUAL_ANGLE_DEG, DUAL_SCALE = (1.6, -0.9), 0.4, 1.004


def dual_transformation(size_px: int = 100) -> np.ndarray:
    """The 3x3 map from the secondary (lower) half to the main (upper) half,
    in chip pixels -- the direction `calibrate.dual` and `ChannelTransform`
    use.  Its inverse is where a molecule in the main half appears below."""
    centre = np.array([(size_px - 1) / 2, (size_px - 1) / 2])
    a = np.deg2rad(DUAL_ANGLE_DEG)
    rotation = DUAL_SCALE * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    forward = np.eye(3)                              # main -> secondary
    forward[:2, :2] = rotation
    forward[:2, 2] = centre - rotation @ centre + np.array(DUAL_SHIFT_PX) + [0, size_px]
    return np.linalg.inv(forward)


def dual_camera_frames(n_frames: int = 2000, seed: int = 0, pixelsize_nm: float = 100.0,
                       size_px: int = 100, photons: float = 7500.0,
                       background: float = 20.0, sigma_nm=(130.0, 145.0),
                       ratios=(0.25, 0.75), conversion: float = 0.5,
                       offset: float = 100.0, read_noise: float = 1.5,
                       astigmatism: Optional[Tuple[float, float]] = None,
                       structure: str = "demo",
                       labelling: Optional[LabellingSettings] = None,
                       blinking: Optional[BlinkingSettings] = None):
    """Frames of a split camera imaging two dyes: ``(frames, truth)``.

    The two-colour counterpart of `camera_frames`: which dye a molecule is
    comes from its structure (the demo's ring is dye 1, the lines and the
    scattered points dye 2).  A dichroic sends each molecule's light to both
    halves of the chip -- the upper half the main channel, the lower the
    secondary, placed by `dual_transformation` -- in a proportion that is
    the dye's: ``ratios`` is the fraction in the secondary half, one per dye.
    So every molecule is a pair of spots, and which dye it is shows only in
    how its photons split, which is what the two-colour fit and the colour
    assignment measure.  The halves see different wavelengths, so
    ``sigma_nm`` is a width per half.  ``frames`` is uint16 ADU,
    (n, 2 * size_px, size_px); ``truth`` has the main-half positions,
    ``dye`` and ``ratio``, and its metadata the transformation.
    """
    from .localizations import ground_truth
    settings = _settings(n_frames, seed, pixelsize_nm, size_px, photons, background,
                         sigma_nm[0], conversion, offset, read_noise, astigmatism,
                         structure, labelling, blinking)
    truth = ground_truth(settings)
    dual = {"ratios": tuple(ratios), "sigma_nm": tuple(sigma_nm)}
    return render(truth, dual=dual), camera_truth(truth, dual=dual)


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


def dual_bead_stacks(n_stacks: int = 3, seed: int = 0, z_range_nm=(-800.0, 800.0),
                     dz_nm: float = 20.0, pixelsize_nm: float = 100.0,
                     size_px: int = 100, sigma_nm=(130.0, 145.0),
                     astigmatism=ASTIGMATISM, photons: float = 20000.0,
                     secondary_share: float = 0.5, background: float = 100.0,
                     conversion: float = 0.5, offset: float = 100.0):
    """Bead z-stacks on the split camera of `dual_camera_frames`: ``(stacks, z)``.

    What a dual-colour bead calibration is measured from: broadband beads,
    seen in both halves, each half's image placed by the same
    `dual_transformation` the acquisition has -- so a calibration from these
    maps the halves as the fit needs them mapped -- with each half's width
    and the same astigmatism.  ``secondary_share`` is the fraction of a
    bead's light in the lower half.  Nine beads per half on a 30 px grid, a
    little off the pixel grid, as in `bead_stacks`; z is the objective
    position, so the beads are drawn at -z (see there).  Each stack is uint16
    ADU, (z, 2 * size_px, size_px).
    """
    rng = np.random.default_rng(seed)
    z = np.arange(z_range_nm[0], z_range_nm[1] + dz_nm / 2, dz_nm)
    forward = np.linalg.inv(dual_transformation(size_px))
    widths = []
    for sigma in sigma_nm:
        sx_nm, sy_nm = astigmatic_sigmas(-z, sigma, *astigmatism)
        widths.append((sx_nm / pixelsize_nm * np.sqrt(2.0), sy_nm / pixelsize_nm * np.sqrt(2.0)))
    stacks = []
    for _ in range(n_stacks):
        image = np.full((len(z), 2 * size_px, size_px), background, dtype=np.float64)
        centres = np.array([(gx + rng.uniform(-0.5, 0.5), gy + rng.uniform(-0.5, 0.5))
                            for gy in (20, 50, 80) for gx in (20, 50, 80)])
        partners = (np.c_[centres, np.ones(len(centres))] @ forward.T)[:, :2]
        planes = np.arange(len(z))
        for half, (xy, share) in enumerate(((centres, 1 - secondary_share),
                                            (partners, secondary_share))):
            sx, sy = widths[half]
            for cx, cy in xy:
                _draw(image, planes, np.tile([cx, cy], (len(z), 1)),
                      np.full(len(z), photons * share), sx, sy)
        stacks.append(_camera(image, rng, conversion, offset, 0.0))
    return stacks, z
