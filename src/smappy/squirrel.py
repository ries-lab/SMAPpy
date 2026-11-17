"""Error maps: where a super-resolved picture disagrees with a widefield one.

NanoJ-SQUIRREL (Culley et al., *Nat Methods* 15, 263 (2018),
https://doi.org/10.1038/nmeth.4605), re-implemented from the paper.  A
super-resolved picture, blurred back to the diffraction limit, should look
like a widefield image of the same field.  Where it does not, something is
missing from it (a structure not labelled, not detected, filtered away) or
something is in it that the sample did not show (two filaments drawn as one,
a bright clump of repeated localizations).  The recipe:

1. the localizations are pictured on the widefield image's pixels;
2. they are blurred by a Gaussian of width ``sigma`` -- the paper's
   *resolution scaling function* (RSF) -- and scaled, ``alpha * blurred +
   beta``, with ``alpha``, ``beta`` and ``sigma`` the ones that make the two
   pictures agree best in the least-squares sense;
3. the error map is the difference, pixel by pixel; the RSE is its root mean
   square and the RSP the Pearson correlation of the two pictures.

**The departures, and why.**

*The comparison is on the reference's own pixels.*  SQUIRREL and NanoPyx
magnify the widefield image to the super-resolved pixels by interpolation
and compare there.  That interpolation smooths the reference's noise and so
shrinks the error; here nothing is done to the reference.  The localizations
are counted on a grid ``magnification`` times finer than the reference's
pixels, blurred there, and each block of ``magnification x magnification``
fine pixels summed back into the reference pixel it lies in -- exact for
counts, and blurring before summing is what a camera does to light.

*Localizations, not a rendered image.*  The picture is a histogram of the
positions, never the window's rendering: its contrast and saturation would
break the linear ``alpha``, ``beta``, and any blur a renderer adds is a
Gaussian the fitted ``sigma`` takes up anyway.

*The shift moves the localizations.*  SQUIRREL registers the images by
cross-correlation and crops the edges the shift exposes.  Here the
cross-correlation of the blurred picture with the reference gives the shift,
and the localizations are moved by it and pictured again, until the
correlation finds no shift left (`REFINE` times at most) -- the same refining
Register layers does, which removes the quadratic peak fit's pull towards
whole pixels.  Nothing is interpolated and nothing is cropped: the fine grid
reaches `PAD` reference pixels beyond the compared ones, so localizations
just outside still blur into them.

*``sigma`` by a bounded search in one dimension.*  For a given ``sigma`` the
best ``alpha`` and ``beta`` are a straight-line fit, solved exactly; only
``sigma`` is searched (Brent's method within bounds), where SQUIRREL searched
all three by a particle swarm on the GPU.  The bound is `MAX_SIGMA_PIXELS`
reference pixels; a ``sigma`` found on it is said to be.

*Drift.*  The localizations are compared as the table holds them, drift
corrected or not.  A widefield image taken before the acquisition matches a
drift curve referenced to the first frame, one taken after it the last; a
constant offset between them is the shift, found anyway.  The mean of the
raw frames (what a fit keeps as its first image) is smeared by whatever
drifted during the acquisition, which the fitted ``sigma`` takes up only in
part.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

#: how much finer than the reference's pixels the localizations are counted
DEFAULT_MAGNIFICATION = 4
#: the widest blur searched, in reference pixels.  A PSF on a camera sampled
#: at Nyquist is about one pixel sigma; five leaves room for defocus and
#: drift without letting the fit blur everything into the background.
#: (SQUIRREL searches up to ten; NanoPyx's bound is a FWHM of four pixels.)
MAX_SIGMA_PIXELS = 5.0
#: the fine grid's margin beyond the compared pixels: the Gaussian's reach,
#: in reference pixels, at the widest sigma (four sigma)
PAD = int(np.ceil(4 * MAX_SIGMA_PIXELS)) + 1
#: how often the shift is measured again with the localizations moved
REFINE = 4
#: the farthest the shift is looked for, by default
MAX_SHIFT_NM = 1000.0
#: fewer compared pixels than this on a side is refused
MIN_PIXELS = 8

# QnR's logistic (eSRRF, Laine et al. 2023, NanoJ-eSRRF's GetQnRmap_): the
# best resolution of the set maps to 0.925, the worst to 0.075
QNR_LOW, QNR_HIGH = 0.075, 0.925


@dataclass
class ErrorMap:
    """A super-resolved picture against its widefield reference.

    ``reference`` and ``model`` are the compared pixels, ``[row (y), column
    (x)]``: the reference as it is, and ``alpha * blur(localizations) +
    beta``.  The grid's outer edge is ``(x0, y0)`` in the reference's
    coordinates; the localizations lie ``(dx_nm, dy_nm)`` from it.
    """
    reference: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    model: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    x0: float = 0.0
    y0: float = 0.0
    pixelsize: float = float("nan")
    pixelsize_y: float = float("nan")
    alpha: float = float("nan")
    beta: float = float("nan")
    sigma_nm: float = float("nan")
    sigma_fitted: bool = True
    at_bound: bool = False
    dx_nm: float = 0.0
    dy_nm: float = 0.0
    rse: float = float("nan")
    rsp: float = float("nan")
    n_locs: int = 0
    magnification: int = DEFAULT_MAGNIFICATION
    message: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.model.size) and np.isfinite(self.rse)

    @property
    def signed(self) -> np.ndarray:
        """``model - reference``: positive where the localizations show more
        than the reference, negative where they miss what it shows."""
        return self.model - self.reference

    @property
    def error(self) -> np.ndarray:
        """The error map, ``|model - reference|``, as SQUIRREL's."""
        return np.abs(self.signed)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(x0, y0, x1, y1) of the compared pixels, in the reference's
        coordinates."""
        ny, nx = self.model.shape
        return (self.x0, self.y0, self.x0 + nx * self.pixelsize,
                self.y0 + ny * self.pixelsize_y)

    def at(self, x, y, signed: bool = False) -> np.ndarray:
        """The map's value at each localization ``(x, y)`` -- table
        coordinates, the shift taken out -- NaN outside the compared pixels."""
        values = self.signed if signed else self.error
        x = np.asarray(x, float) - self.dx_nm
        y = np.asarray(y, float) - self.dy_nm
        out = np.full(np.broadcast(x, y).shape, np.nan)
        if not values.size:
            return out
        ny, nx = values.shape
        column = np.floor((x - self.x0) / self.pixelsize).astype(np.int64)
        row = np.floor((y - self.y0) / self.pixelsize_y).astype(np.int64)
        inside = (column >= 0) & (column < nx) & (row >= 0) & (row < ny)
        out[inside] = values[row[inside], column[inside]]
        return out

    def image(self, name: str = "error map", signed: bool = False):
        """The map as an `ImageData`, placed over the localizations (the
        shift added back, so that it lies on what they show)."""
        from .images import ImageData

        data = self.signed if signed else self.error
        return ImageData(np.ascontiguousarray(data, np.float32),
                         pixelsize=float(self.pixelsize),
                         x0=float(self.x0 + self.dx_nm),
                         y0=float(self.y0 + self.dy_nm), name=name,
                         pixelsize_y=(None if self.pixelsize_y == self.pixelsize
                                      else float(self.pixelsize_y)))


# ------------------------------------------------------------------ the parts

def compared_pixels(shape: Tuple[int, int], x0: float, y0: float,
                    px: float, py: float, x, y) -> Tuple[int, int, int, int]:
    """``(row0, row1, col0, col1)``: the reference pixels the localizations
    cover, their bounding box cut to the image.  A reference larger than the
    localizations' field would otherwise count every pixel outside it as
    missed structure."""
    ny, nx = shape
    col0 = int(np.clip(np.floor((np.min(x) - x0) / px), 0, nx))
    col1 = int(np.clip(np.floor((np.max(x) - x0) / px) + 1, 0, nx))
    row0 = int(np.clip(np.floor((np.min(y) - y0) / py), 0, ny))
    row1 = int(np.clip(np.floor((np.max(y) - y0) / py) + 1, 0, ny))
    return row0, row1, col0, col1


def fine_histogram(x, y, x0: float, y0: float, fine_x: float, fine_y: float,
                   shape: Tuple[int, int]) -> np.ndarray:
    """Localizations counted on a grid of ``shape`` fine pixels whose outer
    edge is ``(x0, y0)``; those outside are left out."""
    ny, nx = shape
    column = np.floor((np.asarray(x, float) - x0) / fine_x).astype(np.int64)
    row = np.floor((np.asarray(y, float) - y0) / fine_y).astype(np.int64)
    keep = (column >= 0) & (column < nx) & (row >= 0) & (row < ny)
    counts = np.bincount(row[keep] * nx + column[keep], minlength=nx * ny)
    return counts.reshape(ny, nx).astype(np.float64)


class _Blurrer:
    """One fine histogram, blurred at any sigma and summed into reference
    pixels.  Its transform is taken once: each sigma the search tries is a
    multiplication and one inverse transform, and the margin of `PAD` pixels
    keeps the transform's wrap-around out of the compared pixels."""

    def __init__(self, fine: np.ndarray, m: int, fine_x: float, fine_y: float,
                 pad: int):
        from scipy.fft import next_fast_len, rfft2
        self.shape = fine.shape
        self.m, self.pad = m, pad
        size = tuple(next_fast_len(n) for n in fine.shape)
        self.size = size
        self.spectrum = rfft2(fine, size)
        fy = np.fft.fftfreq(size[0], d=fine_y)
        fx = np.fft.rfftfreq(size[1], d=fine_x)
        self.fy2, self.fx2 = fy ** 2, fx ** 2

    def __call__(self, sigma_nm: float) -> np.ndarray:
        """The blurred counts on the compared reference pixels."""
        from scipy.fft import irfft2
        s2 = 2 * np.pi ** 2 * float(sigma_nm) ** 2
        transfer = np.exp(-s2 * self.fy2)[:, None] * np.exp(-s2 * self.fx2)[None, :]
        blurred = irfft2(self.spectrum * transfer, self.size)[:self.shape[0],
                                                              :self.shape[1]]
        m = self.m
        ny, nx = self.shape[0] // m, self.shape[1] // m
        coarse = blurred[:ny * m, :nx * m].reshape(ny, m, nx, m).sum(axis=(1, 3))
        p = self.pad
        return coarse[p:ny - p, p:nx - p]


def scale_to(picture: np.ndarray, reference: np.ndarray) -> Tuple[float, float]:
    """``(alpha, beta)`` minimising ``|alpha * picture + beta - reference|^2``:
    a straight-line fit, solved exactly."""
    a = picture.ravel()
    b = reference.ravel()
    da = a - a.mean()
    variance = float(np.dot(da, da))
    if variance <= 0:
        raise ValueError("the localizations make a flat picture on these pixels")
    alpha = float(np.dot(da, b - b.mean()) / variance)
    return alpha, float(b.mean() - alpha * a.mean())


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    da, db = a - a.mean(), b - b.mean()
    norm = float(np.sqrt(np.dot(da.ravel(), da.ravel()) * np.dot(db.ravel(), db.ravel())))
    return float(np.dot(da.ravel(), db.ravel()) / norm) if norm > 0 else float("nan")


def _rse(blurred: np.ndarray, reference: np.ndarray) -> float:
    alpha, beta = scale_to(blurred, reference)
    return float(np.sqrt(np.mean((alpha * blurred + beta - reference) ** 2)))


def fit_sigma(blur: _Blurrer, reference: np.ndarray, upper_nm: float
              ) -> Tuple[float, bool]:
    """The sigma of least RSE within ``[0, upper_nm]``, and whether it lies
    on the upper bound."""
    from scipy.optimize import minimize_scalar
    found = minimize_scalar(lambda s: _rse(blur(s), reference),
                            bounds=(0.0, upper_nm), method="bounded",
                            options={"xatol": 1e-3 * upper_nm})
    sigma = float(found.x)
    return sigma, sigma > 0.98 * upper_nm


# ------------------------------------------------------------------ the map

def error_map(reference: np.ndarray, x, y, x0: float = 0.0, y0: float = 0.0,
              pixelsize: float = 100.0, pixelsize_y: Optional[float] = None,
              magnification: int = DEFAULT_MAGNIFICATION,
              sigma_nm: Optional[float] = None, register: bool = True,
              max_shift_nm: float = MAX_SHIFT_NM, report=None) -> ErrorMap:
    """The error map of localizations ``(x, y)`` (nm) against ``reference``,
    a 2D widefield image whose pixel (0, 0) has its outer edge at ``(x0,
    y0)``.  ``sigma_nm`` None fits the blur; ``register`` finds and takes out
    the shift between the two, within ``max_shift_nm``."""
    reference = np.asarray(reference, float)
    if reference.ndim != 2:
        raise ValueError("the reference must be one plane")
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    px = float(pixelsize)
    py = float(pixelsize if pixelsize_y is None else pixelsize_y)
    m = max(int(magnification), 1)
    out = ErrorMap(pixelsize=px, pixelsize_y=py, magnification=m, n_locs=len(x))
    if not len(x):
        out.message = "no localizations"
        return out
    row0, row1, col0, col1 = compared_pixels(reference.shape, x0, y0, px, py, x, y)
    if row1 - row0 < MIN_PIXELS or col1 - col0 < MIN_PIXELS:
        out.message = (f"the localizations cover {col1 - col0} x {row1 - row0} "
                       f"pixels of the reference; at least {MIN_PIXELS} x "
                       f"{MIN_PIXELS} are needed -- do they lie on it?")
        return out
    compared = reference[row0:row1, col0:col1]
    if not np.all(np.isfinite(compared)):
        compared = np.where(np.isfinite(compared), compared, np.nanmedian(compared))
    out.reference = compared
    out.x0, out.y0 = x0 + col0 * px, y0 + row0 * py
    # the fine grid: the compared pixels and a margin of PAD on every side,
    # within which the localizations still blur into them
    pad = PAD + int(np.ceil(max_shift_nm / min(px, py))) if register else PAD
    gx0, gy0 = out.x0 - pad * px, out.y0 - pad * py
    shape = ((row1 - row0 + 2 * pad) * m, (col1 - col0 + 2 * pad) * m)
    fine_x, fine_y = px / m, py / m
    upper = MAX_SIGMA_PIXELS * min(px, py)
    lags = max(2, int(np.ceil(max_shift_nm / min(px, py))))

    dx = dy = 0.0
    sigma, at_bound = (float(sigma_nm), False) if sigma_nm is not None else (0.0, False)
    for step in range(REFINE + 1 if register else 1):
        fine = fine_histogram(x - dx, y - dy, gx0, gy0, fine_x, fine_y, shape)
        blur = _Blurrer(fine, m, fine_x, fine_y, pad)
        if sigma_nm is None:
            sigma, at_bound = fit_sigma(blur, compared, upper)
        if not register or step == REFINE:
            break
        from .layer_registration import lateral_shift
        try:
            col, row, _ = lateral_shift(compared, blur(sigma), 1.0, lags)
        except ValueError as error:
            out.message = str(error)
            return out
        dx, dy = dx + col * px, dy + row * py
        if report:
            report(f"error map: shift ({dx:.1f}, {dy:.1f}) nm, sigma {sigma:.1f} nm")
        if np.hypot(col * px, row * py) < 0.01 * min(px, py):
            break
    if np.hypot(dx, dy) > max_shift_nm:
        out.message = (f"the shift found, ({dx:.0f}, {dy:.0f}) nm, is beyond "
                       f"max shift ({max_shift_nm:g} nm)")
    blurred = blur(sigma)
    try:
        alpha, beta = scale_to(blurred, compared)
    except ValueError as error:
        out.message = str(error)
        return out
    out.model = alpha * blurred + beta
    out.alpha, out.beta = alpha, beta
    out.sigma_nm, out.sigma_fitted, out.at_bound = sigma, sigma_nm is None, at_bound
    out.dx_nm, out.dy_nm = float(dx), float(dy)
    out.rse = float(np.sqrt(np.mean(out.signed ** 2)))
    out.rsp = pearson(out.model, compared)
    if at_bound and not out.message:
        out.message = (f"sigma reached its bound ({upper:.0f} nm, "
                       f"{MAX_SIGMA_PIXELS:g} pixels): no good minimum -- is "
                       "this the same field?")
    return out


def reference_plane(image, plane: Optional[int] = None) -> np.ndarray:
    """The plane of ``image`` (an `ImageData`) to compare with: ``plane``, or
    the average of the raw frames a fit kept, or the plane it shows."""
    if plane is None:
        frames = image.frames
        if frames is not None and np.any(np.asarray(frames) < 0):
            plane = int(np.flatnonzero(np.asarray(frames) < 0)[0])
        else:
            plane = image.frame
    return np.asarray(image.plane_at(plane), float)


# -------------------------------------------------------------------- QnR

def normalised_resolution(resolution) -> np.ndarray:
    """eSRRF's nFRC: each resolution mapped between 0 and 1 by a logistic
    through the set's best (``QNR_HIGH``) and worst (``QNR_LOW``) -- a rank
    within the set, not an absolute measure.  NaN stays NaN; a set whose
    resolutions are all one value is 0.5."""
    r = np.asarray(resolution, float)
    good = r[np.isfinite(r)]
    if not len(good):
        return np.full(r.shape, np.nan)
    best, worst = float(good.min()), float(good.max())
    if worst <= best:
        return np.where(np.isfinite(r), 0.5, np.nan)
    a1 = np.log((1 - QNR_LOW) / QNR_LOW)
    a2 = np.log((1 - QNR_HIGH) / QNR_HIGH)
    middle = (a2 * worst - a1 * best) / (a2 - a1)
    k = a1 / (middle - worst)
    return 1.0 / (np.exp(-k * (r - middle)) + 1.0)


def qnr(rsp, resolution) -> np.ndarray:
    """eSRRF's quality-and-resolution score, ``2 RSP nFRC / (RSP + nFRC)``,
    for a set of pictures of one field: the harmonic mean of the fidelity
    (RSP) and the resolution's rank in the set."""
    rsp = np.asarray(rsp, float)
    n = normalised_resolution(resolution)
    total = rsp + n
    return np.divide(2 * rsp * n, total, out=np.full(rsp.shape, np.nan),
                     where=np.isfinite(total) & (total > 0))
