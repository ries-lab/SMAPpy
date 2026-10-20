"""An astigmatic start: each bead's z and the astigmatism Zernikes from elliptical Gaussians (plan §10, 5).

The data's rough start (`start.py`) puts each bead's z at its brightest slice and the pupil at zero.
For an astigmatic PSF the spot's shape says more: through focus it is stretched one way above and the
other way below, and round where the bead is in focus.  Per slice, the spot's covariance
(C_xx, C_yy, C_xy, nm²) is measured, and its anisotropic part

    D = C_xx − C_yy  (vertical astigmatism, Noll 6),     X = 2 C_xy  (oblique, Noll 5)

is, in the usual astigmatic width model σ²_{u,v}(z) = σ₀²(1 + (z ± c)²/d²), *linear* in the defocus:
σ²_u − σ²_v = 4σ₀²c·z/d².  So per bead D_k = s₆ (z + z_k) and X_k = s₅ (z + z_k), with z_k the slice's
nominal z (`Batch.z_nm`): z is where both cross zero, and the slopes (s₆, s₅) are the astigmatism.

**Slopes to radians by the model itself.**  A paraxial formula is several tens of percent off at NA 1.4
with an index mismatch, and the pixel, the blur and the estimator's own bias all bend the slope; so
the same estimator is run on the model's own noise-free PSFs, with the start's amplitude, and the
data's slopes are matched to the model's (`astig_start`, a secant step or two).  Whatever the
estimator gets wrong it gets equally wrong on both.

**The covariance, three ways** (`slice_shapes`): SMAPpy's elliptical Gaussian MLE (``gauss_xy``, the
default: C++, axis-aligned, under a second for 70 beads), a Poisson fit of a rotated elliptical Gaussian
(``fit``, torch, 20–40 s: the only one that sees oblique astigmatism, Noll 5) and adaptive moments
(``moments``, which put the bead z 75–150 nm off once coma is present).  The lab's and Zenodo's
astigmatism is Noll 6 to within 0.02 rad, so SMAPpy's fitter needs no angle (docs/uipsf/architecture.md,
"How a bead fit starts").

**What it is used for.**  Measured there: the bead z and astigmatism did not make one-channel fits faster
or safer (stage 1 places beads and low modes together anyway), and the astigmatism into stage 1 stalled a
two-channel fit; the beads' positions in both channels make a transform start 2–3 × closer than the
pairing's centroids (`gaussian_start`'s ``transform``).

Coma is left out: the centroid's lateral walk through focus is what it would be read from, and a
stage that drifts between slices makes the same walk.
"""
from __future__ import annotations

import math

import torch

ASTIG = (5, 6)


# ------------------------------------------------------------------ per slice: the spot's covariance

def _grid(Y, X, pixel_size_nm, dtype):
    xs = (torch.arange(X, dtype=dtype) - (X - 1) / 2) * pixel_size_nm[0]
    ys = (torch.arange(Y, dtype=dtype) - (Y - 1) / 2) * pixel_size_nm[1]
    return xs[None, :].expand(Y, X), ys[:, None].expand(Y, X)


def _background(d):
    """Each spot's background: the median of its border pixels (`start.py`'s rule, per slice)."""
    border = torch.cat([d[..., 0, :], d[..., -1, :], d[..., 1:-1, 0], d[..., 1:-1, -1]], -1)
    return border.median(-1).values


def moments(d, pixel_size_nm, iterations: int = 8, bg=None):
    """Adaptive moments of spots ``d`` [..., Y, X]: (xy [..., 2], C [..., 3] = (C_xx, C_yy, C_xy) nm²,
    photons [...], background per pixel [...]).  Weighted by the current Gaussian, whose moments are then
    half the spot's when
    the spot is Gaussian (the product of two equal Gaussians has half the covariance): C = 2M."""
    d = d.double()
    *lead, Y, X = d.shape
    gx, gy = _grid(Y, X, pixel_size_nm, d.dtype)
    bg = _background(d) if bg is None else bg.double()
    s = d - bg[..., None, None]
    w0 = s.clamp(min=0)
    n0 = w0.sum((-2, -1)).clamp(min=1e-9)
    cx = (w0 * gx).sum((-2, -1)) / n0
    cy = (w0 * gy).sum((-2, -1)) / n0
    p2 = (pixel_size_nm[0] * pixel_size_nm[1])
    cxx = torch.full_like(cx, 4 * p2)
    cyy, cxy = cxx.clone(), torch.zeros_like(cx)
    for _ in range(iterations):
        det = (cxx * cyy - cxy ** 2).clamp(min=1e-6)
        dx, dy = gx - cx[..., None, None], gy - cy[..., None, None]
        q = (cyy[..., None, None] * dx ** 2 - 2 * cxy[..., None, None] * dx * dy
             + cxx[..., None, None] * dy ** 2) / det[..., None, None]
        w = torch.exp(-0.5 * q) * s
        n = w.sum((-2, -1))
        ok = n > 1e-6
        n = n.clamp(min=1e-6)
        cx = torch.where(ok, (w * gx).sum((-2, -1)) / n, cx)
        cy = torch.where(ok, (w * gy).sum((-2, -1)) / n, cy)
        dx, dy = gx - cx[..., None, None], gy - cy[..., None, None]
        mxx, myy, mxy = ((w * dx * dx).sum((-2, -1)) / n, (w * dy * dy).sum((-2, -1)) / n,
                         (w * dx * dy).sum((-2, -1)) / n)
        # C = 2M; held positive definite and above a quarter pixel, so a noisy slice cannot collapse it
        floor = 0.25 * p2
        nxx, nyy = (2 * mxx).clamp(min=floor), (2 * myy).clamp(min=floor)
        nxy = (2 * mxy).clamp(-0.95 * (nxx * nyy).sqrt(), 0.95 * (nxx * nyy).sqrt())
        cxx, cyy, cxy = torch.where(ok, nxx, cxx), torch.where(ok, nyy, cyy), torch.where(ok, nxy, cxy)
    photons = w0.sum((-2, -1))
    return torch.stack([cx, cy], -1), torch.stack([cxx, cyy, cxy], -1), photons, bg


def _gauss(theta, gx, gy, area):
    """A rotated elliptical Gaussian sampled at pixel centres: theta = (x, y, log N, log b, l₁₁, l₂₁, l₂₂)
    with the covariance L Lᵀ, L = [[e^l₁₁, 0], [l₂₁, e^l₂₂]] (positive definite for any theta)."""
    x, y, ln, lb, l11, l21, l22 = theta.unbind(-1)
    a, c = torch.exp(l11), torch.exp(l22)
    cxx, cxy, cyy = a * a, a * l21, l21 * l21 + c * c
    det = cxx * cyy - cxy * cxy
    dx, dy = gx - x[..., None, None], gy - y[..., None, None]
    q = (cyy[..., None, None] * dx ** 2 - 2 * cxy[..., None, None] * dx * dy + cxx[..., None, None] * dy ** 2) \
        / det[..., None, None]
    norm = torch.exp(ln) * area / (2 * math.pi * det.sqrt())
    return norm[..., None, None] * torch.exp(-0.5 * q) + torch.exp(lb)[..., None, None]


def gauss_fit(d, pixel_size_nm, iterations: int = 30, s: float = 0.0):
    """A Poisson fit of a rotated elliptical Gaussian to each spot ``d`` [..., Y, X] (photons; ``s`` the
    read-noise variance, added as the shifted Poisson does): (xy, C, photons) as `moments`, which it
    starts from.  Levenberg–Marquardt, one damping per spot, the 7 × 7 normal equations by hand."""
    d = d.double()
    *lead, Y, X = d.shape
    d = d.reshape(-1, Y, X) + s
    gx, gy = _grid(Y, X, pixel_size_nm, d.dtype)
    area = pixel_size_nm[0] * pixel_size_nm[1]
    bg = _background(d).clamp(min=1e-3)
    xy, C, ph, _ = moments(d - s, pixel_size_nm, bg=(bg - s))
    l11 = 0.5 * torch.log(C[:, 0])
    l21 = C[:, 2] / torch.exp(l11)
    l22 = 0.5 * torch.log((C[:, 1] - l21 ** 2).clamp(min=0.25 * area))
    theta = torch.stack([xy[:, 0], xy[:, 1], torch.log(ph.clamp(min=1.0)), torch.log(bg), l11, l21, l22], -1)
    lam = torch.full((d.shape[0],), 1e-3, dtype=d.dtype)

    def nll(t):
        mu = _gauss(t, gx, gy, area) + s
        return (mu - d * torch.log(mu)).sum((-2, -1))

    from torch.func import jacfwd, vmap
    jac = vmap(jacfwd(lambda t: _gauss(t, gx, gy, area).reshape(-1)))
    f = nll(theta)
    for _ in range(iterations):
        mu = (_gauss(theta, gx, gy, area) + s).reshape(d.shape[0], -1)
        J = jac(theta)                                                   # [N, P, 7]
        r = (d.reshape(d.shape[0], -1) / mu - 1)                          # −∂nll/∂μ
        g = (J * r[..., None]).sum(1)
        H = torch.einsum("npi,np,npj->nij", J, 1 / mu, J)
        Hd = H + lam[:, None, None] * torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2).clamp(min=1e-12))
        step = torch.linalg.solve(Hd, g[..., None])[..., 0]
        new = theta + step
        fn = nll(new)
        better = torch.isfinite(fn) & (fn < f)
        theta = torch.where(better[:, None], new, theta)
        f = torch.where(better, fn, f)
        lam = torch.where(better, lam * 0.3, lam * 10).clamp(1e-7, 1e7)
    x, y, ln, lb, l11, l21, l22 = theta.unbind(-1)
    a, c = torch.exp(l11), torch.exp(l22)
    C = torch.stack([a * a, l21 * l21 + c * c, a * l21], -1)
    return (torch.stack([x, y], -1).reshape(*lead, 2), C.reshape(*lead, 3), torch.exp(ln).reshape(lead),
            (torch.exp(lb) - s).reshape(lead))


def gauss_xy(d, pixel_size_nm, iterations: int = 50):
    """SMAPpy's elliptical Gaussian MLE (`smappy.psf.GaussianPSF(elliptical=True)`, C++, axis-aligned) on
    each spot ``d`` [..., Y, X] (square): (xy, C with C_xy = 0, photons, background), as `moments`."""
    import numpy as np
    from smappy.psf import GaussianPSF
    *lead, Y, X = d.shape
    rois = np.ascontiguousarray(d.reshape(-1, Y, X).detach().cpu().numpy().astype(np.float32))
    psf = GaussianPSF(sigma=1.3, elliptical=True)
    out = psf.unpack(psf.fit(rois, iterations))
    t = {k: torch.as_tensor(np.asarray(v), dtype=torch.float64) for k, v in out.items()}
    lost = ~torch.stack([t[k] for k in ("x_roi", "y_roi", "sigma_x_pix", "sigma_y_pix", "photons")]).isfinite().all(0)
    t["photons"] = t["photons"].masked_fill(lost, 0.0)            # a fit that failed weighs nothing
    for k in ("x_roi", "y_roi", "sigma_x_pix", "sigma_y_pix", "background"):
        t[k] = t[k].masked_fill(lost, 1.0)
    px, py = pixel_size_nm
    xy = torch.stack([(t["x_roi"] - (X - 1) / 2) * px, (t["y_roi"] - (Y - 1) / 2) * py], -1)
    C = torch.stack([(t["sigma_x_pix"] * px) ** 2, (t["sigma_y_pix"] * py) ** 2, torch.zeros_like(t["x_roi"])], -1)
    return xy.reshape(*lead, 2), C.reshape(*lead, 3), t["photons"].reshape(lead), t["background"].reshape(lead)


def slice_shapes(d, pixel_size_nm, method: str = "gauss_xy", s: float = 0.0):
    """Each spot's (xy, C, photons, background): ``method`` "moments", "gauss_xy" (SMAPpy's, no angle) or
    "fit" (`gauss_fit`, with an angle: the reference for what an angle in SMAPpy's fitter would give)."""
    if method == "moments":
        return moments(d, pixel_size_nm)
    if method == "gauss_xy":
        return gauss_xy(d, pixel_size_nm)
    if method == "fit":
        return gauss_fit(d, pixel_size_nm, s=s)
    raise ValueError(f"no shape method {method!r}")


# ------------------------------------------------------------------ through focus: z and the slopes

def precision_weights(C, photons, bg, pixel_size_nm, max_size_nm2=None):
    """How well each slice's anisotropy is measured, up to a constant: var(C_ij) ≈ C²/N_eff, with
    N_eff = N² / (N + 4π b √det C / pixel area) (Thompson et al. 2002's photon count with the background
    under the spot), so the weight is N_eff / (C_xx² + C_yy²).  Zero where the spot is not measured:
    no photons, a fit that ran away (non-finite, or wider than ``max_size_nm2``)."""
    C, photons, bg = C.double(), photons.double(), bg.double().clamp(min=0)
    area = pixel_size_nm[0] * pixel_size_nm[1]
    det = (C[..., 0] * C[..., 1] - C[..., 2] ** 2).clamp(min=1e-9)
    n = photons.clamp(min=0)
    n_eff = n * n / (n + 4 * math.pi * bg * det.sqrt() / area).clamp(min=1e-9)
    w = n_eff / (C[..., 0] ** 2 + C[..., 1] ** 2).clamp(min=1e-9)
    ok = torch.isfinite(w) & torch.isfinite(C).all(-1) & (photons > 0)
    if max_size_nm2 is not None:
        ok &= (C[..., 0] + C[..., 1]) < max_size_nm2
    return torch.where(ok, w, torch.zeros_like(w))


def _huber(r, w, k=1.5):
    """IRLS weights of Huber's loss for residuals ``r`` under weights ``w``, scaled by their weighted MAD."""
    a = (r.abs() * w.sqrt())
    scale = 1.4826 * a.masked_fill(w <= 0, float("nan")).nanmedian(-1, keepdim=True).values.clamp(min=1e-12)
    u = a / scale
    return w * torch.where(u <= k, torch.ones_like(u), k / u.clamp(min=1e-12))


def through_focus(C, z_nm, weights, window_nm: float = 500.0, rounds: int = 4, peak=None):
    """Each bead's z and the shared slopes from its slices' covariances C [E, K, 3] at nominal z
    ``z_nm`` [E, K], with ``weights`` [E, K] (`precision_weights`): (z [E], (s₆, s₅) nm²/nm, per-bead
    slopes [E, 2]).

    Per bead, D = C_xx − C_yy and X = 2C_xy are fitted as s (z + z_k) over the slices within
    ``window_nm`` of its focus (the spot is Gaussian enough there); the window follows the estimate,
    which starts at the slice with the highest ``peak`` [E, K] (the brightest pixel, `start.py`'s
    rule), or else where the spot is smallest.  Not the smallest when the peak is known: far from
    focus a fit can collapse onto one lobe of a spread spot and look smaller than the focus (spherical
    of 0.4 rad, 900 nm out).  The shared
    slopes are the medians of the beads' own; each bead's z is then refitted with them, with Huber's
    weights, so that a slice the fit lost does not move it."""
    C, z_nm, weights = C.double(), z_nm.double(), weights.double()
    D, Xc = C[..., 0] - C[..., 1], 2 * C[..., 2]
    E, K = D.shape
    if peak is not None:
        z = -z_nm[torch.arange(E), peak.double().argmax(1)]
    else:
        size = torch.where(weights > 0, C[..., 0] + C[..., 1], torch.full_like(D, float("inf")))
        smooth = torch.nn.functional.avg_pool1d(size.clamp(max=1e12)[:, None], 3, 1, 1,
                                                count_include_pad=False)[:, 0]
        z = -z_nm[torch.arange(E), smooth.argmin(1)]
    robust = weights.clone()
    for _ in range(rounds):
        w = ((z[:, None] + z_nm).abs() <= window_nm).double() * robust
        sw = w.sum(1).clamp(min=1e-30)
        zm = (w * z_nm).sum(1) / sw
        dz = z_nm - zm[:, None]
        var = (w * dz * dz).sum(1).clamp(min=1e-30)
        slopes = torch.stack([(w * dz * D).sum(1) / var, (w * dz * Xc).sum(1) / var], -1)   # [E, 2]: (s₆, s₅)
        good = (w > 0).sum(1) >= 3
        s = slopes[good].median(0).values if good.any() else slopes.median(0).values
        # each bead's z with the shared slopes: least squares of (D, X) = s (z + z_k)
        z = ((w * (s[0] * (D - s[0] * z_nm) + s[1] * (Xc - s[1] * z_nm))).sum(1)
             / (sw * (s * s).sum().clamp(min=1e-30)))
        r = s[0] * (D - s[0] * (z[:, None] + z_nm)) + s[1] * (Xc - s[1] * (z[:, None] + z_nm))
        robust = _huber(r, weights)
    return z, s, slopes


# ------------------------------------------------------------------ the start

def _model_shapes(psf_fn, pixel_size_nm, method, photons, background):
    """The estimator on the model's own noise-free PSF [K, Y, X], weighted as the data's slices would be
    at the data's typical ``photons`` and ``background``: the slope of a curve that is not quite a line
    depends on how its points are weighted, so model and data must be weighted alike."""
    mu = psf_fn()
    _, C, _, _ = slice_shapes(mu[None], pixel_size_nm, method)
    w = precision_weights(C, torch.full(C.shape[:-1], photons, dtype=C.dtype),
                          torch.full(C.shape[:-1], background, dtype=C.dtype), pixel_size_nm)
    return C, w


def astig_start(spots, z_nm, pixel_size_nm, model_psf, method: str = "fit", window_nm: float = 500.0,
                steps: int = 2, s: float = 0.0, amplitude0: float = 0.3):
    """Noll 5 and 6 (rad) and each bead's z from bead spots ``spots`` [E, K, Y, X] (photons) at nominal
    z ``z_nm`` [K] or [E, K].  ``model_psf(a5, a6)`` returns the model's PSF of a bead at the origin
    [K, Y, X] on the same z grid (``z_nm`` of one bead), with only those two modes.

    The data's slopes are matched to the model's by a secant step on each mode: the model's slopes are
    nearly linear in (a₆, a₅) and nearly decoupled (the pixel's anisotropy couples them a little), so
    a 2 × 2 sensitivity from one probe at ``amplitude0`` and ``steps`` Newton steps from it suffice.
    Returns dict(a5, a6, z [E], slopes, per_bead)."""
    E = spots.shape[0]
    zk = z_nm if z_nm.dim() == 2 else z_nm.expand(E, -1)
    xy, C, ph, bg = slice_shapes(spots, pixel_size_nm, method, s=s)
    Y, X = spots.shape[-2:]
    largest = (min(Y * pixel_size_nm[1], X * pixel_size_nm[0]) / 2.5) ** 2
    w = precision_weights(C, ph, bg, pixel_size_nm, max_size_nm2=largest)
    peak = (spots.double() - bg[..., None, None]).amax((-2, -1))
    z, target, per_bead = through_focus(C, zk, w, window_nm, peak=peak)
    z0 = zk[0].double()

    ok = w > 0
    typical = (float(ph[ok].median()), float(bg[ok].median()))

    peak_m = -(z0[None].abs())                      # the model's bead is in focus at z = 0

    def slopes(a6, a5):
        C_m, w_m = _model_shapes(lambda: model_psf(a5, a6), pixel_size_nm, method, *typical)
        return through_focus(C_m, z0[None], w_m, window_nm, peak=peak_m)[1]

    # sensitivity: ∂(s₆, s₅)/∂(a₆, a₅) from probes at ±amplitude0 (odd in a: the difference cancels the
    # estimator's offset, the pixel's anisotropy, at zero)
    Jc = []
    for probe in ((amplitude0, 0.0), (0.0, amplitude0)):
        sp, sm = slopes(*probe), slopes(-probe[0], -probe[1])
        Jc.append((sp - sm) / (2 * amplitude0))
    J = torch.stack(Jc, 1)                                               # columns: a₆, a₅
    # pinv: an axis-aligned Gaussian (gauss_xy) reads no oblique slope, and leaves a₅ at zero
    Jinv = torch.linalg.pinv(J)
    a = Jinv @ (target - slopes(0.0, 0.0))
    for _ in range(steps):
        a = a + Jinv @ (target - slopes(float(a[0]), float(a[1])))
    # the bead z read again against the model's curve at the found amplitude: where the model's spot
    # is round is not quite z = 0 (the pixel and the blur need not be round, coma tilts the curves)
    C_m, w_m = _model_shapes(lambda: model_psf(float(a[1]), float(a[0])), pixel_size_nm, method, *typical)
    zero = float(through_focus(C_m, z0[None], w_m, window_nm, peak=peak_m)[0][0])
    return dict(a5=float(a[1]), a6=float(a[0]), z=z - zero, slopes=target, per_bead=per_bead, xy=xy, C=C,
                photons=ph, background=bg, weights=w, sensitivity=J)



# ------------------------------------------------------------------ a model's start from it

def channel_probe(psf, z_nm, sigma_nm=None):
    """``model_psf(a5, a6)`` for `astig_start`: channel ``psf``'s (`VectorPSF`) image of a bead in focus
    at the origin through the nominal z ``z_nm`` [K], with only Noll 5 and 6 in the pupil, [K, Y, X]."""
    from ..simulate import parameters, template
    from .forward import ForwardModel
    one = ForwardModel(psf)
    dev, dtype = psf.grid.device, psf.grid.dtype
    b = template(1, torch.as_tensor(z_nm, dtype=dtype, device=dev), psf.roi_shape, dtype=dtype, device=dev)
    modes = psf.pupil.phase_modes

    def model_psf(a5, a6):
        phase = torch.zeros(len(modes), dtype=dtype, device=dev)
        for j, a in ((5, a5), (6, a6)):
            if j in modes:
                phase[modes.index(j)] = a
        values = {"pupil.phase": phase, "emitter.xyz": torch.zeros(1, 3, dtype=dtype, device=dev),
                  "emitter.photons": torch.full((1,), 1e4, dtype=dtype, device=dev),
                  "emitter.background": torch.zeros(1, dtype=dtype, device=dev)}
        if sigma_nm is not None:
            values["psf.sigma"] = torch.as_tensor(sigma_nm, dtype=dtype, device=dev)
        with torch.no_grad():
            return one(parameters(one, b, values, dtype=dtype, device=dev).physical(), b)[0, 0].cpu().double()
    return model_psf


def gaussian_start(model, batch, method: str = "gauss_xy", window_nm: float = 500.0) -> dict:
    """A start for ``model`` on ``batch`` from elliptical Gaussians per slice, every channel alone.

    Returns ``shared`` (each channel's pupil phase: zero but Noll 5 and 6), ``emitters`` (the
    `rough_start` dict with each bead's z from the main channel's through-focus fit and x, y the
    Gaussians' mean position over the slices near focus; with several channels its ``dz`` from the
    channels' z), ``transform`` (each secondary channel's map to the main chip from the beads' Gaussian
    positions in both, a projective fit as SMAPpy's pairing makes it), and ``per_channel`` (the
    estimates)."""
    from .start import rough_start
    z_nm = batch.z_nm if batch.z_nm.dim() == 1 else batch.z_nm[0]
    px = model.pixel_size_nm
    found = []
    for c, ch in enumerate(model.channels):
        r = astig_start(batch.data[:, c].cpu(), batch.z_nm.cpu(), px, channel_probe(ch.psf, z_nm.cpu()),
                        method=method, window_nm=window_nm, s=float(torch.as_tensor(batch.s).mean()))
        found.append(r)
    rough = rough_start(batch, px)
    E = batch.n_emitters
    zk = (batch.z_nm if batch.z_nm.dim() == 2 else batch.z_nm.expand(E, -1)).cpu().double()
    main = found[0]
    near = ((main["z"][:, None] + zk).abs() <= window_nm).double()

    def position(r, w):
        w = w * r["weights"]
        return (w[..., None] * r["xy"]).sum(1) / w.sum(1, keepdim=True).clamp(min=1e-30)   # [E, 2] nm

    w_all = near * torch.stack([r["weights"] > 0 for r in found]).all(0).double()
    xyz = rough["xyz"].clone()
    xy0 = position(main, w_all)
    xyz[:, 0], xyz[:, 1] = xy0[:, 0].to(xyz.dtype), xy0[:, 1].to(xyz.dtype)
    xyz[:, 2] = main["z"].to(xyz.dtype)
    emitters = dict(rough, xyz=xyz)
    shared = {}
    for c, (ch, r) in enumerate(zip(model.channels, found, strict=True)):
        pre = "" if c == 0 else f"ch{c}."
        modes = ch.psf.pupil.phase_modes
        shared[pre + "pupil.phase"] = torch.tensor([r["a5"] if j == 5 else r["a6"] if j == 6 else 0.0 for j in modes])
    out = dict(shared=shared, emitters=emitters, per_channel=found)
    if model.n_channels > 1:
        import numpy as np
        from smappy.calibrate.dual import fit_projective
        emitters["dz"] = {c: float((found[c]["z"] - main["z"]).median()) for c in range(model.n_channels)}
        pix = torch.tensor(px, dtype=torch.float64)
        chip0 = batch.roi_xy[:, 0].cpu().double() + xy0 / pix
        out["transform"] = {}
        for c in range(1, model.n_channels):
            chip = batch.roi_xy[:, c].cpu().double() + position(found[c], w_all) / pix
            h = np.asarray(fit_projective(chip.numpy(), chip0.numpy()), float)
            out["transform"][c] = h / h[2, 2]
    return out
