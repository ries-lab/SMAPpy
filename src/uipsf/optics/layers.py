"""The layers between emitter and objective, with their interfaces tilted: sample, coverslip, immersion.

A plane wave of the pupil leaves the emitter, crosses the sample (n_med) for the emitter's height h above
the coverslip, the coverslip (n_cov, d_cov thick) and the immersion (n_imm) to the objective.  At an
interface the wave vector's component along the interface is kept; only the component along the normal
changes.  With n̂ = (t_x, t_y, 1)/√(1 + t²) the interfaces' normal, into the sample, and the light going
down, k = (k_x, k_y, −k_z,imm) in the immersion, the normal component in layer ℓ is

    k_n,imm = −k·n̂ = (k_z,imm − t_x k_x − t_y k_y)/√(1 + t²),      k_n,ℓ(t) = √(n_ℓ²/λ² − n_imm²/λ² + k_n,imm²).

The pupil carries the conjugate of the wave's phase (`docs/uipsf/conventions.md`: −k_x x + k_z z for an
emitter at (x, y, z)); summed over the layers' segments, with the tangential part the same in all of them,
the layers add, in cycles,

    φ(k) = [k_n,med(t) − k_n,imm]·h + [k_n,cov(t) − k_n,imm]·d_cov − [k_z,cov − k_z,imm]·d_cov,design,

the last term the objective's own correction of the plate it was designed for.  The emitter's position
(x, y, z) stays in the microscope's axes, so the wave keeps its exp(2πi (−k_x x − k_y y + k_z z)) and the
layers' phase comes on top: at t = 0, φ = (k_z,med − k_z,imm)·h, the index mismatch of today's model, and
the coverslip's terms cancel.  With an oil objective (n_cov = n_imm) the coverslip is immersion and its
term is zero at every tilt: only the glass–sample interface's tilt matters, and only for emitters above it.
With a water or silicone objective the tilted 170 µm plate gives coma (and, at second order in t,
astigmatism) even for beads on the coverslip.

This departs from the depth phase k_z,med(t)·h + k_z,cov(t)·d_cov − k_z,imm·f written in the brief by
the k_n,imm terms: without them a tilt of 0.01 rad moves every emitter by d_cov·t = 1.7 µm (a linear phase
in k), which their positions would have to follow.  Here a tilt changes the PSF, not where it is.

**A tilt does not move the image.**  Seen through a tilted interface an emitter appears displaced: along the
interface's normal, by its apparent depth (+17 nm at 1 µm and 1° with oil), and through a tilted plate of
another index by much more (−730 nm per degree through 170 µm of glass with a water objective).  That is a
phase a·k_x + b·k_y, exactly what a lateral position does, so it is taken out of both terms (the
least-squares shift over the aperture, `coefficients`): the PSF stays centred where SMAPpy samples it, an
emitter's (x, y) is where its image is, and a step of the tilt does not drag every bead out of its region.
`aberrations` reports the shift that was taken out.

The interface's height above the focal plane is z₀ − t_x·x − t_y·y: a positive t_x lowers the coverslip
towards +x (`docs/uipsf/conventions.md`).  The interfaces stay parallel to each other, the emitter's h is
measured along their normal, and where the coverslip lies over the field is not modelled: an emitter's h
and the focus are its own (a tilt of 0.005 rad over ±20 µm moves the coverslip by ±100 nm, which each
molecule's height takes up).  Fresnel coefficients and the apodisation stay those of the untilted layers:
they change at first order in t only by a slight imbalance of the aperture's edge.

The square roots take the branch +i·|·| below zero, as the grid's (`grid._sqrt`): light beyond the
critical angle of a layer decays in it.
"""
from __future__ import annotations

import torch


def _root(a: torch.Tensor, eps: float) -> torch.Tensor:
    """√a with the +i branch for a < 0, smoothed where |a| ≲ ``eps``: √(a + iε·exp(−a²/ε²)).

    The kink of √ at zero is the critical angle.  A tilt moves the critical ring through the pupil's
    samples, and each sample that crosses it puts a kink in the objective: from a level start an in situ fit
    of 24 molecules (24 pupil samples) stopped 28 nats above the optimum, the objective rough along the way,
    and at 48 samples a start at the truth stopped short too.  With ε a 64th of how far a moves between
    neighbouring samples at the aperture's edge both reached one optimum.  The smoothing must stay that
    small: near the critical angle the tilt's change of the root grows as 1/√a, and those samples carry most
    of the coma an oil objective sees (at 1°, 1 µm deep, 64 samples: −0.189 rad, the exact root's converged
    value −0.187; with ε 16 times larger, −0.179, and 64 times larger −0.005).  More than 3ε from zero the
    root is the exact one.
    """
    damp = eps * torch.exp(-(a / eps) ** 2)
    return torch.sqrt(torch.complex(a, damp))


def _without_shift(c: torch.Tensor, grid) -> torch.Tensor:
    """``c`` less its least-squares part a·k_x + b·k_y over the aperture: what a lateral shift does."""
    w = grid.aperture
    kx = (grid.kx[None, :] * w).to(c.dtype)
    ky = (grid.ky[:, None] * w).to(c.dtype)
    a = (c * kx).sum((-2, -1)) / (kx * kx).sum()
    b = (c * ky).sum((-2, -1)) / (ky * ky).sum()
    return c - a * grid.kx[None, :].to(c.dtype) - b * grid.ky[:, None].to(c.dtype)


def coefficients(grid, tilt: torch.Tensor | None, centred: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """(c_h, c_0), [P, P] complex, cycles: the wave's phase per nm of height in the sample,
    ``k_z,imm + [k_n,med(t) − k_n,imm]`` (the emitter's own k_z,imm·h and the layers'), and the coverslip's
    phase (module docstring).

    ``tilt`` (t_x, t_y) in radians, None for none.  Outside the aperture the untilted values are kept: the
    fields are zero there, and the immersion's own k_z, imaginary at the corners, stays bounded.

    ``centred``: the part of each that is a lateral shift, a·k_x + b·k_y fitted over the aperture, is taken
    out (module docstring).  At t = 0 there is none: both are even in k."""
    o = grid.optics
    c_h = grid.kz_med
    c_0 = ((grid.kz_cov - grid.kz_imm) * (o.d_cov_nm - o.d_cov_design_nm) if o.coverslip_matters
           else torch.zeros_like(grid.kz_imm))
    if tilt is None:
        return c_h, c_0
    # today's exact values and the change the tilt makes, the smoothed roots' at t less theirs at 0 (`_root`)
    t = tilt.to(grid.dtype)
    h_t, cov_t = _tilted(grid, t)
    h_0, cov_0 = _tilted(grid, torch.zeros_like(t))
    inside = grid.aperture > 0
    c_h = torch.where(inside, c_h + (h_t - h_0), c_h)
    if o.coverslip_matters:
        c_0 = torch.where(inside, c_0 + (cov_t - cov_0) * o.d_cov_nm, c_0)
        if centred:
            c_0 = _without_shift(c_0, grid)
    if centred:
        c_h = torch.where(inside, _without_shift(c_h, grid), grid.kz_med)
    return c_h, c_0


def _tilted(grid, t):
    """k_n,med(t) − k_n,imm(t) and k_n,cov(t) − k_n,imm(t), [P, P] complex, with the smoothed root."""
    o = grid.optics
    cdtype = grid.kz_imm.dtype
    kz = grid.kz_imm.real
    # the wave's component along the normal: the light travels down, k = (k_x, k_y, −k_z,imm), and the pupil
    # carries the conjugate of its phase (exp(−2πi k_x x + 2πi k_z z), `docs/uipsf/conventions.md`)
    kn = (kz - t[0] * grid.kx[None, :] - t[1] * grid.ky[:, None]) / torch.sqrt(1 + t[0] ** 2 + t[1] ** 2)
    lam2 = o.wavelength_nm ** 2
    # a 64th of how far k_n² moves between two samples at the aperture's edge (`_root`)
    eps = 4 * o.NA ** 2 / (lam2 * grid.size) / 64

    def k_normal(n):
        return _root((n * n - o.n_imm ** 2) / lam2 + kn * kn, eps) - kn.to(cdtype)

    return k_normal(o.n_med), k_normal(o.n_cov) if o.coverslip_matters else None


def derivatives(grid, tilt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(∂c_h/∂t, ∂c_0/∂t), [2, P, P] complex: what the solver's linearised Jacobian needs for the tilt
    (`ForwardModel.emitter_linearisation`).  Forward mode through the real and imaginary parts, so that
    no complex tangent is asked of any device."""
    t = tilt.detach().to(grid.dtype)

    def parts(v):
        c_h, c_0 = coefficients(grid, v)
        return torch.stack([c_h.real, c_h.imag, c_0.real, c_0.imag])

    eye = torch.eye(2, dtype=t.dtype, device=t.device)
    d = torch.stack([torch.func.jvp(parts, (t,), (eye[i],))[1] for i in range(2)])       # [2, 4, P, P]
    return torch.complex(d[:, 0], d[:, 1]), torch.complex(d[:, 2], d[:, 3])


def layered(grid, p: dict) -> bool:
    """Whether the PSF needs this module's phase: a tilt among the parameters ``p``, or a coverslip off its
    design thickness.  Otherwise the wave is today's, exactly."""
    return "geometry.tilt" in p or grid.optics.layered


def aberrations(optics, tilt_rad, depth_nm: float = 0.0, modes=range(5, 38), size: int = 128):
    """({Noll j: rad}, (s_x, s_y) nm): the phase a tilt ``tilt_rad`` (t_x, t_y) adds for an emitter
    ``depth_nm`` above the coverslip, in Noll-normalised Zernikes (RMS radians over the aperture) and the
    lateral shift of the image, the one the model takes out (`coefficients`).

    Fitted by least squares over the propagating part of the aperture, together with what the emitter's own
    position takes up: piston, the shifts (k_x, k_y) and the exact high-NA defocus k_z,imm.  Beyond the
    critical angle the tilt changes how fast the light decays, a magnitude, not a phase.  What a tilt does,
    for `docs/uipsf/architecture.md` and `benchmarks/uipsf/synthetic/tilt_aberrations.py`."""
    import math

    import numpy as np

    from .grid import PupilGrid
    from .zernike import zernike

    g = PupilGrid.build(optics, size, device="cpu", dtype=torch.float64)
    c_h, c_0 = coefficients(g, torch.as_tensor(tilt_rad, dtype=torch.float64), centred=False)
    c_h0, c_00 = coefficients(g, torch.zeros(2, dtype=torch.float64), centred=False)
    dphi = (2 * math.pi * ((c_h - c_h0) * depth_nm + (c_0 - c_00))).real.numpy()
    rho, phi = g.rho.numpy(), g.phi.numpy()
    sub = (rho < 1) & (rho * optics.NA < min(optics.n_med, optics.n_cov))
    modes = list(modes)
    columns = [np.ones_like(rho), g.kx.numpy()[None, :] + 0 * rho, g.ky.numpy()[:, None] + 0 * rho,
               g.kz_imm.real.numpy()] + [zernike(j, rho, phi) for j in modes]
    A = np.stack([c[sub] for c in columns], 1)
    coef, *_ = np.linalg.lstsq(A, dphi[sub], rcond=None)
    # a pupil phase 2π k_x·s moves the image by −s (the DFT's exp(2πi k_x (x − x₀)), conventions)
    shift = (-coef[1] / (2 * math.pi), -coef[2] / (2 * math.pi))
    return {j: float(c) for j, c in zip(modes, coef[4:], strict=True)}, (float(shift[0]), float(shift[1]))
