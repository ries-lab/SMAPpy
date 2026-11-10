"""The two check figures of a training's preparation (milestone (e) of docs/decode/poc-plan.md).

Before hours of training, two things are worth a look: whether the frames the training will
simulate look like the data, and whether the ranges it draws from cover what the prefit found.
Both are module-level functions on arrays, so the page draws them on simulated data and the Train
window on the user's.  `draw_ranges` is torch-free; `simulate_frames` runs DECODE's simulation.
"""
from __future__ import annotations

import numpy as np


def photons(frames, camera_section: dict) -> np.ndarray:
    """Camera counts to photons as DECODE's ``Photon2Camera.backward`` converts them."""
    c = camera_section
    out = (np.asarray(frames, np.float64) - c["baseline"]) * c["e_per_adu"]
    if c.get("em_gain"):
        out /= c["em_gain"]
    return (out - c.get("spur_noise", 0.0)) / c.get("qe", 1.0)


def simulate_frames(method_params: dict, calibration, n_frames: int = 8, seed: int = 0) -> np.ndarray:
    """Frames as the training simulates them, in photons: ``(n, rows, columns)``.

    ``method_params`` is a job's ``method.yaml``; ``calibration`` the calibration's path or a loaded
    one, which replaces the file's relative ``calibration_file``.
    """
    import torch

    from .simulation import background, camera, emitter_generator, simulator, structure_prior
    from .utils import param_io
    from .utils.calibration_io import spline_psf
    from .utils.types import RecursiveNamespace

    full = param_io.autofill_dict(method_params, param_io.load_reference())
    param = param_io.autoset_scaling(RecursiveNamespace(**full))
    sim = param.Simulation
    torch.manual_seed(seed)
    psf = spline_psf(calibration, xextent=sim.psf_extent[0], yextent=sim.psf_extent[1], img_shape=sim.img_size,
                     device="cpu", roi_size=sim.roi_size, roi_auto_center=sim.roi_auto_center)
    sampler = emitter_generator.EmitterSamplerBlinking.parse(
        param, structure=structure_prior.RandomStructure.parse(param), frames=(0, n_frames - 1))
    noise = camera.Photon2Camera(qe=param.Camera.qe, spur_noise=param.Camera.spur_noise,
                                 em_gain=param.Camera.em_gain, e_per_adu=param.Camera.e_per_adu,
                                 baseline=param.Camera.baseline, read_sigma=param.Camera.read_sigma,
                                 photon_units=True, device="cpu")
    s = simulator.Simulation(psf=psf, em_sampler=sampler, background=background.UniformBackground.parse(param),
                             noise=noise, frame_range=(0, n_frames - 1))
    _, frames, _ = s.sample()
    return frames.numpy()


def draw_frames(axes, real_photons, simulated_photons, size: int = 40):
    """A real frame's corner beside a simulated frame, on one scale, and their pixel values.

    ``axes`` three axes; the frames in photons.  Where the two look alike -- the spots as bright
    and as wide, the background as grey and as grainy -- the simulation matches the data.
    """
    real = np.asarray(real_photons)[..., :size, :size]
    sim = np.asarray(simulated_photons)[..., :size, :size]
    lo, hi = np.percentile(np.concatenate([real.ravel(), sim.ravel()]), [1, 99.5])
    for ax, img, title in ((axes[0], real, "data"), (axes[1], sim, "simulated")):
        ax.imshow(img[0] if img.ndim == 3 else img, vmin=lo, vmax=hi, cmap="gray")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    bins = np.linspace(lo, max(hi, np.percentile(real, 99.9)), 60)
    axes[2].hist(real.ravel(), bins, histtype="step", density=True, label="data")
    axes[2].hist(sim.ravel(), bins, histtype="step", density=True, label="simulated")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("photons a pixel")
    axes[2].legend()


def draw_ranges(axes, localizations: dict, method_params: dict):
    """The prefit's photons, background and z, with what the training draws over them.

    ``localizations`` the prefit's columns (``photons``, ``background``, optional ``z_nm``), the
    photons best those of the fully-on frames (`decode_smlm.prefit.fully_on`): the training draws
    a molecule's photons a frame while it is on, from a normal distribution, and its frames of
    switching on and off get less.  ``method_params`` is the job's ``method.yaml``.  Background
    and z are drawn uniformly.  A range that misses a large part of the data, or spans far beyond
    it, is worth changing.
    """
    sim = method_params["Simulation"]
    mu, sd = sim["intensity_mu_sig"]
    ph = np.asarray(localizations["photons"], np.float64)
    top = max(np.percentile(ph, 99), mu + 4 * sd)
    bins = np.linspace(0, top, 60)
    axes[0].hist(ph, bins, density=True, histtype="stepfilled", alpha=0.4, label="prefit")
    x = np.linspace(0, top, 300)
    axes[0].plot(x, np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), label="training, while on")
    axes[0].set_xlabel("photons")
    axes[0].legend()
    _uniform(axes[1], np.asarray(localizations["background"], np.float64), sim["bg_uniform"], "background (photons a pixel)")
    z = localizations.get("z_nm")
    zr = sim["emitter_extent"][2]
    if z is not None:
        _uniform(axes[2], np.asarray(z, np.float64), zr, "z (nm)")
    else:
        axes[2].axvspan(*zr, alpha=0.2)
        axes[2].set_xlabel("z (nm): no z in the prefit")


def _uniform(ax, values, rng, label):
    lo, hi = float(rng[0]), float(rng[1])
    span = hi - lo
    bins = np.linspace(min(lo, np.percentile(values, 0.5)) - 0.05 * span,
                       max(hi, np.percentile(values, 99.5)) + 0.05 * span, 60)
    ax.hist(values, bins, density=True, histtype="stepfilled", alpha=0.4, label="prefit")
    ax.plot([lo, lo, hi, hi], [0, 1 / span, 1 / span, 0], label="training")
    ax.set_xlabel(label)
    ax.legend()
