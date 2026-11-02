"""The minimal pupil on real 4Pi beads, against the symmetric Zernikes the 4Pi form had (docs/4pi.md §23).

    python benchmarks/uipsf/real/zenodo_4pi_minimal.py [--device mps] [--variants old,minimal,...]

The beads are uiPSF v1's 18 of Zenodo `4pi_40nm_bead` (`zenodo_4pi_beads.v1_batch`: 4 channels, 3 mirror
steps, 21 slices 50 nm apart), learnt through the window's own path (`calibrate.fourpi_beads.learn`), vectorial,
order 6, a free photon count per observation, point-like beads.  The variants (`VARIANTS`):

- **old**: the form's default before: symmetric Zernikes (Noll 4, 11, 22) per objective, σ round from 60 nm,
  no pixel blur;
- **minimal**: the form's default now: cos θ^γ, one γ for both objectives (*same magnitude*), σ round over the
  pixel's blur;
- **minimal-each**: γ per objective;
- **symmetric-shared**: symmetric Zernikes shared, σ over the pixel's blur -- the magnitude's form alone
  against *minimal*.

Each is exported (`workflows.fourpi.calibration`) and fits the beads back with Spline 4Pi's `Global4PiPSF`
(`bead_z`): z of the envelope and of the phase against the stage, their scatter against the CRLB, the
envelope's bias against the phase's z, and *4Pi z*'s fringe choice (`smappy.fourpi.unwrap`).  One JSON line per
variant to ``--out``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from zenodo_4pi_beads import STEP_NM, v1_batch  # noqa: E402

CROP = 15
CORE = slice(3, -3)
VARIANTS = {
    "old": dict(magnitude="symmetric", blur_nm=60.0, pixel_blur=False, same=False),
    "minimal": dict(magnitude="minimal", blur_nm=20.0, pixel_blur=True, same=True),
    "minimal-each": dict(magnitude="minimal", blur_nm=20.0, pixel_blur=True, same=False),
    "symmetric-shared": dict(magnitude="symmetric", blur_nm=20.0, pixel_blur=True, same=True),
    # the minimal pupil with its aperture learnt (`EffectiveNA`), one for both objectives
    "minimal-na": dict(magnitude="minimal", blur_nm=20.0, pixel_blur=True, same=True, learn_na=True),
}


def effective_na_pupil():
    """`MinimalPupil` with its aperture learnt: ``pupil.na_ratio``, the effective NA over the nominal one, in the
    cutoff's place; with *same magnitude* one for both objectives.  A prototype for the question it answers
    (docs/4pi.md §23), patched into the builder for the run."""
    import torch

    from uipsf.model import fourpi as fourpi_model
    from uipsf.model.pupil import MinimalPupil
    from uipsf.params import Bounded, ParamSpec

    class EffectiveNA(MinimalPupil):
        def specs(self):
            return super().specs() + [ParamSpec("pupil.na_ratio", (), init=0.98, transform=Bounded(0.8, 1.0))]

        def symmetric(self, p, cos_imm=None, kappa=None, cutoff=None):
            if cutoff is None and "pupil.na_ratio" in p:
                rho = self.grid.rho.real if self.grid.rho.is_complex() else self.grid.rho
                width = max(0.01, 1.0 / self.grid.size)
                cutoff = torch.sigmoid((p["pupil.na_ratio"][..., None, None] * self.na / self.grid_na - rho) / width)
            return super().symmetric(p, cos_imm, kappa, cutoff)

        def truncated(self, n_max):
            out = super().truncated(n_max)
            out.__class__ = EffectiveNA
            return out

        def without_ring(self):
            out = super().without_ring()
            out.__class__ = EffectiveNA
            return out

    fourpi_model.MAGNITUDE = fourpi_model.MAGNITUDE + ("pupil.na_ratio",)
    return EffectiveNA


def v1_beads():
    """v1's ROIs as the window's `FourPiBeads` (bead, channel, slice, mirror step, y, x)."""
    from smappy.calibrate.fourpi_beads import FourPiBeads
    batch, data = v1_batch()                                     # data (C, E, M, J, Y, X)
    C, E, M, J, Y, X = data.shape
    z = -(np.arange(J) - (J - 1) / 2) * STEP_NM
    return FourPiBeads(np.transpose(data, (1, 0, 3, 2, 4, 5)).astype(np.float32),
                       batch.roi_xy.numpy().astype(float), z), data


def bead_z(cal, data) -> dict:
    """Every frame fitted with ``cal`` (`Global4PiPSF`, the model's maps not applied: v1's ROIs are cut at its own
    registration); per bead and mirror step along the scan's middle (`CORE`):

    - **phase z**: the phase unwrapped along the slices over k, a line against the stage -- its slope (±1 when the
      model's period is the beads') and the rms about it, against the CRLB |σ_φ / k|;
    - **envelope z**: the same, against its CRLB;
    - **the envelope's bias**: z_env − phase z, each bead's offset removed, the median over beads per slice: its rms
      and its largest value, the bend a model's envelope has against its own fringe;
    - **4Pi z** (`smappy.fourpi.unwrap`, the *4Pi z* step's): the cavity phase per bead and step as the plugin
      measures it per time window, each frame's fringe chosen by its envelope; wrong where it is not the bead's
      majority fringe."""
    from smappy import fourpi
    from smappy.psf import Global4PiPSF
    C, E, M, J, Y, X = data.shape
    o = (Y - CROP) // 2
    rois = np.moveaxis(data[..., o:o + CROP, o:o + CROP], 0, 3).reshape(-1, C, CROP, CROP)
    model = Global4PiPSF(cal)
    v = model.unpack(model.fit(np.ascontiguousarray(rois, np.float32), iterations=100))
    get = lambda name: np.asarray(v[name], float).reshape(E, M, J)[..., CORE]  # noqa: E731
    z_env, z_err, phase, phase_err = get("z_nm"), get("z_err_nm"), get("phase_rad"), get("phase_err_rad")
    k = float(cal.phase_per_nm)
    period = 2 * np.pi / abs(k)
    stage = (-(np.arange(J) - (J - 1) / 2) * STEP_NM)[CORE]
    zp = np.unwrap(phase, axis=-1) / k
    out = {"phase_slope": [], "phase_rms": [], "env_slope": [], "env_rms": []}
    for e in range(E):
        for m in range(M):
            for name, z in (("phase", zp[e, m]), ("env", z_env[e, m])):
                line = np.polyfit(stage, z, 1)
                out[f"{name}_slope"].append(line[0])
                out[f"{name}_rms"].append(np.sqrt(np.mean((z - np.polyval(line, stage)) ** 2)))
    bias = z_env - zp
    bias -= np.median(bias, axis=-1, keepdims=True)
    bend = np.median(bias.reshape(E * M, -1), axis=0)
    # 4Pi z: the cavity phase of each bead and step (the plugin's per time window: here one window each)
    w = 1 / (phase_err ** 2 + (k * z_err) ** 2)
    phase0 = np.angle((w * np.exp(1j * (phase - k * z_env))).sum(-1, keepdims=True))
    un = fourpi.unwrap(phase.ravel(), phase_err.ravel(), z_env.ravel(), z_err.ravel(),
                       np.broadcast_to(phase0, phase.shape).ravel(), k)
    fringe = np.rint((un.z_nm.reshape(phase.shape) - zp) / period).astype(int)
    wrong = np.mean([np.mean(f != np.bincount(f - f.min()).argmax() + f.min()) for f in fringe.reshape(E * M, -1)])
    med = lambda a: float(np.median(a))  # noqa: E731
    return {"period_nm": float(cal.period_nm), "phase_slope": med(out["phase_slope"]),
            "phase_z_rms_nm": med(out["phase_rms"]), "phase_z_crlb_nm": med(phase_err / abs(k)),
            "env_slope": med(out["env_slope"]), "env_z_rms_nm": med(out["env_rms"]), "env_z_crlb_nm": med(z_err),
            "env_bias_rms_nm": float(np.sqrt(np.mean(bend ** 2))), "env_bias_max_nm": float(np.abs(bend).max()),
            "fringe_wrong": float(wrong), "fringe_probability": float(np.nanmedian(un.probability)),
            "envelope_scale": float(un.envelope_scale), "photons": med(np.asarray(v["photons"]))}


def run(name: str, beads, data, device, out_dir: Path, na: float = 1.35, bead_radius_nm: float = 0.0) -> dict:
    from smappy import fourpi
    from smappy.calibrate.fourpi_beads import FourPiBeadSettings, learn

    from uipsf.workflows.fourpi import calibration
    choice = VARIANTS[name]
    import uipsf.model.build as build
    from uipsf.model.pupil import MinimalPupil
    build.MinimalPupil = effective_na_pupil() if choice.get("learn_na") else MinimalPupil
    spec = replace(FourPiBeadSettings().psf, magnitude=choice["magnitude"], blur_nm=choice["blur_nm"],
                   pixel_blur=choice["pixel_blur"])
    choice = {k: v for k, v in choice.items() if k != "learn_na"}
    t = time.time()
    fit = learn(beads, NA=na, bead_radius_nm=bead_radius_nm, wavelength_nm=600, n_imm=1.406, pixel_size_nm=129,
                steps_rad=(2.0, 0.0, -2.0), spec=spec, device=device, same_magnitude=choice["same"],
                progress=lambda e: print(name, e.get("stage"), e.get("iteration"), e.get("objective"), flush=True)
                if e.get("iteration", 0) % 10 == 0 else None)
    seconds = time.time() - t
    shared = {k: v.detach().cpu().double().numpy() for k, v in fit.shared().items()}
    std = {k: v.detach().cpu().double().numpy() for k, v in fit.result.std.items() if k in shared}
    n_shared = int(sum(np.size(v) for k, v in shared.items() if not fit.result.params.specs[k].fixed))
    cal = calibration(fit, z_range_nm=600, dz_nm=20, roi_px=21)
    fourpi.save_fourpi_calibration(out_dir / f"fourpi_minimal_{name}_na{na:g}_r{bead_radius_nm:g}.h5", cal,
                                   overwrite=True)
    pick = lambda key: shared[key].tolist() if key in shared else None  # noqa: E731
    row = {"variant": name, **choice, "NA": na, "bead_radius_nm": bead_radius_nm,
           "objective": float(fit.result.objective), "shared_parameters": n_shared,
           "iterations": len(fit.result.history), "stop": fit.result.stop_reason, "seconds": round(seconds, 1),
           "na_ratio": pick("pupil.na_ratio"), "gamma": pick("pupil.gamma"), "lo_gamma": pick("lo.pupil.gamma"),
           "gamma_std": std["pupil.gamma"].tolist() if "pupil.gamma" in std else None,
           "magnitude": pick("pupil.magnitude"), "lo_magnitude": pick("lo.pupil.magnitude"),
           "sigma_nm": pick("psf.sigma"), "sigma_std": std["psf.sigma"].tolist(),
           "offset_nm": pick("fourpi.offset"), "steps_rad": pick("fourpi.steps"), "deltas_rad": pick("fourpi.delta"),
           "coherence": pick("fourpi.coherence"), "balance": pick("fourpi.balance"),
           "modulation": cal.parameters["modulation"], "pupils": cal.parameters["pupils"],
           "fit": bead_z(cal, data)}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--na", default="1.35", help="the objective's NA, comma-separated for a scan: the minimal "
                                                 "pupil's aperture (an effective NA)")
    ap.add_argument("--bead-radius", type=float, default=0.0)
    ap.add_argument("--out", default=str(Path(__file__).parents[1] / "results" / "fourpi_minimal.jsonl"))
    ap.add_argument("--models", default=str(Path.home() / ".cache/uipsf/fourpi_minimal"))
    a = ap.parse_args()
    models = Path(a.models)
    models.mkdir(parents=True, exist_ok=True)
    beads, data = v1_beads()
    for name, na in ((n, float(v)) for n in a.variants.split(",") for v in a.na.split(",")):
        row = run(name, beads, data, a.device, models, na, a.bead_radius)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "pupils"}), flush=True)


if __name__ == "__main__":
    main()
