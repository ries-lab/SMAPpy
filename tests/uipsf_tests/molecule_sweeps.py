"""Blinking molecules on the coverslip imaged through repeated z sweeps, simulated with a known pupil and written
as OME-TIFF files whose planes carry the stage's z (``PositionZ``): what `smappy.calibrate.molecule_stack`
reads.  Each molecule is on in one frame; the frames are the model's expected images (`uipsf.export.sample_psf`)
over a flat background, Poisson, in ADU on the camera's offset."""
from __future__ import annotations

import numpy as np
from smappy.camera import CameraMetadata

CAMERA = CameraMetadata(conversion=1.0, offset=100.0, pixelsize_um=0.11, read_noise_e=1.0)
#: the phase (nm of wavefront, Noll) of the simulated pupil: astigmatism, coma and spherical
PHASE_NM = {5: -25.0, 6: 60.0, 7: 8.0, 11: 40.0}


def truth_model(optics, height_nm: float = 10.0, gamma: float = 0.5, sigma_nm: float = 0.0, order: int = 4,
                roi: int = 25):
    """(model, shared values): a minimal pupil of ``order`` with `PHASE_NM`, γ, σ and the molecules at
    ``height_nm`` above the glass, as `molecule_stack_model` builds it for one sweep."""
    import torch
    from smappy.calibrate.pupil import PSFModelSettings

    from uipsf.params import Params
    from uipsf.simulate import template
    from uipsf.workflows.calibrate import molecule_stack_model
    spec = PSFModelSettings(zernike_order=order, magnitude="minimal", blur="round", blur_nm=max(sigma_nm, 0.0))
    model = molecule_stack_model(optics, spec, (roi, roi), 1, [0.0], (0.0, 0.0), device="cpu",
                                 dtype=torch.float64)
    b = template(1, torch.zeros(1, 1, dtype=torch.float64), (roi, roi), dtype=torch.float64)
    import dataclasses
    b = dataclasses.replace(b, stack_id=torch.zeros(1, dtype=torch.long), field_xy=torch.zeros(1, 2))
    pupil = model.psf.pupil
    phase = torch.tensor([2 * np.pi * PHASE_NM.get(int(j), 0.0) / optics.wavelength_nm for j in pupil.phase_modes],
                         dtype=torch.float64)
    values = {"pupil.phase": phase, "pupil.gamma": torch.tensor(gamma, dtype=torch.float64),
              "psf.sigma": torch.tensor([sigma_nm], dtype=torch.float64),
              "geometry.height": torch.tensor(height_nm, dtype=torch.float64)}
    p = Params(model.specs(b, {"xyz": torch.zeros(1, 3), "photons": torch.ones(1), "background": torch.ones(1)}),
               1, dtype=torch.float64, values=values).physical()
    shared = {k: v for k, v in p.items() if not k.startswith("emitter.")}
    return model, shared


def write_sweeps(folder, optics, files: int = 2, sweeps: int = 3, step_nm: float = 100.0, span_nm: float = 1000.0,
                 per_frame: int = 6, size: int = 128, photons=(3000.0, 6000.0), background: float = 20.0,
                 focus_um=(50.0, 50.2), sweep_jitter_nm: float = 20.0, plane=(0.0, 0.0), height_nm: float = 10.0,
                 gamma: float = 0.5, seed: int = 0, up_down: bool = False):
    """``files`` OME-TIFFs in ``folder``, each ``sweeps`` sweeps of the stage over ±``span_nm`` about its file's
    focus ``focus_um`` (each sweep's own off by up to ``sweep_jitter_nm``), ``per_frame`` molecules on in
    each frame at random places away from the edges; alternately up and down with ``up_down``.  The focal
    plane's slope ``plane`` (nm per µm, x and y).  Returns the truth: the paths, each sweep's focus (nm,
    relative to its file's), the model and its shared values."""
    from pathlib import Path

    import tifffile
    import torch

    from uipsf.export import sample_psf
    rng = np.random.default_rng(seed)
    model, shared = truth_model(optics, height_nm, gamma)
    roi = 25
    h = roi // 2
    px = np.asarray(optics.pixel_size_nm, float)
    stage = np.arange(-span_nm, span_nm + 1, step_nm)
    paths, foci = [], []
    for k in range(files):
        frames, z_um = [], []
        for s in range(sweeps):
            jitter = float(rng.uniform(-sweep_jitter_nm, sweep_jitter_nm))
            foci.append(jitter)
            steps = stage if not (up_down and s % 2) else stage[::-1]
            for zk in steps:
                image = np.full((size, size), background)
                for _ in range(per_frame):
                    x, y = rng.uniform(h + 2, size - h - 3, 2)
                    u, v = (x - (size - 1) / 2) * px[0] / 1000, (y - (size - 1) / 2) * px[1] / 1000
                    # the stage above focus by d leaves the molecule d below, SMAPpy's z −d; the sweep's focus and
                    # the plane in `CoverslipStackGeometry`'s sign: z = −zk − f − (a·u + b·v)
                    d = zk + jitter + (plane[0] * u + plane[1] * v)
                    xi, yi = int(round(x)), int(round(y))
                    xyz = torch.tensor([[(x - xi) * px[0], (y - yi) * px[1], -d]], dtype=torch.float64)
                    spot = sample_psf(model, shared, xyz, roi, torch.tensor([height_nm], dtype=torch.float64))[0]
                    image[yi - h:yi + h + 1, xi - h:xi + h + 1] += rng.uniform(*photons) * spot.numpy()
                frames.append(rng.poisson(image) / CAMERA.conversion + CAMERA.offset)
                z_um.append(focus_um[k % len(focus_um)] + zk / 1000)
        path = Path(folder) / f"file{k}_sweeps.ome.tif"        # not x_1.ome.tif: one series of files
        planes = {"PositionZ": [float(z) for z in z_um], "PositionZUnit": ["µm"] * len(z_um)}
        tifffile.imwrite(path, np.asarray(frames, np.uint16)[:, None, None], ome=True,
                         metadata={"axes": "TZCYX", "Plane": planes})
        paths.append(str(path))
    return dict(paths=paths, foci_nm=np.asarray(foci), model=model, shared=shared, height_nm=height_nm, gamma=gamma)


#: the simulated split chip: two halves side by side, the secondary shifted by (3, −2) px, unmirrored
DUAL_SHIFT_PX = (3.0, -2.0)


def dual_transform(height: int = 128, width: int = 128):
    """The `dual.ChannelTransform` of `write_dual_sweeps`' chip: secondary (right) chip pixels onto the main
    (left) half's."""
    from smappy.calibrate.dual import ChannelTransform
    H = np.array([[1.0, 0.0, -width - DUAL_SHIFT_PX[0]], [0.0, 1.0, -DUAL_SHIFT_PX[1]], [0.0, 0.0, 1.0]])
    geometry = {"image_shape": [height, 2 * width], "coordinate_system": "roi-local", "split_position": width,
                "layout": "right-left", "main_channel": "left", "mirror_axis_xy": None, "sources": []}
    return ChannelTransform(H, geometry, {})


def write_dual_sweeps(folder, optics, sweeps: int = 2, step_nm: float = 100.0, span_nm: float = 700.0,
                      per_frame: int = 5, size: int = 128, share: float = 0.6, background: float = 20.0,
                      focus_um: float = 50.0, sweep_jitter_nm: float = 20.0, height_nm: float = 10.0,
                      gamma: float = 0.5, seed: int = 0):
    """One OME-TIFF of a split chip, [T, size, 2·size]: each molecule in the left half and, ``share`` of its
    light the other way round, in the right one where `dual_transform` puts it, the same pupil in both
    (`truth_model`).  Returns the truth as `write_sweeps` does, with the transform."""
    from pathlib import Path

    import tifffile
    import torch

    from uipsf.export import sample_psf
    rng = np.random.default_rng(seed)
    model, shared = truth_model(optics, height_nm, gamma)
    transform = dual_transform(size, size)
    roi, h = 25, 12
    px = np.asarray(optics.pixel_size_nm, float)
    frames, z_um, foci = [], [], []
    for _ in range(sweeps):
        jitter = float(rng.uniform(-sweep_jitter_nm, sweep_jitter_nm))
        foci.append(jitter)
        for zk in np.arange(-span_nm, span_nm + 1, step_nm):
            image = np.full((size, 2 * size), background)
            for _ in range(per_frame):
                x, y = rng.uniform(h + 6, size - h - 6, 2)
                d = zk + jitter
                N = rng.uniform(4000.0, 8000.0)
                for at, part in (((x, y), share), (tuple(transform.to_secondary(np.array([[x, y]]))[0]), 1 - share)):
                    xi, yi = int(round(at[0])), int(round(at[1]))
                    xyz = torch.tensor([[(at[0] - xi) * px[0], (at[1] - yi) * px[1], -d]], dtype=torch.float64)
                    spot = sample_psf(model, shared, xyz, roi, torch.tensor([height_nm], dtype=torch.float64))[0]
                    image[yi - h:yi + h + 1, xi - h:xi + h + 1] += N * part * spot.numpy()
            frames.append(rng.poisson(image) / CAMERA.conversion + CAMERA.offset)
            z_um.append(focus_um + zk / 1000)
    path = Path(folder) / "dual_sweeps.ome.tif"
    tifffile.imwrite(path, np.asarray(frames, np.uint16)[:, None, None], ome=True,
                     metadata={"axes": "TZCYX", "Plane": {"PositionZ": [float(z) for z in z_um],
                                                          "PositionZUnit": ["µm"] * len(z_um)}})
    return dict(paths=[str(path)], foci_nm=np.asarray(foci), model=model, shared=shared, height_nm=height_nm,
                gamma=gamma, transform=transform)
