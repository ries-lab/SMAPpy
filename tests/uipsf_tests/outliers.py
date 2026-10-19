"""Simulated bead stacks with planted outliers, for the bead-rejection tests."""
import torch

import uipsf
from uipsf.simulate import parameters, simulate, template

OUTLIERS = {"double": 0, "jump": 1, "bright": 2, "saturated": 3}


def bead_set(n_good=12, seed=0, outliers=True, P=32, roi=(15, 15), K=21, photons=(2000.0, 2500.0)):
    """(model, batch, truth values): ``n_good`` matched beads after the planted outliers (if any)."""
    optics = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))
    grid = uipsf.PupilGrid.build(optics, P, device="cpu")
    pupil = uipsf.ZernikePupil(grid, n_max=4)
    model = uipsf.ForwardModel(uipsf.VectorPSF(grid, pupil, roi, sigma_nm=(30.0, 30.0)))
    g = torch.Generator().manual_seed(seed)
    n_out = len(OUTLIERS) if outliers else 0
    E = n_good + n_out
    phase = torch.zeros(len(pupil.phase_modes))
    for j, v in {6: 0.5, 7: 0.15, 11: 0.2}.items():
        phase[pupil.phase_modes.index(j)] = v
    xyz = (torch.rand(E, 3, generator=g) - 0.5) * torch.tensor([150.0, 150.0, 200.0])
    N = photons[0] + (photons[1] - photons[0]) * torch.rand(E, generator=g)
    if outliers:
        N[OUTLIERS["bright"]] *= 3
    values = {"pupil.phase": phase, "psf.sigma": torch.tensor([40.0, 35.0]), "emitter.xyz": xyz,
              "emitter.photons": N, "emitter.background": torch.full((E,), 20.0)}
    z = uipsf.Batch.z_from_objective(torch.linspace(-500, 500, K))
    b0 = template(E, z, roi)
    batch = simulate(model, parameters(model, b0, values), b0, seed=seed + 1)
    if outliers:
        data = batch.data.clone()
        # a second, dimmer bead 4 pixels away inside the first one's ROI
        v2 = dict(values, **{"emitter.xyz": xyz + torch.tensor([4 * 127.0, 1 * 116.0, 0.0]),
                             "emitter.photons": 0.6 * N, "emitter.background": torch.zeros(E)})
        extra = simulate(model, parameters(model, b0, v2), b0, seed=seed + 2).data
        data[OUTLIERS["double"]] += extra[OUTLIERS["double"]]
        # the stage jumps by 300 nm halfway through this bead's stack
        v3 = dict(values, **{"emitter.xyz": xyz + torch.tensor([0.0, 0.0, 300.0])})
        jumped = simulate(model, parameters(model, b0, v3), b0, seed=seed + 3).data
        data[OUTLIERS["jump"], :, K // 2:] = jumped[OUTLIERS["jump"], :, K // 2:]
        mask = torch.ones_like(data, dtype=torch.bool)
        e = OUTLIERS["saturated"]
        brightest = data[e].flatten().topk(int(0.03 * data[e].numel())).indices
        m = mask[e].flatten()
        m[brightest] = False
        mask[e] = m.reshape(mask[e].shape)
        batch = uipsf.Batch(data=data, z_nm=batch.z_nm, s=batch.s, mask=mask)
    return model, batch, values
