"""Learning two channels from simulated beads (plan §9 M4): the transform, the photon share, the defocus
offset and both pupils come back within their CRLBs, from the start a pairing gives.

The transform starts where a pairing puts it (`smappy.calibrate.pupil.dual_rois`): a projective fit to the
beads' centroids, here 0.37 px off.  Measured on the biplane beads: a start 2.2 px off by a translation,
or 1.4 px at the field's edge by a 0.4° rotation, reaches the same optimum; a translation alone, the
rotation, scale and perspective all left out (~3 px at the edge), stopped in a wrong minimum, the
secondary pupil bent to take the misplacement (docs/uipsf/architecture.md, M4)."""
import pytest
import torch

import uipsf
from uipsf.workflows import learn_beads
from uipsf_tests.conftest import two_channel_beads


def _from_centroids(batch, pixel_nm):
    """The start a pairing gives (`smappy.calibrate.dual`): a projective fit to the beads' centroids in each channel
    (the background-free mean of the central slices), in chip pixels."""
    from smappy.calibrate.dual import fit_projective
    d = batch.data.double()
    K = d.shape[2]
    m = d[:, :, K // 2 - 2:K // 2 + 3].mean(2)                          # [E, C, Y, X]
    m = (m - m.flatten(2).median(2).values[..., None, None]).clamp(min=0)
    Y, X = m.shape[-2:]
    gx, gy = torch.arange(X, dtype=d.dtype) - (X - 1) / 2, torch.arange(Y, dtype=d.dtype) - (Y - 1) / 2
    w = m.sum((-2, -1))
    xy = torch.stack([(m.sum(-2) * gx).sum(-1) / w, (m.sum(-1) * gy).sum(-1) / w], -1)
    chip = (batch.roi_xy.double() + xy).numpy()                         # [E, C, 2]
    h = fit_projective(chip[:, 1], chip[:, 0])
    return h / h[2, 2]                         # smappy's fit leaves the scale free; H₂₂ = 1 here


@pytest.mark.slow
@pytest.mark.parametrize("kind, mirrored, n_max", [("biplane", False, 4), ("ratiometric", True, 6)])
def test_two_channels_are_learned_to_their_crlbs_from_a_pairings_transform(kind, mirrored, n_max):
    optics = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))
    model, truth, batch = two_channel_beads(optics, n=12, kind=kind, dtype=torch.float32, P=48, roi=(17, 17),
                                            K=21, z_range=1000.0, n_max=n_max, photons=20000.0, mirrored=mirrored)
    tp = truth.physical()
    model.channels[1].transform = _from_centroids(batch, optics.pixel_size_nm)
    model.channels[1].share = model.channels[1].dz_nm = None       # from the data
    cal = learn_beads(model, batch)
    fp, sd = cal.result.params.physical(), cal.result.std
    assert len(cal.beads.used) == 12 and cal.atlas.flagged() == []
    for k in ("ch1.transform", "ch1.share", "ch1.dz", "pupil.phase", "ch1.pupil.phase", "pupil.magnitude",
              "ch1.pupil.magnitude", "psf.sigma", "ch1.psf.sigma"):
        z = ((fp[k] - tp[k]).double() / sd[k]).abs()
        assert z.max() < 4, (k, float(z.max()))
    # The transform's error where the beads are: a fraction of a nanometre.
    from uipsf.model.channels import map_points, unpack_transform
    main = batch.roi_xy[:, 0].double()
    moved = map_points(torch.linalg.inv(unpack_transform(fp["ch1.transform"].double())), main)
    true = map_points(torch.linalg.inv(unpack_transform(tp["ch1.transform"].double())), main)
    assert (moved - true).abs().max() * 120 < 2.0                  # nm


def _transform_error_px(H, H_true, main):
    from uipsf.model.channels import map_points
    a = map_points(torch.linalg.inv(torch.as_tensor(H, dtype=torch.float64)), main)
    b = map_points(torch.linalg.inv(torch.as_tensor(H_true, dtype=torch.float64)), main)
    return float((a - b).norm(dim=-1).max())


@pytest.mark.parametrize("kind, mirrored", [("biplane", False), ("ratiometric", True)])
def test_a_pairs_transform_starts_from_the_beads_gaussian_positions_closer_than_from_their_centroids(kind, mirrored):
    """Measured on five seeds of each (benchmarks/uipsf/real/starts.py): 0.11-0.25 px from the Gaussians,
    0.37-0.59 from the centroids, where a start 2.5 px off fails on mirrored pairs."""
    from uipsf.model.channels import unpack_transform
    from uipsf.workflows.calibrate import gaussian_transforms
    optics = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))
    model, truth, batch = two_channel_beads(optics, n=12, kind=kind, dtype=torch.float32, P=48, roi=(17, 17),
                                            K=21, z_range=1000.0, n_max=4, photons=20000.0, mirrored=mirrored)
    H_true = unpack_transform(truth.physical()["ch1.transform"].double())
    main = batch.roi_xy[:, 0].double()
    found = gaussian_transforms(model, batch)
    gauss = _transform_error_px(found[1], H_true, main)
    centroids = _transform_error_px(_from_centroids(batch, optics.pixel_size_nm), H_true, main)
    assert gauss < 0.3 and gauss < 0.6 * centroids
    # too few pairs to fix eight entries well: the pairing's transform stays
    assert gaussian_transforms(model, batch.subset(torch.arange(4))) == {}
