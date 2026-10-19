"""The batch the model and the solver read (`data/dataset.py`); its making is smappy's (test_bead_rois.py)."""


def test_thinning_keeps_the_largest_multiple_of_the_step_not_above_the_maximum_and_the_focus():
    import torch

    from uipsf import Batch
    z = Batch.z_from_objective(torch.arange(-1000.0, 1001.0, 20.0))              # 101 slices, 20 nm
    data = torch.arange(101.0)[None, None, :, None, None].expand(2, 1, 101, 3, 3)
    b = Batch(data=data, z_nm=z, mask=torch.ones(2, 1, 101, 3, 3, dtype=torch.bool))
    for max_dz, step in ((10.0, 20.0), (20.0, 20.0), (50.0, 40.0), (59.9, 40.0), (60.0, 60.0), (100.0, 100.0)):
        t = b.thinned(max_dz)
        assert abs(t.z_step_nm - step) < 1e-4, max_dz
        assert 0.0 in t.z_nm.tolist()                                           # the focus is kept
        assert torch.equal(t.data[0, 0, :, 0, 0], -t.z_nm / 20 + 50)            # data and z stay paired
        assert t.mask.shape[2] == t.data.shape[2]
    assert b.thinned(None) is b
