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


def _stack_2um():
    import torch

    from uipsf import Batch
    z = Batch.z_from_objective(torch.arange(-2000.0, 2001.0, 20.0))              # 201 slices, 20 nm, ±2 µm
    data = torch.arange(201.0)[None, None, :, None, None].expand(2, 1, 201, 3, 3)
    return Batch(data=data, z_nm=z, mask=torch.ones(2, 1, 201, 3, 3, dtype=torch.bool))


def test_a_z_range_keeps_the_slices_within_it_of_the_focus():
    b = _stack_2um()
    t = b.within(1000.0)
    assert t.z_nm.abs().max() == 1000.0 and t.data.shape[2] == 101 == t.mask.shape[2]
    assert t.z_nm.tolist() == b.z_nm[50:151].tolist()
    assert b.within(None) is b and b.within(5000.0) is b


def test_learning_takes_every_slice_within_the_largest_z_step_and_the_z_range():
    """The bead calibration's *largest z step* and *learn z range*: 20 nm stacks learnt at 50 nm keep every
    2nd slice, at 60 nm every 3rd; ±1 µm of a ±2 µm stack keeps that much, the focus always among them."""
    from smappy.calibrate.pupil import BeadSettings

    from uipsf.workflows.calibrate import learning_batch
    b = _stack_2um()
    for max_dz, step, z_range, reach in ((50.0, 40.0, None, 2000.0), (60.0, 60.0, None, 1980.0),
                                         (50.0, 40.0, 1000.0, 1000.0), (20.0, 20.0, 600.0, 600.0)):
        t = learning_batch(b, BeadSettings(max_dz_nm=max_dz, z_range_nm=z_range))
        z = t.z_nm.tolist()
        assert all(abs(abs(b2 - a) - step) < 1e-4 for a, b2 in zip(z, z[1:], strict=False)), max_dz
        assert 0.0 in z and max(abs(v) for v in z) == reach
    assert learning_batch(b, None).z_step_nm == 40.0         # settings without them: 50 nm, the whole stack


def test_by_default_learning_keeps_1_5_um_about_the_focus_and_a_shorter_stack_whole():
    from smappy.calibrate.pupil import BeadSettings

    from uipsf.workflows.calibrate import learning_batch
    b = _stack_2um()
    assert learning_batch(b, BeadSettings()).z_nm.abs().max() == 1480.0         # every 2nd of 20 nm slices
    short = b.within(1000.0)
    assert learning_batch(short, BeadSettings()).z_nm.abs().max() == 1000.0     # ±1 µm: whole
    assert BeadSettings.from_options({}).z_range_nm is None   # a model file of before: the whole stack


def test_beads_of_a_few_stacks_are_fitted_with_the_drift_of_every_stack():
    """M6's five stacks: the beads left out, all from the first two, were localised with the learnt drift of
    five stacks and failed ("expanded size (2) must match (5)"); a subset keeps the acquisition's stacks."""
    import torch

    import uipsf
    from uipsf.model.geometry import BeadStackGeometry
    from uipsf.noise import Poisson
    from uipsf.workflows.scoring import localise
    from uipsf_tests.conftest import small_beads
    optics = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.518, n_med=1.33, pixel_size_nm=(110, 110))
    model, truth, batch = small_beads(optics, n=6, geometry=BeadStackGeometry(drift="linear"))
    from dataclasses import replace
    batch = replace(batch, stack_id=torch.tensor([0, 1, 2, 3, 4, 4]))
    few = batch.subset(torch.tensor([0, 1]))
    assert few.n_stacks == 5 and few.subset(torch.tensor([0])).n_stacks == 5
    spec = {s.name: s for s in model.specs(few)}["geometry.drift_rate"]
    assert spec.shape == (5, 2)
    shared = {"geometry.drift_rate": torch.zeros(5, 2, dtype=batch.data.dtype)}
    shared.update({n: v for n, v in truth.physical().items() if n.startswith(("pupil.", "psf."))})
    fit, _ = localise(model, shared, few, Poisson())
    assert fit.params.n_emitters == 2
