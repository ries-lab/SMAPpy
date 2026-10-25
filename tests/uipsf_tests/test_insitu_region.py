"""The region each molecule is fitted in (October 2026, docs/uipsf/architecture.md, "The region size").

At 15 and 21 px M2's January molecules learnt a primary astigmatism 0.3 rad stronger than the beads' and a
secondary one of -0.2, at 25 px the beads' (Noll 6 -0.73, beads -0.68); the same molecules cut from 25 px
regions to 15 learnt the stronger one again: the pixels, not the selection.  A simulated movie learnt at 15 px
does not show it -- which is why the simulated tests and benchmarks, all at 15 px, never did: with the model
exact, every region gives the truth.  On January two solutions nearly tie (the beads' astigmatism with an
anisotropic blur, or more astigmatism with a round one) and only the spots' faint outer light tells them
apart.  So the guard is the default: *region* on auto spans 3 µm.
"""
import numpy as np
import pytest
import torch
from smappy.calibrate import insitu as I

from uipsf.workflows.insitu_rounds import RoundSettings, calibrate_insitu


def test_the_region_on_auto_spans_three_micrometres_of_the_sample():
    assert I.InSituSettings().roi_px is None
    assert I.region_px((116.0, 127.0)) == I.region_px((127.0, 116.0)) == 25          # M2, Evolve and iXon
    assert I.region_px((110.0, 110.0)) == 29                                          # Zenodo's FD pair
    assert I.region_px((65.0, 65.0)) == 47
    assert I.region_px((250.0, 250.0)) == I.MIN_REGION_PX
    for p in [(65.0, 65.0), (100.0, 100.0), (116.0, 127.0), (250.0, 250.0)]:
        n = I.region_px(p)
        assert n % 2 == 1 and (n * np.mean(p) >= I.REGION_NM or n == I.MIN_REGION_PX)


def test_a_region_set_by_hand_is_kept_and_auto_becomes_pixels_where_the_movie_comes_in():
    assert I.InSituSettings(roi_px=17).resolved((116.0, 127.0)).roi_px == 17
    assert I.InSituSettings().resolved((116.0, 127.0)).roi_px == 25


@pytest.mark.long
def test_on_an_exact_model_a_15_and_a_21_px_region_learn_the_same_astigmatism():
    # M2's optics and its beads' astigmatism, 0-1300 nm deep, focus 430 nm, ~3000 photons on 55 per pixel,
    # learnt from the truth: whatever moves is a bias of the region.  ~1 min on 4 cores.
    from smappy.calibrate.pupil import BeadSettings
    from smappy.simulate import BlinkingSettings

    import uipsf
    from uipsf.movie import MovieSettings, simulate_movie
    from uipsf.workflows import insitu_model
    optics = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.33, pixel_size_nm=(116, 127))
    truth = {5: -0.05, 6: -0.68, 7: 0.05, 8: 0.08, 11: 0.10}

    def model(roi, focus):
        return insitu_model(optics, BeadSettings(zernike_order=4, magnitude="fixed", blur_nm=55.0), (roi, roi),
                            focus, z_max_nm=1300.0, device="cpu")

    def shared(m):
        modes = m.psf.pupil.phase_modes
        return {"pupil.phase": torch.tensor([truth.get(j, 0.0) for j in modes]),
                "pupil.magnitude": torch.zeros(len(m.psf.pupil.magnitude_modes)),
                "psf.sigma": torch.tensor([55.0, 55.0])}

    m = model(31, 430.0)
    movie = simulate_movie(m, dict(shared(m), **{"geometry.focus": torch.tensor(430.0)}), MovieSettings(
        size_px=(96, 96), n_frames=400, fluorophores=450, heights_nm=(0.0, 1300.0), on_coverslip=0.1,
        background=55.0, illumination_fwhm_px=200.0, seed=3, spot_px=31,
        blinking=BlinkingSettings(photons=6000.0, photons_std=2500.0, on_time=1.5, blinks=3.0)))
    learnt = {}
    for roi in (15, 21):
        sel = I.InSituSettings(molecules=600, roi_px=roi)
        rois = I.insitu_rois(movie.frames, optics.pixel_size_nm, sel)
        mr = model(roi, 530.0)
        run = calibrate_insitu(mr, rois, start=shared(mr), selection=sel, settings=RoundSettings(max_rounds=1))
        cal = run.final.calibration
        j = cal.model.psf.pupil.phase_modes.index(6)
        learnt[roi] = (float(cal.shared()["pupil.phase"][j]), float(cal.result.std["pupil.phase"][j]))
    for value, sd in learnt.values():
        assert abs(value - truth[6]) < max(4 * sd, 0.03)
    assert abs(learnt[15][0] - learnt[21][0]) < 0.05           # January's 15 px gap: 0.3 rad
