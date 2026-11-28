"""The aberration maps of a field-dependent model: each mode's coefficient, astigmatism's axis, coma's
direction and the total RMS wavefront across the field (`diagnostics.field.field_maps`)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import uipsf
from uipsf.diagnostics.field import SHOW_ALL, SHOW_MOST, chosen_items, field_maps, map_items, pair_vector, show_choices
from uipsf.model import Channel, FieldPolynomial

f64 = torch.float64
OPTICS = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.518, n_med=1.33, pixel_size_nm=(110, 110))
PX = (110.0, 110.0)
BASE = {5: 0.1, 6: 0.6, 7: 0.05, 11: 0.2}
#: Noll 6 grows 0.3 rad per 50 µm in x, Noll 5 0.2 in y, Noll 8 0.1 in x
FIELD = {(6, 1, 0): 0.3, (5, 0, 1): 0.2, (8, 1, 0): 0.1}


def source(points=((0.0, 0.0), (1000.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (400.0, 700.0))):
    """A model file's stand-in (`LearnedPSF`'s ``model``, ``shared`` and ``beads``) with a known field."""
    grid = uipsf.PupilGrid.build(OPTICS, 64, device="cpu", dtype=f64)
    pupil = uipsf.ZernikePupil(grid, n_max=4)
    field = FieldPolynomial((500.0, 500.0), PX)
    model = uipsf.ForwardModel([Channel(uipsf.VectorPSF(grid, pupil, (11, 11), sigma_nm=(30.0, 30.0)), field=field)])
    phase = torch.tensor([BASE.get(j, 0.0) for j in pupil.phase_modes], dtype=f64)
    shared = {"pupil.phase": phase, "field.phase": torch.tensor([FIELD.get(t, 0.0) for t in field.layout], dtype=f64),
              "psf.sigma": torch.tensor([30.0, 30.0], dtype=f64)}
    beads = [{"bead": i, "state": "used", "field_x": x, "field_y": y} for i, (x, y) in enumerate(points)]
    beads.append({"bead": 99, "state": "rejected", "field_x": 5000.0, "field_y": 5000.0})
    return SimpleNamespace(model=model, shared=shared, beads=beads), pupil


def expected(x, y):
    u, v = (x - 500.0) * 110 / 50000, (y - 500.0) * 110 / 50000
    return {5: BASE[5] + 0.2 * v, 6: BASE[6] + 0.3 * u, 7: BASE[7], 8: 0.1 * u, 11: BASE[11]}


def test_the_maps_span_the_used_emitters_and_hold_each_modes_coefficient_there():
    src, _ = source()
    maps = field_maps(src)
    assert maps.x_px[0] == 0 and maps.x_px[-1] == 1000 and maps.y_px[-1] == 1000     # not the rejected bead
    for (iy, ix) in ((0, 0), (-1, -1), (0, -1), (12, 5)):
        want = expected(maps.x_px[ix], maps.y_px[iy])
        for j, value in want.items():
            assert maps.of(j)[iy, ix] == pytest.approx(value, abs=1e-9), j
    assert sorted(maps.varying) == [5, 6, 7, 8]


def test_astigmatism_is_an_axis_and_coma_an_arrow_from_each_pair_of_modes():
    src, _ = source()
    maps = field_maps(src)
    items = {i.label.split(" (")[0]: i for i in map_items(maps)}
    astig, coma = items["astigmatism"], items["coma"]
    assert astig.modes == (6, 5) and astig.m == 2 and coma.modes == (8, 7) and coma.m == 1
    size, angle = pair_vector(maps, astig)
    c = expected(maps.x_px[-1], maps.y_px[0])
    assert size[0, -1] == pytest.approx(math.hypot(c[6], c[5]))
    assert angle[0, -1] == pytest.approx(math.atan2(c[5], c[6]) / 2)
    # the axis: the wavefront c6 cos 2φ + c5 sin 2φ is largest along θ
    phi = angle[0, -1]
    assert c[6] * math.cos(2 * phi) + c[5] * math.sin(2 * phi) == pytest.approx(size[0, -1])
    size, angle = pair_vector(maps, coma)
    assert angle[0, -1] == pytest.approx(math.atan2(c[7], c[8]))


def test_the_families_varying_most_come_first_and_a_constant_one_is_not_offered():
    src, _ = source()
    maps = field_maps(src)
    labels = [i.label for i in map_items(maps)]
    assert labels[0].startswith("astigmatism") and labels[1].startswith("coma") and len(labels) == 2
    assert show_choices(maps) == (SHOW_MOST, SHOW_ALL) + tuple(labels)
    assert [i.label for i in chosen_items(maps, labels[1])] == [labels[1]]
    # coma varies 0.11 rad over the field, astigmatism ~0.4: coma is still above a quarter of it
    assert len(chosen_items(maps, SHOW_MOST)) == 2


def test_the_total_rms_wavefront_is_the_root_sum_of_squares_of_nolls_coefficients():
    src, _ = source()
    maps = field_maps(src)
    for iy, ix in ((0, 0), (-1, -1), (7, 3)):
        want = math.sqrt(sum(v * v for v in expected(maps.x_px[ix], maps.y_px[iy]).values()))
        assert maps.rms[iy, ix] == pytest.approx(want, rel=0.02)


def test_a_model_without_a_field_has_no_maps_and_the_page_says_so():
    from matplotlib.figure import Figure

    from uipsf.diagnostics.figures import draw_field_maps
    src, _ = source()
    src.model.channels[0].field = None
    assert field_maps(src) is None
    fig = Figure()
    draw_field_maps(fig, src)
    assert "no field dependence" in fig.axes[0].texts[0].get_text()


@pytest.mark.parametrize("show", [SHOW_MOST, SHOW_ALL, "coma (Noll 7, 8)"])
def test_the_page_draws_the_rms_and_the_chosen_families(show):
    from matplotlib.figure import Figure

    from uipsf.diagnostics.figures import draw_field_maps
    src, _ = source()
    fig = Figure()
    draw_field_maps(fig, src, show=show, units="nm")
    titles = [a.get_title() for a in fig.axes if a.get_title()]
    assert titles[0].startswith("total RMS wavefront")
    assert any(t.startswith("coma") for t in titles)
    assert any(t.startswith("astigmatism") for t in titles) == (show != "coma (Noll 7, 8)")
    assert np.any([len(a.collections) for a in fig.axes])          # the arrows


def test_a_model_file_draws_the_same_maps_as_the_fit_it_was_saved_from(tmp_path):
    from matplotlib.figure import Figure
    from smappy.calibrate.psf_methods import aberration_map_page

    from uipsf.export import load_model, save_model
    from uipsf.workflows import learn_beads

    from .test_field import field_beads
    model, truth, batch = field_beads(n=4, truth_field={(6, 1, 0): 0.3}, dtype=torch.float32)
    cal = learn_beads(model, batch, first_order=None)
    save_model(cal, tmp_path / "field.uipsf.h5")
    learned = load_model(tmp_path / "field.uipsf.h5", device="cpu")
    fitted, kept = field_maps(cal), field_maps(learned)
    assert np.allclose(fitted.points, kept.points) and np.allclose(fitted.coefficients, kept.coefficients, atol=1e-6)
    assert np.allclose(fitted.rms, kept.rms, atol=1e-6)
    page = aberration_map_page(learned)
    assert page.controls["show"][0] == SHOW_MOST and page.controls["units"][0] == "nm"
    status = page.draw(Figure(), [], set(), {"show": SHOW_ALL, "units": "nm"})
    assert status.startswith("total RMS wavefront") and "4 used emitters" in status
