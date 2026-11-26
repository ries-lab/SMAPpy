"""The LUTs' groups and the rule each group keeps (`smappy.lut`)."""

import numpy as np

from smappy import lut as luts
from smappy.render import DisplaySettings, FieldOfView, RenderedImage


def lightness(rgb):
    """CIE L* of sRGB colours, 0 (black) to 100 (white)."""
    rgb = np.asarray(rgb, np.float64)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    y = linear @ np.array([0.2126, 0.7152, 0.0722])
    return np.where(y > 0.008856, 116 * np.cbrt(y) - 16, 903.3 * y)


def lab_path(table):
    """The colour differences summed along a ramp, in CIE Lab: roughly how many
    steps of it can be told apart."""
    rgb = np.asarray(table, np.float64)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = linear @ m.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    lab = np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]),
                    200 * (f[:, 1] - f[:, 2])], axis=1)
    return np.linalg.norm(np.diff(lab, axis=0), axis=1).sum()


def group(heading):
    return dict(luts.groups())[heading]


def test_every_lut_is_in_one_group_and_the_defaults_are_in_theirs():
    listed = [name for _, names in luts.groups(ids=True) for name in names]
    assert sorted(listed) == sorted(luts.names()) and len(set(listed)) == len(listed)
    assert luts.kind(luts.INTENSITY_LUT) == "several colours"
    assert luts.kind(luts.FIELD_LUT) == "attribute"


def test_the_groups_for_what_the_layer_is_coloured_by_come_first():
    assert [h for h, _ in luts.groups()] == ["one colour", "several colours", "attribute"]
    assert [h for h, _ in luts.groups(attribute=True)] == ["attribute", "one colour",
                                                           "several colours"]


def test_intensity_ramps_rise_from_black_and_several_colours_reach_white():
    for heading in ("one colour", "several colours"):
        for name in group(heading):
            table = luts.get(name)
            assert np.all(table[0] < 0.02), name
            assert np.all(np.diff(lightness(table)) > -0.5), name
    gray = lab_path(luts.get("gray"))
    for name in group("several colours"):
        table = luts.get(name)
        assert np.all(table[-1] > 0.98), name
        # the point of them: more visible steps than a ramp of one colour
        assert lab_path(table) > 2.4 * gray, name


def test_attribute_ramps_have_no_darkness_of_their_own_but_the_diverging_middle():
    """The density multiplies a column's colour, so a dark entry would read as
    few localizations; only a diverging ramp through black is dark, and only
    at its middle."""
    for name in group("attribute"):
        table = luts.get(name)
        value = table.max(axis=1)
        if name == "magenta-black-green":
            assert np.all(table[127:129] < 0.01)
            assert value[0] == 1 and value[-1] == 1
            continue
        assert value.min() > 0.99, name
    for name in ("purple-white-orange", "blue-white-red"):
        assert np.all(luts.get(name)[127:129] > 0.99), name


def test_the_hue_circles_close_without_drawing_a_colour_twice():
    for name in ("hue circle", "four-hue circle"):
        table = luts.get(name)
        step = np.abs(np.diff(table, axis=0)).max()
        closing = np.abs(table[-1] - table[0]).max()
        assert 0 < closing <= 1.01 * step, name


def test_white_paper_draws_a_ramp_of_several_colours_backwards():
    """On white, red hot runs from white for nothing to black for the most,
    through yellow and red: the picture `on_white` makes of the table on black
    is red hot turned round."""
    ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)[None, :]
    rendered = RenderedImage(FieldOfView(0, 0, 1, 256, 1), ramp)
    display = DisplaySettings(lut="red hot", imax=1.0, white_background=True)
    picture = display.apply(rendered)[0]
    assert np.allclose(picture, luts.get("red hot")[::-1], atol=1e-6)
    # summed with other layers on black and turned over once, the same
    on_black = display.apply(rendered, white_background=False)
    assert np.allclose(luts.on_white(on_black)[0], picture, atol=1e-6)
    assert np.all(np.diff(lightness(picture)) < 0.5)


def test_white_paper_leaves_a_ramp_of_one_colour_to_the_turn_over():
    for name in ("gray", "red"):
        table = luts.intensity_table(name, white=True)
        assert np.allclose(table, luts.get(name)), name
    # on gray the two ways are one: white to black
    assert np.allclose(luts.on_white(luts.get("gray")), luts.get("gray")[::-1], atol=1e-6)


def test_a_diverging_ramp_through_black_goes_through_white_on_white_paper():
    on_paper = luts.on_white(luts.get("magenta-black-green"))
    assert np.all(on_paper[127:129] > 0.99)
    assert np.allclose(on_paper[[0, -1]], luts.get("magenta-black-green")[[0, -1]])


# ---------------------------------------------------------------- the ids table

def test_neighbouring_ids_get_colours_far_apart_where_a_ramp_gives_them_alike():
    ids = np.arange(1, 1001)
    step = lambda c: np.linalg.norm(np.diff(c, axis=0), axis=1)
    by_ids = step(luts.colors(ids, luts.IDS_LUT, 1, 1000))
    by_rainbow = step(luts.colors(ids, "rainbow", 1, 1000))
    assert by_ids.mean() > 0.7 and by_ids.mean() > 100 * by_rainbow.mean()
    # and not only on average: no two neighbours share a colour
    assert by_ids.min() > 0.05


def test_ids_whose_hues_land_close_differ_in_lightness():
    table = luts.get(luts.IDS_LUT)[1:]                   # entry 0 is "no id"
    light = lightness(table)
    # every three entries next to each other in hue span a third of L*
    window = np.lib.stride_tricks.sliding_window_view(light, 3)
    assert (window.max(axis=1) - window.min(axis=1)).min() > 30


def test_minus_one_and_nan_are_grey_in_the_ids_table():
    out = luts.colors(np.array([-1, np.nan, 3.0]), luts.IDS_LUT, 0, 10)
    assert np.allclose(out[:2], luts.NO_VALUE)
    assert not np.allclose(out[2], luts.NO_VALUE)


def test_an_ids_colour_depends_on_the_id_and_the_offset_alone():
    some = np.array([5, 17, 400])
    alone = luts.colors(some, luts.IDS_LUT, 0, 1, offset=0.3)
    among = luts.colors(np.r_[some, np.arange(1000, 1200)], luts.IDS_LUT, -50, 7000,
                        invert=True, offset=0.3)[:3]
    assert np.array_equal(alone, among)
    assert not np.array_equal(alone, luts.colors(some, luts.IDS_LUT, 0, 1, offset=0.55))


def test_the_ids_table_on_a_range_draws_what_its_colours_are():
    """The GPU only looks up on a range: `as_ramp` makes the places so."""
    values = np.r_[-1, np.arange(0, 3000)]
    places, (lo, hi), invert = luts.as_ramp(values, luts.IDS_LUT, 0, 1, True, offset=0.7)
    assert invert is False
    table = luts.get(luts.IDS_LUT)
    index = np.clip((places - lo) * (len(table) - 1) / (hi - lo), 0, len(table) - 1)
    assert np.array_equal(table[index.astype(int)],
                          luts.colors(values, luts.IDS_LUT, 0, 1, offset=0.7))


def test_the_ids_table_is_offered_for_ids_the_file_and_the_channel_only():
    for column in ("cluster_id", "track_id", "tid", "tid_file", "filenumber", "channel"):
        assert luts.offers_ids(column), column
    for column in ("x_nm", "frame", "n_in_group", "n_in_cluster", "photons", None):
        assert not luts.offers_ids(column), column
    assert luts.IDS_LUT not in dict(luts.groups(attribute=True))["attribute"]
    assert dict(luts.groups(attribute=True, ids=True))["attribute"][-1] == luts.IDS_LUT


def test_a_layer_drawn_with_the_ids_table_colours_each_cluster_its_own_way():
    from smappy.locs import Localizations
    from smappy.render import RenderSettings, render_locs
    x = np.repeat([1000.0, 2000.0, 3000.0], 50)
    locs = Localizations({"x_nm": x.astype(np.float32),
                          "y_nm": np.full(150, 1000.0, np.float32),
                          "cluster_id": np.repeat([1, 2, -1], 50)}, {"units": "nm"})
    fov = FieldOfView(0.0, 0.0, 100.0, 40, 20)
    settings = RenderSettings(mode="hist", color_field="cluster_id", color_range=(1, 2))
    pixel = lambda r, x_nm: r.color[10, int(x_nm / 100)] / r.weight[10, int(x_nm / 100)]
    one = render_locs(locs, fov, settings, DisplaySettings(lut=luts.IDS_LUT, ids_offset=0.2))
    assert np.allclose(pixel(one, 1000), luts.colors([1], luts.IDS_LUT, 0, 1, offset=0.2)[0])
    assert np.allclose(pixel(one, 3000), luts.NO_VALUE)
    other = render_locs(locs, fov, settings, DisplaySettings(lut=luts.IDS_LUT, ids_offset=0.7))
    assert not np.allclose(pixel(one, 1000), pixel(other, 1000))
