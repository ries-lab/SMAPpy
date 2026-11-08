"""The local FRC map: each tile's own resolution, from the tiles the whole
field's FRC already draws."""
import numpy as np

from smappy.frc import frc_resolution, local_frc
from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.precision import (LOCAL_COLUMN, LOCAL_ERROR_COLUMN,
                                      LocalizationPrecision, PrecisionSettings,
                                      local_frc_image)


def _two_halves(seed=0, n_emitters=6000, per_emitter=12, n_frames=4000,
                width=8000.0, sharp=6.0, blurred=24.0):
    """Emitters over an 8 x 4 um field, localized with a precision of
    ``sharp`` nm on the left half and ``blurred`` nm on the right."""
    rng = np.random.default_rng(seed)
    ex = rng.uniform(0, width, n_emitters)
    ey = rng.uniform(0, width / 2, n_emitters)
    which = np.repeat(np.arange(n_emitters), per_emitter)
    sigma = np.where(ex[which] < width / 2, sharp, blurred)
    # kept inside the field, so the tiles are the 4 x 2 the test reads
    x = np.clip(ex[which] + rng.normal(0, 1, len(which)) * sigma, 0, width)
    y = np.clip(ey[which] + rng.normal(0, 1, len(which)) * sigma, 0, width / 2)
    frame = rng.integers(0, n_frames, len(which))
    return x, y, frame, width


def test_the_map_tells_a_sharp_half_from_a_blurred_one():
    x, y, frame, width = _two_halves()
    found = local_frc(x, y, frame, pixelsize=3.0, tile_nm=2000.0, repeats=3)
    assert found.ok and found.resolution.shape == (2, 4)
    columns = (np.arange(4) + 0.5) * found.tile_nm + found.x0
    left = found.resolution[:, columns < width / 2]
    right = found.resolution[:, columns > width / 2]
    assert np.all(np.isfinite(left)) and np.all(np.isfinite(right))
    assert right.min() > 2 * left.max()
    # each tile agrees with the FRC of its own half, measured whole
    for half, tiles in ((x < width / 2, left), (x >= width / 2, right)):
        whole = frc_resolution(x[half], y[half], frame[half], pixelsize=3.0,
                               repeats=1)
        assert abs(np.median(tiles) / whole.resolution - 1) < 0.15
    assert np.all(np.isfinite(found.error[np.isfinite(found.resolution)]))


def test_a_tile_with_too_few_localizations_has_no_value():
    x, y, frame, _ = _two_halves(n_emitters=3000)
    # a sparse corner: the first tile keeps only a few hundred
    corner = (x < 2000) & (y < 2000)
    keep = ~corner | (np.arange(len(x)) % 40 == 0)
    found = local_frc(x[keep], y[keep], frame[keep], pixelsize=3.0,
                      tile_nm=2000.0, repeats=1, min_locs=1000)
    row, column = found.tile_of([500.0], [500.0])
    assert found.n_locs[row[0], column[0]] < 1000
    assert np.isnan(found.resolution[row[0], column[0]])
    assert found.n_sparse >= 1 and found.n_measured >= 4


def test_smoothing_interpolates_between_tiles_but_never_into_an_empty_one():
    x, y, frame, width = _two_halves(n_emitters=4000)
    found = local_frc(x, y, frame, pixelsize=3.0, tile_nm=2000.0, repeats=1)
    found.resolution[0, 0] = np.nan
    # on the border between the halves the smoothed value lies between them
    left, right = found.resolution[1, 1], found.resolution[1, 2]
    middle = found.at([width / 2], [3000.0], smooth=True)[0]
    assert min(left, right) < middle < max(left, right)
    assert found.at([width / 2 - 1], [3000.0])[0] == left
    assert np.isnan(found.at([100.0], [100.0], smooth=True)[0])


def test_the_plugin_writes_the_map_to_the_localizations_and_as_an_image():
    x, y, frame, width = _two_halves(n_emitters=4000)
    locs = Localizations({"x_nm": x.astype(np.float32),
                          "y_nm": y.astype(np.float32),
                          "frame": frame.astype(np.int32)})
    settings = PrecisionSettings(pairwise=False, crlb=False, frc=True,
                                 frc_map=True, frc_repeats=1)
    plugin = LocalizationPrecision()
    assert plugin.changes_locs_for(settings)
    assert plugin.describe(None, settings).shown["frc_local"]
    result = plugin.run(Context(locs=locs), settings)
    out = result.locs
    assert len(out) == len(locs)
    value = np.asarray(out[LOCAL_COLUMN])
    assert np.isfinite(value).mean() > 0.9
    assert np.all(np.isfinite(np.asarray(out[LOCAL_ERROR_COLUMN])[np.isfinite(value)]))
    sharp = np.nanmedian(value[x < width / 2 - 2000])
    blurred = np.nanmedian(value[x > width / 2 + 2000])
    assert blurred > 2 * sharp
    image, = result.image_layers
    local = result.data["frc_map"]
    assert image.data.shape == local.resolution.shape
    assert image.pixelsize == local.tile_nm and image.x0 == local.x0
    assert "local FRC" in result.plots


def test_without_write_to_localizations_the_table_is_left_alone():
    settings = PrecisionSettings(frc_map=True)
    settings.frc_local.write_column = False
    assert not LocalizationPrecision().changes_locs_for(settings)
    assert not LocalizationPrecision().changes_locs_for(PrecisionSettings())


def test_the_image_layer_puts_an_unmeasured_tile_at_zero():
    x, y, frame, _ = _two_halves(n_emitters=3000)
    found = local_frc(x, y, frame, pixelsize=3.0, tile_nm=2000.0, repeats=1)
    found.resolution[0, 0] = np.nan
    image = local_frc_image(found)
    assert image.data[0, 0] == 0 and np.all(np.isfinite(image.data))


def test_in_a_session_the_map_is_a_column_and_a_layer_over_the_localizations():
    from smappy import plugins
    from smappy.session import Session

    x, y, frame, width = _two_halves(n_emitters=3000)
    session = Session(Localizations({"x_nm": x, "y_nm": y, "frame": frame}, {}))
    n = len(session.layers)
    session.run(plugins.get("Analysis/Measure/Localization Precision")(),
                PrecisionSettings(pairwise=False, crlb=False, frc=False,
                                  frc_map=True, frc_repeats=1))
    assert LOCAL_COLUMN in session.locs and LOCAL_ERROR_COLUMN in session.locs
    assert len(session.layers) == n + 1
    image = session.layers[-1].image
    x0, y0, x1, y1 = image.bounds
    assert x0 <= x.min() and x1 >= x.max() and y0 <= y.min() and y1 >= y.max()
