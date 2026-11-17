"""Image quality: the error map against a widefield image (SQUIRREL), the
local FRC map beside it, QnR across layers, and an image saved as a TIFF."""
import numpy as np
import pytest

from smappy import plugins
from smappy.images import ImageData, load_images, save_tiff
from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.image_quality import (ERROR_COLUMN, LOCAL_COLUMN,
                                          LOCAL_ERROR_COLUMN, SIGNED_COLUMN,
                                          ImageQuality, ImageQualitySettings)
from smappy.session import Session
from smappy.squirrel import error_map, normalised_resolution, qnr

PIXEL = 100.0


def _filaments(seed=0, n_pixels=64, n_lines=8):
    """Straight filaments over a field of ``n_pixels`` camera pixels: the
    points on them, and the line each point is on."""
    rng = np.random.default_rng(seed)
    width = n_pixels * PIXEL
    points, line = [], []
    for k in range(n_lines):
        angle = rng.uniform(0, np.pi)
        cx, cy = rng.uniform(0.25 * width, 0.75 * width, 2)
        t = rng.uniform(-0.6 * width, 0.6 * width, 3000)
        points.append(np.c_[cx + t * np.cos(angle), cy + t * np.sin(angle)])
        line.append(np.full(len(t), k))
    points, line = np.vstack(points), np.concatenate(line)
    inside = np.all((points > 0) & (points < width), axis=1)
    return points[inside], line[inside], width


def _widefield(points, width, sigma=130.0, scale=3.0, background=20.0, seed=1):
    """The camera's picture of the filaments: blurred by ``sigma`` nm, summed
    into pixels, scaled, on a background, with shot noise."""
    from scipy.ndimage import gaussian_filter
    fine = 10
    n = int(round(width / PIXEL))
    counts, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=n * fine,
                                  range=[[0, width], [0, width]])
    blurred = gaussian_filter(counts, sigma / (PIXEL / fine))
    camera = blurred.reshape(n, fine, n, fine).sum(axis=(1, 3))
    rng = np.random.default_rng(seed)
    return rng.poisson(scale * camera + background).astype(float)


def _localizations(points, n=60000, precision=10.0, shift=(0.0, 0.0), seed=2):
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(points), n)
    xy = points[pick] + rng.normal(0, precision, (n, 2)) + np.asarray(shift)
    return xy[:, 0], xy[:, 1], pick


def test_the_error_map_finds_the_blur_the_shift_and_the_scale():
    points, _, width = _filaments()
    reference = _widefield(points, width, sigma=130.0, background=20.0)
    x, y, _ = _localizations(points, shift=(70.0, -40.0))
    found = error_map(reference, x, y, pixelsize=PIXEL)
    assert found.ok and not found.message
    # the localizations' own 10 nm come on top of the blur the fit finds
    assert abs(found.sigma_nm - np.sqrt(130.0 ** 2 - 10.0 ** 2)) < 6
    assert abs(found.dx_nm - 70.0) < 5 and abs(found.dy_nm + 40.0) < 5
    assert abs(found.beta - 20.0) < 2
    assert found.rsp > 0.95
    # noise alone is left: the RSE is about the shot noise of the pixels
    assert found.rse < 1.5 * np.sqrt(reference.mean())


def test_a_filament_missing_from_the_localizations_lights_up_in_the_map():
    points, line, width = _filaments(n_lines=6)
    reference = _widefield(points, width)
    x, y, pick = _localizations(points)
    kept = line[pick] != 0                      # filament 0 was not detected
    found = error_map(reference, x[kept], y[kept], pixelsize=PIXEL)
    on = found.at(points[line == 0, 0], points[line == 0, 1], signed=True)
    elsewhere = found.at(points[line != 0, 0], points[line != 0, 1], signed=True)
    on_alone = np.isfinite(on) & (np.abs(found.at(points[line == 0, 0],
                                                  points[line == 0, 1])) > 0)
    # missing: the model shows less than the reference there
    assert np.nanmedian(on[on_alone]) < -3 * np.nanmedian(np.abs(elsewhere))
    whole = error_map(reference, x, y, pixelsize=PIXEL)
    assert whole.rsp > found.rsp and whole.rse < found.rse


def test_a_blur_given_is_used_as_it_is():
    points, _, width = _filaments()
    reference = _widefield(points, width)
    x, y, _ = _localizations(points)
    found = error_map(reference, x, y, pixelsize=PIXEL, sigma_nm=200.0,
                      register=False)
    assert found.sigma_nm == 200.0 and not found.sigma_fitted
    fitted = error_map(reference, x, y, pixelsize=PIXEL)
    assert fitted.rse < found.rse


def test_localizations_off_the_reference_are_refused_with_a_reason():
    points, _, width = _filaments()
    reference = _widefield(points, width)
    x, y, _ = _localizations(points)
    found = error_map(reference, x + 2 * width, y, pixelsize=PIXEL)
    assert not found.ok and "lie on it" in found.message


def test_the_map_is_placed_over_the_localizations_it_was_measured_on():
    points, _, width = _filaments()
    reference = _widefield(points, width)
    x, y, _ = _localizations(points, shift=(150.0, 0.0))
    found = error_map(reference, x, y, x0=500.0, y0=-200.0, pixelsize=PIXEL)
    image = found.image()
    assert image.data.shape == found.model.shape
    assert abs(image.x0 - (found.x0 + found.dx_nm)) < 1e-6
    # the reference moved by (500, -200) and the localizations by (150, 0)
    assert abs(found.dx_nm + 350.0) < 5 and abs(found.dy_nm - 200.0) < 5
    assert np.isnan(found.at([-1e6], [0.0])[0])


def test_qnr_ranks_by_fidelity_and_resolution_together():
    n = normalised_resolution([30.0, 60.0, 45.0])
    assert abs(n[0] - 0.925) < 1e-9 and abs(n[1] - 0.075) < 1e-9
    assert n[1] < n[2] < n[0]
    # as good as the best resolution, but unfaithful: ranked below a
    # faithful one of middling resolution
    scores = qnr([0.5, 0.95, 0.95], [30.0, 60.0, 45.0])
    assert np.argmax(scores) == 2
    assert np.all(np.isnan(normalised_resolution([np.nan, np.nan])))
    assert np.allclose(normalised_resolution([40.0, 40.0]), 0.5)


def _session(points, width, two_layers=False):
    x, y, _ = _localizations(points, n=40000)
    rng = np.random.default_rng(3)
    channel = (np.arange(len(x)) % 2).astype(np.int32)
    session = Session(Localizations({"x_nm": x, "y_nm": y,
                                     "frame": rng.integers(0, 4000, len(x)),
                                     "channel": channel}, {}))
    if two_layers:
        session.layers[0].set_ranges({"channel": (0, 0)})
        session.add_layer(like=0).set_ranges({"channel": (1, 1)})
    session.add_image(ImageData(_widefield(points, width).astype(np.float32),
                                PIXEL, name="widefield"))
    return session, channel


def test_in_a_session_each_layer_gets_its_map_as_a_column_and_an_image():
    points, _, width = _filaments()
    session, channel = _session(points, width, two_layers=True)
    plugin = plugins.get("Analysis/Measure/Image quality")()
    settings = ImageQualitySettings()
    settings.squirrel.signed = True
    state = plugin.describe(Context(locs=session.locs, session=session), settings)
    assert state.hints["squirrel.reference"] == "widefield"
    n = len(session.layers)
    result = session.run(plugin, settings)
    maps = result.data["error_maps"]
    assert len(maps) == 2 and all(m.ok for m in maps.values())
    # each layer's error map plus its signed one
    assert len(session.layers) == n + 4
    error = np.asarray(session.locs[ERROR_COLUMN])
    signed = np.asarray(session.locs[SIGNED_COLUMN])
    assert np.isfinite(error).mean() > 0.95
    assert np.allclose(np.abs(signed[np.isfinite(signed)]),
                       error[np.isfinite(signed)], atol=1e-3)
    # half the localizations each: twice the scale of the reference
    assert all(abs(m.rsp - 0.95) < 0.05 for m in maps.values())
    assert "RSP" in result.text and set(result.plots) == set(maps)


def test_qnr_is_measured_across_the_layers_when_asked():
    points, _, width = _filaments()
    session, _ = _session(points, width, two_layers=True)
    settings = ImageQualitySettings(qnr=True, write_column=False,
                                    image_layer=False)
    settings.frc_local.repeats = 1
    result = session.run(ImageQuality(), settings)
    assert set(result.data["qnr"]) == {"layer 1", "layer 2"}
    assert "QnR" in result.plots and "best:" in result.text
    assert ERROR_COLUMN not in session.locs


def test_the_error_map_needs_an_image_and_says_so():
    points, _, width = _filaments()
    x, y, _ = _localizations(points, n=5000)
    session = Session(Localizations({"x_nm": x, "y_nm": y}, {}))
    plugin = ImageQuality()
    state = plugin.describe(Context(locs=session.locs, session=session),
                            ImageQualitySettings())
    assert state.notice is not None and "widefield" in state.notice.text
    with pytest.raises(ValueError, match="widefield image"):
        session.run(plugin, ImageQualitySettings())


def test_the_maps_are_saved_as_tiffs_beside_the_file(tmp_path):
    points, _, width = _filaments()
    session, _ = _session(points, width)
    session.path = tmp_path / "cells_sml.h5"
    settings = ImageQualitySettings(save_tiff=True, write_column=False,
                                    image_layer=False)
    result = session.run(ImageQuality(), settings)
    written = sorted(p.name for p in tmp_path.glob("*.tif"))
    assert written == ["cells_sml_error_map_layer_1.tif"]
    back = load_images(tmp_path / written[0])[0]
    found = result.data["error_maps"]["layer 1"]
    assert np.allclose(back.data, found.error, rtol=1e-5, atol=1e-4)
    assert abs(back.pixelsize - PIXEL) < 1e-6
    assert abs(back.x0 - (found.x0 + found.dx_nm)) < 1e-3


# ------------------------------------------------------------ the local FRC

def _two_halves(seed=0, n_emitters=4000, per_emitter=12, n_frames=4000,
                width=8000.0, sharp=6.0, blurred=24.0):
    rng = np.random.default_rng(seed)
    ex = rng.uniform(0, width, n_emitters)
    ey = rng.uniform(0, width / 2, n_emitters)
    which = np.repeat(np.arange(n_emitters), per_emitter)
    sigma = np.where(ex[which] < width / 2, sharp, blurred)
    x = np.clip(ex[which] + rng.normal(0, 1, len(which)) * sigma, 0, width)
    y = np.clip(ey[which] + rng.normal(0, 1, len(which)) * sigma, 0, width / 2)
    frame = rng.integers(0, n_frames, len(which))
    return x, y, frame, width


def test_the_local_frc_map_is_written_to_the_localizations_and_as_an_image():
    x, y, frame, width = _two_halves()
    locs = Localizations({"x_nm": x.astype(np.float32),
                          "y_nm": y.astype(np.float32),
                          "frame": frame.astype(np.int32)})
    settings = ImageQualitySettings(error_map=False, frc_map=True)
    settings.frc_local.repeats = 1
    plugin = ImageQuality()
    assert plugin.changes_locs_for(settings)
    state = plugin.describe(None, settings)
    assert state.shown["frc_local"] and not state.shown["squirrel"]
    result = plugin.run(Context(locs=locs), settings)
    value = np.asarray(result.locs[LOCAL_COLUMN])
    assert np.isfinite(value).mean() > 0.9
    assert np.all(np.isfinite(np.asarray(result.locs[LOCAL_ERROR_COLUMN])[np.isfinite(value)]))
    assert np.nanmedian(value[x > width / 2 + 2000]) > 2 * np.nanmedian(value[x < width / 2 - 2000])
    image, = result.image_layers
    local, = result.data["frc_maps"].values()
    assert image.data.shape == local.resolution.shape
    assert image.pixelsize == local.tile_nm and image.x0 == local.x0


def test_nothing_ticked_is_refused_and_no_column_means_no_table_change():
    plugin = ImageQuality()
    assert not plugin.changes_locs_for(ImageQualitySettings(write_column=False))
    with pytest.raises(ValueError, match="nothing to measure"):
        plugin.run(Context(locs=Localizations({"x_nm": np.zeros(3),
                                               "y_nm": np.zeros(3)})),
                   ImageQualitySettings(error_map=False))


def test_localization_precision_no_longer_draws_the_local_map():
    from smappy.plugins.precision import PrecisionSettings
    assert not hasattr(PrecisionSettings(), "frc_map")


# ------------------------------------------------------ an image as a TIFF

def test_a_stack_is_saved_whole_or_one_plane_and_read_back_in_place(tmp_path):
    image = ImageData(np.random.default_rng(0).random((3, 5, 7)).astype(np.float32),
                      pixelsize=110.0, x0=250.0, y0=-30.0, name="stack",
                      metadata={"axis": "z", "z_step_nm": 50.0})
    back = load_images(save_tiff(image, tmp_path / "whole.tif"))[0]
    assert back.data.shape == (3, 5, 7) and np.allclose(back.data, image.data)
    assert abs(back.pixelsize - 110.0) < 1e-6 and (back.x0, back.y0) == (250.0, -30.0)
    assert back.z_step_nm == pytest.approx(50.0)
    one = load_images(save_tiff(image, tmp_path / "one", plane=1))[0]
    assert one.data.shape == (5, 7) and np.allclose(one.data, image.data[1])


def test_save_image_layer_writes_the_chosen_image_beside_the_file(tmp_path):
    session = Session(Localizations({"x_nm": np.zeros(3), "y_nm": np.zeros(3)}, {}))
    session.path = tmp_path / "cells_sml.h5"
    session.add_image(ImageData(np.arange(12, dtype=np.float32).reshape(3, 4),
                                PIXEL, name="widefield.tif"))
    plugin = plugins.get("File/Export/Image layer")()
    settings = plugin.Settings()
    state = plugin.describe(Context(locs=session.locs, session=session), settings)
    assert state.hints["image"] == "widefield.tif"
    assert state.hints["path"].endswith("cells_sml_widefield.tif")
    result = session.run(plugin, settings)
    back = load_images(result.written)[0]
    assert np.array_equal(back.data, np.arange(12).reshape(3, 4))
