"""The camera frames a fit keeps with its table, and the image layers that show them."""
import numpy as np
import pytest

from smappy.camera import to_photons
from smappy.io.hdf5 import load_images, save_localizations
from smappy.plugins import Context
from smappy.plugins.fit import (Finished, GaussianFit, GaussianFitSettings,
                                OutputSettings, SourceSettings)
from smappy.rawframes import AVERAGE, RawFrameKeeper, spaced
from smappy.render import FieldOfView
from smappy.session import Session
from smappy.simulate import CameraOutputSettings, LabellingSettings, SimulationSettings

N_FRAMES = 120


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    """A small camera simulation, fitted once: (recipe, fitted file, result)."""
    from smappy.plugins import get
    folder = tmp_path_factory.mktemp("raw")
    settings = SimulationSettings(
        output="camera", n_frames=N_FRAMES, seed=3,
        labelling=LabellingSettings(efficiency=0.1),
        camera=CameraOutputSettings(path=str(folder / "run.sim.yaml")))
    made = get("File/Simulate/Blinking Structure")().run(Context(), settings)
    out = folder / "out.hdf5"
    result = GaussianFit().run(Context(), GaussianFitSettings(
        source=SourceSettings(path=made.data["path"], chunk=25),
        output=OutputSettings(path=str(out), raw_frames=10)))
    return made.data["path"], out, result


def _photons(recipe):
    from smappy.io.tiff import open_stack
    from smappy.plugins.fit import CameraSettings
    source = open_stack(recipe)
    camera = CameraSettings().resolve(source)
    frames = np.concatenate([b for _, b in source.frames(chunk=50)])
    return to_photons(frames, camera), camera


def test_spaced_frames_start_at_the_first_and_end_at_the_last():
    assert spaced(0, 120, 10).tolist() == [0, 13, 26, 40, 53, 66, 79, 93, 106, 119]
    assert spaced(5, 8, 10).tolist() == [5, 6, 7]
    assert spaced(7, 100, 1).tolist() == [7]
    assert spaced(0, 100, 0).size == 0


def test_a_live_keeper_keeps_evenly_spaced_frames_from_the_first():
    keeper = RawFrameKeeper(8, start=3)                 # no end: live
    for first in range(3, 103, 10):
        keeper.see(first, np.full((10, 2, 2), 1.0))
    kept = sorted(keeper.kept)
    assert kept[0] == 3 and len(kept) <= 8
    assert len(set(np.diff(kept))) == 1                  # evenly spaced
    assert keeper.n == 100


def test_the_fit_keeps_the_average_and_the_chosen_frames_in_photons(simulated):
    recipe, out, result = simulated
    photons, camera = _photons(recipe)
    raw, = result.data["images"]
    assert raw.kind == "raw"
    assert raw.frames.tolist() == [AVERAGE] + spaced(0, N_FRAMES, 10).tolist()
    assert raw.metadata["n_averaged"] == N_FRAMES
    data = np.asarray(raw.data)
    assert data.dtype == np.float32
    np.testing.assert_allclose(data[0], photons.mean(axis=0), rtol=1e-4, atol=1e-3)
    for plane, number in zip(data[1:], raw.frames[1:]):
        np.testing.assert_allclose(plane, photons[number], rtol=1e-5)
    # and the file has the same, with each plane's frame number
    stored, = load_images(out)
    assert stored.frames.tolist() == raw.frames.tolist()
    np.testing.assert_allclose(np.asarray(stored.data), data)
    np.testing.assert_allclose(stored.plane_at(3), data[3])


def test_the_kept_frames_lie_under_their_localizations(simulated):
    """The spot a localization came from is centred on it, to a fraction of a
    pixel: placed half a pixel off, the mean offset would be 0.5 px."""
    _, _, result = simulated
    raw, = result.data["images"]
    locs = result.locs
    px, py = raw.pixelsize_xy
    offsets = []
    for index, number in enumerate(raw.frames):
        if number < 0:
            continue
        plane = raw.plane_at(index)
        here = locs[(locs["frame"] == number) & (locs["photons"] > 1000)]
        for x, y in zip(here["x_nm"], here["y_nm"]):
            ix, iy = int((x - raw.x0) // px), int((y - raw.y0) // py)
            patch = plane[iy - 2:iy + 3, ix - 2:ix + 3]
            if patch.shape != (5, 5):
                continue
            patch = patch - np.median(plane)
            gy, gx = np.mgrid[iy - 2:iy + 3, ix - 2:ix + 3]
            cx = raw.x0 + (np.sum(gx * patch) / patch.sum() + 0.5) * px
            cy = raw.y0 + (np.sum(gy * patch) / patch.sum() + 0.5) * py
            offsets.append(((cx - x) / px, (cy - y) / py))
    offsets = np.array(offsets)
    assert len(offsets) > 20
    np.testing.assert_allclose(np.median(offsets, axis=0), 0, atol=0.15)
    # and resampling onto a view of one pixel gives that pixel back
    fov = FieldOfView(raw.x0 + 10.5 * px - 5, raw.y0 + 7.5 * py - 5, 10.0, 1, 1)
    assert raw.resample(fov, 2).weight[0, 0] == pytest.approx(raw.plane_at(2)[7, 10])


def test_no_frames_are_kept_when_asked_for_none(simulated, tmp_path):
    recipe, _, _ = simulated
    out = tmp_path / "none.hdf5"
    result = GaussianFit().run(Context(), GaussianFitSettings(
        source=SourceSettings(path=recipe, stop=20),
        output=OutputSettings(path=str(out), raw_frames=0)))
    assert result.data["images"] == [] and load_images(out) == []


def test_the_kept_frames_survive_the_rewrite_of_a_finished_table(simulated, tmp_path,
                                                                 monkeypatch):
    recipe, _, _ = simulated
    monkeypatch.setattr(GaussianFit, "finish",
                        lambda self, ctx, settings, locs: Finished(locs, changed=True))
    out = tmp_path / "finished.hdf5"
    GaussianFit().run(Context(), GaussianFitSettings(
        source=SourceSettings(path=recipe, stop=30),
        output=OutputSettings(path=str(out), raw_frames=4)))
    stored, = load_images(out)
    assert stored.frames.tolist() == [AVERAGE, 0, 10, 19, 29]


def test_opening_a_fitted_file_offers_its_frames_and_an_image_layer_shows_the_average(
        simulated):
    _, out, result = simulated
    session = Session()
    session.load(out)
    raw, = session.raw_images()
    assert raw.metadata["filenumber"] == 0
    assert all(not l.is_image for l in session.layers)   # a source, not a layer
    layer = session.add_image()
    assert layer.image is raw and layer.frame == 0
    np.testing.assert_allclose(raw.plane_at(0),
                               np.asarray(result.data["images"][0].data[0]))


def test_saving_keeps_the_frames_and_every_image_a_layer_shows(simulated, tmp_path):
    from smappy.images import ImageData
    _, out, _ = simulated
    session = Session()
    session.load(out)
    widefield = ImageData(np.arange(12, dtype=np.float32).reshape(3, 4), 100.0,
                          x0=50.0, y0=-20.0, name="widefield.tif", path="/x/widefield.tif")
    session.add_image(widefield)
    unused = ImageData(np.zeros((2, 2), np.float32), 10.0, name="tried.tif")
    session.images.append(unused)
    saved = tmp_path / "saved.hdf5"
    session.save(saved)

    again = Session()
    again.load(saved)
    names = [im.name for im in again.images]
    assert "widefield.tif" in names and "tried.tif" not in names
    assert len(again.raw_images()) == 1
    wf = next(im for im in again.images if im.name == "widefield.tif")
    assert (wf.pixelsize, wf.x0, wf.y0, wf.kind) == (100.0, 50.0, -20.0, "image")
    np.testing.assert_array_equal(np.asarray(wf.data), widefield.data)


def test_saving_over_the_open_file_keeps_its_frames(simulated, tmp_path):
    import shutil
    _, out, _ = simulated
    copy = tmp_path / "copy.hdf5"
    shutil.copy(out, copy)
    session = Session()
    session.load(copy)
    before = np.asarray(session.raw_images()[0].data)
    session.save(copy)                      # starts the file again
    stored, = load_images(copy)
    np.testing.assert_array_equal(np.asarray(stored.data), before)


def test_appended_files_keep_their_own_frames_and_removing_one_drops_its(simulated,
                                                                        tmp_path):
    import shutil
    _, out, _ = simulated
    second = tmp_path / "second.hdf5"
    shutil.copy(out, second)
    session = Session()
    session.load(out)
    session.load(second, append=True)
    assert [im.metadata["filenumber"] for im in session.raw_images()] == [0, 1]
    both = tmp_path / "both.hdf5"
    session.save(both)
    again = Session()
    again.load(both)
    assert [im.metadata["filenumber"] for im in again.raw_images()] == [0, 1]
    session.remove_file(0)
    assert [im.metadata["filenumber"] for im in session.raw_images()] == [0]


def test_a_table_saved_without_images_opens_without_any(tmp_path):
    from smappy.locs import Localizations
    path = tmp_path / "plain.hdf5"
    save_localizations(path, Localizations({"x_nm": np.zeros(3), "y_nm": np.zeros(3),
                                            "frame": np.arange(3)}, {}))
    session = Session()
    session.load(path)
    assert session.images == []
    with pytest.raises(ValueError, match="no image"):
        session.add_image()
