"""Image stacks: what their planes are, read from the file or asked; a slice
or a projection in 2D; a plane, a MIP or a volume in the 3D view."""
import numpy as np
import pytest
import tifffile

from smappy.images import ImageData, assign_axes, load_images
from smappy.render import DisplaySettings, FieldOfView


def _zstack(centres, nz=21, n=40, px=100.0, step=50.0, z0=-500.0):
    """Gaussian blobs at ``centres`` (x, y, z nm) in a z-stack, no background."""
    zz = z0 + np.arange(nz) * step
    yy = xx = (np.arange(n) + 0.5) * px
    Z, Y, X = np.meshgrid(zz, yy, xx, indexing="ij")
    vol = sum(np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * 150.0 ** 2)
                     - (Z - z) ** 2 / (2 * 100.0 ** 2)) for x, y, z in centres)
    image = ImageData(vol.astype(np.float32), px)
    image.axis, image.z0_nm, image.z_step_nm = "z", z0, step
    return image


def test_a_hyperstacks_channels_are_split_and_its_z_step_read(tmp_path):
    data = np.random.default_rng(0).random((5, 2, 8, 8)).astype(np.float32)
    path = tmp_path / "zc.tif"
    tifffile.imwrite(path, data, imagej=True, resolution=(10, 10),
                     metadata={"axes": "ZCYX", "spacing": 0.05, "unit": "um"})
    images = load_images(path)
    assert [im.name for im in images] == ["zc.tif: C1", "zc.tif: C2"]
    assert all(im.is_zstack and im.z_step_nm == pytest.approx(50.0) for im in images)
    assert images[0].pixelsize == pytest.approx(100.0)
    assert np.array_equal(images[1].data, data[:, 1])


def test_a_time_series_knows_its_frame_time(tmp_path):
    path = tmp_path / "t.tif"
    tifffile.imwrite(path, np.zeros((4, 8, 8), np.float32), imagej=True,
                     resolution=(10, 10), metadata={"axes": "TYX", "finterval": 0.1,
                                                    "unit": "um"})
    image, = load_images(path)
    assert image.axis == "t" and image.frame_time_s == pytest.approx(0.1)
    assert image.label(2) == "t 0.2 s"


def test_plain_pages_are_unknown_until_said_interleaved_or_in_blocks(tmp_path):
    planes = (np.arange(6)[:, None, None] * np.ones((6, 4, 4))).astype(np.float32)
    path = tmp_path / "pages.tif"
    tifffile.imwrite(path, planes, photometric="minisblack")
    image, = load_images(path, pixelsize=100.0)
    assert image.axes_unknown and image.axis == ""
    c1, c2 = assign_axes(image, "z", n_channels=2, interleaved=True, z_step_nm=30.0)
    assert list(c1.data[:, 0, 0]) == [0, 2, 4] and list(c2.data[:, 0, 0]) == [1, 3, 5]
    assert c1.is_zstack and not c1.axes_unknown and c1.name == "pages.tif: C1"
    b1, _ = assign_axes(image, "t", n_channels=2, interleaved=False, frame_time_s=0.5)
    assert list(b1.data[:, 0, 0]) == [0, 1, 2] and b1.axis == "t"
    with pytest.raises(ValueError, match="do not make 4 channels"):
        assign_axes(image, "z", n_channels=4)


def test_a_projection_is_of_the_range_of_planes_asked_for():
    rng = np.random.default_rng(1)
    image = ImageData(rng.random((6, 5, 5)).astype(np.float32), 100.0)
    assert np.allclose(image.projection("max"), image.data.max(axis=0))
    assert np.allclose(image.projection("mean", (1, 3)), image.data[1:4].mean(axis=0))
    assert np.allclose(image.shown_plane("sum", 0, (2, 2)), image.data[2])
    assert np.allclose(image.shown_plane("slice", 4), image.data[4])


def test_the_plane_at_a_height_is_the_nearest_inside_the_stack():
    image = _zstack([(0, 0, 0)])
    assert image.z_range() == (-525.0, 525.0)
    assert image.index_at_z(-500) == 0 and image.index_at_z(20) == 10
    assert image.index_at_z(-530) is None and image.index_at_z(600) is None
    assert image.label(10) == "z 0 nm"


def test_a_file_opened_in_a_session_is_a_layer_per_channel_in_channel_colours(tmp_path):
    from smappy.locs import Localizations
    from smappy.session import Session
    path = tmp_path / "zc.tif"
    tifffile.imwrite(path, np.zeros((3, 2, 8, 8), np.float32), imagej=True,
                     resolution=(10, 10), metadata={"axes": "ZCYX", "unit": "um"})
    session = Session(Localizations({"x_nm": np.zeros(2), "y_nm": np.zeros(2)}, {}))
    session.open_image(path)
    layers = [layer for layer in session.layers if layer.is_image]
    assert [layer.display.lut for layer in layers] == ["green", "magenta"]


def test_a_slice_that_follows_z_shows_the_plane_the_z_filter_is_centred_on():
    from smappy.layer_state import snapshot_layers
    from smappy.locs import Localizations
    from smappy.session import Session
    locs = Localizations({"x_nm": np.zeros(3), "y_nm": np.zeros(3),
                          "z_nm": np.array([-100.0, 0.0, 100.0])}, {})
    session = Session(locs)
    layer = session.add_image(_zstack([(0, 0, 0)]))
    layer.set_image_view(follow_z=True)
    session.layers[0].set_bound("z_nm", 150.0, 250.0)
    snap = snapshot_layers(session.layers)[-1]
    assert snap.frame == 14                         # z = -500 + 14 x 50 = 200 nm
    layer.set_image_view(follow_z=False)
    assert snapshot_layers(session.layers)[-1].frame == layer.frame


def _peak(weight):
    """The centroid of the brightest part, in pixels (x, y)."""
    w = np.where(weight > 0.5 * weight.max(), weight, 0.0)
    ys, xs = np.indices(w.shape)
    return (w * xs).sum() / w.sum(), (w * ys).sum() / w.sum()


@pytest.mark.parametrize("angles", [(0, 0, None), (0, 90, None), (30, 50, None),
                                    (30, 50, 8000.0)])
@pytest.mark.parametrize("mode", ["max", "volume"])
def test_a_blob_in_a_volume_is_drawn_where_the_projection_puts_it(angles, mode):
    from smappy.view3d import Projection, render_volume_3d
    centre = (1000.0, 1500.0, 300.0)
    image = _zstack([centre])
    az, el, focal = angles
    proj = Projection(azimuth=az, elevation=el, focal=focal, pivot=[2000, 2000, 0], zoom=20)
    fov = proj.fov(200, 160)
    _, planes = render_volume_3d(image, DisplaySettings(), proj, None, fov, mode)
    xv, yv, _ = proj.apply([centre[0]], [centre[1]], [centre[2]])
    expected = ((xv[0] - fov.x0) / fov.pixelsize - 0.5, (yv[0] - fov.y0) / fov.pixelsize - 0.5)
    assert np.allclose(_peak(planes.weight), expected, atol=1.5)


def test_from_the_top_the_volumes_mip_is_the_stacks_own():
    from smappy.view3d import Projection, render_volume_3d
    image = _zstack([(1000, 1500, 300), (2500, 2500, -200)])
    proj = Projection(pivot=[2000, 2000, 0], zoom=100.0)
    fov = FieldOfView(-2000.0, -2000.0, 100.0, 40, 40)       # the stack's own pixels
    _, planes = render_volume_3d(image, DisplaySettings(), proj, None, fov, "max")
    assert np.allclose(planes.weight, image.projection("max"), atol=0.02)


def test_opacity_lets_the_near_blob_hide_the_far_one():
    from smappy.view3d import Projection, render_volume_3d
    one = _zstack([(2000, 2000, 300)])
    two = _zstack([(2000, 2000, 300), (2000, 2000, -300)])
    fov = FieldOfView(-500.0, -500.0, 50.0, 20, 20)

    def centre(image, opacity):
        proj = Projection(pivot=[2000, 2000, 0], opacity=opacity)
        _, planes = render_volume_3d(image, DisplaySettings(), proj, None, fov, "volume")
        return planes.weight[10, 10]

    assert centre(two, 0.0) == pytest.approx(2 * centre(one, 0.0), rel=0.02)
    assert centre(two, 1.0) < 1.2 * centre(one, 1.0)       # the back one is hidden


def test_as_a_plane_a_zstack_shows_the_plane_at_the_pivots_height():
    from smappy.view3d import Projection, Slab, render_image_3d
    image = _zstack([(2050, 2050, 300)])                   # on a pixel's centre
    fov = FieldOfView(-500.0, -500.0, 50.0, 20, 20)
    at = lambda z: render_image_3d(image, 0, DisplaySettings(), Projection(pivot=[2050, 2050, z]),
                                   None, fov)[1].weight[10, 10]
    assert at(300.0) > 0.95 and at(0.0) < 0.05
    assert at(700.0) == 0.0                                  # above the stack


def _micromanager_tiff(path, roi="271-195-193-178", pixel_um=0.127, shape=(178, 193)):
    """What Micro-Manager 2 writes: XResolution per centimetre
    (ResolutionUnit 3) beside an ImageJ unit of "um", the ROI and the pixel
    size in each image's own tags."""
    import json
    per_cm = round(1e4 / pixel_um)
    tags = {"ROI": roi, "PixelSizeUm": pixel_um, "Binning": 1}
    tifffile.imwrite(path, np.ones(shape, np.uint16), imagej=True,
                     resolution=(per_cm, per_cm), resolutionunit="CENTIMETER",
                     metadata={"unit": "um"},
                     extratags=[(51123, "s", 0, json.dumps(tags), True)])


def test_a_resolution_per_centimetre_is_read_as_such_whatever_imagejs_unit_says(tmp_path):
    path = tmp_path / "cm.tif"
    tifffile.imwrite(path, np.ones((4, 4), np.uint16), imagej=True,
                     resolution=(78740, 78740), resolutionunit="CENTIMETER",
                     metadata={"unit": "um"})
    assert load_images(path)[0].pixelsize == pytest.approx(127.0, rel=1e-4)
    # ImageJ's own files say "none" and the unit is ImageJ's
    tifffile.imwrite(path, np.ones((4, 4), np.uint16), imagej=True,
                     resolution=(10, 10), metadata={"unit": "um"})
    assert load_images(path)[0].pixelsize == pytest.approx(100.0)


def test_a_micromanager_image_takes_its_own_pixel_size_and_lies_where_its_roi_is(tmp_path):
    from smappy.metadata import CameraMetadata
    from smappy.rawframes import RawFrameKeeper
    path = tmp_path / "wf_MMStack_Pos0.ome.tif"
    _micromanager_tiff(path)
    image = load_images(path)[0]
    assert image.pixelsize == pytest.approx(127.0)
    # exactly where the fitter puts a movie's frames taken with the same ROI
    camera = CameraMetadata(pixelsize_um=0.127, roi=(271, 195, 193, 178),
                            conversion=1.0, offset=0.0)
    kept = RawFrameKeeper(keep=1)
    kept.see(0, np.ones((1, 178, 193), np.float32))
    raw = kept.image(camera)
    assert (image.x0, image.y0) == pytest.approx((raw.x0, raw.y0))
    assert (image.x0, image.y0) == pytest.approx((270.5 * 127, 194.5 * 127))
    assert image.bounds == pytest.approx(raw.bounds)
