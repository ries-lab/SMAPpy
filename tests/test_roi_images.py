"""ROI images: every ROI drawn at one scale, turned to its line, written as
a PDF of pages or a TIFF stack."""
import numpy as np
import pytest
import yaml

from smappy.io.formats import FileInfo
from smappy.plugins.roi_images import (RoiImages, RoiImagesSettings, line_angle,
                                       page_grid, to_picture)
from smappy.session import Session
from smappy.simulate import simulate
from smappy.simulate.settings import SimulationSettings, StructureSettings

ARROW = [[-50, 0, 0], [50, 0, 0], [50, 40, 0]]


def arrows(tmp_path, n=12, seed=3):
    """Arrows turned at random, an ROI on each with a line along its arrow."""
    spec = {"name": "arrow", "elements": [{"points": ARROW}],
            "copies": {"n": n, "placement": "grid", "rotation": "random",
                       "field": [0, 0, 4000, 4000]}}
    path = tmp_path / "arrow.yaml"
    path.write_text(yaml.safe_dump(spec))
    locs = simulate(SimulationSettings(n_frames=3000, seed=seed,
                                       structure=StructureSettings(file=str(path))))
    session = Session()
    session.add_file(locs, FileInfo(name="arrows", path="arrows", format="simulated",
                                    n=len(locs)))
    session.show_grouped(0, False)
    project = session.rois
    project.set_geometry(250)
    file_id = next(iter(project.sources))
    copies = locs.metadata["copies"]
    for x, y, gamma in zip(copies["x_nm"], copies["y_nm"], copies["gamma_deg"]):
        roi = project.add_roi(file_id, [x, y])
        a = np.radians(gamma)
        project.set_line(roi.id, "direction", [[x, y], [x + 60 * np.cos(a),
                                                        y + 60 * np.sin(a)]])
    return session


def test_a_point_on_the_line_lands_to_the_right_of_the_centre():
    angle = line_angle([[0, 0], [0, 10]])                  # straight up in data y
    assert angle == pytest.approx(np.pi / 2)
    px = to_picture([[0, 30]], [0, 0], angle, half=50.0, pixelsize=1.0)
    np.testing.assert_allclose(px, [[80.0, 50.0]], atol=1e-9)
    assert line_angle([[1, 1], [1, 1]]) is None


def test_pictures_are_laid_out_page_by_page():
    assert page_grid(10, 2, 3) == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9]]
    assert page_grid(0, 2, 3) == []


def _alike(pictures):
    """The mean correlation of each picture's brightness with the others'."""
    grey = [p.image.astype(float).sum(axis=2).ravel() for p in pictures]
    grey = [(g - g.mean()) / (g.std() or 1.0) for g in grey]
    r = [np.mean(a * b) for i, a in enumerate(grey) for b in grey[i + 1:]]
    return float(np.mean(r))


def test_turned_to_their_lines_the_arrows_all_point_the_same_way(tmp_path):
    session = arrows(tmp_path)
    settings = RoiImagesSettings(path=str(tmp_path / "rois.pdf"), size_nm=250.0,
                                 pixelsize_nm=5.0)
    as_they_lie = session.run(RoiImages(), settings).data["pictures"]
    settings.turn_to = "direction"
    turned = session.run(RoiImages(), settings).data["pictures"]
    assert len(turned) == 12 and turned[0].image.shape == (50, 50, 3)
    assert _alike(turned) > 0.4 > _alike(as_they_lie)
    # the line is drawn along the picture's x, from its centre to the right
    line = turned[0].lines["direction"]
    np.testing.assert_allclose(line[:, 1], 25.0, atol=1e-6)
    assert line[1, 0] - line[0, 0] == pytest.approx(60 / 5.0)


def test_the_pdf_has_a_page_per_grid_and_the_tiff_a_plane_per_roi(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    session = arrows(tmp_path)
    rois = list(session.rois.rois.values())
    rois[0].use = False
    settings = RoiImagesSettings(path=str(tmp_path / "rois.pdf"), size_nm=300.0,
                                 pixelsize_nm=5.0, columns=2, rows=2)
    result = session.run(RoiImages(), settings)
    pdf = (tmp_path / "rois.pdf").read_bytes()
    assert pdf.startswith(b"%PDF") and pdf.count(b"/Type /Page\n") + \
        pdf.count(b"/Type /Page ") + pdf.count(b"/Type/Page") >= 3     # 11 ROIs, 4 a page
    assert len(result.data["pictures"]) == 11
    settings.path = str(tmp_path / "rois.tif")
    session.run(RoiImages(), settings)
    with tifffile.TiffFile(tmp_path / "rois.tif") as tif:
        stack = tif.asarray()
        labels = tif.imagej_metadata["Labels"]
        x_res = tif.pages[0].tags["XResolution"].value
    assert stack.shape == (11, 60, 60, 3) and stack.dtype == np.uint8
    assert labels[0] == f"ROI {rois[1].number}"
    assert x_res[0] / x_res[1] == pytest.approx(1000 / 5.0)
    # the outline is drawn into the pixels, in the ROI manager's yellow
    assert ((stack[..., 0] == 255) & (stack[..., 1] == 255) & (stack[..., 2] == 0)).any()


def test_the_titles_carry_what_was_measured(tmp_path):
    session = arrows(tmp_path, n=4)
    from smappy.chain import Step
    session.rois.evaluate([Step(plugin="Analysis/Measure/Localization Statistics")])
    settings = RoiImagesSettings(path=str(tmp_path / "rois.pdf"), pixelsize_nm=5.0,
                                 title_columns=["n_localizations"])
    pictures = session.run(RoiImages(), settings).data["pictures"]
    assert all(p.title.startswith(f"ROI {p.number}, n_localizations ") for p in pictures)


def test_the_form_counts_the_pictures_and_offers_the_lines(tmp_path):
    session = arrows(tmp_path, n=4)
    state = RoiImages().describe(session.context(),
                                 RoiImagesSettings(path="x.pdf", pixelsize_nm=5.0))
    assert "4 ROIs of 150 x 150 pixels, 1 page" in state.notice.text   # 3 x 250 nm
    assert ("direction", "direction") in state.choices["turn_to"]
    tiff = RoiImages().describe(session.context(), RoiImagesSettings(path="x.tif"))
    assert not tiff.active["columns"]


def test_without_rois_it_says_where_they_come_from(tmp_path):
    session = Session(simulate(SimulationSettings(n_frames=200)))
    with pytest.raises(ValueError, match="no ROI"):
        session.run(RoiImages(), RoiImagesSettings(path=str(tmp_path / "x.pdf")))
