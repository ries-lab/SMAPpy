"""Analysis selection, provenance and an end-to-end synthetic NPC workflow."""

import numpy as np
import pytest

from smappy.io.hdf5 import save_localizations
from smappy.locs import Localizations
from smappy.group import GroupSettings
from smappy.roi_manager import ROIProject, DensityPeaks, Histograms
from smappy.roi_manager.pipeline import Step, errors, merged_values
from smappy.plugins.roi import HistogramSettings, histograms


def table(x, y=None):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    return Localizations({'x_nm': x, 'y_nm': np.zeros(n) if y is None else np.asarray(y, dtype=float),
                          'xy_err_nm': np.arange(n, dtype=float) + 10,
                          'photons': np.arange(n, dtype=float) * 100 + 100,
                          'frame': np.arange(n, dtype=np.int64)}, {'units': 'nm'})


def test_circle_square_polygon_boundaries_filters_and_file_isolation():
    p = ROIProject()
    a = p.add_source(table([0, 100, 150, 150.01, -150, 100, 0], [0, 100, 0, 0, 0, -100, 151]))
    b = p.add_source(table([0]))
    p.set_filters({})
    roi = p.add_roi(a.id, [0, 0])
    assert p.indices(roi).tolist() == [0, 1, 2, 4, 5]
    p.set_geometry(200, 'square')
    assert p.indices(roi).tolist() == [0, 1, 5]
    roi.polygon = [[0, 0], [150, 0], [0, 150]]
    assert p.indices(roi).tolist() == [0, 2]
    p.set_filters({'photons': [200, None], 'filenumber': [99, 99]})
    assert p.indices(roi).tolist() == [2]
    assert not p.state(b.id).filter.mask.any()
    p.state(a.id).display.contrast = 1
    assert p.indices(roi).tolist() == [2]


def test_statistics_review_and_stale_results():
    p = ROIProject()
    s = p.add_source(table([0, 10, 140]))
    p.set_filters({})
    roi = p.add_roi(s.id, [0, 0], reviewed=False)
    assert not p.evaluate()['records']
    roi.reviewed = True
    run = p.evaluate()
    values = merged_values(run['records'][roi.id])
    assert values['n_localizations'] == 3
    assert [step['label'] for step in run['pipeline']] == ['Statistics']
    assert values['mean_precision_nm'] == 11
    assert values['mean_photons'] == 200
    assert len(p.results()) == 1
    roi.comment = 'Good pore'
    assert not p.stale_steps(roi.id)
    p.navigation['detail_center'] = [1000, 1000]
    p.state(s.id).display.contrast = 2
    assert not p.stale_steps(roi.id)
    p.set_geometry(200)
    assert p.stale_steps(roi.id)
    assert not p.results()
    p.evaluate()
    assert p.results()[0]['n_localizations'] == 2
    p.set_filters({'photons': [200, None]})
    assert p.stale_steps(roi.id)
    p.evaluate()
    assert p.results()[0]['n_localizations'] == 1
    roi.use = False
    assert not p.results()
    assert not p.evaluate()['records']
    assert len(p.runs) == 5


def test_polygon_ignores_global_size_but_translation_moves_all_geometry():
    p = ROIProject()
    s = p.add_source(table([0, 10, 1000]))
    roi = p.add_roi(s.id, [0, 0], polygon=[[-20, -20], [20, -20], [20, 20], [-20, 20]])
    roi.direction = [[0, 0], [10, 0]]
    p.evaluate()
    p.set_geometry(500, 'square')
    assert not p.stale_steps(roi.id)
    p.move_roi(roi.id, [1000, 0])
    assert roi.direction == [[1000, 0], [1010, 0]]
    assert p.indices(roi).tolist() == [2]
    assert p.stale_steps(roi.id)


def test_empty_and_nonfinite_measurements():
    p = ROIProject()
    locs = table([0, 10, 100])
    locs['photons'][1] = np.nan
    p.set_filters({})
    s = p.add_source(locs)
    p.add_roi(s.id, [0, 0])
    p.add_roi(s.id, [1000, 1000])
    p.evaluate()
    rows = p.results()
    assert rows[0]['mean_photons'] == 200
    assert rows[0]['mean_photons_n'] == 2
    assert rows[1]['n_localizations'] == 0
    assert np.isnan(rows[1]['mean_photons'])
    hist = histograms(rows, HistogramSettings(bins=10))
    assert hist['n_localizations']['counts'].sum() == 2
    assert hist['mean_photons']['counts'].sum() == 1
    assert hist['mean_photons']['missing'] == 1


def test_grouped_render_and_analysis_share_filters_after_regrouping():
    p = ROIProject()
    p.set_filters({'photons': [250, None]})
    s = p.add_source(table([0, 1, 1000]))
    roi = p.add_roi(s.id, [0, 0])
    assert len(p.extract(roi)) == 0
    p.grouped = True
    assert len(p.extract(roi)) == 1
    assert p.extract(roi)['photons'][0] == 300
    assert p.state(s.id).filter.ranges['photons'] == (250, None)
    p.evaluate()
    p.group_settings = GroupSettings(dx=.1)
    assert len(p.extract(roi)) == 0
    assert p.stale_steps(roi.id)
    assert p.state(s.id).filter.ranges['photons'] == (250, None)


def test_save_reload_relink_and_changed_source_detection(tmp_path):
    source_path = tmp_path / 'locs.h5'
    locs = table([0, 10, 150])
    save_localizations(source_path, locs)
    p = ROIProject()
    s = p.add_file(source_path)
    roi = p.add_roi(s.id, [0, 0])
    roi.comment = 'NPC α'
    p.evaluate()
    p.navigation = {'detail_center': [300, 400]}
    path = p.save(tmp_path / 'project.h5')
    loaded = ROIProject.load(path)
    assert loaded.rois[roi.id].comment == 'NPC α'
    assert loaded.results() == p.results()
    assert loaded.navigation == p.navigation
    renamed = tmp_path / 'renamed.h5'
    source_path.rename(renamed)
    loaded = ROIProject.load(path, {s.id: renamed})
    assert loaded.results() == p.results()
    save_localizations(renamed, table([99, 10, 150]))
    with pytest.raises(ValueError, match='contents changed'):
        ROIProject.load(path, {s.id: renamed})
    with pytest.raises(ValueError, match='overwrite'):
        p.save(source_path)


def test_units_validation_and_invalid_geometry():
    p = ROIProject()
    with pytest.raises(ValueError, match='nm coordinates'):
        p.add_source(Localizations({'x_pix': np.array([1.]), 'y_pix': np.array([2.])}))
    s = p.add_source(Localizations({'x_pix': np.array([1.]), 'y_pix': np.array([2.])},
                                    {'pixelsize_nm': 100.}))
    assert s.state.locs['x_nm'][0] == 100
    with pytest.raises(ValueError):
        p.set_geometry(-1)
    with pytest.raises(ValueError):
        p.add_roi(s.id, [np.nan, 0])
    with pytest.raises(ValueError):
        p.add_roi(s.id, [0, 0], polygon=[[0, 0], [1, 0], [2, 0]])


def pores():
    rng = np.random.default_rng(4)
    centers = np.array([[300, 300], [800, 300], [300, 800], [800, 800]], dtype=float)
    xy = []
    for i, center in enumerate(centers):
        if i % 2 == 0:
            angle = np.linspace(0, 2 * np.pi, 30, endpoint=False)
            offsets = 50 * np.column_stack((np.cos(angle), np.sin(angle))) + rng.normal(0, 4, (30, 2))
        else:
            offsets = rng.normal(0, 20, (30, 2))
        xy.extend(center + offsets)
    xy = np.asarray(xy)
    locs = table(xy[:, 0], xy[:, 1])
    locs.columns['xy_err_nm'][:] = 12
    locs.columns['photons'][:] = 1000
    return locs, centers


def test_find_rings_and_blobs_review_evaluate_histogram():
    locs, centers = pores()
    p = ROIProject()
    s = p.add_source(locs)
    rois = p.find(s.id)
    assert len(rois) == 4
    for roi in rois:
        assert np.min(np.linalg.norm(centers - roi.center, axis=1)) < 15
    assert not p.find(s.id)  # repeat finding does not duplicate existing ROIs
    assert not p.evaluate()['records']
    for roi in rois:
        roi.reviewed = True
    p.evaluate()
    assert [row['n_localizations'] for row in p.results()] == [30] * 4
    hist = histograms(p.results(), HistogramSettings(bins=10))
    assert hist['n_localizations']['counts'].sum() == 4
    p.set_filters({'photons': [2000, None]})
    assert not p.find(s.id)
    assert not p.results()


def test_a_failing_step_costs_its_columns_and_not_the_run():
    from smappy.plugins import Plugin, Result

    class Mean(Plugin):
        name, version = 'example', '1'
        scope = 'site'

        def run(self, ctx, settings):
            if not len(ctx.locs):
                raise ValueError('empty ROI')
            return Result(data={'x': float(ctx.locs['x_nm'].mean())})

    p = ROIProject()
    s = p.add_source(table([0]))
    empty = p.add_roi(s.id, [1000, 0])
    good = p.add_roi(s.id, [0, 0])
    step = Step(label='Mean', path='Test/Mean', plugin=Mean(), settings=None)
    run = p.evaluate(steps=[step])
    assert list(errors(run['records'][empty.id])) == ['Mean']
    assert not errors(run['records'][good.id])
    rows = p.results()
    assert [row['roi_id'] for row in rows] == [good.id]
    assert rows[0]['x'] == 0.0


def test_two_steps_share_a_row_and_collisions_are_qualified():
    from smappy.plugins import Plugin, Result

    class Count(Plugin):
        scope = 'site'

        def run(self, ctx, settings):
            return Result(data={'n': len(ctx.locs), 'only_here': 1})

    p = ROIProject()
    s = p.add_source(table([0, 10]))
    p.set_filters({})
    roi = p.add_roi(s.id, [0, 0])
    steps = [Step(label='first', path='T/C', plugin=Count(), settings=None),
             Step(label='second', path='T/C', plugin=Count(), settings=None)]
    p.evaluate(steps=steps)
    row = p.results()[0]
    # 'n' is produced by both steps, so each is qualified; nothing is lost
    assert row['first.n'] == 2 and row['second.n'] == 2
    assert 'n' not in row
    assert row['first.only_here'] == 1 and row['second.only_here'] == 1
