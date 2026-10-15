"""Analysis selection, provenance and an end-to-end synthetic NPC workflow."""

import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.group import GroupSettings
from smappy.roi_manager import Histograms
from smappy.roi_manager.pipeline import Step, errors, merged_values
from smappy.plugins.roi import HistogramSettings, histograms
from test_roi_session import rois_of, set_bounds              # noqa: E402


def table(x, y=None):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    return Localizations({'x_nm': x, 'y_nm': np.zeros(n) if y is None else np.asarray(y, dtype=float),
                          'xy_err_nm': np.arange(n, dtype=float) + 10,
                          'photons': np.arange(n, dtype=float) * 100 + 100,
                          'frame': np.arange(n, dtype=np.int64)}, {'units': 'nm'})


def test_circle_square_polygon_boundaries_filters_and_file_isolation():
    p, a = rois_of(table([0, 100, 150, 150.01, -150, 100, 0], [0, 100, 0, 0, 0, -100, 151]),
                   table([0]), filters={})
    b = p.source_of(1)
    roi = p.add_roi(a.id, [0, 0])
    assert p.indices(roi).tolist() == [0, 1, 2, 4, 5]
    p.set_geometry(200, 'square')
    assert p.indices(roi).tolist() == [0, 1, 5]
    roi.polygon = [[0, 0], [150, 0], [0, 150]]
    assert p.indices(roi).tolist() == [0, 2]
    set_bounds(p, {'photons': (200, None)})
    assert p.indices(roi).tolist() == [2]
    assert not p.state(b.id).filter.mask.any()
    p.state(a.id).display.contrast = 1
    assert p.indices(roi).tolist() == [2]


def test_statistics_use_and_stale_results():
    p, s = rois_of(table([0, 10, 140]), filters={})
    roi = p.add_roi(s.id, [0, 0])
    roi.use = False
    assert not p.evaluate()['records']
    roi.use = True
    run = p.evaluate()
    values = merged_values(run['records'][roi.id])
    assert values['n_localizations'] == 3
    assert [step['label'] for step in run['pipeline']] == ['Localization Statistics']
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
    set_bounds(p, {'photons': (200, None)})
    assert p.stale_steps(roi.id)
    p.evaluate()
    assert p.results()[0]['n_localizations'] == 1
    roi.use = False
    assert not p.results()
    assert not p.evaluate()['records']
    assert len(p.runs) == 5


def test_polygon_ignores_global_size_but_translation_moves_all_geometry():
    p, s = rois_of(table([0, 10, 1000]))
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
    locs = table([0, 10, 100])
    locs['photons'][1] = np.nan
    p, s = rois_of(locs, filters={})
    p.add_roi(s.id, [0, 0])
    p.add_roi(s.id, [1000, 1000])
    p.evaluate()
    rows = p.results()
    assert rows[0]['mean_photons'] == 200
    assert rows[1]['n_localizations'] == 0
    assert np.isnan(rows[1]['mean_photons'])
    hist = histograms(rows, HistogramSettings(bins=10))
    assert hist['n_localizations']['counts'].sum() == 2
    assert hist['mean_photons']['counts'].sum() == 1
    assert hist['mean_photons']['missing'] == 1


def test_grouped_render_and_analysis_share_filters_after_regrouping():
    p, s = rois_of(table([0, 1, 1000]), filters={'photons': (250, None)})
    p.session.set_group_settings(GroupSettings(dx=1.5, dt=1))
    roi = p.add_roi(s.id, [0, 0])
    assert len(p.extract(roi)) == 0
    set_bounds(p, grouped=True)
    assert len(p.extract(roi)) == 1
    assert p.extract(roi)['photons'][0] == 300
    assert p.state(s.id).filter.ranges['photons'] == (250, None)
    p.evaluate()
    p.session.set_group_settings(GroupSettings(dx=.1, dt=1))
    p.sync()
    assert len(p.extract(roi)) == 0
    assert p.stale_steps(roi.id)
    assert p.state(s.id).filter.ranges['photons'] == (250, None)


def test_invalid_geometry_is_refused():
    p, s = rois_of(table([0]))
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
    p, s = rois_of(locs)
    rois = p.find(s.id)
    assert len(rois) == 4
    for roi in rois:
        assert np.min(np.linalg.norm(centers - roi.center, axis=1)) < 15
    assert not p.find(s.id)  # repeat finding does not duplicate existing ROIs
    p.evaluate()
    assert [row['n_localizations'] for row in p.results()] == [30] * 4
    hist = histograms(p.results(), HistogramSettings(bins=10))
    assert hist['n_localizations']['counts'].sum() == 4
    set_bounds(p, {'photons': (2000, None)})
    assert not p.find(s.id)
    assert not p.results()


def test_a_failing_step_costs_its_columns_and_not_the_run():
    from smappy.plugins import Plugin, Result

    class Mean(Plugin):
        changes_locs = False
        name, version = 'example', '1'
        per_region = True

        def run(self, ctx, settings):
            locs = ctx.selection.apply(ctx.locs)
            if not len(locs):
                raise ValueError('empty ROI')
            return Result(values={'x': float(locs['x_nm'].mean())})

    p, s = rois_of(table([0]))
    empty = p.add_roi(s.id, [1000, 0])
    good = p.add_roi(s.id, [0, 0])
    step = Step(label='Mean', path='Test/Mean', plugin=Mean(), settings=None)
    run = p.evaluate(steps=[step])
    assert list(errors(run['records'][empty.id])) == ['Mean']
    assert not errors(run['records'][good.id])
    rows = p.results()
    assert [row['roi_id'] for row in rows] == [good.number]
    assert rows[0]['x'] == 0.0


def test_two_steps_share_a_row_and_collisions_are_qualified():
    from smappy.plugins import Plugin, Result

    class Count(Plugin):
        changes_locs = False
        per_region = True

        def run(self, ctx, settings):
            return Result(values={'n': len(ctx.selection), 'only_here': 1})

    p, s = rois_of(table([0, 10]), filters={})
    roi = p.add_roi(s.id, [0, 0])
    steps = [Step(label='first', path='T/C', plugin=Count(), settings=None),
             Step(label='second', path='T/C', plugin=Count(), settings=None)]
    p.evaluate(steps=steps)
    row = p.results()[0]
    # 'n' is produced by both steps, so each is qualified; nothing is lost
    assert row['first.n'] == 2 and row['second.n'] == 2
    assert 'n' not in row
    assert row['first.only_here'] == 1 and row['second.only_here'] == 1


def test_found_rois_are_evaluated_without_a_review_step():
    """find() then evaluate(), as a script writes it, measures what was
    found: there is no review flag for evaluate() to pass over."""
    locs, _ = pores()
    p, s = rois_of(locs)
    p.find(s.id)
    p.evaluate()
    assert len(p.results()) == 4


def test_the_default_histograms_draw_every_number_the_statistics_gave():
    locs, _ = pores()
    p, s = rois_of(locs)
    p.session.layers[0].show_grouped(True)
    p.session.layers[0].locs                 # merged once: `n_in_group` is written
    p.session.layers[0].show_grouped(False)
    p.find(s.id)
    p.evaluate()
    drawn = histograms(p.results(), HistogramSettings(bins=10))
    assert set(drawn) == {'n_localizations', 'mean_photons', 'median_photons',
                          'photons_n0', 'mean_precision_nm', 'median_precision_nm',
                          'sigma_c_nm', 'on_time_frames', 'n_blinks'}


def test_results_live_in_a_table_per_evaluation_and_runs_are_only_a_log():
    p, s = rois_of(table([0, 10, 1000]), filters={})
    a = p.add_roi(s.id, [0, 0])
    b = p.add_roi(s.id, [1000, 0])
    run = p.evaluate()
    assert set(run['records']) == {a.id, b.id}           # the caller still gets them
    assert 'records' not in p.runs[-1] and p.runs[-1]['regions'] == 2
    kept = p.tables['Localization Statistics']
    assert kept.ids == [a.id, b.id]
    assert kept.columns()['n_localizations'].tolist() == [2, 1]
    assert kept.step(a.id)['plugin'] == 'Analysis/Measure/Localization Statistics'


def test_a_removed_roi_keeps_its_numbers_for_an_undo_but_not_in_the_file():
    p, s = rois_of(table([0, 1000]), filters={})
    a = p.add_roi(s.id, [0, 0])
    b = p.add_roi(s.id, [1000, 0])
    p.evaluate()
    with p.edit('Remove ROI 2'):
        del p.rois[b.id]
    assert len(p.results()) == 1
    p.undo()
    assert len(p.results()) == 2                         # back, with its numbers
    del p.rois[b.id]
    saved = p.tables_dict(p.rois)
    assert [doc['ids'] for doc in saved] == [[a.id]]
    assert b.id in p.tables['Localization Statistics']   # the live table is untouched


def test_an_older_files_runs_with_their_records_become_tables():
    p, s = rois_of(table([0, 10]), filters={})
    roi = p.add_roi(s.id, [0, 0])
    step = {'label': 'old', 'plugin': 'X/Y', 'version': '1', 'parameters': {}}
    p.absorb_run({'id': 'r', 'pipeline': [step],
                  'records': {roi.id: {'signature': 'd', 'steps': {
                      'old': {'values': {'n': 2}, 'signature': 's'}}}}})
    assert p.runs[-1] == {'id': 'r', 'pipeline': [step], 'regions': 1}
    assert p.tables['old'].entry(roi.id) == {'values': {'n': 2}, 'signature': 's'}
    assert p.tables['old'].step(roi.id) == step


def test_the_region_histograms_are_a_titled_grid_and_say_which_measures_had_no_values(
        monkeypatch):
    """Twelve panels stacked in one column overlapped and named nothing."""
    from matplotlib.figure import Figure
    from smappy.plugins import Context
    from smappy.plugins.roi import Histograms, grid_shape, measure_title
    rng = np.random.default_rng(0)
    rows = [{"roi_id": i, "file_id": "", "radius_nm": float(rng.normal(50, 3)),
             "n_corners": int(rng.integers(0, 9)), "angle_deg": float(rng.uniform(0, 45)),
             "efficiency": float("nan")} for i in range(40)]
    monkeypatch.setattr(Histograms, "rows", lambda self, ctx, s: (rows, None))
    result = Histograms().run(Context(), HistogramSettings(bins=10))
    assert "no values for efficiency" in result.text
    figure = Figure()
    result.plot.draw(figure)
    titles = [a.get_title() for a in figure.axes if a.get_visible()]
    assert titles == ["radius (nm)", "n corners", "angle (°)"]
    assert grid_shape(12) == (3, 4) and grid_shape(4) == (2, 2)
    assert measure_title("sigma_c_nm") == "sigma c (nm)"
