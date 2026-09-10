"""Three linked views with an independent controls window, like smappy-view."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.widgets import Button, TextBox, RadioButtons, CheckButtons, PolygonSelector

from ..render import FieldOfView
from ..viewer import _window_title
from .core import ROI, ROIProject, digest, point, polygon_vertices, positive
from .plugins import DensityPeaks, Histograms


class ROIManager:
    page_size = 10

    def __init__(self, project=None, project_path=None):
        self.project = project if project is not None else ROIProject()
        self.file_id = next(iter(self.project.sources), None)
        self.selected_id = None
        self.candidate = None
        self.detail_center = [0., 0.]
        self.detail_size = 3000.
        self.preview_size = 900.
        self.grid_index = -1
        self.page = 0
        self.selector = None
        self.line_start = None
        self.mode = 'select'
        self.hist_figure = None
        self.hist_pick = {}
        self.hist_signature = None
        self.widgets = []
        self.table_ids = {}
        self.message = 'Click overview, click detail, recenter in preview, then Add.'
        self.figure = plt.figure(figsize=(14, 8))
        _window_title(self.figure, 'SMAPpy ROI manager')
        self.axes = [self.figure.add_axes([.055 + i * .325, .47, .26, .44]) for i in range(3)]
        self.table_ax = self.figure.add_axes([.045, .08, .93, .26])
        self.status = self.figure.text(.045, .02, self.message, fontsize=10)
        self.controls = plt.figure(figsize=(10, 9))
        _window_title(self.controls, 'ROI manager — controls')
        self.figure.canvas.mpl_connect('button_press_event', self._click)
        self.figure.canvas.mpl_connect('pick_event', self._pick)
        self.figure.canvas.mpl_connect('key_press_event', self._key)
        self.figure.canvas.mpl_connect('close_event', self._close)
        self._build_controls(project_path)
        self._restore_navigation()
        self.draw()

    @property
    def active(self):
        return self.candidate or self.project.rois.get(self.selected_id)

    def _button(self, rect, label, callback):
        button = Button(self.controls.add_axes(rect), label)
        button.on_clicked(lambda event: self._action(callback))
        self.widgets.append(button)
        return button

    def _box(self, rect, label, value):
        box = TextBox(self.controls.add_axes(rect), label, initial=str(value))
        self.widgets.append(box)
        return box

    def _action(self, callback):
        try:
            callback()
        except Exception as error:
            self.message = f'{type(error).__name__}: {error}'
        self.draw()

    def _build_controls(self, project_path):
        c = self.controls
        c.text(.04, .965, 'Files and project', weight='bold')
        self.source_box = self._box([.18, .91, .62, .035], 'Source .h5 ', '')
        self._button([.82, .91, .14, .035], 'Add file', self.add_file)
        self.path_box = self._box([.18, .855, .48, .035], 'Project .h5 ', project_path or 'rois.h5')
        self._button([.68, .855, .13, .035], 'Save', self.save)
        self._button([.83, .855, .13, .035], 'Open', self.open_project)
        self._button([.04, .80, .16, .035], 'Previous file', lambda: self.change_file(-1))
        self._button([.22, .80, .16, .035], 'Next file', lambda: self.change_file(1))
        self.file_label = c.text(.41, .81, '', fontsize=9)
        c.text(.04, .76, 'Geometry and navigation (nm)', weight='bold')
        self.size_box = self._box([.18, .705, .12, .035], 'ROI size ', self.project.size_nm)
        self.shape_widget = RadioButtons(c.add_axes([.32, .69, .13, .065]), ['circle', 'square'],
                                         active=int(self.project.shape == 'square'))
        self.widgets.append(self.shape_widget)
        self.detail_box = self._box([.59, .705, .11, .035], 'Detail width ', self.detail_size)
        self.preview_box = self._box([.84, .705, .12, .035], 'Preview width ', self.preview_size)
        self._button([.04, .645, .17, .035], 'Apply geometry', self.apply_geometry)
        self._button([.23, .645, .14, .035], 'Previous tile', lambda: self.step_grid(-1))
        self._button([.39, .645, .14, .035], 'Next tile', lambda: self.step_grid(1))
        self.group_widget = CheckButtons(c.add_axes([.56, .635, .18, .055]), ['Grouped'], [self.project.grouped])
        self.widgets.append(self.group_widget)
        self._button([.77, .645, .19, .035], 'Apply grouping', self.apply_grouping)
        c.text(.04, .605, 'Global localization filters — blank means unbounded', weight='bold')
        fields = ['loc_precision_nm', 'photons', 'logl_rel', 'z_nm', 'frame']
        fields += [k for k in self.project.filters if k not in fields]
        # Show the standard fields; extra script-defined fields remain in the project.
        self.filter_boxes = {}
        for i, field in enumerate(fields[:5]):
            y = .555 - i * .045
            lo, hi = self.project.filters.get(field, (None, None))
            c.text(.05, y + .01, field, fontsize=10)
            self.filter_boxes[field] = (
                self._box([.24, y, .12, .032], '', '' if lo is None else lo),
                self._box([.39, y, .12, .032], 'to ', '' if hi is None else hi))
        self._button([.56, .51, .18, .04], 'Apply filters', self.apply_filters)
        self.contrast_box = self._box([.82, .515, .14, .035], 'Contrast ', 3)
        self._button([.77, .46, .19, .035], 'Apply contrast', self.apply_contrast)
        group = self.project.group_settings
        c.text(.56, .40, f'Grouping: {group.dx:g} nm / {group.dt} frame gap.\nFilters apply after grouping.\nAbsent fields are listed above the views.', fontsize=9)
        c.text(.04, .32, 'Find clusters in current file', weight='bold')
        self.finder_boxes = {}
        for i, (name, value) in enumerate(DensityPeaks.defaults.items()):
            x = .17 + (i % 3) * .30
            y = .275 - (i // 3) * .05
            label = {'bin_nm': 'Bin nm ', 'sigma_nm': 'Sigma nm ', 'separation_nm': 'Spacing nm ',
                     'count_radius_nm': 'Count radius ', 'min_count': 'Min count '}[name]
            self.finder_boxes[name] = self._box([x, y, .105, .032], label, value)
        self._button([.78, .225, .18, .035], 'Find candidates', self.find)
        self._button([.04, .16, .13, .04], 'Add ROI', self.add_candidate)
        self._button([.19, .16, .13, .04], 'Polygon', self.start_polygon)
        self._button([.34, .16, .13, .04], 'Direction', self.start_direction)
        self._button([.49, .16, .14, .04], 'Clear shape', self.clear_shape)
        self._button([.65, .16, .14, .04], 'Clear line', self.clear_direction)
        self._button([.81, .16, .15, .04], 'Cancel draft', self.cancel_candidate)
        self._button([.04, .105, .13, .04], 'Accept ROI', self.accept)
        self._button([.19, .105, .18, .04], 'Accept file ROIs', self.accept_file)
        self._button([.39, .105, .13, .04], 'Toggle use', self.toggle_use)
        self._button([.54, .105, .13, .04], 'Remove ROI', self.remove)
        self._button([.69, .105, .12, .04], 'Evaluate all', self.evaluate)
        self._button([.83, .105, .13, .04], 'Histograms', self.histograms)
        self.comment_box = self._box([.13, .052, .39, .032], 'Comment ', '')
        self._button([.54, .052, .13, .032], 'Set comment', self.set_comment)
        self._button([.69, .052, .12, .032], 'List back', lambda: self.turn_page(-1))
        self._button([.83, .052, .13, .032], 'List next', lambda: self.turn_page(1))

    def _restore_navigation(self):
        nav = self.project.navigation
        if nav.get('file_id') in self.project.sources:
            self.file_id = nav['file_id']
        self.detail_size = float(nav.get('detail_size', 3000))
        self.preview_size = float(nav.get('preview_size', 900))
        if 'detail_center' in nav:
            self.detail_center = point(nav['detail_center'])
        elif self.file_id:
            x0, y0, x1, y1 = self.project.state(self.file_id).index.bounds
            self.detail_center = [(x0 + x1) / 2, (y0 + y1) / 2]
        self.detail_box.set_val(str(self.detail_size))
        self.preview_box.set_val(str(self.preview_size))
        selected = nav.get('selected_id')
        if selected in self.project.rois:
            self.select_roi(selected)

    def add_file(self):
        source = self.project.add_file(self.source_box.text.strip())
        self.file_id = source.id
        self.selected_id = None
        self.cancel_candidate()
        self.grid_index = -1
        self.step_grid(1)
        self.message = f'Loaded {source.name}'

    def save(self):
        self.project.navigation = {'file_id': self.file_id, 'detail_center': self.detail_center,
                                   'detail_size': self.detail_size, 'preview_size': self.preview_size,
                                   'selected_id': self.selected_id}
        path = self.project.save(self.path_box.text.strip())
        self.message = f'Saved {path}'
        if self.candidate is not None:
            self.message += ' — draft is not saved; use Add ROI first.'

    def open_project(self):
        project = ROIProject.load(self.path_box.text.strip())
        self.project = project
        self.file_id = next(iter(project.sources), None)
        self.selected_id = None
        self.cancel_candidate()
        self.size_box.set_val(str(project.size_nm))
        self.shape_widget.set_active(int(project.shape == 'square'))
        if self.group_widget.get_status()[0] != project.grouped:
            self.group_widget.set_active(0)
        for name, (low, high) in self.filter_boxes.items():
            lo, hi = project.filters.get(name, (None, None))
            low.set_val('' if lo is None else str(lo))
            high.set_val('' if hi is None else str(hi))
        self.page = 0
        self.grid_index = -1
        self._restore_navigation()
        self.message = f'Opened {self.path_box.text.strip()}'

    def change_file(self, step):
        ids = list(self.project.sources)
        if not ids:
            return
        self.file_id = ids[(ids.index(self.file_id) + step) % len(ids)]
        self.selected_id = None
        self.cancel_candidate()
        self.page = 0
        self.grid_index = -1
        self.step_grid(1)

    def apply_geometry(self):
        detail = positive(self.detail_box.text, 'Detail width')
        preview = positive(self.preview_box.text, 'Preview width')
        self.project.set_geometry(self.size_box.text, self.shape_widget.value_selected)
        self.detail_size, self.preview_size = detail, preview
        self.grid_index = -1
        self.message = 'Global geometry updated. Changed analysis inputs require reevaluation.'

    def apply_filters(self):
        ranges = dict(self.project.filters)
        for name, boxes in self.filter_boxes.items():
            ranges[name] = tuple(float(b.text) if b.text.strip() else None for b in boxes)
        self.project.set_filters(ranges)
        self.message = 'Filters updated for rendering, finding and analysis.'

    def apply_grouping(self):
        old = self.project.grouped
        self.project.grouped = self.group_widget.get_status()[0]
        try:
            for file_id in self.project.sources:
                self.project.state(file_id)
        except Exception:
            self.project.grouped = old
            if self.group_widget.get_status()[0] != old:
                self.group_widget.set_active(0)
            raise
        self.message = 'Grouped' if self.project.grouped else 'Ungrouped'

    def apply_contrast(self):
        contrast = positive(self.contrast_box.text, 'Contrast')
        for source in self.project.sources.values():
            source.state.display.contrast = contrast

    def step_grid(self, step):
        if not self.file_id:
            return
        x0, y0, x1, y1 = self.project.state(self.file_id).index.bounds
        nx = max(1, int(np.ceil((x1 - x0) / self.detail_size)))
        ny = max(1, int(np.ceil((y1 - y0) / self.detail_size)))
        self.grid_index = (self.grid_index + step) % (nx * ny)
        row, col = divmod(self.grid_index, nx)
        # A serpentine path avoids jumping across the file at the end of each row.
        if row % 2:
            col = nx - 1 - col
        self.detail_center = [x0 + (col + .5) * self.detail_size,
                              y0 + (row + .5) * self.detail_size]
        self.message = f'Tile {self.grid_index + 1}/{nx * ny}; ROI positions are unchanged.'

    def _stop_tools(self):
        if self.selector is not None:
            self.selector.set_active(False)
            self.selector.disconnect_events()
            self.selector = None
        self.mode = 'select'
        self.line_start = None

    def cancel_candidate(self):
        self._stop_tools()
        self.candidate = None

    def new_candidate(self, center):
        self._stop_tools()
        self.selected_id = None
        self.candidate = ROI(self.file_id, point(center), reviewed=True)
        self.comment_box.set_val('')
        self.message = 'Draft: recenter in preview; Add ROI to keep it.'

    def add_candidate(self):
        if self.candidate is None:
            raise ValueError('Click the detail view to create a candidate first')
        roi = self.candidate
        self.project.rois[roi.id] = roi
        self.candidate = None
        self.selected_id = roi.id
        self._stop_tools()
        self.page = self._file_rois().index(roi) // self.page_size
        self.message = f'Added ROI {roi.id[:8]}'

    def select_roi(self, roi_id):
        self.cancel_candidate()
        roi = self.project.rois[roi_id]
        self.selected_id = roi_id
        self.file_id = roi.file_id
        self.detail_center = list(roi.center)
        self.page = self._file_rois().index(roi) // self.page_size
        self.comment_box.set_val(roi.comment)
        record, stale = self.project.latest(roi_id)
        self.message = f'ROI {roi.id[:8]}: right-click preview to recenter.'
        if stale:
            self.message += ' Results are outdated.'
        if record and 'error' in record:
            self.message += ' ' + record['error']

    def _require_active(self):
        if self.active is None:
            raise ValueError('Select or create an ROI first')
        return self.active

    def start_polygon(self):
        self._require_active()
        self._stop_tools()
        self.mode = 'polygon'
        self.message = 'Click polygon vertices in preview; close at the first vertex. Esc cancels.'
        # draw() must not clear the preview while the selector owns its artists.
        self.selector = PolygonSelector(self.axes[2], self._polygon_done, useblit=True)

    def _polygon_done(self, vertices):
        def apply():
            self._require_active().polygon = polygon_vertices(vertices)
            self._stop_tools()
            self.message = 'Polygon is the analysis boundary; global size no longer applies to it.'
        self._action(apply)

    def start_direction(self):
        self._require_active()
        self._stop_tools()
        self.mode = 'direction'
        self.message = 'Click the start and end of the direction line in the preview.'

    def clear_shape(self):
        self._require_active().polygon = None
        self._stop_tools()

    def clear_direction(self):
        self._require_active().direction = None
        self._stop_tools()

    def set_comment(self):
        self._require_active().comment = self.comment_box.text

    def accept(self):
        self._require_active().reviewed = True

    def accept_file(self):
        for roi in self._file_rois():
            roi.reviewed = True
        self.message = 'All ROIs in this file are reviewed; their use flags are unchanged.'

    def toggle_use(self):
        roi = self._require_active()
        roi.use = not roi.use

    def remove(self):
        if self.candidate is not None:
            self.cancel_candidate()
        elif self.selected_id:
            del self.project.rois[self.selected_id]
            self.selected_id = None
        self.message = 'Removed ROI. Save the project to retain this change.'

    def find(self):
        if not self.file_id:
            raise ValueError('Add a localization file first')
        parameters = {k: float(box.text) for k, box in self.finder_boxes.items()}
        found = self.project.find(self.file_id, parameters=parameters)
        if found:
            self.select_roi(found[0].id)
        self.message = f'Found {len(found)} new candidates. Review before evaluation.'

    def evaluate(self):
        run = self.project.evaluate()
        errors = sum('error' in r for r in run['records'].values())
        self.message = f"Evaluated {len(run['records'])} reviewed, included ROIs; {errors} errors."

    def histograms(self):
        rows = self.project.results()
        if not rows:
            raise ValueError('No current results. Accept ROIs and Evaluate all first.')
        if self.hist_figure is not None:
            plt.close(self.hist_figure)
        self.hist_figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        self.hist_signature = digest(rows)
        _window_title(self.hist_figure, 'ROI statistics')
        self.hist_pick = {}
        data = Histograms().analyze(rows)
        for ax, (field, histogram) in zip(axes, data.items()):
            edges = histogram['edges']
            bars = ax.bar(edges[:-1], histogram['counts'], width=np.diff(edges), align='edge')
            for i, bar in enumerate(bars):
                bar.set_picker(True)
                values = histogram['values']
                inside = ((values >= edges[i]) & (values < edges[i + 1] if i < len(bars) - 1
                                                  else values <= edges[i + 1]))
                ids = [rid for rid, ok in zip(histogram['roi_ids'], inside) if ok]
                self.hist_pick[bar] = ids
            ax.set_xlabel(field.replace('_', ' '))
            ax.set_ylabel('ROIs')
            ax.ticklabel_format(axis='x', style='plain', useOffset=False)
            ax.set_title(f"{len(histogram['values'])} finite; {histogram['missing']} missing")
        self.hist_figure.suptitle('Reviewed + included + current results — click a bar to inspect; repeat to cycle')
        self.hist_figure.tight_layout(rect=[0, 0, 1, .9])
        self.hist_figure.canvas.mpl_connect('pick_event', self._hist_click)
        self.hist_figure.canvas.draw_idle()
        self.hist_figure.show(warn=False)

    def _hist_click(self, event):
        ids = [rid for rid in self.hist_pick.get(event.artist, []) if rid in self.project.rois]
        if ids:
            i = (ids.index(self.selected_id) + 1) % len(ids) if self.selected_id in ids else 0
            self.select_roi(ids[i])
            self.message = f'Histogram bin: ROI {i + 1} of {len(ids)}. Click again to cycle.'
            self.draw()

    def _file_rois(self):
        return [r for r in self.project.rois.values() if r.file_id == self.file_id]

    def turn_page(self, step):
        pages = max(1, int(np.ceil(len(self._file_rois()) / self.page_size)))
        self.page = (self.page + step) % pages

    def _click(self, event):
        if event.xdata is None or event.ydata is None or not self.file_id:
            return
        toolbar = getattr(self.figure.canvas, 'toolbar', None)
        if toolbar is not None and toolbar.mode:
            return
        if self.mode == 'polygon':
            return
        p = [event.xdata, event.ydata]
        def act():
            if event.inaxes is self.axes[0] and event.button == 1:
                self.detail_center = p
                self.grid_index = -1
            elif event.inaxes is self.axes[1] and event.button == 1:
                self.new_candidate(p)
            elif event.inaxes is self.axes[2] and self.active is not None:
                roi = self.active
                if self.mode == 'direction' and event.button == 1:
                    if self.line_start is None:
                        self.line_start = p
                    else:
                        if np.linalg.norm(np.asarray(p) - self.line_start) == 0:
                            raise ValueError('Direction line needs two distinct points')
                        roi.direction = [self.line_start, p]
                        self._stop_tools()
                elif event.button == 3 or (self.candidate is not None and event.button == 1):
                    delta = np.asarray(p) - roi.center
                    if self.candidate is not None:
                        roi.center = point(p)
                        for name in ('polygon', 'direction'):
                            if getattr(roi, name) is not None:
                                setattr(roi, name, (np.asarray(getattr(roi, name)) + delta).tolist())
                    else:
                        self.project.move_roi(roi.id, p)
        self._action(act)

    def _pick(self, event):
        roi_id = self.table_ids.get(event.artist)
        if roi_id:
            self.select_roi(roi_id)
            self.draw()

    def _key(self, event):
        if event.key == 'escape':
            self._stop_tools()
            self.draw()
        elif event.key in ('left', 'right'):
            rois = self._file_rois()
            if rois:
                ids = [r.id for r in rois]
                i = ids.index(self.selected_id) if self.selected_id in ids else -1
                self.select_roi(ids[(i + (1 if event.key == 'right' else -1)) % len(ids)])
                self.draw()
        elif event.key == 'enter' and self.candidate is not None:
            self._action(self.add_candidate)

    def _outline(self, ax, roi, color, width=1):
        if roi.polygon is not None:
            patch = Polygon(roi.polygon, closed=True, fill=False, edgecolor=color, linewidth=width)
        elif self.project.shape == 'circle':
            patch = Circle(roi.center, self.project.size_nm / 2, fill=False, edgecolor=color, linewidth=width)
        else:
            half = self.project.size_nm / 2
            patch = Rectangle(np.asarray(roi.center) - half, self.project.size_nm,
                              self.project.size_nm, fill=False, edgecolor=color, linewidth=width)
        ax.add_patch(patch)

    def draw(self):
        self.status.set_text(self.message)
        if self.hist_figure is not None and plt.fignum_exists(self.hist_figure.number):
            if digest(self.project.results()) != self.hist_signature:
                self.hist_figure.suptitle('Results changed — click Histograms to refresh this snapshot', color='#a04000')
                self.hist_figure.canvas.draw_idle()
        if self.file_id:
            source = self.project.sources[self.file_id]
            state = self.project.state(self.file_id)
            ids = list(self.project.sources)
            missing = [k for k in self.project.filters if k not in state.locs]
            self.file_label.set_text(f'{ids.index(self.file_id) + 1}/{len(ids)}  {source.name[:45]}')
            self.figure.suptitle(f'{source.name} — {len(state.filter):,} filtered localizations'
                                 + (f" — absent filters: {', '.join(missing)}" if missing else ''))
            x0, y0, x1, y1 = state.index.bounds
            span = max(x1 - x0, y1 - y0, self.project.size_nm) * 1.04
            centers = [[(x0 + x1) / 2, (y0 + y1) / 2], self.detail_center,
                       self.active.center if self.active else self.detail_center]
            spans = [span, self.detail_size, self.preview_size]
            titles = ['File overview — click to position detail', 'Detail — click to define ROI',
                      'ROI preview — right-click to recenter']
            for i, (ax, center, width, title) in enumerate(zip(self.axes, centers, spans, titles)):
                if i == 2 and self.selector is not None:
                    continue
                ax.clear()
                pixel = width / max(100, min(700, int(ax.bbox.width)))
                fov = FieldOfView.from_range((center[0] - width / 2, center[0] + width / 2),
                                             (center[1] - width / 2, center[1] + width / 2), pixel)
                image, _ = state.image(fov)
                ax.imshow(image, extent=fov.extent, origin='upper', interpolation='nearest')
                ax.set_title(title, fontsize=10)
                ax.set_xlabel('x (nm)')
                ax.set_ylabel('y (nm)')
                if i < 2:
                    for roi in self._file_rois():
                        if abs(roi.center[0] - center[0]) > width / 2 + self.project.size_nm or abs(roi.center[1] - center[1]) > width / 2 + self.project.size_nm:
                            continue
                        color = '#888888' if not roi.use else ('#62e3a2' if roi.reviewed else '#ffc857')
                        self._outline(ax, roi, color, 1.5 if roi.id == self.selected_id else .7)
                if i == 0:
                    ax.add_patch(Rectangle(np.asarray(self.detail_center) - self.detail_size / 2,
                                            self.detail_size, self.detail_size, fill=False,
                                            edgecolor='#58c5ff', linewidth=1.5))
                if self.active and i > 0:
                    self._outline(ax, self.active, '#58c5ff', 1.5)
                if i == 2 and self.active:
                    roi = self.active
                    ax.plot(*roi.center, '+', color='#58c5ff')
                    if roi.direction:
                        ax.annotate('', xy=roi.direction[1], xytext=roi.direction[0],
                                    arrowprops={'arrowstyle': '->', 'color': '#58c5ff'})
                    count = len(self.project.indices(roi))
                    ax.set_title(f"{'Draft' if self.candidate else roi.id[:8]} — {count} selected localizations", fontsize=10)
                ax.set_xlim(fov.x0, fov.x1)
                ax.set_ylim(fov.y1, fov.y0)
        else:
            for ax, title in zip(self.axes, ['File overview', 'Detail', 'ROI preview']):
                ax.clear()
                ax.set_title(title)
            self.file_label.set_text('Add a localization .h5 file to begin')
        self._draw_table()
        self.figure.canvas.draw_idle()
        self.controls.canvas.draw_idle()

    def _draw_table(self):
        ax = self.table_ax
        ax.clear()
        ax.axis('off')
        rois = self._file_rois()
        self.page = min(self.page, max(0, (len(rois) - 1) // self.page_size))
        visible = rois[self.page * self.page_size:(self.page + 1) * self.page_size]
        ax.set_title(f'ROIs in current file — {len(rois)} total — page {self.page + 1} — click row to inspect', fontsize=10, loc='left')
        self.table_ids = {}
        rows = []
        for roi in visible:
            record, stale = self.project.latest(roi.id)
            status = 'outdated' if stale else ('error' if record and 'error' in record else ('ready' if record else 'not run'))
            values = record.get('values', {}) if record else {}
            rows.append([roi.id[:8], 'yes' if roi.reviewed else 'unreviewed', 'yes' if roi.use else 'no',
                         status, values.get('n_localizations', '—'),
                         self._number(values.get('mean_precision_nm')), self._number(values.get('mean_photons'))])
        if not rows:
            ax.text(.01, .6, 'No ROIs yet. Click in the detail view or Find candidates.', transform=ax.transAxes)
            return
        table = ax.table(cellText=rows, colLabels=['ROI', 'Reviewed', 'Use', 'Result', 'N locs', 'Mean precision (nm)', 'Mean photons'],
                         bbox=[0, 0, 1, .95], cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (row, col), cell in table.get_celld().items():
            if row:
                roi = visible[row - 1]
                cell.set_picker(True)
                self.table_ids[cell] = roi.id
                if roi.id == self.selected_id:
                    cell.set_facecolor('#d9efff')

    @staticmethod
    def _number(value):
        return '—' if value is None else f'{value:.3g}'

    def _close(self, event):
        self._stop_tools()
        plt.close(self.controls)
        if self.hist_figure is not None:
            plt.close(self.hist_figure)
