"""Optional Tk file selector and matplotlib calibration review window."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import queue
import threading

import numpy as np

from .core import CalibrationSettings, collect_beads, build_calibration
from .input import discover_acquisitions


def calibration_save_defaults(paths):
    """Name outputs after the dataset containing the first stack directory."""
    if not paths:
        return Path.cwd(), 'beads_calibration'
    first = Path(paths[0]).expanduser().resolve()
    stack_folder = first if first.is_dir() else first.parent
    destination = stack_folder.parent
    return destination, (destination.name or stack_folder.name or 'beads')+'_calibration'


def calibration_save_path(path):
    """Native save dialogs may append an extension to an existing extension."""
    path = Path(path)
    name = path.name
    while name.lower().endswith('.h5'):
        name = name[:-3]
    return path.with_name((name or 'beads_calibration')+'.h5')


def _stack_slice(volume, orientation, index):
    """Return one displayed plane from a z,y,x stack."""
    if orientation == 'XY':
        return volume[index]
    if orientation == 'XZ':
        return volume[:, index, :]
    if orientation == 'YZ':
        return volume[:, :, index]
    raise ValueError(f'unknown stack orientation {orientation!r}')


def _stack_contrast(volume, image, auto, factor=1.):
    """Return display limits, either per-slice to its max or fixed for the stack."""
    values = image if auto else volume
    vmin = max(0., float(np.min(values)))
    vmax = float(np.max(values)) if auto else float(np.percentile(values, 99.9))
    vmax *= factor
    if factor != 1.:
        vmin = 0.
    if vmax <= vmin:
        vmax = vmin+np.finfo(float).eps
    return vmin, vmax


class StackBrowser:
    """Small three-axis browser for one z,y,x average stack."""
    def __init__(self, parent, volume, calibration):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.volume = np.asarray(volume)
        self.calibration = calibration
        self.window = tk.Toplevel(parent)
        self.window.title('Average bead stack browser')
        self.window.geometry('760x700')
        controls = ttk.Frame(self.window, padding=6)
        controls.pack(fill='x')
        ttk.Label(controls, text='View').pack(side='left')
        self.orientation = tk.StringVar(value='XY')
        selector = ttk.Combobox(controls, textvariable=self.orientation,
                                values=('XY', 'XZ', 'YZ'), width=5, state='readonly')
        selector.pack(side='left', padx=5)
        selector.bind('<<ComboboxSelected>>', self.change_orientation)
        self.auto_contrast = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text='Auto contrast to slice maximum',
                        variable=self.auto_contrast, command=self.draw).pack(side='left', padx=8)
        self.contrast_factor = tk.DoubleVar(value=1.)
        factorbar = ttk.Frame(self.window, padding=5)
        factorbar.pack(fill='x')
        ttk.Label(factorbar, text='Contrast upper-limit factor (0.5 reveals dim structure)').pack(side='left')
        tk.Scale(factorbar, variable=self.contrast_factor, from_=.05, to=2., resolution=.05,
                 orient='horizontal', command=lambda value: self.draw()).pack(side='left', fill='x', expand=True)
        self.position = tk.IntVar()
        self.scale = tk.Scale(self.window, variable=self.position, orient='horizontal',
                              from_=0, resolution=1, showvalue=True, command=lambda value: self.draw())
        self.scale.pack(fill='x', padx=8)
        self.figure = Figure(figsize=(7, 6), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        widget = self.canvas.get_tk_widget()
        widget.pack(fill='both', expand=True)
        widget.bind('<MouseWheel>', self.mouse_wheel)
        widget.bind('<Button-4>', lambda event: self.step(-1))
        widget.bind('<Button-5>', lambda event: self.step(1))
        self.window.bind('<Left>', lambda event: self.step(-1))
        self.window.bind('<Down>', lambda event: self.step(-1))
        self.window.bind('<Right>', lambda event: self.step(1))
        self.window.bind('<Up>', lambda event: self.step(1))
        self.change_orientation()
        widget.focus_set()

    def slice_count(self):
        return {'XY': self.volume.shape[0], 'XZ': self.volume.shape[1],
                'YZ': self.volume.shape[2]}[self.orientation.get()]

    def change_orientation(self, event=None):
        count = self.slice_count()
        self.scale.configure(to=count-1)
        self.position.set(count//2)
        self.draw()

    def step(self, amount):
        value = int(np.clip(self.position.get()+amount, 0, self.slice_count()-1))
        self.position.set(value)
        self.draw()

    def mouse_wheel(self, event):
        self.step(-1 if event.delta > 0 else 1)

    def draw(self):
        orientation = self.orientation.get()
        index = int(np.clip(self.position.get(), 0, self.slice_count()-1))
        self.figure.clear()
        ax = self.figure.subplots()
        image = _stack_slice(self.volume, orientation, index)
        vmin, vmax = _stack_contrast(self.volume, image, self.auto_contrast.get(), self.contrast_factor.get())
        if orientation == 'XY':
            ax.imshow(image, cmap='inferno', vmin=vmin, vmax=vmax)
            ax.set(xlabel='x (pixels)', ylabel='y (pixels)',
                   title=f'XY · emitter z = {self.calibration.z_index_to_nm(index):.1f} nm')
        else:
            horizontal = 'x' if orientation == 'XZ' else 'y'
            fixed = 'y' if orientation == 'XZ' else 'x'
            ax.imshow(image, cmap='inferno', vmin=vmin, vmax=vmax, aspect='auto',
                      extent=(0, image.shape[1]-1,
                              self.calibration.z_index_to_nm(image.shape[0]-1),
                              self.calibration.z_index_to_nm(0)))
            ax.set(xlabel=f'{horizontal} (pixels)', ylabel='Emitter z (nm)',
                   title=f'{orientation} · {fixed} = {index} px')
        self.canvas.draw_idle()


class CalibrationWindow:
    def __init__(self, root, paths=(), settings=None):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.root = root
        self.root.title('SMAPpy — bead PSF calibration')
        self.root.geometry('1250x850')
        self.events = queue.Queue()
        self.busy = False
        self.result = None
        self.excluded = set()
        self.paths = []
        self.result_paths = []
        self.pending_paths = []
        self.controls = []
        self.diagnostics = None
        self.profile_data = None
        self.showing_quality = False
        self.settings_vars = {}
        self.advanced_visible = False
        self.recursive = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value='Add files or directories; all selected stacks form one calibration.')
        # Reserve footer space before the expanding output pane gets its allocation.
        self.status_label = ttk.Label(root, textvariable=self.status, wraplength=1200, padding=8)
        self.status_label.pack(side='bottom', fill='x')
        self.status_label.bind('<Configure>', lambda event: self.status_label.configure(
            wraplength=max(200, event.width-16)))
        pane = ttk.Panedwindow(root, orient='horizontal')
        pane.pack(fill='both', expand=True)
        left = ttk.Frame(pane, padding=8)
        right = ttk.Frame(pane, padding=8)
        pane.add(left, weight=1)
        pane.add(right, weight=3)
        toolbar = ttk.Frame(left)
        toolbar.pack(fill='x')
        self.button(toolbar, 'Add files', self.add_files)
        self.button(toolbar, 'Add directory…', self.add_directory)
        self.button(toolbar, 'Remove', self.remove_paths)
        ttk.Checkbutton(left, text='Search subdirectories', variable=self.recursive).pack(anchor='w')
        ttk.Label(left, text='Acquisitions (add directories repeatedly to combine them)').pack(anchor='w')
        self.file_list = tk.Listbox(left, height=8, selectmode='extended', exportselection=False, width=47)
        self.file_list.pack(fill='x')
        scrollbar = ttk.Scrollbar(left, orient='horizontal', command=self.file_list.xview)
        scrollbar.pack(fill='x')
        self.file_list.configure(xscrollcommand=scrollbar.set)
        settings_box = ttk.LabelFrame(left, text='Calibration settings', padding=5)
        settings_box.pack(fill='x', pady=8)
        labels = {'roi_size': 'PSF size (odd pixels)', 'padding': 'Alignment padding (pixels)',
                  'dz_nm': 'Z spacing override (nm; blank = metadata)',
                  'detection_sigma_px': 'Detection smoothing (pixels)',
                  'detection_threshold_sigma': 'Detection threshold (noise sigma)',
                  'min_distance_px': 'Minimum bead separation (pixels)',
                  'max_xy_shift_px': 'Maximum xy shift (pixels)',
                  'max_z_shift_nm': 'Maximum z shift (± nm)',
                  'alignment_range_nm': 'Central alignment range (nm)',
                  'registration_iterations': 'Alignment passes',
                  'rejection_mad': 'Shape rejection (empirical σ)',
                  'smooth_z_nm': 'Z smoothing sigma (nm)',
                  'smooth_xy_px': 'XY smoothing sigma (pixels)', 'min_beads': 'Minimum accepted beads',
                  'brightness_range': 'Maximum brightness range (factor; blank = off)',
                  'saturation_adu': 'Saturation level (ADU; blank = automatic)'}
        defaults = settings or CalibrationSettings()
        basic = {'roi_size', 'dz_nm', 'max_z_shift_nm', 'alignment_range_nm',
                 'rejection_mad', 'brightness_range'}
        basic_row = advanced_row = 0
        self.advanced_frame = ttk.Frame(settings_box)
        for f in fields(defaults):
            parent = settings_box if f.name in basic else self.advanced_frame
            row = basic_row if f.name in basic else advanced_row
            ttk.Label(parent, text=labels[f.name]).grid(row=row, column=0, sticky='w')
            value = getattr(defaults, f.name)
            var = tk.StringVar(value='' if value is None else str(value))
            self.settings_vars[f.name] = var
            ttk.Entry(parent, textvariable=var, width=9).grid(row=row, column=1, sticky='e')
            if f.name in basic:
                basic_row += 1
            else:
                advanced_row += 1
        self.advanced_button = ttk.Button(settings_box, text='Show optional parameters',
                                          command=self.toggle_advanced)
        self.advanced_button.grid(row=basic_row, column=0, columnspan=2, sticky='w', pady=(4, 0))
        self.advanced_grid_row = basic_row+1
        actions = ttk.Frame(left)
        actions.pack(fill='x')
        self.button(actions, 'Detect + calibrate', self.run)
        self.button(actions, 'Save calibration…', self.save)
        ttk.Label(left, text='xy: pixels · z: nm · z=0: aligned stack center',
                  justify='left').pack(anchor='w', pady=8)
        review = ttk.Frame(right)
        review.pack(side='bottom', fill='x')
        self.button(review, 'Toggle exclusion', self.toggle)
        self.button(review, 'Rebuild from beads', self.rebuild)
        self.fit_quality_button = self.button(review, 'Calculate fit quality', self.fit_quality)
        self.button(review, 'Browse average stack…', self.browse_stack)
        ttk.Label(review, text='Select bead rows to inspect / exclude').pack(side='left', padx=5)
        self.table = ttk.Treeview(right, columns=('id', 'stack', 'xy', 'brightness',
                                                  'shift', 'correlation', 'error', 'state'),
                                  show='headings', height=8, selectmode='extended')
        for name, label, width in [('id', 'Bead ID', 55), ('stack', 'Stack', 55), ('xy', 'x, y (px)', 95),
                                   ('brightness', 'Brightness (ADU)', 105),
                                   ('shift', 'z shift (nm)', 90), ('correlation', 'Correlation', 80),
                                   ('error', 'Shape residual', 90),
                                   ('state', 'Status', 140)]:
            self.table.heading(name, text=label)
            self.table.column(name, width=width, stretch=True)
        self.table.pack(side='bottom', fill='x')
        self.create_plot(right)
        self.table.bind('<<TreeviewSelect>>', lambda event: self.draw_current())
        if paths:
            self.add_paths(paths)
        root.after(100, self.poll)

    def button(self, parent, label, callback):
        from tkinter import ttk
        b = ttk.Button(parent, text=label, command=callback)
        b.pack(side='left', padx=2, pady=2)
        self.controls.append(b)
        return b

    def create_plot(self, parent):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.figure = Figure(figsize=(8, 5), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def open_dual(self):
        import tkinter as tk
        from .dual_gui import DualCalibrationWindow
        DualCalibrationWindow(tk.Toplevel(self.root), self.paths)

    def toggle_advanced(self):
        self.advanced_visible = not self.advanced_visible
        if self.advanced_visible:
            self.advanced_frame.grid(row=self.advanced_grid_row, column=0, columnspan=2,
                                     sticky='ew')
            self.advanced_button.configure(text='Hide optional parameters')
        else:
            self.advanced_frame.grid_remove()
            self.advanced_button.configure(text='Show optional parameters')

    def settings(self):
        defaults = CalibrationSettings()
        values = {}
        for name, var in self.settings_vars.items():
            text = var.get().strip()
            if name in {'dz_nm', 'brightness_range', 'saturation_adu'}:
                values[name] = float(text) if text else None
            else:
                values[name] = type(getattr(defaults, name))(text)
        settings = CalibrationSettings(**values)
        settings.validate()
        return settings

    def error(self, exc):
        from tkinter import messagebox
        messagebox.showerror('Calibration', str(exc), parent=self.root)

    def add_paths(self, paths):
        try:
            acquisitions = discover_acquisitions(paths, self.recursive.get())
            self.paths = sorted(set(self.paths).union(acquisitions), key=str)
            self.file_list.delete(0, 'end')
            for path in self.paths:
                self.file_list.insert('end', str(path))
            self.status.set(f'{len(self.paths)} acquisitions selected for one calibration.')
        except Exception as exc:
            self.error(exc)

    def add_files(self):
        from tkinter import filedialog
        paths = filedialog.askopenfilenames(parent=self.root, title='Select bead acquisitions',
                  filetypes=[('Acquisitions', '*.tif *.tiff *.index'), ('All files', '*')])
        if paths:
            self.add_paths(paths)

    def add_directory(self):
        from tkinter import filedialog
        path = filedialog.askdirectory(parent=self.root, title='Add a directory of bead stacks')
        if path:
            self.add_paths([path])

    def remove_paths(self):
        for i in reversed(self.file_list.curselection()):
            del self.paths[i]
            self.file_list.delete(i)
        self.status.set(f'{len(self.paths)} acquisitions selected.')

    def work(self, fn, kind='result'):
        if self.busy:
            return
        self.busy = True
        for control in self.controls:
            control.configure(state='disabled')
        def worker():
            try:
                self.events.put((kind, fn()))
            except Exception as exc:
                self.events.put(('error', exc))
        threading.Thread(target=worker, daemon=True).start()

    def progress(self, message):
        self.events.put(('progress', message))

    def run(self):
        try:
            settings = self.settings()
            paths = list(self.paths)
            if not paths:
                raise ValueError('Add at least one acquisition')
            self.excluded = set()
            self.pending_paths = paths
            self.work(lambda: build_calibration(collect_beads(paths, settings, self.progress),
                                                progress=self.progress))
        except Exception as exc:
            self.error(exc)

    def toggle(self):
        if self.result is None or self.busy:
            return
        for value in self.table.selection():
            i = int(value)
            if i in self.excluded:
                self.excluded.remove(i)
            else:
                self.excluded.add(i)
        self.refresh_table()
        self.status.set('Exclusion changed. Rebuild to apply it before saving.')

    def rebuild(self):
        if self.result is None:
            return
        try:
            if self.paths != self.result_paths:
                raise ValueError('Acquisitions changed: use Detect + calibrate.')
            if self.settings() != self.result.beads.settings:
                raise ValueError('Settings changed: use Detect + calibrate to apply them.')
            excluded = set(self.excluded)
            self.pending_paths = list(self.result_paths)
            self.work(lambda: build_calibration(self.result.beads, excluded, self.progress))
        except Exception as exc:
            self.error(exc)

    def fit_quality(self):
        if self.result is None:
            return
        if self.diagnostics is not None:
            self.draw_quality()
            return
        from .validation import fit_bead_diagnostics, aligned_midline_profiles
        self.status.set('Refitting bead planes with five z starts…')
        def diagnostics():
            fitted = fit_bead_diagnostics(self.result)
            profiles = aligned_midline_profiles(self.result, normalize=False)
            return fitted, profiles
        self.work(diagnostics, 'diagnostics')

    def browse_stack(self):
        if self.result is not None:
            StackBrowser(self.root, self.result.raw_psf, self.result.calibration)

    def redraw_profiles(self):
        if self.showing_quality:
            self.draw_quality()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'progress':
                    self.status.set(value)
                    continue
                self.busy = False
                for control in self.controls:
                    control.configure(state='normal')
                if kind == 'error':
                    self.status.set('Calibration failed; review settings and input.')
                    self.error(value)
                elif kind == 'result':
                    self.result = value
                    self.result_paths = list(self.pending_paths)
                    self.diagnostics = None
                    self.profile_data = None
                    self.showing_quality = False
                    self.fit_quality_button.configure(text='Calculate fit quality')
                    self.refresh_table()
                    self.draw_current()
                    self.status.set(f'{value.accepted.sum()}/{len(value.accepted)} beads accepted. '
                                    f'PSF {value.calibration.psf.shape}, dz={value.calibration.dz:g} nm. '
                                    + ' '.join(value.messages))
                elif kind == 'diagnostics':
                    self.diagnostics, self.profile_data = value
                    self.result.refits = self.diagnostics
                    self.showing_quality = True
                    self.fit_quality_button.configure(text='Calculate fit quality', state='disabled')
                    self.draw_quality()
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def refresh_table(self):
        selection = self.table.selection()
        self.table.delete(*self.table.get_children())
        r = self.result
        for i, record in enumerate(r.beads.records):
            state = 'manual exclusion' if i in self.excluded else r.reasons[i]
            if hasattr(r, 'transform_accepted') and i not in self.excluded and r.transform_accepted[i]:
                state = ('PSF + transformation' if r.accepted[i] else
                         'Transformation only: '+state)
            self.table.insert('', 'end', iid=str(i), values=(i, record['stack']+1,
                f"{record['x_px']}, {record['y_px']}", f"{record['brightness_adu']:.4g}",
                f'{r.shifts[i, 0]*r.calibration.dz:.1f}',
                f'{r.correlations[i]:.3f}', f'{r.residuals[i]:.3f}', state))
        self.table.selection_set([i for i in selection if self.table.exists(i)])

    def draw_current(self):
        if self.showing_quality:
            self.draw_quality()
        else:
            self.draw()

    def draw(self):
        if self.result is None:
            return
        r = self.result
        selected = {int(i) for i in self.table.selection()}
        stack = r.beads.records[min(selected)]['stack'] if selected else 0
        self.figure.clear()
        axes = self.figure.subplots(2, 2)
        ax = axes[0, 0]
        im = r.beads.projections[stack]
        ax.imshow(im, cmap='gray', vmax=np.percentile(im, 99.5))
        for i, rec in enumerate(r.beads.records):
            if rec['stack'] == stack:
                good = r.accepted[i] and i not in self.excluded
                ax.plot(rec['x_px'], rec['y_px'], 'o', mfc='none',
                        mec='gold' if i in selected else 'lime' if good else 'red',
                        mew=2.5 if i in selected else 1, ms=9 if i in selected else 6)
                ax.text(rec['x_px']+2, rec['y_px'], str(i), color='yellow', fontsize=7)
        ax.set_title(f'Stack {stack+1}: bead IDs (green accepted, red rejected)', fontsize=9)
        psf = r.calibration.psf
        colors = ['green' if a and i not in self.excluded else 'red'
                  for i, a in enumerate(r.accepted)]
        xy_shift = np.hypot(r.shifts[:, 1], r.shifts[:, 2])
        brightness = np.array([rec['brightness_adu'] for rec in r.beads.records])
        axes[0, 1].scatter(xy_shift, brightness, c=colors)
        axes[0, 1].set(xlabel='Applied XY shift magnitude (pixels)', ylabel='Brightness (ADU)',
                       title='Brightness vs XY shift', yscale='log')
        if r.beads.records:
            lower = r.beads.records[0].get('brightness_lower_adu')
            upper = r.beads.records[0].get('brightness_upper_adu')
            if lower is not None and np.isfinite(lower):
                axes[0, 1].axhspan(lower, upper, color='green', alpha=.08)
        zmin, zmax = r.calibration.z_index_to_nm(np.array([psf.shape[0]-1, 0]))
        axes[1, 0].imshow(psf[:, psf.shape[1]//2, :], cmap='inferno', aspect='auto',
                          extent=(0, psf.shape[2]-1, zmin, zmax))
        axes[1, 0].set(xlabel='x (pixels)', ylabel='Emitter z (nm)', title='PSF xz')
        quality_ax = axes[1, 1]
        z_shift = r.shifts[:, 0]*r.calibration.dz
        quality_ax.scatter(z_shift, r.residuals, c=colors)
        quality_ax.set(xlabel='Applied z shift (nm)', ylabel='Relative shape residual',
                       title='Bead alignment')
        if selected:
            chosen = np.array(sorted(selected))
            axes[0, 1].scatter(xy_shift[chosen], brightness[chosen], s=100, facecolors='none',
                               edgecolors='gold', linewidths=2, zorder=5)
            quality_ax.scatter(z_shift[chosen], r.residuals[chosen], s=100, facecolors='none',
                               edgecolors='gold', linewidths=2, zorder=5)
        self.canvas.draw_idle()

    def draw_quality(self):
        d = self.diagnostics
        self.figure.clear()
        axes = self.figure.subplots(2, 2)
        selected = {int(i) for i in self.table.selection()}
        bead_ids = sorted(set(d['bead_id'])-{-1})
        colors = []
        if bead_ids:
            hues = np.linspace(0, 1, len(bead_ids), endpoint=False)
            import matplotlib.colors as mcolors
            colors = mcolors.hsv_to_rgb(np.column_stack((hues, np.full(len(hues), .7),
                                                          np.full(len(hues), .8))))
        for color, bead_id in zip(colors, bead_ids):
            use = d['bead_id'] == bead_id
            order = np.argsort(d['expected_z_nm'][use])
            width = 2.5 if bead_id in selected else .8
            alpha = 1 if bead_id in selected else .75
            axes[0, 0].plot(d['expected_z_nm'][use][order], d['fitted_z_nm'][use][order],
                            color=color, linewidth=width, alpha=alpha)
            axes[0, 1].plot(d['expected_z_nm'][use][order], d['centered_error_nm'][use][order],
                            color=color, linewidth=width, alpha=alpha)
        average = d['bead_id'] == -1
        if np.any(average):
            order = np.argsort(d['expected_z_nm'][average])
            axes[0, 0].plot(d['expected_z_nm'][average][order], d['fitted_z_nm'][average][order],
                            color='black', linewidth=3, label='average bead stack')
            axes[0, 1].plot(d['expected_z_nm'][average][order],
                            d['centered_error_nm'][average][order], color='black', linewidth=3)
        limits = [min(d['expected_z_nm']), max(d['expected_z_nm'])]
        axes[0, 0].plot(limits, limits, 'k--')
        axes[0, 0].set(xlabel='Expected emitter z (nm)', ylabel='Fitted z (nm)',
                       title='In-sample bead refits')
        axes[0, 0].legend(loc='best', fontsize=7)
        axes[0, 1].set(xlabel='Expected emitter z (nm)',
                       ylabel='Error after bead offset removal (nm)', title='Centered fit error')
        profiles = self.profile_data
        x_profiles = profiles['x_profiles'].copy()
        z_profiles = profiles['z_profiles'].copy()
        average_x = profiles['average_x'].copy()
        average_z = profiles['average_z'].copy()
        profile_colors = {bead_id: color for bead_id, color in zip(bead_ids, colors)}
        for row, bead_id in enumerate(profiles['bead_id']):
            color = profile_colors[int(bead_id)]
            width = 2.5 if bead_id in selected else .8
            alpha = 1 if bead_id in selected else .75
            axes[1, 0].plot(profiles['x_px'], x_profiles[row], color=color,
                            linewidth=width, alpha=alpha)
            axes[1, 1].plot(profiles['z_nm'], z_profiles[row], color=color,
                            linewidth=width, alpha=alpha)
        axes[1, 0].plot(profiles['x_px'], average_x, color='black', linewidth=3,
                        label='average bead stack')
        axes[1, 1].plot(profiles['z_nm'], average_z, color='black', linewidth=3)
        ylabel = 'Intensity (calibration scale)'
        axes[1, 0].set(xlabel='x from center (pixels)', ylabel=ylabel,
                       title='Aligned bead lateral midlines')
        axes[1, 1].set(xlabel='Emitter z (nm)', ylabel=ylabel,
                       title='Aligned bead axial midlines')
        axes[1, 0].legend(loc='best', fontsize=7)
        for ax in axes[1]:
            ax.relim(); ax.autoscale_view(); ax.margins(x=.02, y=.05)
        self.canvas.draw_idle()
        individual = d['bead_id'] >= 0
        self.status.set(f"In-sample refit median absolute centered error: "
                        f"{np.nanmedian(abs(d['centered_error_nm'][individual])):.1f} nm; "
                        f"{np.count_nonzero(~d['fit_valid'][individual])} failed, "
                        f"{np.count_nonzero(d['at_z_boundary'][individual])} at z boundary. "
                        'This is a consistency check, not independent accuracy.')

    def save(self):
        if self.result is None:
            return
        from tkinter import filedialog, messagebox
        try:
            applied = {i for i, reason in enumerate(self.result.reasons) if reason == 'manual exclusion'}
            if (self.excluded != applied or self.settings() != self.result.beads.settings
                    or self.paths != self.result_paths):
                raise ValueError('Recalculate changed settings/exclusions before saving.')
            folder, filename = calibration_save_defaults(self.result_paths)
            path = filedialog.asksaveasfilename(parent=self.root, defaultextension='.h5',
                         initialdir=str(folder), initialfile=filename,
                         filetypes=[('SMAPpy calibration', '*.h5')])
            if path:
                path = calibration_save_path(path)
                exists = Path(path).exists()
                if exists and not messagebox.askyesno('Replace calibration?', str(path), parent=self.root):
                    return
                self.result.save(path, overwrite=exists)
                self.status.set(f'Saved {path}')
        except Exception as exc:
            self.error(exc)


def show_calibration(paths=(), settings=None):
    import tkinter as tk
    from .unified_gui import UnifiedCalibrationWindow
    root = tk.Tk()
    UnifiedCalibrationWindow(root, paths, settings, dual=False)
    root.mainloop()
