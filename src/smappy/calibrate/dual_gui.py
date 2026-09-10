"""Paired calibration review using the existing single-channel window controls."""
from dataclasses import asdict, replace
import numpy as np

from .gui import CalibrationWindow, StackBrowser
from .core import CalibrationSettings
from .dual import (DualColorSettings, LAYOUTS, collect_dual_beads,
                   build_dual_calibration, channel_result, map_points)


class DualCalibrationWindow(CalibrationWindow):
    def __init__(self, root, paths=(), settings=None):
        import tkinter as tk
        from tkinter import ttk
        s = settings or DualColorSettings()
        geometry = ttk.LabelFrame(root, text='Split-frame dual-color geometry', padding=6)
        geometry.pack(fill='x')
        self.layout = tk.StringVar(value=s.layout)
        self.main_channel = tk.StringVar(value=s.main_channel)
        self.split = tk.StringVar(value='' if s.split_position is None else str(s.split_position))
        self.min_pairs = tk.StringVar(value=str(s.min_pairs))
        self.threshold = tk.StringVar(value=str(s.reprojection_threshold_px))
        self.axis_limit = tk.StringVar(value=str(s.transform_axis_limit_px))
        self.display_channel = tk.StringVar(value='main')
        for label, var, values in [('Layout', self.layout, LAYOUTS),
                                   ('Main channel', self.main_channel, ('upper', 'lower')),
                                   ('Inspect PSF', self.display_channel, ('main', 'secondary'))]:
            ttk.Label(geometry, text=label).pack(side='left', padx=3)
            box = ttk.Combobox(geometry, textvariable=var, values=values, state='readonly', width=20)
            box.pack(side='left')
            if var is self.layout:
                box.bind('<<ComboboxSelected>>', self.layout_changed)
            elif var is self.main_channel:
                self.main_selector = box
            else:
                box.bind('<<ComboboxSelected>>', self.channel_changed)
        row = ttk.Frame(root, padding=6)
        row.pack(fill='x')
        for i, (label, var) in enumerate([
                ('Split position (blank = midpoint)', self.split),
                ('Minimum transformation pairs (≥4)', self.min_pairs),
                ('Initial RANSAC radius (px)', self.threshold),
                ('Round 2: |dx| and |dy| limit (px)', self.axis_limit)]):
            ttk.Label(row, text=label).grid(row=i//2, column=2*(i%2), sticky='w', padx=5)
            ttk.Entry(row, textvariable=var, width=6).grid(row=i//2, column=2*(i%2)+1, sticky='w')
        base = CalibrationSettings(**{k: v for k, v in asdict(s).items() if k in CalibrationSettings.__dataclass_fields__})
        super().__init__(root, paths, base)
        self.root.title('SMAPpy — split-frame dual-color calibration')
        self.layout_changed()

    def layout_changed(self, event=None):
        choices = ('left', 'right') if 'right-left' in self.layout.get() else ('upper', 'lower')
        self.main_selector.configure(values=choices)
        if self.main_channel.get() not in choices:
            self.main_channel.set(choices[0])

    def channel_changed(self, event=None):
        self.diagnostics = None
        self.showing_quality = False
        self.fit_quality_button.configure(text='Calculate fit quality', state='normal')
        self.draw()

    def settings(self):
        values = asdict(super().settings())
        s = DualColorSettings(**values, layout=self.layout.get(), main_channel=self.main_channel.get(),
                split_position=int(self.split.get()) if self.split.get().strip() else None,
                min_pairs=int(self.min_pairs.get()), reprojection_threshold_px=float(self.threshold.get()),
                transform_axis_limit_px=float(self.axis_limit.get()))
        s.validate()
        return s

    def run(self):
        try:
            settings, paths = self.settings(), list(self.paths)
            self.excluded = set()
            self.pending_paths = paths
            self.work(lambda: build_dual_calibration(collect_dual_beads(paths, settings, self.progress),
                                                     progress=self.progress))
        except Exception as exc:
            self.error(exc)

    def rebuild(self):
        if self.result is None:
            return
        try:
            if self.paths != self.result_paths or self.settings() != self.result.beads.settings:
                raise ValueError('Inputs/settings changed: use Detect + calibrate')
            excluded = set(self.excluded)
            previous = self.result
            self.pending_paths = list(self.result_paths)
            def rebuild():
                rebuilt = build_dual_calibration(previous.beads, excluded, self.progress)
                rebuilt.initial_transformation = previous.initial_transformation
                rebuilt.initial_residuals = previous.initial_residuals
                return rebuilt
            self.work(rebuild)
        except Exception as exc:
            self.error(exc)

    def browse_stack(self):
        if self.result is not None:
            view = channel_result(self.result, 0 if self.display_channel.get() == 'main' else 1)
            StackBrowser(self.root, view.raw_psf, view.calibration)

    def fit_quality(self):
        if self.result is None:
            return
        if self.diagnostics is not None:
            self.showing_quality = not self.showing_quality
            self.draw_current()
            return
        from .validation import fit_bead_diagnostics, aligned_midline_profiles
        ch = 0 if self.display_channel.get() == 'main' else 1
        view = channel_result(self.result, ch)
        def diagnostics():
            fitted = fit_bead_diagnostics(view)
            self.result.registration.refits[str(ch)] = fitted
            return fitted, aligned_midline_profiles(view, normalize=False)
        self.work(diagnostics, 'diagnostics')

    def draw(self):
        if self.result is None:
            return
        r = self.result
        selected = {int(i) for i in self.table.selection()}
        si = r.beads.records[min(selected)]['stack'] if selected else 0
        self.figure.clear()
        axes = self.figure.subplots(2, 2)
        image = r.beads.projections[si]
        axes[0, 0].imshow(image, cmap='gray', vmax=np.percentile(image, 99.5))
        for i, rec in enumerate(r.beads.records):
            if rec['stack'] != si:
                continue
            color = ('gold' if i in selected else 'red' if i in self.excluded else
                     'lime' if r.accepted[i] else 'darkorange' if r.transform_accepted[i] else 'red')
            points = np.array([c['full_image_xy'] for c in rec['channel_records']])
            axes[0, 0].plot(*points.T, 'o-', color=color, mfc='none', lw=.5, ms=5)
            axes[0, 0].text(*points[0], str(i), color=color, fontsize=7)
        for ch, indices in enumerate(r.beads.unmatched):
            points = [r.beads.channels[ch].records[i]['full_image_xy'] for i in indices
                      if r.beads.channels[ch].records[i]['stack'] == si]
            if points:
                axes[0, 0].plot(*np.array(points).T, '+', color='cyan')
        axes[0, 0].set_title(f'Acquisition {si+1}: pairs; cyan = unmatched')
        colors = ['green' if a and i not in self.excluded else 'red' for i, a in enumerate(r.transform_accepted)]
        delta = map_points(r.calibration.transformation, r.beads.secondary_points)-r.beads.main_points
        vectors = axes[0, 1].quiver(*r.beads.main_points.T, *delta.T, color=colors, angles='xy')
        axes[0, 1].quiverkey(vectors, .85, .07, .1, '0.1 px', coordinates='axes', labelpos='N')
        axes[0, 1].margins(.15)
        axes[0, 1].set(title=f'Projective residuals: median {np.median(r.transform_residuals[r.transform_accepted]):.3f} px',
                        xlabel='Main x (pixels)', ylabel='Main y (pixels)')
        ch = 0 if self.display_channel.get() == 'main' else 1
        view = channel_result(r, ch)
        psf = view.calibration.psf
        axes[1, 0].imshow(psf[:, psf.shape[1]//2], cmap='inferno', aspect='auto')
        axes[1, 0].set(title=self.display_channel.get()+' PSF, native xz', xlabel='x (pixels)', ylabel='z plane')
        axes[1, 1].scatter(r.shifts[:, 0]*r.calibration.dz, r.residuals, c=colors)
        axes[1, 1].set(xlabel='Shared pair z shift (nm)', ylabel='Joint shape residual')
        self.canvas.draw_idle()


def show_dual_calibration(paths=(), settings=None):
    import tkinter as tk
    from .unified_gui import UnifiedCalibrationWindow
    root = tk.Tk()
    UnifiedCalibrationWindow(root, paths, settings, dual=True)
    root.mainloop()
