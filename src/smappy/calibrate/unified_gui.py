"""One window for both calibration modes, with linked diagnostic pages."""
from dataclasses import asdict
import numpy as np
from scipy.stats import gaussian_kde
from .gui import CalibrationWindow, StackBrowser, _stack_slice
from .dual_gui import DualCalibrationWindow
from .core import CalibrationSettings
from .dual import DualColorSettings, channel_result, map_points


def residual_density(points):
    points = np.asarray(points)
    if len(points)<3 or np.linalg.matrix_rank(points-points.mean(0))<2:
        return np.ones(len(points))
    try:
        return gaussian_kde(points.T)(points.T)
    except np.linalg.LinAlgError:
        return np.ones(len(points))


class UnifiedCalibrationWindow(DualCalibrationWindow):
    def __init__(self, root, paths=(), settings=None, dual=False):
        import tkinter as tk
        from tkinter import ttk
        self.mode = tk.StringVar(value='Dual color' if dual else 'Single channel')
        self.modebar = ttk.Frame(root, padding=5)
        self.modebar.pack(fill='x')
        ttk.Label(self.modebar, text='Calibration mode').pack(side='left')
        selector = ttk.Combobox(self.modebar, textvariable=self.mode, state='readonly',
                                values=('Single channel', 'Dual color'))
        selector.pack(side='left', padx=5)
        selector.bind('<<ComboboxSelected>>', self.mode_changed)
        s = settings if isinstance(settings, DualColorSettings) else DualColorSettings(**asdict(settings or CalibrationSettings()))
        super().__init__(root, paths, s)
        # The inherited geometry controls stay alive when hidden, preserving choices.
        self.geometry_frames = root.winfo_children()[1:3]
        for widget in self.geometry_frames[0].winfo_children()[-2:]:
            widget.destroy()  # obsolete Inspect PSF label and selector
        self.mode_selector = selector
        self.active_dual = dual
        self.update_mode()
        root.title('SMAPpy — bead calibration')
        width = min(1450, max(800, root.winfo_screenwidth()-80))
        height = min(980, max(600, root.winfo_screenheight()-110))
        root.geometry(f'{width}x{height}+30+35')

    @property
    def is_dual(self):
        return self.mode.get()=='Dual color'

    def update_mode(self):
        after = self.modebar
        for frame in self.geometry_frames:
            if self.is_dual:
                frame.pack(fill='x', after=after)
                after = frame
            else:
                frame.pack_forget()
        self.notebook.tab(1, state='normal' if self.is_dual else 'disabled')
        self.notebook.tab(3, state='normal' if self.is_dual else 'disabled')

    def mode_changed(self, event=None):
        if self.busy:
            self.mode.set('Dual color' if self.active_dual else 'Single channel')
            return
        self.active_dual = self.is_dual
        self.result = self.diagnostics = self.profile_data = None
        self.showing_quality = False
        self.excluded.clear()
        self.table.delete(*self.table.get_children())
        self.notebook.select(0)
        for fig, canvas in self.pages:
            fig.clear(); canvas.draw_idle()
        self.fit_quality_button.configure(text='Calculate fit quality', state='normal')
        self.update_mode()
        self.status.set('Mode changed; files and common settings retained. Detect + calibrate to recalculate.')

    def settings(self):
        return super().settings() if self.is_dual else CalibrationWindow.settings(self)

    def run(self):
        self.notebook.select(0)
        return super().run() if self.is_dual else CalibrationWindow.run(self)

    def rebuild(self):
        return super().rebuild() if self.is_dual else CalibrationWindow.rebuild(self)

    def create_plot(self, parent):
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.notebook = ttk.Notebook(parent); self.notebook.pack(fill='both', expand=True)
        self.pages = []
        for title in ('Overview', 'Transformation', 'Fit quality', 'Field diagnostics', 'Bead diagnostics'):
            frame = ttk.Frame(self.notebook); self.notebook.add(frame, text=title)
            fig = Figure(figsize=(10, 6), constrained_layout=True)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(fill='both', expand=True)
            canvas.mpl_connect('pick_event', self.pick_pair)
            self.pages.append((fig, canvas))
        self.figure, self.canvas = self.pages[0]
        self.notebook.bind('<<NotebookTabChanged>>', lambda e: self.draw_current())

    def pick_pair(self, event):
        ids = getattr(event.artist, 'pair_ids', None)
        if ids is not None and len(event.ind):
            identity = str(int(ids[event.ind[0]]))
            self.table.selection_set(identity); self.table.see(identity)

    def acquisitions(self):
        # Aggregate diagnostics always pool sources; bead selection changes only
        # the source image and highlights the same stable IDs in pooled plots.
        return set(range(len(self.result.beads.sources)))

    def draw_current(self):
        if self.result is None:
            return
        index = self.notebook.index(self.notebook.select())
        self.figure, self.canvas = self.pages[index]
        self.fit_quality_button.configure(text='Calculate fit quality',
                    state='disabled' if self.busy or self.diagnostics is not None else 'normal')
        if index==2:
            if self.diagnostics is None:
                self.figure.clear(); self.figure.text(.5,.5,'Click Calculate fit quality to refit bead planes.',ha='center'); self.canvas.draw_idle()
            else:
                self.draw_quality()
        elif index in (1, 3) and self.is_dual:
            self.draw_transformation()
        elif index == 4:
            self.draw_bead_diagnostics()
        else:
            self.draw()

    def draw(self):
        self.figure, self.canvas = self.pages[0]
        if not self.is_dual:
            return CalibrationWindow.draw(self)
        if self.result is None:
            return
        r = self.result
        chosen = {int(i) for i in self.table.selection()}
        acquisitions = self.acquisitions()
        selected_stack = r.beads.records[min(chosen)]['stack'] if chosen else None
        si = selected_stack if selected_stack in acquisitions else min(acquisitions)
        self.figure.clear(); axes = self.figure.subplots(1,2)
        axes[0].imshow(r.beads.projections[si],cmap='gray',vmax=np.percentile(r.beads.projections[si],99.5))
        for i, rec in enumerate(r.beads.records):
            color = ('gold' if i in chosen else 'red' if i in self.excluded else
                     'limegreen' if r.accepted[i] else 'darkorange' if r.transform_accepted[i] else 'red')
            if rec['stack']==si:
                xy=np.array([c['full_image_xy'] for c in rec['channel_records']])
                line,=axes[0].plot(*xy.T,'o-',color=color,mfc='none',lw=.5,picker=5); line.pair_ids=np.array([i,i])
                axes[0].text(*xy[0],str(i),color=color,fontsize=7)
            if rec['stack'] in acquisitions:
                artist=axes[1].scatter(*r.beads.main_points[i],c=color,s=45 if i in chosen else 20,picker=5); artist.pair_ids=np.array([i])
        for ch, ids in enumerate(r.beads.unmatched):
            for i in ids:
                rec=r.beads.channels[ch].records[i]; xy=np.array(rec['full_image_xy'])
                if rec['stack']==si: axes[0].plot(*xy,'+',color='cyan')
                if ch==0 and rec['stack'] in acquisitions: axes[1].plot(*(xy+self.origin(rec['stack'])),'+',color='dodgerblue',ms=9)
        axes[0].set_title(f'Acquisition {si+1}: paired / unpaired (cyan)')
        axes[1].set(title='Main FoV: PSF + transform (green), transform only (orange)\nRejected (red), unpaired (+)',xlabel='Main x (pixels)',ylabel='Main y (pixels)')
        h,w=r.beads.geometry['image_shape']; split=r.beads.geometry['split_position']
        last=r.beads.settings.main_channel in ('right','lower'); horizontal='right-left' in r.beads.settings.layout
        xlim=(split if last else 0,w if last else split) if horizontal else (0,w)
        ylim=(0,h) if horizontal else (split if last else 0,h if last else split)
        origins=np.array([self.origin(i) for i in acquisitions])
        axes[1].set_xlim(origins[:,0].min()+xlim[0],origins[:,0].max()+xlim[1])
        axes[1].set_ylim(origins[:,1].max()+ylim[1],origins[:,1].min()+ylim[0]); axes[1].set_aspect('equal')
        self.canvas.draw_idle()

    def origin(self, stack):
        return np.array(self.result.beads.sources[stack]['roi'][:2] if self.result.beads.geometry['coordinate_system']=='camera-chip' else (0,0))

    def draw_transformation(self):
        r=self.result; acquisitions=self.acquisitions()
        ids=np.array([i for i,rec in enumerate(r.beads.records) if rec['stack'] in acquisitions],int)
        target=map_points(r.calibration.transformation,r.beads.secondary_points); delta=target-r.beads.main_points
        good=np.array([r.transform_accepted[i] and i not in self.excluded for i in ids], dtype=bool)
        self.figure.clear()
        if self.notebook.index(self.notebook.select()) == 3:
            self.draw_field_diagnostics(ids, good, target, acquisitions)
            return
        axes=self.figure.subplots(1,2)
        for subset, accepted in ((ids[good],True),(ids[~good],False)):
            if not len(subset): continue
            if accepted:
                density=residual_density(delta[subset]); order=np.argsort(density); subset=subset[order]
                artist=axes[0].scatter(*delta[subset].T,c=density[order],cmap='viridis',picker=5,label='Used (density)')
            else:
                artist=axes[0].scatter(*delta[subset].T,facecolors='none',edgecolors='red',picker=5,label='Rejected')
            artist.pair_ids=subset
        axes[0].axhline(0,color='gray',lw=.5); axes[0].axvline(0,color='gray',lw=.5); axes[0].legend(fontsize=8)
        spread=np.std(delta[ids[good]],axis=0,ddof=1) if good.sum()>1 else [np.nan,np.nan]
        axes[0].set(title=f'Round 2 residuals\nσx={spread[0]:.3f}, σy={spread[1]:.3f} px',xlabel='dx (pixels)',ylabel='dy (pixels)')
        chosen=[int(i) for i in self.table.selection() if int(i) in ids]
        if chosen:
            axes[0].scatter(*delta[chosen].T,s=110,facecolors='none',edgecolors='gold',lw=2)
        delta1 = r.transform_fit.round1_dxdy
        artist = axes[1].scatter(*delta1[ids].T, c=np.where(good, 'green', 'red'), picker=5)
        artist.pair_ids = ids
        limit = r.beads.settings.transform_axis_limit_px
        axes[1].plot([-limit, limit, limit, -limit, -limit],
                     [-limit, -limit, limit, limit, -limit], 'k--', lw=.8)
        axes[1].axhline(0, color='gray', lw=.5); axes[1].axvline(0, color='gray', lw=.5)
        axes[1].set(title=f'Round 1 screening\n|dx|, |dy| ≤ {limit:g} px',
                    xlabel='dx (pixels)', ylabel='dy (pixels)')
        if chosen:
            axes[1].scatter(*delta1[chosen].T, s=110, facecolors='none', edgecolors='gold', lw=2)
        # A physical display floor avoids autoscaling numerical roundoff to a
        # misleadingly huge cloud for effectively exact synthetic transforms.
        for ax, values in zip(axes, (delta[ids], delta1[ids])):
            extent = max(limit*1.15, float(np.max(np.abs(values)))*1.1) if len(values) else limit*1.15
            ax.set(xlim=(-extent, extent), ylim=(-extent, extent))
            ax.set_aspect('equal', adjustable='box')
            ax.title.set_fontsize(10)
        self.canvas.draw_idle()

    def draw_field_diagnostics(self, ids, good, target, acquisitions):
        r = self.result
        axes = self.figure.subplots(1, 2)
        axes[0].scatter(*r.beads.main_points[ids].T, facecolors='none', edgecolors='black', label='Main')
        artist = axes[0].scatter(*target[ids].T, marker='+', c=np.where(good, 'green', 'red'),
                                 picker=5, label='Mapped ch. 2')
        artist.pair_ids = ids
        for ch, indices in enumerate(r.beads.unmatched):
            for i in indices:
                rec = r.beads.channels[ch].records[i]
                if rec['stack'] not in acquisitions:
                    continue
                xy = np.array(rec['full_image_xy'])+self.origin(rec['stack'])
                if ch:
                    xy = map_points(r.calibration.transformation, xy[None])[0]
                axes[0].plot(*xy, 'x', color='magenta' if ch else 'dodgerblue')
        axes[0].set(title='Overlay; unpaired ×\nMain (blue), secondary (magenta)',
                    xlabel='Main x (pixels)', ylabel='Main y (pixels)')
        axes[0].legend(fontsize=7)
        valid = ids[np.isfinite(r.residuals[ids]) & np.array([
            r.beads.records[i]['brightness_accepted'] and i not in self.excluded for i in ids], dtype=bool)]
        if len(valid):
            artist = axes[1].scatter(*r.beads.main_points[valid].T, c=r.residuals[valid],
                                     cmap='magma', picker=5)
            artist.pair_ids = valid
            self.figure.colorbar(artist, ax=axes[1], label='Joint shape residual')
            psf_ids = valid[r.accepted[valid]]
            axes[1].scatter(*r.beads.main_points[psf_ids].T, s=65, facecolors='none',
                            edgecolors='limegreen', label='PSF accepted')
            axes[1].legend(fontsize=7)
        coverage = r.calibration.parameters['coverage_area_px2']
        axes[1].set(title=f'Shape across FoV\nPooled hull (px²): transform {coverage["transformation"]:.0f}\nPSF {coverage["psf"]:.0f}',
                    xlabel='Main x (pixels)', ylabel='Main y (pixels)')
        chosen = [int(i) for i in self.table.selection() if int(i) in ids]
        for ax in axes:
            if chosen:
                ax.scatter(*r.beads.main_points[chosen].T, s=110, facecolors='none', edgecolors='gold', lw=2)
            ax.invert_yaxis(); ax.set_aspect('equal'); ax.title.set_fontsize(10)
        self.canvas.draw_idle()

    def browse_stack(self):
        if not self.is_dual: return CalibrationWindow.browse_stack(self)
        if self.result is not None: PairedStackBrowser(self.root,self.result)

    def draw_bead_diagnostics(self):
        r = self.result
        selected = {int(i) for i in self.table.selection()}
        ids = np.arange(len(r.beads.records))
        colors = np.array(['red' if i in self.excluded else 'limegreen' if r.accepted[i]
                           else 'darkorange' if self.is_dual and r.transform_accepted[i]
                           else 'red' for i in ids])
        self.figure.clear()
        axes = self.figure.subplots(2, 2).ravel() if self.is_dual else self.figure.subplots(1, 2)
        xy = np.hypot(r.shifts[:, 1], r.shifts[:, 2])

        def scatter(ax, x, y, log=False):
            x, y = np.asarray(x), np.asarray(y)
            valid = np.isfinite(x) & np.isfinite(y)
            if log:
                valid &= y > 0
                ax.set_yscale('log')
            artist = ax.scatter(x[valid], y[valid], c=colors[valid], picker=5)
            artist.pair_ids = ids[valid]
            chosen = valid & np.array([i in selected for i in ids], dtype=bool)
            ax.scatter(x[chosen], y[chosen], s=100, facecolors='none',
                       edgecolors='gold', linewidths=2, zorder=5)

        for ch in range(2 if self.is_dual else 1):
            records = ([rec['channel_records'][ch] for rec in r.beads.records]
                       if self.is_dual else r.beads.records)
            scatter(axes[ch], xy, [rec['brightness_adu'] for rec in records], log=True)
            if records:
                lower, upper = (records[0].get(key) for key in ('brightness_lower_adu', 'brightness_upper_adu'))
                if lower is not None and upper is not None and np.isfinite([lower, upper]).all():
                    axes[ch].axhspan(lower, upper, color='green', alpha=.08)
            name = ('Main' if ch == 0 else 'Secondary')+' brightness' if self.is_dual else 'Brightness'
            axes[ch].set(title=name+' vs XY shift', ylabel='Brightness (ADU)',
                         xlabel=('Shared' if self.is_dual else 'Applied')+' XY shift (pixels)')
        ax = axes[2 if self.is_dual else 1]
        scatter(ax, r.shifts[:, 0]*r.calibration.dz, r.residuals)
        ax.set(title='Bead alignment', xlabel='Applied z shift (nm)', ylabel='Relative shape residual')
        if self.is_dual:
            scatter(axes[3], r.correlations, r.residuals)
            axes[3].set(title='Shape and correlation', xlabel='Joint correlation', ylabel='Relative shape residual')
        for ax in axes:
            ax.title.set_fontsize(9); ax.tick_params(labelsize=8)
            ax.xaxis.label.set_size(9); ax.yaxis.label.set_size(9)
        self.figure.suptitle('All acquisitions · green: PSF'+
            (' · orange: transformation only' if self.is_dual else '')+' · red: rejected', fontsize=9)
        self.canvas.draw_idle()

    def fit_quality(self):
        if self.result is None: return
        if self.diagnostics is not None:
            self.draw_quality(); return
        if not self.is_dual: return CalibrationWindow.fit_quality(self)
        from .validation import fit_bead_diagnostics
        result=self.result
        def diagnostics():
            fitted=[]
            for ch in range(2):
                view=channel_result(result,ch)
                fitted.append(fit_bead_diagnostics(view))
            result.registration.refits={str(ch):d for ch,d in enumerate(fitted)}
            return fitted,paired_profiles(result)
        self.work(diagnostics,'diagnostics')

    def draw_quality(self):
        self.figure,self.canvas=self.pages[2]
        if self.notebook.index(self.notebook.select())!=2: self.notebook.select(2)
        if not self.is_dual:
            self.fit_quality_button.configure(text='Calculate fit quality', state='disabled')
            return CalibrationWindow.draw_quality(self)
        self.figure.clear(); axes=self.figure.subplots(2,4)
        selected={int(i) for i in self.table.selection()}
        acquisitions = self.acquisitions()
        for ch,name in enumerate(('Main','Secondary')):
            d,profiles=self.diagnostics[ch],self.profile_data[ch]
            for i in np.unique(d['bead_id']):
                if i >= 0 and self.result.beads.records[int(i)]['stack'] not in acquisitions:
                    continue
                use=d['bead_id']==i
                style=dict(color='black',lw=2.5) if i==-1 else dict(lw=2 if i in selected else .7,alpha=.8)
                axes[ch,0].plot(d['expected_z_nm'][use],d['fitted_z_nm'][use],**style)
                axes[ch,1].plot(d['expected_z_nm'][use],d['centered_error_nm'][use],**style)
            limits=[min(d['expected_z_nm']),max(d['expected_z_nm'])]
            axes[ch,0].plot(limits,limits,'k--',lw=.5)
            axes[ch,0].set(title=name+' z fits',xlabel='Expected z (nm)',ylabel='Fitted z (nm)')
            axes[ch,1].axhline(0,color='gray',lw=.5)
            axes[ch,1].set(title=name+' centered z error',xlabel='Expected z (nm)',ylabel='Error (nm)')
            psf=channel_result(self.result,ch).calibration.psf
            for col,axis,model in ((2,'x',psf[len(psf)//2,psf.shape[1]//2]),(3,'z',psf[:,psf.shape[1]//2,psf.shape[2]//2])):
                coord=profiles[axis+('_px' if axis=='x' else '_nm')]
                curves=profiles[axis+'_profiles'].copy(); average=profiles['average_'+axis].copy(); model=model.copy()
                for i,curve in zip(profiles['bead_id'],curves):
                    if self.result.beads.records[int(i)]['stack'] in acquisitions:
                        axes[ch,col].plot(coord,curve,lw=2 if i in selected else .6,alpha=.6)
                axes[ch,col].plot(coord,average,'k',lw=2,label='Average')
                axes[ch,col].plot(coord,model,'--',color='royalblue',lw=2,label='Spline knots')
                axes[ch,col].set(title=name+' '+axis+' profile',xlabel=axis+(' (pixels)' if axis=='x' else ' (nm)'),ylabel='Intensity (calibration scale)')
                axes[ch,col].legend(fontsize=7)
                axes[ch,col].relim(); axes[ch,col].autoscale_view()
                axes[ch,col].margins(x=.02, y=.05)
        for ax in axes.ravel():
            ax.title.set_fontsize(9); ax.tick_params(labelsize=8)
            ax.xaxis.label.set_size(9); ax.yaxis.label.set_size(9)
        self.fit_quality_button.configure(text='Calculate fit quality', state='disabled'); self.canvas.draw_idle()

    def redraw_profiles(self):
        if self.diagnostics is not None and self.notebook.index(self.notebook.select())==2: self.draw_quality()


class PairedStackBrowser(StackBrowser):
    def __init__(self,parent,result):
        import tkinter as tk
        from tkinter import ttk
        self.result=result; self.linked=tk.BooleanVar(value=True); self.native=tk.BooleanVar(value=False)
        super().__init__(parent,result.registration.channel_raw_psfs[0],result.calibration.main)
        self.window.title('Paired average PSFs — synchronized slices'); self.window.geometry('1050x750')
        controls=ttk.Frame(self.window,padding=5); controls.pack(side='bottom',fill='x')
        ttk.Checkbutton(controls,text='Linked channel contrast',variable=self.linked,command=self.draw).pack(side='left')
        ttk.Checkbutton(controls,text='Native camera orientation',variable=self.native,command=self.draw).pack(side='left')
        self.draw()

    def draw(self):
        if not hasattr(self,'canvas'): return
        volumes=[np.asarray(v) for v in self.result.registration.channel_raw_psfs]
        mirror=self.result.beads.geometry['mirror_axis_xy']
        if self.native.get() and mirror is not None: volumes[1]=np.flip(volumes[1],2-mirror)
        orientation=self.orientation.get(); index=int(np.clip(self.position.get(),0,self.slice_count()-1))
        images=[_stack_slice(v,orientation,index) for v in volumes]
        maxima=[float(np.max(im if self.auto_contrast.get() else vol)) for im,vol in zip(images,volumes)]
        if self.linked.get(): maxima=[max(maxima)]*2
        self.figure.clear(); axes=self.figure.subplots(1,2,sharex=True,sharey=True)
        for ch,(ax,im,maximum) in enumerate(zip(axes,images,maxima)):
            options=dict(cmap='inferno',vmin=0,vmax=max(maximum*self.contrast_factor.get(),1e-15))
            if orientation!='XY': options.update(aspect='auto',extent=(0,im.shape[1]-1,self.calibration.z_index_to_nm(len(volumes[ch])-1),self.calibration.z_index_to_nm(0)))
            ax.imshow(im,**options)
            ax.set(title=('Main' if ch==0 else 'Secondary')+' · '+orientation,xlabel='y (pixels)' if orientation=='YZ' else 'x (pixels)',ylabel='y (pixels)' if orientation=='XY' else 'Emitter z (nm)')
        self.figure.suptitle(('Native camera' if self.native.get() else 'Joint registration')+f' orientation · slice {index}')
        self.canvas.draw_idle()


def paired_profiles(result):
    """Display aligned pairs on one amplitude scale; never normalize channels separately."""
    from scipy import ndimage
    r = result.registration
    ids = np.flatnonzero(r.accepted)
    reference = np.asarray(r.channel_raw_psfs)
    p, crop = r.beads.settings.padding, r.z_crop_start
    samples = []
    bright = reference > np.quantile(reference, .75)
    for i in ids:
        sample = np.stack([ndimage.shift(r.beads.original_volumes[i,ch],
                r.shifts[i]+r.beads.channel_offsets[i,ch], order=3, mode='constant') for ch in range(2)])
        sample = sample[:,crop:sample.shape[1]-crop,p:-p,p:-p]
        valid = bright & (reference > 1e-12)
        factor = np.median(sample[valid]/reference[valid])
        samples.append(sample/max(float(factor),1e-15))
    samples = np.asarray(samples)
    output = []
    for ch in range(2):
        volume = samples[:,ch]
        average = reference[ch].copy()
        mirror = result.beads.geometry['mirror_axis_xy']
        if ch and mirror is not None:
            volume = np.flip(volume,3-mirror)
            average = np.flip(average,2-mirror)
        z,y,x = (n//2 for n in average.shape)
        output.append({'bead_id':ids, 'x_px':np.arange(average.shape[2])-x,
            'z_nm':result.calibration.main.z_index_to_nm(np.arange(average.shape[0])),
            'x_profiles':volume[:,z,y,:], 'z_profiles':volume[:,:,y,x],
            'average_x':average[z,y,:], 'average_z':average[:,y,x]})
    return output
