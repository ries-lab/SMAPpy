---
title: Render tab
summary: The layers of the picture -- which localizations each one shows, and how it is drawn.
widget: smappy.gui.render_tab.RenderTab
covers: [smappy.session.Layer, smappy.session.default_bounds, smappy.filter.LocFilter, smappy.group.group, smappy.group.connect, smappy.group.combine, smappy.render.render_locs, smappy.render.render_sigmas, smappy.render.psf_sigmas, smappy.render.camera_pixelsize_nm, smappy.render.camera_grid, smappy.render.SigmaSettings, smappy.render.to_rgb, smappy.render.RenderAxes, smappy.render.render_tracks, smappy.render.track_segments, smappy.render.render_segments, smappy.render.render_voronoi, smappy.render.VoronoiCells, smappy.render.voronoi_cells_for, smappy.lut.on_white, smappy.lut.invert_sum, smappy.lut.complement]
---

## What it does

A localization table is a list of positions, not an image.  To see the
structure, every localization is drawn as a small spot on a fine grid and the
spots are added up: where many molecules were found, the picture is bright.
The Render tab decides what goes into that picture and how it is drawn.

The picture is made of **layers**.  Each layer shows the localizations its own
filter keeps, drawn its own way -- one layer per channel or per file, say, or
the same data twice with different filters.  The visible layers are added up
into the image in the main window, as in SMAP.  The *overlay* button on the
picture's toolbar shows them *split* instead, a panel per layer side by side,
or *split + overlay* with the added-up picture after them (SMAP's *split* and
*comp*).
The panels go across or one above another, whichever shows the data larger in
the window, and move and zoom together; the [ROI](plugin:Panels/Glossary)
(region of interest: a rectangle, line or polygon drawn on the picture) is
drawn and edited on the first and its outline is on all of them.  Beside it, *pixel* says what one
screen pixel is: the zoom, as a number.  Type a size to zoom to it, or press
*pixel* to go to 10 nm.  Double-click the picture to move that place to the
middle.  At the end of the toolbar is how many localizations each layer draws
in the view -- counted again after every zoom -- or in the ROI when one is
drawn; *3D* opens the [3D view](plugin:Panels/3D view).  Every control below the layer strip edits one layer,
the one selected in the strip.

What a layer shows is also what the rest of the program works on.  A plugin
measures the localizations the layer's filter keeps (inside the drawn
ROI, if there is one), and
the ROI manager takes its filter, and whether blinks are merged, from a layer
too.  So a filter set here to clean up the picture also decides what is
measured.

From top to bottom the tab has: a small *overview* of the whole field, the
layer strip, the *filter*, the *image* section (only for a layer that is a
pixel image), the *display* -- how the localizations are drawn -- and, closed
by default, the *axes*, which turn the picture into a plot of any column
against any other.

## How it works

```figure-setup
import dataclasses
from smappy.simulate import simulate
from smappy.session import Session
from smappy.render import FieldOfView, RenderAxes, render_locs
locs = simulate(n_frames=10000, seed=1)
layer = Session(locs).layers[0]  # what the tab edits: default bounds, grouped
zoom = FieldOfView.from_range((4700, 5500), (4400, 5200), 4.0)
whole = FieldOfView.from_range((0, 10000), (0, 10000), 20.0)

def draw(ax, title, fov=zoom, **change):
    """Render the layer with some of its settings changed, then put them back."""
    state = layer.state
    kept = (state.settings, state.display)
    state.settings = dataclasses.replace(state.settings, **{
        k: v for k, v in change.items() if hasattr(state.settings, k)})
    state.display = dataclasses.replace(state.display, **{
        k: v for k, v in change.items() if hasattr(state.display, k)})
    rgb, _ = layer.render(fov)
    state.settings, state.display = kept
    ax.imshow(rgb, extent=fov.extent)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
```

**1. Layers.**  A file opens as one layer.  **+** adds another, which starts
as a copy of the selected one -- a second layer is nearly always the first
with one thing changed.  A layer can also be a pixel image, placed under the
localizations by its pixel size and position: the camera frames the fit kept
with the file (their average, and a few single frames), or a widefield or
diffraction-limited picture of the same cells.  The layers are drawn one by one
and their colours added.

**2. The filter.**  A [filter](plugin:Panels/Glossary) (also called bounds)
is a range per column: keep the localizations
whose precision is at most 25 nm, whose z lies within 500 nm of focus, and so
on.  A localization is shown only if it is inside every range.  The tab shows
one column at a time, as a histogram with the kept range shaded, and the
range can be dragged or typed.  A new layer opens with a few bounds already
set (see *In detail*), so that fits which are clearly bad are left out from
the start -- and are left out visibly, in the shaded histogram, rather than
silently.

```figure A simulated ring-and-cross structure.  Left: the localization precision of all localizations, with the range the default bound keeps shaded.  Middle: every localization.  Right: what the default bounds take out -- mostly the dim, imprecise localizations scattered around the lines.
layer.show_grouped(False)
fig.set_size_inches(7.5, 2.6)
a, b, c = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.3, 1, 1]})
a.hist(np.asarray(layer.locs["xy_err_nm"]), bins=np.linspace(0, 60, 121), color="0.5")
a.axvspan(0, layer.filter.ranges["xy_err_nm"][1], color=(0.24, 0.47, 0.86), alpha=0.25)
a.set_xlabel("xy_err_nm (nm)", fontsize=8); a.set_yticks([]); a.tick_params(labelsize=7)
field = FieldOfView.from_range((4000, 6000), (4000, 6000), 10.0)
bounds = dict(layer.filter.ranges)
for name in bounds:
    layer.remove_bound(name)
draw(b, f"no bounds: {len(layer.filter)}", field, mode="hist")
for name, (lo, hi) in bounds.items():
    layer.set_bound(name, lo, hi)
state = layer.state
settings = dataclasses.replace(state.settings, mode="hist")
out = ~layer.filter.mask
c.imshow(state.display.apply(render_locs(layer.locs, field, settings, state.display,
                                         select=out)), extent=field.extent)
c.set_title(f"taken out: {int(out.sum())}", fontsize=9)
c.set_xticks([]); c.set_yticks([])
layer.show_grouped(True)
```

**3. Merging blinks.**  A fluorophore that stays on for several camera frames is
localized in each of them, so one blink appears as a small cluster of
localizations.  [Merge blinks](plugin:Panels/Glossary) (also called grouping
or linking) combines the localizations in consecutive frames that come from
one blinking event -- that lie within a small box of each other -- into one
localization, at their weighted mean position, with the photons of all of
them and a correspondingly better precision.  A picture of merged blinks is
cleaner and counts molecules more fairly: a long blink no longer weighs more
than a short one.  Layers merge blinks by default (*merge blinks*).  A large
file does not wait for the merge: it opens at once, the blinks are merged in
the background -- the picture says "merging blinks" with the stage and a
clock, and so does the status bar -- and they are drawn when they are there.
No single localizations are drawn in their place meanwhile.  The distance and the
number of dark frames allowed are set under *merge settings...*, beside it,
for every layer at once.  The distance is either
fixed or, with *merge by* set to *precision*, scaled by how precisely each
localization was measured: bright ones are held close together, dim ones may
lie further apart.

```figure The same localizations (with the default bounds, drawn as a histogram) before and after merging blinks, and how many frames each blink lasted.
fig.set_size_inches(7.5, 2.6)
a, b, c = fig.subplots(1, 3)
layer.show_grouped(False)
draw(a, f"localizations: {len(layer.filter)}", mode="hist")
layer.show_grouped(True)
draw(b, f"blinks merged: {len(layer.filter)}", mode="hist")
fig.subplots_adjust(wspace=0.35)
c.hist(np.asarray(layer.locs["n_in_group"]), bins=np.arange(0.5, 12.5), color="0.5")
c.set_xlabel("n_in_group (frames)", fontsize=8); c.tick_params(labelsize=7)
```

**4. Drawing the localizations.**  *render* chooses the spot each
localization is drawn as.  *histogram* counts localizations per pixel: exact, but
at a fine pixel size mostly single bright pixels.  *Gaussian (fixed)* draws every
localization as the same Gaussian blob, of width *sigma x (fixed)*.
*Gaussian (precision)*, the default, draws each one as a Gaussian as wide as its own
[localization precision](plugin:Panels/Glossary) (also called the CRLB or
localization uncertainty: how far its fitted position is expected to lie from
the true one) times the *precision factor*
([Baddeley et al. 2010](https://doi.org/10.1017/S143192760999122X)): a
precise localization is a sharp dot, an uncertain one a faint wide blur, so
the picture shows how well each position is known.  *diffraction-limited*
draws what the camera saw instead: each localization as a spot as
wide as its fitted [PSF](plugin:Panels/Glossary) (point spread function: the
image one molecule makes on the camera), on the camera's own pixels -- the widefield picture
of the same molecules, to set beside the super-resolved one.  *tracks* draws
[tracks](plugin:Panels/Glossary) instead of spots: each track -- the
localizations sharing a value of the *track id* column, `track_id` from
[Link tracks](plugin:Analysis/Tracking/Link tracks) -- as the line through
its localizations in time order, *line width* pixels wide.  Coloured by a
column, the line takes the colour of each step: by `frame`, a track runs
through the LUT as it goes on; by `track_D_um2_per_s` from
[Tracking analysis](plugin:Analysis/Tracking/Tracking analysis), slow tracks
and fast ones stand apart.  *Voronoi density* (SharpViSu,
[Andronov et al. 2016](https://doi.org/10.1093/bioinformatics/btw123))
divides the plane into a cell per localization -- its
[Voronoi tessellation](plugin:Panels/Glossary): the part of the plane closer
to it than to any other -- and colours each cell by its density, one over
its area.  A dense structure is evenly bright and sparse background is dim,
with no spot width to choose: the cells are as large as the gaps between
the localizations.  The cells along the edge of the field are open, and
are left empty.

```figure The crossing of the two lines, 800 nm across, drawn with each of the three super-resolution modes.
fig.set_size_inches(7.5, 2.6)
axes = fig.subplots(1, 3)
draw(axes[0], "histogram", mode="hist")
draw(axes[1], "Gaussian (fixed), sigma 10 nm", mode="gauss", sigma=10.0)
draw(axes[2], "Gaussian (precision), factor 0.5")
```

```figure The crossing in *Gaussian (precision)* and *Voronoi density*, and the whole field in *Voronoi density* (contrast 2): each cell is as bright as it is small, so the lines are filled with a mosaic of bright cells, with no gaps between the spots, and the scattered background is dim.
fig.set_size_inches(7.5, 2.6)
axes = fig.subplots(1, 3)
draw(axes[0], "Gaussian (precision)")
draw(axes[1], "Voronoi density", mode="voronoi", contrast=2.0)
draw(axes[2], "Voronoi density, whole field", fov=whole, mode="voronoi", contrast=2.0)
```

```figure The whole field, 10 µm across, in *Gaussian (precision)* and *diffraction-limited*: on 100 nm camera pixels the ring and the lines are as wide as the PSF, as the camera saw them.
fig.set_size_inches(5.2, 2.6)
axes = fig.subplots(1, 2)
draw(axes[0], "Gaussian (precision)", fov=whole)
draw(axes[1], "diffraction-limited", fov=whole, mode="dl")
```

```figure Simulated tracks, 6 µm across, drawn as *tracks*: left in one colour, right coloured by frame, so each track runs from blue to red as it goes on and a crossing of two tracks at different times is told apart.
from smappy.simulate import simulate_tracks
from smappy.render import RenderSettings, DisplaySettings
moving = simulate_tracks(n_frames=200, field_um=6.0, density_um2=0.1,
                         d_um2_s=(0.3,), track_frames=40.0, seed=2)
view = FieldOfView.from_range((0, 6000), (0, 6000), 15.0)
fig.set_size_inches(5.2, 2.6)
axes = fig.subplots(1, 2)
for ax, colour in zip(axes, (None, "frame")):
    settings = RenderSettings(mode="tracks", track_column="track_id_true",
                              line_width=1.0, color_field=colour)
    display = DisplaySettings(lut="hot" if colour is None else "jet", contrast=1.5)
    ax.imshow(display.apply(render_locs(moving, view, settings, display)),
              extent=view.extent)
    ax.set_title("one colour" if colour is None else "coloured by frame", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
```

**5. Brightness and colour.**  A super-resolution image has a few very bright
pixels -- a long blink, a fiducial bead -- and a great many faint ones, so its
brightest pixel is a useless scale.  *contrast* instead saturates a fixed
fraction of the pixels: at contrast 3 the brightest one in a thousand are
white, at 2 one in a hundred (brighter), at 4 one in ten thousand (darker).
*gamma* below 1 lifts the faint parts further.  The *LUT* (look-up table,
also called a colour map) is the colour scale brightness is shown in.  Instead of brightness alone, the
colour can also encode a column (*colour by*, e.g. z), with the brightness
still showing density.

```figure The same crossing at three contrasts, and with gamma 0.5.
fig.set_size_inches(7.5, 5.2)
axes = fig.subplots(2, 2).ravel()
draw(axes[0], "contrast 2", contrast=2.0)
draw(axes[1], "contrast 3 (default)")
draw(axes[2], "contrast 4", contrast=4.0)
draw(axes[3], "contrast 3, gamma 0.5", gamma=0.5)
```

```figure The whole field coloured by z: the ring is tilted, so its top and bottom lie at different depths, and the two lines are at slightly different heights.  The same on a white background, and the plain picture with the LUT inverted.
fig.set_size_inches(7.5, 2.6)
axes = fig.subplots(1, 3)
by_z = dict(color_field="z_nm", color_range=(-300.0, 300.0), lut="turbo", contrast=2.0)
draw(axes[0], "colour by z_nm", whole, **by_z)
draw(axes[1], "colour by z_nm, white background", whole, white_background=True, **by_z)
draw(axes[2], "hot, inverted", whole, invert="sum", contrast=2.0)
```

**6. Any column against any other.**  The *axes* section says which column
each axis of the picture is.  Normally that is x and y, and the picture is a
picture of the sample.  Choose z for the vertical axis and it is a side view;
choose frame and photons and it is a plot of brightness over the acquisition
-- drawn with the same layers, filters, colours and ROIs as the image.  This
is how bleaching, a change of laser power or a drop in quality over time is
seen at a glance.

```figure Left: z against x -- a side view of the tilted ring and the two lines (z stretched for the figure).  Right: photons against frame, one dot per blink.
fig.set_size_inches(7.5, 2.8)
left, right = fig.subplots(1, 2)
fig.subplots_adjust(wspace=0.3)
state = layer.state
kept = state.settings

def versatile(ax, axes, mode):
    state.settings = dataclasses.replace(kept, axes=axes, mode=mode, sigma=0.0)
    x, y = axes.coordinates(layer.locs, layer.filter.indices)
    (x0, x1), (y0, y1) = np.quantile(x, (0.01, 0.99)), np.quantile(y, (0.01, 0.99))
    px, py = 0.08 * (x1 - x0), 0.15 * (y1 - y0)
    fov = FieldOfView.fit((x0 - px, x1 + px), (y0 - py, y1 + py), 400, 300)
    rgb, _ = layer.render(fov)
    ax.imshow(rgb, origin="lower", aspect="auto",
              extent=(fov.x0 * axes.x_scale, fov.x1 * axes.x_scale,
                      fov.y0 * axes.y_scale, fov.y1 * axes.y_scale))
    ax.set_xlabel(axes.x, fontsize=8); ax.set_ylabel(axes.y, fontsize=8)
    ax.tick_params(labelsize=7)

versatile(left, RenderAxes(x="x_nm", y="z_nm"), "precision")
left.set_ylim(-700, 700)
# the scales *fit* picks here: 5 frames and 20 photons per render unit
versatile(right, RenderAxes(x="frame", y="photons", x_scale=5.0, y_scale=20.0), "hist")
state.settings = kept
```

## In detail

**The rendering kernel.**  Each localization contributes the integral of its
Gaussian over each pixel, not the Gaussian's value at the pixel centre:

$$K_{jk} \propto \left[\mathrm{erf}\left(\frac{x_{j+1} - x}{\sqrt{2}\,\sigma_x}\right) - \mathrm{erf}\left(\frac{x_j - x}{\sqrt{2}\,\sigma_x}\right)\right]\left[\mathrm{erf}\left(\frac{y_{k+1} - y}{\sqrt{2}\,\sigma_y}\right) - \mathrm{erf}\left(\frac{y_k - y}{\sqrt{2}\,\sigma_y}\right)\right]$$

with $x_j$, $y_k$ the pixel edges.  The kernel is cut off at 2.7 $\sigma$ and
normalised by its own sum over the pixels it reaches, so every localization adds exactly one to the
image, whatever its width and wherever it sits in its pixel; a histogram is
the limit $\sigma \to 0$ of the same kernel.  A pixel $j$ covers
$[x_0 + jp,\, x_0 + (j+1)p)$ for pixel size $p$.

**The width in *Gaussian (precision)*.**  With $\sigma_i$ the localization
precision of localization $i$ (`xy_err_nm`), $f$ the *precision factor*
(0.5 by default) and $p$ the rendered pixel size,

$$s_i = \min\left(\max\left(f\,\sigma_i,\ 0.7\,p\right),\ 10\ \mathrm{median}_k\left(f\,\sigma_k\right)\right).$$

The floor of 0.7 pixels keeps a very precise localization from vanishing
between pixels; it follows the zoom, since the main view renders at the
screen's resolution.  The cap keeps a handful of absurd precisions from
smearing over the picture and from dominating the render time.  A missing
precision is drawn at the floor.  On an axis that is z, the axial precision
`z_err_nm` is used where the fit produced one.  On an axis that is not a
position at all (photons, frame), there is no precision, and the explicit
width is used -- zero, plain binning, unless set.

**The *diffraction-limited* mode.**  As SMAP's DL: the picture is rendered on a grid of
camera pixels -- their edges on multiples of the pixel size $a$, which comes
from the fit's metadata (`pixelsize_nm`, or the camera's), a simulation's
optics, or 100 nm -- with each localization a Gaussian of width `sigma_nm`
(and `sigma_y_nm` vertically, where the fit had two), and then blown up to
the view pixel by pixel, without interpolation, so that the camera's pixels
stay visible.  A width that is missing, not positive or above 1500 nm (a
failed fit) is one camera pixel.  It needs the positions on both axes; the
3D view, whose turned picture no camera saw, draws it as *Gaussian (fixed)* at the
median PSF width.

**The *tracks* mode.**  The rows the layer's filter keeps with an id above
0 are sorted by id and then by `frame` (by `time_s` where the table has
it); each two consecutive ones of a track are a segment.  A pixel takes
from a segment the share of it a line of width $w$ pixels covers,
approximated by the distance $d$ of the pixel's centre from the segment:
$\min(1, \max(0, w/2 + 1/2 - d))$, so the line is antialiased and its
ends are round.  Two segments of a track meet at a shared localization, so
over a pixel or so a joint is up to twice as bright.  A segment weighs one
(or the mean of its ends' weight column) and, coloured by a column, takes
the mean of its two ends' values.  The view culls whole tracks by their
bounding boxes rather than localizations by position, since a segment
across the view can have both ends outside it.  The line width is in
pixels of the picture, so an exported picture at a finer pixel size has
thinner lines in nanometres.  The 3D view draws the localizations of a
*tracks* layer as points, as *Gaussian (precision)* where the table has
precisions.

**The *Voronoi density* mode.**  The cells are those of every localization
the layer's filter keeps, from the same Delaunay triangulation as
[Neighbourhood](plugin:Analysis/Process/Neighbourhood)'s Voronoi density:
localizations at exactly the same position share a cell and are each
counted, and a cell open at the edge of the field has no area.  A cell
$i$ of area $A_i$ holding $n_i$ localizations has the density
$\rho_i = n_i / A_i$, in localizations per µm² (per $10^6$ square units on
axes that are not nm).  A pixel of area $a$ takes the mean density over
it.  Where the cell its centre lies in -- the cell of the localization
nearest to the centre -- is larger than the pixel, that is the cell's
density, SharpViSu's picture.  Cells smaller than the pixel are counted
into the pixel their localizations are in instead, since
$\rho_i A_i = n_i$: zoomed out, where a pixel holds many cells, the
picture is the histogram over the pixel's area rather than one cell picked
at random, which would be noise.  With $N$ the localizations of the small
cells in the pixel and $S$ their areas,

$$v = \frac{N + \rho_c \max(a - S, 0)}{a},$$

where $\rho_c$ is the density of the cell at the centre if it is larger
than the pixel, and 0 if it is small or open.  On a uniform field the
picture is flat at the field's density at any zoom.  Coloured by a column,
a counted localization brings its own colour and a cell the mean of its
localizations' values.  The *weight* column is not used: the density is of
localizations.

The cells are computed once for a layer's table and filter (about 12 µs a
localization on 4 cores: two million in half a minute) and kept for the
last four, so panning, zooming and exporting only look the pixels up, a
few tenths of a second for a million pixels.  A filter changed, or a fit
still adding localizations, computes them again.  The cells are never of
the localizations in view only, as those along the view's edge would
change with every pan.  Just outside the field's edge, a cell that closes
beyond it is long and thin, so of low density; further out the open cells
leave the picture empty.  The 3D view draws a *Voronoi density* layer as
points, as *Gaussian (precision)* where the table has precisions.

**Contrast and gamma.**  With $I$ the rendered intensity of a pixel and $c$
the *contrast*, the pixel value that becomes full scale is the quantile

$$I_{\mathrm{max}} = Q_{1 - 10^{-c}}(I),$$

that is, $10^{-c}$ of the pixels are saturated (if that quantile is zero, as
in a very sparse image, the maximum is used).  The displayed value is
$v = \min(I / I_{\mathrm{max}}, 1)^{\gamma}$ with $\gamma$ the *gamma*, and
$v$ picks one of the LUT's 256 colours.  Contrast and gamma change only this
last step, so moving them does not render again.

**Colour by a column.**  Each localization gets the LUT colour of its value
$u$ in the *colour range* $[u_{\mathrm{lo}}, u_{\mathrm{hi}}]$ (values outside
take the end colours).  Colours are added up during rendering, together with
the plain count; the displayed hue is the average colour of the pixel and its
brightness the count, put through contrast and gamma as above.  A dense pixel
therefore keeps its hue instead of bleaching to white.  Choosing a column
starts its range at its 0.5th to 99.5th percentile and switches the LUT to
turbo; going back to intensity switches it to hot.  Because the colour is part
of the rendering here, changing the LUT of a coloured layer renders it again.

**Inverting, and the white background.**  With $c = (r, g, b)$ a LUT
colour, *invert*'s SMAP choice gives $\min(r + g + b - c, 1)$ per channel
(SMAP's `lutinvert`), and its grey choice gives $\max(c) + \min(c) - c$, the
opposite hue at the same lightness, so that a layer over its inverse turns
grey where they coincide.  *white background* turns the brightness over and
keeps the hue, $c + 1 - \max(c) - \min(c)$: black becomes white, a saturated
red stays red.  It is applied once, to the sum of the layers.

**Adding up layers.**  The RGB images of the visible layers are added and
clipped at 1.  Two layers in red and green therefore show yellow where both
are bright.

**The filter.**  A bound $(\mathrm{lo}, \mathrm{hi})$ on a column keeps
$\mathrm{lo} \leq u \leq \mathrm{hi}$; an empty end is no bound.  A
localization with no value there (NaN) is excluded by any bound on that
column -- a localization without a z is not in a z slab (a slice of fixed
thickness in z, as the 3D view uses).  The shown set is
the intersection of all bounds and of the file choice.  A new layer starts
with `xy_err_nm` at most 25 nm, `logl_rel` at least -2 and `z_nm`
between -500 and 500 nm, plus `sigma_nm` at most 180 nm on a 2D table (in 3D the PSF width changes with z by design), each only if the
table has that column, and any bounds the file carries from
[Remove Localizations](plugin:Analysis/Process/Remove Localizations).  A
layer has one set of bounds, applied to the table it shows: on a layer that
merges blinks a bound on `xy_err_nm` or `photons` is a bound on a blink's combined
precision or summed photons, so the same bound keeps more blinks than it
keeps frames.  The histogram spans the bulk of the
column in 120 bins (of a random sample of two million, for a larger table):
the 1st to 99th percentile, cut to three interquartile ranges beyond the
quartiles, so that the long tail of a precision column -- a few faint spots
with errors of hundreds of nanometres -- does not squeeze the rest into one
bar.  As in SMAP, dragging an edge of the shaded range to the end of
the histogram removes that end of the bound, so the tail is not cut off, and
anywhere else it is the number.  Only the dragged end changes: a bound that
lies beyond the histogram (the 25 nm one, on a sharp table) stays until its
own edge is moved or its number typed.

**Merging blinks.**  The linking follows SMAP's: the localizations are sorted by
frame and x, and each one not yet linked starts a new blink.  From the
blink's running position it looks in the next frame for the first
unlinked localization inside a box of half-width $d$ (*within*, 50 nm)
in both x and y -- the first found, not the nearest -- and moves the running
position halfway towards it.  A blink may skip up to *dark frames* (1)
before it is closed.  With *merge by* *precision* (SMAP's locprec mode) the
box becomes a circle whose radius depends on the localizations: one joins
the blink if its distance $r$ from the running position satisfies

$$r^2 < k^2 \left( \sigma_h^2 + \sigma^2 \right)$$

with $\sigma$ its lateral precision (`xy_err_nm`), $\sigma_h$ the precision
of the running position (which improves as it averages, $\sigma_h^2 \to
(\sigma_h^2 + \sigma^2)/4$ at each link), and $k$ 2.5 -- and $r$ is always
allowed up to *at least* (10 nm) and never beyond *at most* (150 nm).  For
one emitter $r^2 / (\sigma_h^2 + \sigma^2)$ follows a chi-squared
distribution with two degrees of freedom, so a link is kept with
probability $1 - e^{-k^2/2}$: 96% at $k = 2.5$, 99.97% at $k = 4$.  A
localization without a precision may link up to *at most*; a table without
a precision column is linked by the fixed box, with a warning.  Unlike
SMAP, which took the first localization's precision for both terms, the
precision of the candidate and of the running position are used, and the
bounds clamp $r$ itself rather than each $k \sigma$.  Files and channels
are linked separately.  Each blink
then becomes one row: positions weighted by $1/\sigma^2$ (x by `x_err_nm`,
y by `y_err_nm`, z by `z_err_nm` where the table has them), photons and
background summed, precisions combined as $1/\sqrt{\sum_i 1/\sigma_i^2}$,
the error of a summed quantity as $\sqrt{\sum_i \sigma_i^2}$, `logl_rel` the
best, `frame` the first.  `n_in_group`, the number of frames the blink was
on, and `group_id` are written onto both tables.  Merging is done once per
table and kept; switching *merge blinks* off and on costs nothing after the
first time.  It is done when the merged table is first read: for a table of
300,000 localizations or more, in the background (a few seconds for a few
million), the layer drawing nothing until it is ready -- after a file
opens, a table is changed, *merge blinks* is ticked or the merge settings
change.  A plugin run started meanwhile waits for it and starts by itself
when it is done.

**The axes.**  A coordinate is the column divided by that axis's scale,
$u / s$, and the render grid stays square in these units -- so the zoom, the
ROIs and the 3D box work unchanged, and the two scales are what makes a pixel
represent, say, 5 frames across and 20 photons up.  Choosing a column (or
*fit*) sets each scale that is not a position to a round 1, 2 or 5 times a
power of ten, at or below its 1-99 % span divided by 1000 (or by the span of a
position axis shown beside it), and frames the view on that span.  A position
axis keeps a scale of 1, so that a picture of x against z is still a picture
of a place.  On any axis that is not a position the rendering width is set to
zero (plain binning), and put back on the way back to x and y.

## Controls

### overview
The whole field of view, small, with a yellow frame where the main view is;
click in it to move the main view there.  It draws itself when a table is
opened; it can be taken out into a window of its own.

### update
Draws the overview again with the current layers -- after a filter, a LUT or
a layer has changed.

### +
Adds a layer: *localizations* or *image layer*.  The layer strip also has one
button per layer: click to edit it, right-click to show or hide it without
selecting it; bold means drawn, struck through means hidden.

### visible
Whether the selected layer is drawn.  A hidden layer keeps its settings and
is still what a plugin reads when it asks for that layer.

### name
The selected layer's name, shown in its button's tooltip and in the
layer lists of the dialogs that ask for one.

### localizations
A new localization layer on the same table, starting as a copy of the
selected one -- its bounds, files, display and *merge blinks* -- and the tab moves
onto it.

### image layer
Adds an image layer.  When a file open has camera frames kept with it, the
new layer shows their average; otherwise the first image opened.  With
neither -- during a live fit, whose frames come when it ends -- the layer
draws nothing and shows the first image that arrives, the fit's frames or a
file opened with its *source*.

### -
Removes the selected layer.  There is always at least one.

### ⤓
Copies settings from another layer onto the selected one.  Only the display is
ticked at first: the bounds and the files are usually what makes two layers
two.

### filter
The quick buttons (*precision*, *fit quality*, *frame*, *z*, *photons*,
*PSF*, *channel*, *file*) pick the usual columns -- `xy_err_nm`, `logl_rel`,
`frame`, `z_nm`, `photons`, `sigma_nm`, `channel`, `filenumber`, each where
the table has it, and named in the button's tooltip; the drop-down has every numeric column.  A column
with a bound on it is bold and marked with a dot, and the line under the
quick buttons lists every bound of the layer (`xy_err_nm ≤ 25`): click one to
edit it, × to remove it, so a bound on a column not on show is not
forgotten.  The status bar says how much the filters hide.  The count reads
*shown / all*, and how many are in the ROI when one is drawn.  For `filenumber` the histogram is replaced by a
list of the files to tick; right-click a file to remove it from the session.

### min
The lower bound; empty is none.

### max
The upper bound; empty is none.

### clear
Removes the bounds on the column shown.

### all
Ticks every file.

### none
Unticks every file -- the layer then shows nothing until one is ticked.

### image
Shown instead of the filter for an image layer: where the image lies under
the localizations.

### source
What the layer shows.  The list has the camera frames each open file kept
(*name: raw frames* -- the average of every fitted frame, then single frames
spaced over the acquisition, in photons), every image opened, and *open
file...* for another TIFF or PNG.  The kept frames are placed by the camera's
pixel size and ROI, so they lie under the localizations without adjusting
anything.  Saving the localizations keeps the kept frames and every image a
layer shows in the file, with their pixel size and position, and opening it
again offers them here.

### pixel size (nm)
Must be the image's real pixel size in the sample, or it will not line up with
the localizations.

### x0 (nm)
Where the image's first pixel lies in x, in the localizations' coordinates.
Adjust it (and *y0*) to register the image to the localizations.

### y0 (nm)
Where the image's first pixel lies in y.

### frame
For a stack, which frame is shown; the label beside it says which --
*average of 2000 frames*, or *frame 17*, the number the `frame` column uses.
Each layer has its own, so two layers can show the average and one frame.

### display
How the selected layer is drawn.  An image layer uses only the LUT, contrast,
gamma and white background.

### render
*Gaussian (precision)* for the usual picture.  *Gaussian (fixed)* when
every spot should look alike, e.g. to compare layers of different quality;
*histogram* for counting, or at a pixel size well above the precision.
*diffraction-limited* for the widefield picture:
zoomed out it looks like the average of the raw frames, which is a check that
the fit found what was there.  *tracks* for single-particle tracking, once
the tracks are linked.  *Voronoi density* to see structures of very
different densities at once, or to judge densities by eye; the picture of
a layer with millions of localizations takes a while the first time.

### track id
The id column of the tracks, in *tracks* only: `track_id` from Link tracks,
`tid` from a MINFLUX file, or any id column -- the clusters' `cluster_id`
draws each cluster's localizations joined in time order.

### line width
1 pixel is crisp at any zoom; 2 to 3 for a figure at a coarse pixel size, or
to make sparse tracks visible zoomed out.

### colour by
*intensity* colours by brightness through the LUT; *field* colours by the
column chosen beside it.  A visible layer coloured by a column puts a
colour bar in the render window's top right corner: the LUT over the
*colour range*, its ends in the column's unit, and above it the layer and
the column (*layer 2: z_nm*).  Each scale has its own bar, stacked one
under the other; layers with the same column, LUT, inversion and range
share one, labelled with all of them.  A picture saved or copied from the
render window draws the same bars into it (*Export Image*).

### colour range
The values that get the two ends of the LUT.  Narrow it to spread the
colours over the part of the range that matters -- for z, the depth of the
structure rather than of the outliers.

### auto
Sets the colour range to the column's 0.5th to 99.5th percentile.

### LUT
Hot for intensity, turbo for a column.  For several layers, one pure colour
each (red, green, cyan, magenta) adds up legibly.

### invert
The opposite colour at the same brightness: red becomes cyan.  Beside it,
which inversion: *SMAP* as SMAP does it (inverted hot runs through cyan to
white), *grey* so that a layer over its inverse goes grey where they overlap.
For black on white, pick the gray_inverted LUT or *white background* instead.

### contrast
3 is a good start.  Too low, and the structure saturates into white bands;
too high, and all but the brightest spots disappear.  The same number
transfers between datasets, because it is a fraction of pixels, not a
brightness.

### merge blinks
Draws one localization per blink; on by default.  The table is merged the
first time it is needed -- for a large one in the background, the tick
showing the setting and the picture "merging blinks" until the blinks are
ready; after that it is free.  Unticked meanwhile, the layer shows the
localizations at once, and the merge still finishes and is kept.  A plugin
measures what the layer shows -- blinks, when they are merged -- except one
that needs the single localizations (a drift correction, Localization
Precision, anything that changes the table), which says so on its page.

### merge settings...
How blinks are merged, the session's settings, for every layer at once:
*merge by* -- *fixed distance*, a box of half-width *within* (nm), or
*precision*, a radius of *k* times the two localizations' combined
precision, held between *at least* and *at most* (nm) -- and *dark frames*,
how many frames a blink may be dark and still continue.  A fixed box wide
enough to keep the dim localizations of a blink together also merges bright
neighbours that their precision could tell apart; merging by precision
avoids both, at the price of cutting the 4% of links that fall beyond 2.5
sigma (raise *k* to keep more).  OK merges the table again -- once, since
every layer shows the same merged table; a large one in the background, the
layers drawing nothing until then -- and records the new settings in
the file's history, since the merged table cannot tell which ones produced
it.  The drift corrections take these settings too, unless their own
*within* and *dark frames* are set.

### more
The rendering widths, gamma and the white background.

### sigma x (fixed)
The width in *Gaussian (fixed)*, in the units of the horizontal axis; 0 bins.  A
few times the pixel size, or about the typical precision.

### sigma y (fixed)
The vertical width, for axes that are not the same quantity.  Empty follows
*sigma x* where the two axes are the same quantity and bins where they are
not.

### precision factor
0.5 draws each localization at half its precision: sharper than the
statistics strictly allow, which is what makes fine structure readable.  1
is the honest width; much below 0.5 the spots fall apart into single pixels.

### gamma
Below 1 lifts the faint parts, above 1 suppresses them.  1 keeps the
brightness proportional to density.

### white background
The picture on white paper, the hue kept: for figures and print.  It is set
on every layer at once.

### axes
Which column each axis of the picture is.  Set on all layers at once; closed,
it is the ordinary picture.  While a custom pair is shown, a note says what
is across and up, and the scale bars are in the axes' own units.

### x
The horizontal axis, and after *scale* how many units of that column one
render unit is.  *auto* is the table's x position.

### scale
Units of the axis's column per render unit.  For positions leave it at 1;
for photons against frame, *fit* picks scales that make each axis's 1-99 %
fill the picture.

### y
The vertical axis and its scale.  *auto* is y.

### z (3D)
The third axis in the 3D view and its scale.  *auto* is z.

### same scale on both axes
Ticked by itself when the two axes are the same quantity, so that distances
stay true; x against z stretched would misrepresent the structure.

### fit
Scales each axis so its 1-99 % fills the picture, and frames the view on it.
It is done by itself when a column is chosen.

### reset
Back to x against y, the ordinary picture.

## Differences from SMAP

Based on SMAP's layer panel (`gui.GuiChannel`), its renderer
(`renderSMAP`, `drawerSMAP`) and its `VersatileRenderer` plugin
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).

* **The modes are renamed.**  SMAP's *Gauss* (width from the precision) is
  *Gaussian (precision)* here, its *constGauss* is *Gaussian (fixed)*, and
  its *DL* is *diffraction-limited* (stored as `precision`, `gauss` and `dl`
  in a layer's settings), with
  the same widths and fallback.  SMAP's tiff, raw and Other modes and its intensity coding (photons, blinks) are not in the tab.
* **Voronoi density is a mode**, as SMAP's VoronoiRenderer plugin, which
  ran SharpViSu's code on the localizations in view.  Here the cells are of
  everything the filter keeps, so the edge of the view is no edge of the
  cells, and cells smaller than a pixel are counted rather than sampled.
* **Tracks are a mode.**  SMAP had no track renderer: tracks were points
  coloured by their id, or lines in a plugin's own figure.
* **The width from the precision.**  SMAP draws at 0.4 times the precision,
  with a floor of 3 nm or 0.7 pixels and a cap at 400 nm; here the factor is
  0.5, the floor is 0.7 pixels alone, and the cap is ten times the median
  width.  The Gaussian is integrated over each pixel instead of looked up in
  a sampled template.
* **Colour by a column** keeps the hue of a dense pixel, where SMAP adds the
  colours and a dense region of mixed colours bleaches towards white.
* **The filter** shows one column at a time with its histogram, over every
  numeric column.  It opens with a precision bound of 25 nm (SMAP 30),
  `logl_rel` from -2 (SMAP -1 to 0) and a PSF bound of 180 nm (SMAP 175) only
  for 2D data; z within 500 nm is the same.
* **Contrast** is SMAP's quantile contrast with the same meaning; the default
  is 3, where SMAP uses 3.5, which is darker.
* **Merging blinks** (SMAP's grouping) links within 50 nm and one dark
  frame, where SMAP's File tab starts at 35 nm, and its settings sit under
  *merge settings...*, beside *merge blinks*, rather than in the File tab.  The summed photons' error adds in quadrature and each
  coordinate is weighted by its own error; SMAP uses the precision rule and
  the lateral precision for all of them.
* **The versatile renderer** is the renderer itself, set in the *axes*
  section, rather than a plugin that draws into a result window: the plot has
  the layers, filters, ROIs and 3D view of the main picture.  SMAP's line
  overlay (mean or median of y along x) is not there.
* **White background** and the second inversion (*grey*) are new.

## References

* Andronov L, Lutz Y, Vonesch JL, Klaholz BP. SharpViSu: integrated
  analysis and segmentation of super-resolution microscopy data.
  *Bioinformatics* 32, 2239-2241 (2016).
  [doi:10.1093/bioinformatics/btw123](https://doi.org/10.1093/bioinformatics/btw123)
  -- the Voronoi rendering.
* Baddeley D, Cannell MB, Soeller C. Visualization of localization
  microscopy data. *Microsc Microanal* 16, 64 (2010).
  [doi:10.1017/S143192760999122X](https://doi.org/10.1017/S143192760999122X)
  -- the Gaussian rendering with a width from the localization precision.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
