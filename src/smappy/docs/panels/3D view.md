---
title: 3D view
summary: A box of the data, the slab, shown from any angle, with the Render tab's layers, filters and colours.
widget: smappy.gui.view3d.View3DWindow
covers: [smappy.view3d.Slab.from_region, smappy.view3d.Slab.mask, smappy.view3d.Projection.apply, smappy.view3d.Projection.preset, smappy.view3d.Projection.fit, smappy.view3d.render_layer_3d, smappy.view3d.composite_depth, smappy.view3d.render_3d, smappy.session.Session.slab_from_roi, smappy.session.Session.selects_slab]
---

## What it does

A 3D localization table is a cloud of points in a volume.  The usual image
of it is the view from the top: z is thrown away, or turned into a colour.
Many structures only make sense from the side.  The two rings of a nuclear
pore lie on top of each other when seen from above, and a microtubule seen
from above does not show whether it is hollow.

The 3D view shows a box of the data, the **slab**, from any angle.  You cut
out a small volume around what you want to see, with an ROI in the 2D view,
and turn it with the mouse.  The picture is drawn the same way as the 2D
image: the layers, filters, look-up tables and colour fields of the
[Render tab](plugin:Panels/Render tab) all apply.  Only the direction you
look from is new.

The slab can also be the selection for plugins (*plugins use the slab (while
open)*).  A measurement then sees only what is inside the box, for example
one ring of a pore picked out by its depth.

Use it on tables with a `z_nm` column.  A 2D table can be opened too, but it
is flat: from the side it is a line.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings)
from smappy.view3d import Slab, Projection, render_layer_3d
from smappy.render import RenderSettings, DisplaySettings
from smappy.regions import Region
sim = SimulationSettings(n_frames=6000, seed=3,
                         structure=StructureSettings(preset="npc"),
                         labelling=LabellingSettings(efficiency=0.6))
locs = simulate(sim)
locs = locs[np.asarray(locs["photons"]) > 1000]
x, y, z = (np.asarray(locs[k]) for k in ("x_nm", "y_nm", "z_nm"))
# two neighbouring pores, and a line ROI drawn through both
pores = np.column_stack((locs.metadata["copies"]["x_nm"], locs.metadata["copies"]["y_nm"]))
d = np.hypot(*(pores[:, None] - pores[None]).transpose(2, 0, 1))
d[d < 300] = np.inf
i, j = np.unravel_index(np.argmin(d), d.shape)
a, b = pores[i], pores[j]
u = (b - a) / np.hypot(*(b - a))
roi = Region.line(a - 250 * u, b + 250 * u, 250.0)
slab = Slab.from_region(roi, (-100.0, 100.0))
inside = slab.mask(x, y, z)
EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
         (0, 4), (1, 5), (2, 6), (3, 7)]

def view(ax, projection, title):
    projection.fit(slab, 480, 240)
    projection.zoom *= 0.6                  # closer than the whole box
    fov = projection.fov(480, 240)
    rgb, _ = render_layer_3d(locs, np.ones(len(locs), bool), projection, slab, fov,
                             RenderSettings(), DisplaySettings())
    ax.imshow(rgb, extent=(fov.x0, fov.x1, fov.y1, fov.y0))
    c = slab.corners()
    xv, yv, _ = projection.apply(c[:, 0], c[:, 1], c[:, 2])
    for e in EDGES:
        ax.plot(xv[list(e)], yv[list(e)], color="#ffff00", lw=0.6, alpha=0.7)
    ax.set_xlim(fov.x0, fov.x1); ax.set_ylim(fov.y1, fov.y0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
```

**1. The slab.**  The slab is a box in the data.  Its footprint comes from
the ROI drawn in the 2D view.  A rectangle gives an upright box.  A line ROI
gives a box turned in the plane: its length is the line, its width the line's
width, and its *angle* the line's direction.  Any other ROI gives its bounding
rectangle, and no ROI the whole field.  The depth of the box is the z range
the layer's filter lets through, or all of the data's z if there is no z
filter.  While *follow 2D ROI* is ticked, the box follows the ROI as it is
drawn or dragged.

```figure Simulated nuclear pores (Nup96) seen from the top.  A line ROI drawn through two neighbouring pores becomes a slab turned in the plane (yellow): its long axis is the line, its width the line's width.  The localizations inside it (black) are all the 3D view draws.
w = 1000
cx, cy = slab.center[:2]
near = (np.abs(x - cx) < w) & (np.abs(y - cy) < w)
fig.set_size_inches(4.2, 3.6)
ax = fig.subplots()
ax.scatter(x[near & ~inside], y[near & ~inside], s=0.6, c="0.7", linewidths=0)
ax.scatter(x[inside], y[inside], s=0.6, c="k", linewidths=0)
c = slab.corners()
ax.plot(c[[0, 2, 6, 4, 0], 0], c[[0, 2, 6, 4, 0], 1], color="#c9a400", lw=1.2)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.plot([cx - w + 100, cx - w + 600], [cy - w + 100] * 2, "k", lw=2)
ax.text(cx - w + 350, cy - w + 160, "500 nm", ha="center", va="bottom", fontsize=7)
```

**2. Turning it.**  Only what is inside the slab is drawn.  Each of those
localizations is turned by a rotation $R$ into view coordinates: two across
the screen, and a third, the *depth*, pointing at the viewer.  Dragging with
the mouse changes $R$.  The buttons *top*, *front* and *side* look at the
slab from above, along its width and along its length.  The box's edges are
drawn over the image, and a small tripod in the corner shows where the slab's
own x (red), y (green) and z (blue) point.

**3. Drawing it.**  The turned localizations are rendered with the 2D
renderer, each layer with its own settings, as if the turned slab were a flat
table.  Everything between the front and the back of the box adds up on the
screen.  Optionally the far side is dimmed (*dim with depth*), the front
hides the back (*opacity*), or the colour says how near each localization
is (*colour by depth*).

```figure The slab above, from the top, from the front (along its width, so z is vertical) and tilted by 45°.  From the top each pore is a ring; from the front the two rings of each pore, 50 nm apart in z, are seen edge on.
fig.set_size_inches(7.5, 2.1)
axes = fig.subplots(1, 3)
top = Projection(); top.preset("top", slab.angle)
front = Projection(); front.preset("front", slab.angle)
tilted = Projection(); tilted.preset("front", slab.angle); tilted.elevation = 45.0
for ax, p, title in zip(axes, (top, front, tilted),
                        ("top", "front", "tilted by 45°")):
    view(ax, p, title)
```

**4. While the mouse moves.**  A drag draws a quick preview: at half the
resolution, and from a sample of the localizations sized so that the picture
keeps up with the mouse on the machine it runs on.  When the mouse stops,
the full image is drawn from every localization in the slab.  An image that
is saved is always the full one.

**5. The slab as the selection.**  With *plugins use the slab (while open)*
ticked, every plugin's selection is cut to the slab, on top of the layer's
filter and the 2D ROI.  This holds only while the 3D window is open, so that
nobody measures a box they can no longer see.

## In detail

**What is inside the slab.**  The slab has a centre $\mathbf{c}$, a size
$(L, W, D)$ and an angle $\alpha$ in the plane.  For a localization at
$(x, y, z)$, its position in the slab's own frame is

$$u = \cos\alpha\,(x - c_x) + \sin\alpha\,(y - c_y), \qquad v = -\sin\alpha\,(x - c_x) + \cos\alpha\,(y - c_y), \qquad w = z - c_z ,$$

and it is inside when $|u| \leq L/2$, $|v| \leq W/2$ and $|w| \leq D/2$.  The
edges count as inside.  A table without z is taken to lie in the slab's
centre plane, so only the footprint decides.  The box is tilted only about z:
its top and bottom are always planes of constant z.

**The rotation.**  With $\mathbf{p}$ a data point and $\mathbf{q}$ the centre
of rotation (the pivot), the view coordinates are

$$\mathbf{v} = R\,(\mathbf{p} - \mathbf{q}), \qquad R = R_z(\rho)\, R_x(\epsilon)\, R_z(\phi) ,$$

with $\phi$ the azimuth (a turn about the data's z axis), $\epsilon$ the
elevation (a tilt about the screen's x axis; 0 is the view from the top) and
$\rho$ the roll (a turn about the line of sight).  $v_x$ and $v_y$ are the
screen, $v_z$ the depth, positive towards the viewer.  The presets set
$(\phi, \epsilon, \rho)$ to $(0, 0, 0)$ for *top*, $(0, 90^{\circ}, 0)$ for
*front* and $(90^{\circ}, 90^{\circ}, 0)$ for *side*, with the slab's angle subtracted from
$\phi$ so that they are views of the slab and not of the data's axes.

With *fix roll* ticked (the default) a horizontal drag changes the azimuth, a
vertical one the elevation, and the roll stays zero, so the horizon stays
level.  Unticked, the drag turns about the screen's own axes, like a
trackball.  The mouse turns 0.4° per pixel, and the arrow keys 5° per press.
With *rotate about the screen centre* ticked, a turn is about the point in the
middle of the screen rather than the slab's centre, so that after a pan the
view does not swing away from what is being looked at.

**Perspective.**  Off, the projection is orthographic: parallel edges stay
parallel.  On, with $f$ the eye's distance, the screen coordinates are scaled
by

$$s = \frac{1}{\max(1 - v_z / f,\ 0.05)} ,$$

so that what is nearer is larger.  The distance starts at ten times the
slab's longest side and follows the slab until it is typed in.

**Depth.**  The depth range is that of the slab's eight corners, not of the
localizations, so that colours and dimming do not change with the filter.
*Dim with depth* weights each localization by
$\exp\left(-(d_{\mathrm{front}} - v_z)/\lambda\right)$, $\lambda$ the length
set, so that the front face is at full weight.  *Opacity* $o$ cuts the depth
range into *slices* slabs and draws them from the back to the front: with
$I_k$ the image of slice $k$ and $I_{\max}$ the brightness the plain image
is shown at, each slice covers what is behind it by
$c_k = o \cdot \mathrm{min}(I_k / I_{\max}, 1)$,

$$I \leftarrow I\,(1 - c_k) + I_k .$$

At $o = 0$ this is the plain sum.  The same compositing is done for the
colour.

**The preview.**  While dragging, the grid is two times coarser and at most
about 13 localizations per pixel are drawn, fewer if the last frames took
longer than 1/27 s.  The sample is evenly spaced through the table, which is
in acquisition order, so it is a shorter acquisition rather than a
rearranged one.

**The depth histogram.**  Under the controls is a histogram, in 64 bins, of
the depths of up to 200 000 of the slab's localizations, over all visible
layers.  It shows where along the line of sight the data are, for placing
the slab.

**Other axes.**  When the Render tab draws one column against another (its
*axes* section), the slab is a box in those units and is rebuilt over the
whole field when the axes change.  The scale bar is then hidden, and each arm
of the tripod is labelled with its own column and a round length in its
units.

## Controls

The window has the image and a toolbar; the controls are a window of their
own beside it (*controls*).

In the image: drag to turn, shift-drag (or the middle button) to pan, the
wheel to zoom.  Alt-wheel or page up / page down moves the eye and the
centre of rotation along the line of sight.  Ctrl-wheel moves the slab along
the line of sight by a tenth of its smallest side per step; shift-wheel makes
it thicker or thinner along the axis nearest the line of sight.  The dots on
the box are its faces: drag one to move that face.  Moving a face, like
typing a range, stops the slab from following the ROI.

### Save
*PNG as displayed...* saves the picture as it is on screen.  The two TIFF
entries ask for a pixel size and render the slab again, at full resolution,
to fit the box exactly: in colour, or as the summed intensity in floating
point for measuring.  The TIFF records the projection and the slab.

### top
The slab from above.  Also in the controls window.

### front
The slab along its width: its length across the screen, z vertical.

### side
The slab along its length: its width across the screen, z vertical.

### fit
Centres the slab and zooms so that it fills the window.

### controls
Shows or hides the controls window.  Closing that window unticks it.

### follow 2D ROI
The slab takes its footprint from the 2D ROI every time the ROI changes.
Editing the ranges or dragging a face unticks it.

### from ROI
Takes the footprint from the ROI once, and the depth from the z filter, now.

### x
The slab's extent along its own first axis, in nm; for a slab turned by
*angle*, that is along the line, not along the data's x.  The *y* and *z*
rows are the other two axes.  Changing one side leaves the other where it is.

### angle
The slab's turn in the plane.  A line ROI sets it to the line's direction.

### fix roll: turn about z, tilt z forward / back
Keeps the data's z axis in the vertical plane of the screen, which is the
easy way to keep one's bearings.  Untick for free rotation.

### rotate about the screen centre
Off, the slab's centre is always the centre of rotation, which is right as
long as the slab is in view.

### dim with depth
The attenuation length $\lambda$; 0 is off.  A value around the slab's
thickness along the line of sight makes the front stand out without losing
the back.

### opacity
0 is a plain sum and fastest.  Values of 0.3 to 0.7 give a sense of what is
in front; 1 hides everything behind a bright slice.

### slices
How many depth slices *opacity* composites.  More give a smoother result and
cost proportionally more time.

### perspective
Tick for a perspective view; the number is the eye's distance in µm.  Nearer
exaggerates depth; much farther is nearly orthographic.

### colour by depth
Colours every layer by the depth in the current view, over the depth range
of the slab's corners, in place of the layer's own colour field.

### engine
*CPU* and *GPU* draw the same image; *GPU* is faster on large tables.
*GPU points* draws each localization as a small round sprite with
transparency, and *GPU spheres* as a shaded sphere with ambient occlusion.
Without a usable GPU the CPU is used, and the label under the menu says so.
The controls an engine does not use are greyed out.

### point size
For *GPU points* and *GPU spheres*: the radius in nm.  0 uses the median
localization precision of what is shown.

### point alpha
For *GPU points*: how opaque one sprite is.  Many sprites pile up on one
pixel, so a useful value is small (the default 0.05).

### occlusion
For *GPU spheres*: how strongly spheres shade their neighbours; 0 is off.

### occlusion radius
For *GPU spheres*: how far the shading reaches.  0 takes three sphere radii,
or a twentieth of the slab's smallest side if that is larger.

### show box
Draws the slab's edges and the face handles.  The handles are needed to drag
a face.

### show scale bar and axes
The scale bar is a length in the plane of the screen.  The tripod shows the
slab's own axes, projected like the data, so an axis pointing at the viewer
is short.

### plugins use the slab (while open)
Off by default.  On, the selection name every plugin reports ends with the
slab's size, so the history says that a box was used.  Closing the window
lifts the restriction; opening it again restores it.

## Differences from SMAP

Based on SMAP's `sr3D/Viewer3DV01`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).

* **Any rotation.**  SMAP's viewer needs a line ROI and turns the view only
  about the line (its *polar angle*); turning in the plane means moving the
  ROI.  Here the slab can come from a line, a rectangle or the whole field,
  and the view turns freely about all three axes.
* **The box is its own object.**  In SMAP the volume is the line ROI and a z
  range typed into the plugin.  Here the slab is kept by the session, has
  handles and ranges of its own, and can be the plugins' selection.
* **No stereo, no movies.**  SMAP's anaglyph, side-by-side and goggles
  modes, and its rotation and translation movies, are not ported.
  Perspective is.
* **Depth cues.**  SMAP offers a maximum-intensity projection, a
  transparency and balls.  Here the plain sum is the default, and dimming,
  front-to-back opacity in slices, colour by depth, and GPU points and
  spheres are the options.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
* Thevathasan JV, Kahnwald M, Cieśliński K, et al. Nuclear pores as
  versatile reference standards for quantitative superresolution microscopy.
  *Nat Methods* 16, 1045 (2019).
  [doi:10.1038/s41592-019-0574-9](https://doi.org/10.1038/s41592-019-0574-9)
  -- the Nup96 geometry of the simulated pores.
