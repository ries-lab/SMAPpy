---
title: 3D view
summary: A box of the data, the slab, shown from any angle, with the Render tab's layers, filters and colours.
widget: smappy.gui.view3d.View3DWindow
covers: [smappy.view3d.Slab.from_region, smappy.view3d.Slab.mask, smappy.view3d.Projection.apply, smappy.view3d.Projection.preset, smappy.view3d.Projection.fit, smappy.view3d.render_layer_3d, smappy.view3d.render_tracks_3d, smappy.view3d.render_image_3d, smappy.view3d.image_plane_points, smappy.view3d.render_volume_3d, smappy.view3d.composite_depth, smappy.view3d.render_3d, smappy.view3d.engine_for, smappy.session.Session.slab_from_roi, smappy.session.Session.selects_slab, smappy.view_path.motion_axis, smappy.view_path.motion_angle, smappy.view_path.motion_path]
---

## What it does

A 3D localization table is a cloud of points in a volume.  The usual image
of it is the view from the top: z is thrown away, or turned into a colour.
Many structures only make sense from the side.  The two rings of a nuclear
pore lie on top of each other when seen from above, and a microtubule seen
from above does not show whether it is hollow.

The 3D view shows a box of the data, the **slab**, from any angle.  You cut
out a small volume around what you want to see, with an
[ROI](plugin:Panels/Glossary) (region of interest: a rectangle, circle, line
or polygon drawn) in the 2D view, and turn it with the mouse.  The picture is drawn the same way as the 2D
image: the layers, filters, look-up tables and colour fields of the
[Render tab](plugin:Panels/Render tab) all apply.  Only the direction you
look from is new.

The slab can also be the [selection](plugin:Panels/Glossary) for plugins --
what a plugin measures: the localizations that pass the layer's filter, inside
the drawn ROI (*plugins use the slab*).  A measurement then sees only what is inside the box, for example
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
slab from above, along its width and along its length.  Opened while a line
ROI is drawn, the window looks at the line from the front -- along it, z
up -- and fits it to the window.  The box's edges are
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

**4. Tracks and images.**  A layer the Render tab draws as *tracks* is
drawn as tracks here too: each track is the line through its localizations
in time order, now running through the volume.  A segment is shown when both
of its ends are inside the slab, so a track is cut where it leaves the box
and starts again where it comes back.  An **image layer** (a wide-field
image under the localizations, say) is a flat picture.  It is laid in the
plane of constant z through the centre of rotation, the point the view turns
about, and turns with the localizations around it.  Moving the centre of
rotation along the line of sight (alt-wheel) carries the image up or down
through the data; it is shown only where it is inside the slab.


```figure Simulated 3D tracks (Brownian particles diffusing at 0.5 µm²/s) over an image layer, from the top and tilted by 70°.  From the top the image lies under the tracks as in the 2D view; tilted, it is a plane through the middle of the volume and the tracks wander above and below it.
from smappy.simulate import simulate_tracks
from smappy.images import ImageData
from smappy.view3d import render_tracks_3d, render_image_3d
tracks = simulate_tracks(use_z=True, field_um=3.0, density_um2=0.4, n_frames=120,
                         z_range_nm=800.0, seed=1)
picture = ImageData(np.add.outer(np.sin(np.arange(30) / 3.0), np.cos(np.arange(30) / 4.0)
                                 ).astype(np.float32) + 2.0, 100.0, 0.0, 0.0)
box = Slab.from_bounds(0, 3000, 0, 3000, -600, 600)
fig.set_size_inches(7.0, 2.8)
for ax, elevation, title in zip(fig.subplots(1, 2), (0.0, 70.0), ("top", "tilted by 70°")):
    p = Projection(elevation=elevation, azimuth=20.0 if elevation else 0.0)
    p.fit(box, 360, 260)
    fov = p.fov(360, 260)
    under, _ = render_image_3d(picture, 0, DisplaySettings(lut="gray", imax=4.5), p, box, fov)
    lines, _ = render_tracks_3d(tracks, np.ones(len(tracks), bool), p, box, fov,
                                RenderSettings(mode="tracks", line_width=1.5),
                                DisplaySettings(contrast=0.5))
    ax.imshow(np.clip(0.5 * under + lines, 0, 1), extent=(fov.x0, fov.x1, fov.y1, fov.y0))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)
```

**5. Z-stacks.**  An image layer that is a
[z-stack](plugin:Panels/Glossary) -- camera images of the same field with
the focus stepped in z, its planes a known step apart (the Render tab's
*planes are*) -- has a height for every plane, and the Render tab's *in 3D*
says how it is drawn.  As a *plane* it is the plane at the height of the
centre of rotation, so moving that along the line of sight steps through
the stack.  As *max (MIP)* each pixel of the picture shows the brightest
value on its line of sight through the stack: the stack's
[MIP](plugin:Panels/Glossary), seen from any angle.  As a *volume* the
values along the line of sight are added up, after the stack's background
is taken off, and the *opacity* lets what is near hide what is behind it.
Either way only what is inside the slab is drawn, and the localizations sit
in the same box: a bead and the localizations fitted from it should be in
the same place.

```figure A simulated z-stack of astigmatic beads (each spot stretched one way below focus and the other way above it), from the top and tilted, as *max (MIP)* and as *volume*.  From the top the MIP is the beads' flat picture; tilted, each bead is a column through the stack, its two stretched ends at the two sides of focus.
from smappy.simulate.camera import bead_stacks
from smappy.images import ImageData
from smappy.view3d import render_volume_3d
stacks, zs = bead_stacks(1, z_range_nm=(-600.0, 600.0), dz_nm=40.0, size_px=64)
beads = ImageData(stacks[0].astype(np.float32), 100.0)
beads.axis, beads.z0_nm, beads.z_step_nm = "z", float(zs[0]), 40.0
box = Slab.from_bounds(0, 6400, 0, 6400, -620, 620)
fig.set_size_inches(7.0, 2.4)
views = (("top, max", 0.0, 0.0, "max"), ("tilted, max", 25.0, 65.0, "max"),
         ("tilted, volume", 25.0, 65.0, "volume"))
for ax, (title, az, el, mode) in zip(fig.subplots(1, 3), views):
    p = Projection(azimuth=az, elevation=el, opacity=0.3)
    p.fit(box, 240, 200)
    fov = p.fov(240, 200)
    rgb, _ = render_volume_3d(beads, DisplaySettings(lut="gray"), p, box, fov, mode)
    ax.imshow(rgb, extent=(fov.x0, fov.x1, fov.y1, fov.y0))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)
```

**6. While the mouse moves.**  A drag draws a quick preview: at half the
resolution, and from a sample of the localizations sized so that the picture
keeps up with the mouse on the machine it runs on.  When the mouse stops,
the full image is drawn from every localization in the slab.  An image that
is saved is always the full one.

**8. Rocking it.**  A still projection hides depth: two structures one
above the other look like one.  Turned a little, the near one moves against
the far one, and the eye reads the difference as depth.  The circling arrow
(*rock*, or Space) turns the view back and forth by a few degrees, on and
on, until it is stopped; stopped, it eases back to the view it started from.
It can also *spin* the view round and round.  Both turn about the data's z
when z stands upright on the screen (the front and side views), so the
horizon stays level, and about the screen's vertical when one looks from
the top, where a turn about z would only turn the picture flat.  Another
axis can be chosen under *turn about*: x, say, to tip a structure seen from
the front towards you and back.

**7. The slab as the selection.**  With *plugins use the slab* ticked, every plugin's selection is cut to the slab, on top of the layer's
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

With *keep horizon level* ticked (the default) a horizontal drag changes the azimuth, a
vertical one the elevation, and the roll stays zero, so the horizon stays
level.  Unticked, the drag turns about the screen's own axes, like a
trackball.  The mouse turns 0.4° per pixel, and the arrow keys 5° per press.
With *turn about screen centre* ticked, a turn is about the point in the
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
range into 32 slices and draws them from the back to the front: with
$I_k$ the image of slice $k$ and $I_{\max}$ the brightness the plain image
is shown at, each slice covers what is behind it by
$c_k = o \cdot \mathrm{min}(I_k / I_{\max}, 1)$,

$$I \leftarrow I\,(1 - c_k) + I_k .$$

At $o = 0$ this is the plain sum.  The same compositing is done for the
colour.

**Tracks.**  The segments are found once over every localization the
layer's filter keeps -- consecutive in time within a track, as in the 2D
view -- and a turn only projects their two ends.  Finding them over the slab
instead would join a track's points on either side of an excursion out of
it.  A segment's weight and colour are the mean of its ends' (its depth's
with *colour by depth*), *dim with depth* weights it by its mean depth, and
the line keeps the layer's width in pixels of the picture.  *Opacity* does
not apply to lines: they are added to the sum.  Tracks are always drawn on
the processor, whichever engine draws the localizations.

**An image layer.**  The screen pixel at $(U, V)$ sees the plane
$z = q_z$ (the pivot's height) at the depth $d$ that solves
$R_{02} U + R_{12} V + R_{22} d = 0$, orthographic; in perspective the view
coordinates are $(U, V)\,t$ with $t = 1 - d/f$, and the plane fixes

$$t = \frac{R_{22} f}{R_{22} f - (R_{02} U + R_{12} V)} .$$

The data point $\mathbf{q} + R^{T} \mathbf{v}$ is then looked up in the
image, nearest pixel, through the image's own placement.  A plane seen edge
on ($R_{22} = 0$), or behind the eye ($t \leq 0.05$), is not drawn.  A
z-stack shows its plane nearest $q_z$, and nothing when $q_z$ is more than
half a step beyond its first or last plane; any other image shows what the
layer shows in 2D (its plane, or its projection).  The brightness is scaled
on the whole plane -- for a z-stack on its MIP, so every plane is on one
scale -- not on what the screen shows of it, so the image does not brighten
as it turns edge on.

**A z-stack as a volume.**  Each screen pixel's line of sight,
$\mathbf{q} + R^{T} \left( U t, V t, d \right)$, is sampled at evenly
spaced depths $d$ between where it meets the stack's box and the slab, about
two samples per voxel -- the smaller of the pixel size and the z step --
and at most 30 million samples for the frame (3 million while dragging).
The stack is interpolated linearly between voxel centres, and is zero
outside.  *max (MIP)* keeps the largest sample.  *volume* goes from the
back to the front over the samples $v_k$ above the stack's median $b$ (its
background, from up to a million voxels), each a fraction $s$ of a voxel
long:

$$I \leftarrow I \left( 1 - c_k \right) + s \left( v_k - b \right), \qquad c_k = 1 - \left( 1 - o \cdot \mathrm{min}\left( \frac{v_k - b}{I_{\max}}, 1 \right) \right)^{s} ,$$

with $o$ the *opacity* and $I_{\max}$ the brightness the MIP is shown at.
At $o = 0$ it is the sum along the line of sight in voxels.  The exponent
$s$ makes the hiding the same per length of the sample however finely it is
sampled.  A volume is drawn on the processor, about 1.3 s for a frame of
700 x 530 pixels and 0.1 s while dragging, whichever engine draws the
localizations.

**The rock and the spin.**  The turn $t$ seconds after the start is
$A \sin(2\pi t / T)$ for a rock, with $A$ *rock by* and $T$ the *period*, so
it slows into each turnaround, and $\omega t$ for a spin, $\omega$ *spin
at*.  It is taken from the clock, not counted per frame, so the speed is the
one set however long a frame takes, and each frame is a preview, as while
dragging.  With *turn about* on *from the view*, the axis is the data's z
when z is within 45 degrees of upright on the screen, else the screen's
vertical; a chosen axis is used as it is.  Either goes through the middle of
the screen (or the slab's centre, with *turn about screen centre* off).  A
rock stopped by its button or Space eases back to its start over half a
second, with a smoothstep; a press in the picture, an arrow key or a view
button stops it where it is and takes the view from there.  *Make a movie of
this...* writes the same motion as a [player](plugin:Panels/Render tab)
view path: a rock as keys at $+A$, $0$, $-A$ and $0$, a quarter period
apart, a spin as a whole turn, both without *ease in and out*, played in a
loop.

**The preview.**  While dragging, the grid is two times coarser and at most
about 13 localizations per pixel are drawn, fewer if the last frames took
longer than 1/27 s.  The sample is evenly spaced through the table, which is
in acquisition order, so it is a shorter acquisition rather than a
rearranged one.

**Which engine draws it.**  The plain sum (*sum*) is drawn on the GPU when
the visible layers hold at least 100 000 localizations around the slab, and
on the CPU below that, or when there is no GPU.  The two give the same image.
Only the speed differs: the GPU pays a fixed few milliseconds a frame to hand
the image back, which a small slab does not earn back.  Measured, a 300 nm
crop turned at 246 frames per second on the CPU and 117 on the GPU, and a
10 µm crop of a 46 million localization file at 3.3 and 34.  *Points* and
*spheres* are drawn only on the GPU; without one they fall back to the sum,
and the line under *look* says why: *wgpu is not installed* (the package
that drives the GPU -- Metal on a Mac, Vulkan or DirectX elsewhere --
installed with SMAPpy since 0.5.8; for an older install, `pip install
wgpu`), or *no GPU here*, with the reason on hover.

**Other axes.**  When the Render tab draws one column against another (its
*axes* section), the slab is a box in those units and is rebuilt over the
whole field when the axes change.  The scale bar is then hidden, and each arm
of the tripod is labelled with its own column and a round length in its
units.

## Controls

The window is the picture and a row of buttons above it: the three views,
pictured as a cube with the face looked at filled, then ⤢ (*fit*) and ⟳
(*rock*), and at the right edge ⚙ (*options*), from which the settings drop
down, and ⋯, which holds saving, the mouse and the keys, and the help.  A
button's name is in its tooltip.

In the image: drag to turn, shift-drag (or the middle button) to pan, the
wheel to zoom.  Alt-wheel or page up / page down moves the eye and the
centre of rotation along the line of sight.  Ctrl-wheel moves the slab along
the line of sight by a tenth of its smallest side per step; shift-wheel makes
it thicker or thinner along the axis nearest the line of sight.  The dots on
the box are its faces: drag one to move that face.  Moving a face, like
typing a range, stops the slab from following the ROI.

The *3D* button on the 2D picture's title bar (or View > 3D view, Ctrl+3)
shows and hides this window.  Hidden, it keeps its turn, zoom and slab for
the next time.

### top
The slab from above.

### front
The slab along its width: its length across the screen, z vertical.

### side
The slab along its length: its width across the screen, z vertical.

### fit
⤢.  Centres the slab and zooms so that it fills the window.

### rock
⟳.  Starts and stops the motion; Space does the same from anywhere in the
window -- unless the Render tab's player was used since the motion last
ran, when Space plays and pauses the player instead.  The small arrow beside it opens the choices below.  Stopped, a rock
eases back to where it began; a spin stays where it got to.  Dragging, an
arrow key or a view button stops it at once.

### rock back and forth
The view turns a little to one side and the other, and back.  The best way
to see depth: what moves against the rest is nearer or farther.

### spin
The view turns round and round, at *spin at*: a structure shown from every
side, as for a talk.

### turn about
*from the view* is right most of the time: the data's z seen from the
front or the side, the screen's vertical from the top.  The data's *x* or
*y* rock the structure about one of its own axes -- x seen from the front
tips the top of the box towards you and away.  *screen vertical* and
*screen horizontal* turn about the screen's own axes, whatever the data's
orientation.  A motion under way starts again about the new axis.

### rock by
How far a rock turns either way.  10 to 20 degrees shows depth without
losing one's bearings; more than about 30 starts to show the structure from
another side rather than in depth.

### period
How long one rock there and back takes.  2 to 4 s is easy to follow; under
1 s, a large slab drawn on a slow machine shows only a few frames per rock.

### spin at
Degrees per second; negative turns the other way.  30 takes 12 s for a whole
turn.

### Make a movie of this...
Opens the Render tab's player with the same motion as its view path -- one
rock or one whole turn, looped, recorded from the 3D view -- so that ▶ plays
it and *Record...* writes it to a file.  It replaces the path that was
there.

### options
⚙.  Drops down the settings below.  A click outside closes it; what was set
stays.

### Save image...
Under ⋯.  Opens [Save image](plugin:File/Export/Image) set to this window:
the projection rendered again at a pixel size of your choice -- over the
canvas, or the slab's box -- in colour or as the summed intensity in
floating point for measuring, with a scale bar and colour bars, and the
projection and the slab written into the file.  It is drawn on the
processor, the summed look of the *auto*, *cpu* and *gpu* engines.

### PNG as displayed...
Under ⋯.  Saves the picture as it is on screen, whichever engine drew it.

### Mouse and keys...
Under ⋯.  Lists what the mouse and the keys do in the picture.

### Help
Under ⋯.  This page; F1 opens it too.

### look
How each localization is drawn.  *sum* is the 2D renderer's blur, added
up, and is right for nearly everything.  *points* draws each one as a small
transparent dot, which reads well for a sparse cloud.  *spheres* draws a
shaded ball, with the balls darkening each other where they crowd
(ambient occlusion), which shows the shape of a dense structure.  Only the
fields the chosen look uses are shown.

### dim with depth
The attenuation length $\lambda$; *off* at 0.  A value around the slab's
thickness along the line of sight makes the front stand out without losing
the back.  For *sum* and *spheres*.

### opacity
For *sum*, and for a z-stack drawn as a *volume*.  0 is a plain sum and fastest.  Values of 0.3 to 0.7 give a sense
of what is in front; 1 hides everything behind a bright slice.

### size
For *points* and *spheres*: the radius.  At 0 (*precision*), the median
[localization precision](plugin:Panels/Glossary) (also called the CRLB) of
what is shown.

### alpha
For *points*.  Many dots pile up on one pixel, so a useful value is small
(the default 0.05).

### occlusion
For *spheres*: how strongly the balls shade their neighbours; 0 is off.  How
far the shading reaches is set by itself: three ball radii, or a twentieth
of the slab's smallest side if that is larger.

### perspective
Tick for a perspective view; the number is the eye's distance in µm.  It
follows the slab (ten times its longest side) until a distance is typed.
Nearer exaggerates depth; much farther is nearly orthographic.

### colour by depth
Colours every layer by its distance from the viewer, over the depth range of
the slab's corners, in place of the layer's own colour field.  The colours
are each layer's colour LUT from the Render tab.  It is z-colouring with z
always pointing out of the screen, so it changes as the view turns.

### show box
Draws the slab's edges and the face dots.  The dots are needed to drag a
face.

### scale bar and axes
The scale bar is a length in the plane of the screen.  The tripod shows the
slab's own axes, projected like the data, so an axis pointing at the viewer
is short.

### plugins use the slab
Off by default.  On, the selection name every plugin reports ends with the
slab's size, so the history says that a box was used.  It holds only while
this window is showing: hiding the window lifts the restriction, and showing
it again restores it.

### slab
Folded.  The slab's own settings, below.

### follow 2D ROI
The slab takes its footprint from the 2D ROI every time the ROI changes, and
the view follows the slab: centred on it, and turned with it when a line ROI
is turned, so what one looks at stays in the middle, seen from the same side.
In the 2D view, `[` and `]` (or `,` and `.`) turn a line ROI about its
middle by 5 degrees, with Shift by 1.

It works the other way too.  Dragging a side of the box edits the ROI: the
line's width (the box's second axis) or its ends (the first), or the
rectangle's sides; dragging the top or the bottom sets the depth, which is
kept as the ROI moves until *from ROI* is pressed.  Over a polygon ROI, which
no box is, a drag unticks it, as editing the ranges always does.

### from ROI
Takes the footprint from the ROI once, and the depth from the z filter, now
-- also forgetting a depth dragged in the box.

### x
The slab's extent along its own first axis, in nm; for a slab turned by
*angle*, that is along the line, not along the data's x.  The *y* and *z*
rows are the other two axes.  Changing one side leaves the other where it is.

### angle
The slab's turn in the plane.  A line ROI sets it to the line's direction.

### view
Folded.  How a drag turns the view, below.

### keep horizon level
Keeps the data's z axis in the vertical plane of the screen, which is the
easy way to keep one's bearings.  Untick for free rotation, like a
trackball.

### turn about screen centre
Off, the slab's centre is always the centre of rotation, which is right as
long as the slab is in view.

### centre on the ROI
While *follow 2D ROI* is on, a line or rectangle moved or turned in the 2D
window is centred in the 3D view and turned with it, so it never slides off
the screen.  Untick it to keep the camera still and let the box travel
through the picture -- what a movie along a line, made with the player,
wants.

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
