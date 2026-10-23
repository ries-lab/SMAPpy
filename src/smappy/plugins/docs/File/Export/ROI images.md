---
version: "1"
covers: [smappy.plugins.roi_images.roi_layers, smappy.plugins.roi_images.render_roi, smappy.plugins.roi_images.roi_picture, smappy.plugins.roi_images.write_pdf, smappy.plugins.roi_images.write_tiff, smappy.plugins.roi_images.to_picture]
---

## What it does

Draws every ROI of the [ROI manager](plugin:Panels/ROI manager) as a picture,
all at the same size and pixel, and writes them out, as *format* says: as a
**PDF**, a grid of pictures on each page, or as a **TIFF stack**, one ROI
per plane.  Left empty, the file goes beside the localization file, named
after it (`cells.h5` gives `cells_rois.pdf`); the field shows that name in
grey.

Use it for a gallery of the sites a segmentation found, to look through
before trusting what was measured in them; for a figure of many structures
side by side; or to step through the ROIs in Fiji.  Over each picture it
can draw the ROI's outline and its named lines, and above it the ROI's
number and values measured in it -- the number of localizations, a radius, a
count of corners -- so a site that measured oddly can be looked at.

It needs ROIs: draw them in the ROI manager, or find them with
[Segment regions](plugin:Analysis/Regions/Segment regions).

## How it works

**The pictures.**  Each picture is a square of *size* around the ROI's
centre, rendered by the same renderer as the view, at a *pixel* of the same
size for every ROI.  It shows the visible localization
[layers](plugin:Panels/Glossary) as they show the data: their filters,
merged blinks or not, their colours.  An ROI belongs to one file, and its
picture shows that file's localizations only.

**Turned to a line.**  With *turn to* set to a line's name, each picture is
turned so that the ROI's line of that name points to the right.  Draw a
line along each structure in the ROI manager (a *direction*, for example),
and the pictures show the structures all the same way round.  An ROI
without that line is drawn as it lies.

```figure-setup
import tempfile, yaml
from pathlib import Path
from smappy.layer_state import LayerSnapshot
from smappy.render import RenderSettings
from smappy.simulate import simulate
from smappy.simulate.settings import SimulationSettings, StructureSettings
from smappy.plugins.roi_images import render_roi
spec = {"name": "arrow", "elements": [{"points": [[-50, 0, 0], [50, 0, 0], [50, 40, 0]]}],
        "copies": {"n": 6, "placement": "grid", "rotation": "random",
                   "field": [0, 0, 3000, 2000]}}
structure = Path(tempfile.mkdtemp()) / "arrow.yaml"
structure.write_text(yaml.safe_dump(spec))
locs = simulate(SimulationSettings(n_frames=3000, seed=3,
                                   structure=StructureSettings(file=str(structure))))
layer = (LayerSnapshot(name="arrows", locs=locs, mask=np.ones(len(locs), bool),
                       settings=RenderSettings()),)
copies = locs.metadata["copies"]
```

```figure Six simulated arrows (three labels each) at random turns.  Top: the ROI pictures as the arrows lie.  Bottom: turned so that each ROI's line, drawn along its arrow, points right -- the arrows now look alike.
fig.set_size_inches(6, 2.3)
axes = fig.subplots(2, 6)
for k, (x, y, gamma) in enumerate(zip(copies["x_nm"], copies["y_nm"], copies["gamma_deg"])):
    for row, angle in enumerate((0.0, np.radians(gamma))):
        ax = axes[row, k]
        ax.imshow(render_roi(layer, (x, y), 250.0, 4.0, angle))
        ax.set_axis_off()
fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, top=1, bottom=0)
```

**On top.**  The ROI's outline in yellow, as the ROI manager draws it, and
its named lines, each in a colour of its own.  Above each picture its title:
*ROI* and its number, then the values ticked under *title*, from the
evaluation chosen under *results from*.  An ROI whose results no longer
describe it (it was moved since) gets its number alone.

**The PDF.**  The pictures fill a grid of *columns* by *rows* on each page,
in the ROI manager's order (as *Sort ROIs* left it).  The outlines and
lines are drawn as lines over the pictures, so they stay sharp at any zoom;
each picture has a scale bar, and its length is written at the foot of the
page.

**The TIFF.**  One plane per ROI, in colour, with the outline, lines and
scale bar drawn into the pixels.  It is written as ImageJ writes a stack:
Fiji reads the pixel size, and shows each plane's title (the ROI's number
and values) above it.

## In detail

**Brightness.**  Each picture is coloured as the layer colours the view:
with the layer's *contrast*, which scales each picture by its own brightest
pixels.  Two pictures' brightness therefore do not compare; set the layer
to an absolute scale for that.

**Turning** turns the localizations about the ROI's centre before they are
rendered, rather than the finished image, so no pixel is interpolated.  An
image layer (a camera frame, a widefield image) cannot be turned that way
and is left out of turned pictures.

**Speed.**  For each ROI only the localizations that can reach its square
are rendered, found through the layer's spatial index: a thousand ROIs of a
large table take a few seconds, not a thousand renders of the table.

**Axes.**  The pictures are of the positions.  With the Render tab's axes
set to other columns (photons against frame, say) it refuses rather than
cutting squares out of a picture that is not of space.

## Parameters

### size_nm
On *auto*, the width of the ROI manager's ROI image -- three times the ROI
size unless set there.  Large enough to show the ROI's surroundings, small
enough that the structure fills the picture.

### pixelsize_nm
The rendering pixel.  Around a third of the localization precision shows
single localizations as blobs; smaller makes larger files and no more
detail.

### turn_to
Lists the line names of the ROI manager.

### path
A file set keeps its name and takes the format's extension.

### title_columns
Lists the columns of the chosen results.  None are ticked to begin with:
the titles are the ROI numbers alone, and no results are read.

### columns
With *rows*, how many pictures share a page: the size of each picture on
paper follows from the page.

## Output

The file, and in the log where it went.  A PDF of $\lceil n / (\mathrm{columns} \times \mathrm{rows}) \rceil$
pages for $n$ ROIs; or a TIFF stack of $n$ planes of
$\mathrm{size} / \mathrm{pixel}$ pixels square.  The form says beforehand how
many pictures, how large and on how many pages.
