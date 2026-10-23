---
title: Microscope manager
summary: Keeps your microscopes -- which devices they have, what each state of a device is called, and what follows from it (the pixel from the 3D lens, the emission from the filter, the NA) -- so that every file is read with the optics it was taken with, without typing them.
widget: smappy.gui.microscope_editor.MicroscopeManager
covers: [smappy.microscope_db.Device.state, smappy.microscope_db.Follows.resolve, smappy.microscope_db.Microscope.resolve, smappy.microscope_db.Microscope.as_of, smappy.microscope_db.save_microscope, smappy.microscope_db.one_line, smappy.io.emu.draft, smappy.io.emu.update, smappy.io.tiff.with_microscope, smappy.calibrate.pupil.OpticsSettings.resolved, smappy.io.calibration.microscope_differences, smappy.gui.microscope_editor.MicroscopeManager, smappy.gui.microscope_view.FoundMicroscope]
---

## What it does

A PSF model needs the optics of the microscope that took the beads: the
objective's NA, the immersion oil's refractive index, the emission
wavelength and the pixel on the sample, for each axis.  Micro-Manager records
none of them.  It records every device's state instead -- a servo at 28000, a
slider at *Position 1*, a filter wheel at 3 -- and those states decide the
optics: on the Ries lab's M2 the cylindrical lens of the
[astigmatic](plugin:Panels/Glossary) 3D path stretches the pixel to 127 ×
116 nm, and the filter in the wheel sets the wavelength.

So smappy keeps a **microscope database**, beside the camera database: for
each microscope, the tags that recognise it, the few devices that matter
with their states named, and the values that follow from them.  A file is
then read with its microscope's optics wherever it is opened:

- Wherever a stack is opened -- every fitter, *PSF calibration* -- the same
  control shows it: a section, folded, whose title is one line, *M2 · 3DA on
  · iXon 1 · 127 × 116 nm*, orange when something the function uses is
  missing or no microscope was recognised.  Open, it shows the microscope
  and the camera, recognised or chosen by hand, and only the values the
  function uses, each as it was found, with where it comes from in its
  tooltip.  The values are locked; the lock opens them, a value typed over
  the file's is marked and kept with the run's settings, and ↺ gives it
  back.  When nothing recognises a file, the last microscope used stands
  in, and the section says so.
- A microscope brings its camera: a file whose camera says nothing that
  identifies it is read with the camera the microscope is set up with.  The
  fitters use the microscope's pixel for the state of its 3D lens, over the
  file's own pixel size, which Micro-Manager does not tie to the lens.
- A calibration keeps the state its beads were taken in, and a fit warns when
  its data were taken otherwise: the 3D lens out where the beads had it in,
  another filter, another microscope.

This window is where the entries are made.  Opened from *File > Microscope
manager...*, or from *Microscope manager...* in any *microscope* section.  A value typed where a file is opened (an NA in *PSF
calibration*, a pixel size in the fitter) still wins over the entry, for that
run.

## How it works

**A device per parameter.**  Each parameter follows at most one device: the
pixel follows the 3D lens, the wavelength the filter wheel, and the NA follows
nothing.  A microscope with a lens of two states, a wheel of six and a slider
of four is then entered as twelve states, not 48 combinations.

**Named states.**  A device's states are named (*on*, *off*, *700/100*), and
the parameters are written per name.  A device read as a label (an ELL9's
*Position 1*) matches exactly; one read as a number -- a servo, a stage --
takes the nearest state, so a servo re-trimmed by a few hundred counts still
reads as the state it was sent to.  A filter's wavelength is read from its
name: *676/37* is 676 nm.

**EMU.**  An EMU (ht-SMLM) configuration knows which Micro-Manager property
each button drives and what its states are called.  *Import EMU
configuration...* turns each button into a device under its name, its states
named as on the buttons, and asks which one puts the cylindrical lens in --
EMU has no role for it.  Importing a newer configuration into an entry updates
its devices' properties and values and keeps every name and every value
typed.  A dichroic put in by hand (the splitting 665LP) is in neither EMU nor
the metadata, and is not a device here.

**Versions.**  Saving a changed entry keeps the old one with the day it
applied until, so files taken before a change are read as they were taken.

## Controls

### name
The microscope's name.  An entry of yours with the name of one that comes
with smappy takes its place.

### comment
Free text: where the values came from.

### New
An empty entry, to fill by hand from a file's tags.

### Import EMU configuration...
A new entry from an EMU configuration file (`.uicfg`), or the entry shown
updated from a newer one.

### identify
Tags that must all match.  Take a device that stays with the microscope -- a
laser's or the focus lock's serial number -- not the camera, which moves
between microscopes.

### devices
One row per device: its name; the Micro-Manager tags it is read from (an
older name after `|`, for files from before a rename); *exact*, *nearest* or
*value* (a continuous position kept as it is); a *tolerance* beyond which a
number is no state at all; the states as `name: reading; ...`; the EMU button
it came from.  When two states share one reading, the first listed wins.

### parameters
One row per value: `NA`, `n_immersion`, `wavelength_nm`, `pixel_nm`
(`127 116`: x and y), `modality`, `astigmatism_nm`.  *follows* empty: the value.
`astigmatism_nm` is where *PSF calibration*'s in situ learning starts without a
model: the vertical astigmatism (Noll 6) in nm of wavefront, read from a bead
model of the microscope, per state of the 3D lens (`on: 62; off: 0`).  Its sign
is what decides which way is up, so a camera mounted turned by 90 degrees has
the other.  *follows* a
device: `state: value; ...`, or `from the name` for a filter.


Under the tables, in grey, the objective's pupils kept with the entry, one
per state of the 3D lens, each with the calibration it came from and its
date.  They are not typed: *PSF calibration*'s *Add to microscope...* keeps
one from a pupil model, and an edit of the rest of the entry carries them
through.  In situ learning starts from the one for its movie's lens state.
### channels
The arms of a split, the transmitted one first -- it is the one used alone,
when the splitting dichroic is out.  Each names the device that holds its
filter and the half of the chip it lands on: `left`, `right`, `upper`,
`lower`, for this camera: a camera mounted turned puts the arm on another
side, which is why a new camera is a new setup.  Find it from a dual-colour
acquisition of two far-red dyes: the transmitted half is the brighter.

### camera
The camera the microscope is set up with, from the camera database.  The pixel
per state of the 3D lens and the sides of the split are for this camera.  A
new camera is a new setup: change the camera, the pixel and the sides, save,
and the setup before is kept as a version for older files.  A file taken with
a camera the entry's day does not have is read with the setup that had it,
with a warning.

### Edit camera...
The camera editor on the chosen camera, and on the file being checked with:
the camera's own numbers, its conversion and offset per readout mode.

### Add row
A row in the table shown.

### Remove row
The selected row of the table shown.

### Check with a file...
A stack to check the entry with: below, as the entry is edited, whether it
recognises the file, each device's state and each value, with where it came
from.

### old version until
The day the entry as it was stopped applying: kept for older files when a
changed entry is saved.  After an import, the configuration file's date (a
copied file carries the day it was copied); else today.

### Save
Writes the entry into your own `microscopes.json` in smappy's settings folder,
beside your cameras.  The entries that come with smappy are never written.

## Output

The entry in your `microscopes.json`, which every later file is read with:
the fitters and *PSF calibration* see it at once.
