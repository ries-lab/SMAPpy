---
title: Microscope manager
summary: Keeps your microscopes -- which devices they have, what each state of a device is called, and what follows from it (the pixel from the 3D lens, the emission from the filter, the NA) -- so that every file is read with the optics it was taken with, without typing them.
widget: smappy.gui.microscope_editor.MicroscopeManager
covers: [smappy.microscope_db.Device.state, smappy.microscope_db.Follows.resolve, smappy.microscope_db.Microscope.resolve, smappy.microscope_db.Microscope.as_of, smappy.microscope_db.save_microscope, smappy.microscope_db.one_line, smappy.io.emu.draft, smappy.io.emu.update, smappy.io.tiff.with_microscope, smappy.calibrate.pupil.OpticsSettings.resolved, smappy.io.calibration.microscope_differences, smappy.gui.microscope_editor.MicroscopeManager, smappy.gui.microscope_view.FoundMicroscope, smappy.gui.microscope_editor.draw_channels, smappy.gui.microscope_editor.draw_pupil, smappy.gui.microscope_editor.unnamed_filters, smappy.microscope_db.Microscope.arrangement, smappy.microscope_db.Microscope.split_layout, smappy.microscope_db.Microscope.pupil_for, smappy.microscope_db.keep_pupil]
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

**The channels on the chip.**  A microscope with a split camera sends each
arm of the split to one side of the chip.  Which side, and which arm the
optics mirror, is a fact about the microscope and its camera, so it is kept
here, and the two-colour fitters and the dual bead calibration cut the chip
as the entry says -- nothing guesses it from the data.  The *channels* tab
draws the cut on the first frames of the file being checked, so a wrong side
or split shows at once: each arm's light should fall inside its own box.

**Versions.**  Saving a changed entry keeps the old one with the day it
applied until, so files taken before a change are read as they were taken.
The *versions* tab lists them.

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

### objective
The objective's *name*, *NA*, *immersion* (oil, water, silicone; any other
typed) and *n_immersion*, the immersion's refractive index -- what *PSF
calibration* and in situ learning read.  No magnification: what a fit needs is the pixel on the
sample, which is the camera's, measured, and differs with the 3D lens on some
microscopes; the nominal magnification says little with relay optics behind
the objective.  An entry from before this table had the NA and the index among
the *parameters*, which are still read when the table leaves them empty.

### devices
One row per device: its name; the Micro-Manager tags it is read from (an
older name after `|`, for files from before a rename); *exact*, *nearest* or
*value* (a continuous position kept as it is); a *tolerance* beyond which a
number is no state at all; the states as `name: reading; ...`; the EMU button
it came from.  When two states share one reading, the first listed wins.

### parameters
One row per value: `n_coverslip`, `coverslip_design_um`,
`wavelength_nm`, `pixel_nm` (`127 116`: x and y), `modality`,
`astigmatism_nm`.  *follows* empty: the value.
`n_coverslip` and `coverslip_design_um` are the objective's: the coverslip's
refractive index it expects and the thickness it is corrected for.  Without
them the coverslip is taken to be the immersion (an oil objective, where
neither matters) and 170 µm thick; a water or silicone objective needs
`n_coverslip` (about 1.52 for glass) for *PSF calibration* to model the glass
plate and to offer learning its tilt.  A microscope with several objectives
lets them follow the device that holds the objective.
`astigmatism_nm` is where *PSF calibration*'s in situ learning starts without a
model: the vertical astigmatism (Noll 6) in nm of wavefront, read from a bead
model of the microscope, per state of the 3D lens (`on: 62; off: 0`).  Its sign
is what decides which way is up, so a camera mounted turned by 90 degrees has
the other.  *follows* a
device: `state: value; ...`, or `from the name` for a filter.


### filters
A filter whose name does not say what it passes: a long pass (*665LP*), an
empty slot (*Empty*) behind the splitting dichroic.  Type the wavelength a PSF
should be drawn at, or the band it lets through (*band from*, *band to*; the
centre is taken).  A filter named as its band, *676/37*, needs no row.  The
grey line under the table names the states of the filter devices that give no
wavelength either way: a PSF calibration in one of them would refuse.

### channels
The arms of a split, the transmitted one first -- it is the one used alone,
when the splitting dichroic is out.  Each names the device that holds its
filter and the side of the chip it lands on: `left`, `right`, `upper`,
`lower`, for this camera: a camera mounted turned puts the arm on another
side, which is why a new camera is a new setup.  Find it from a ratiometric
acquisition of two far-red dyes (a 665LP splitting them): the transmitted side
is the brighter.  *mirror* is the axis the optics flip an arm along against the
other -- `x` for a right-left split with one more mirror in one arm, as on the
lab's microscopes; empty for none.  *region* is the arm's rectangle on the
chip, `x y width height` in pixels (per camera as `iXon 1: 228 0 284 512;
...`), where it lies exactly: the four phase images of a 4Pi, or a two-arm
split that does not meet in the middle of the frame.  It is the one place the
split's position is written: with regions the fitters cut there, without them
the frame as it was acquired is halved -- which is how the lab centres its
camera ROI on the split.  An arm's side then follows from its region, and a
side that contradicts it is refused.

Below the arms, the split drawn on the chip: each arm's box where the fitters
cut it, the main one solid and red, on the brightest of the first 50 frames of
the file being checked.  Arms with regions are drawn without a file.  Move a
region until its box sits on the light.

### main channel
The arm the other is registered onto, and whose PSF is a dual calibration's
main.  Empty: the first arm.

### layout
The split's axis, *right-left* or *up-down*, while the arms have neither sides
nor regions -- a microscope whose optics are known but whose arms have not
been told apart.  The left or upper side is then taken as the main one.  It is
greyed once an arm has a side or a region: the arms say it then.  A mirror is
the arm's (*mirror*), not the layout's.

### PSF
The objective's pupils kept with the entry, one per state of the 3D lens and
colour (green, red, far red: not per filter), each with the camera it was
learnt with, the calibration it came from and its day, and the chosen one
drawn: its phase over the pupil in nm of wavefront, its three largest
Zernike modes and γ, the apodisation (a minimal pupil) or the number of
magnitude modes in the title.  They are not typed: *PSF calibration*'s *Add
to microscope...* keeps one from a pupil model, and M2 and M5 come with one
each, learnt from their beads.

A pupil is read for every file of its camera, whatever its day -- it is the
objective's -- but not for another camera's: a camera mounted turned sees the
astigmatism the other way.  In situ learning starts from the one for its
movie's lens state, and Spline 3D and *Simulate* can fit and draw with it
where there is no bead calibration.

### Remove pupil
Drops the chosen pupil, for a model that turned out wrong.  It is gone once
the entry is saved, for older files too.

### versions
The entry's older setups, newest last: each the day it applied until and
what it said otherwise (the sections and names that differed).  A day can be
corrected in place, when a change of the microscope turns out to have been
made on another day than the one saved: files between the two days are read
with the wrong setup until it is.

### Remove version
Forgets the chosen version: the files of its days are then read as the next
newer setup says.

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
