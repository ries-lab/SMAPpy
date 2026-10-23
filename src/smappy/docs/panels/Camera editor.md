---
title: Camera editor
summary: Adds the camera of the file being fitted to your camera database -- or gives a known camera a readout mode it lacked -- so that the next file it takes is converted to photons without typing anything.
widget: smappy.gui.camera_editor.CameraEditor
covers: [smappy.camera_db.draft, smappy.camera_db.read_map, smappy.camera_db.with_mode, smappy.camera_db.problems, smappy.camera_db.save_camera, smappy.camera_db.CameraDatabase.joined_with, smappy.camera_db.Camera.resolve, smappy.plugins.fit._FitPlugin._notice, smappy.gui.camera_editor.CameraEditor]
---

## What it does

A fit needs three numbers about the camera before it can turn a movie into
photons and nanometres: the [conversion](plugin:Panels/Glossary) (also
called sensitivity or gain: photoelectrons per count), the
[offset](plugin:Panels/Glossary) (also called baseline or bias: the count of
a pixel that saw no light) and the *pixel size* in the sample.  A count is
the camera's raw pixel value, in ADU (analog-to-digital units).  Micro-Manager records the pixel size if it was calibrated, and never
the other two.  They depend on the camera's readout mode -- the readout rate,
the pre-amp gain, the amplifier -- and that the file *does* record.  So
smappy keeps a **camera database**: for each camera, the tag that recognises
it, and for each readout mode, its conversion and offset.  A file from a
camera in the database is fitted with the right numbers without anyone
typing them.

A file from a camera that is *not* in the database has to have them typed
into the fitter, every time.  The fitters say so as soon as the file is
chosen, in an orange line under their settings, with a button that opens this
window: *Add camera...* for a camera the database does not know, *Add readout
mode...* for a known camera recorded in a mode it has no numbers for.  A
camera the database does not know, whose numbers are not typed in either, is
asked about again at Run, and *Open camera editor* there opens this window
instead of fitting.  It
also opens from *Edit camera...* in the
[Microscope manager](plugin:Panels/Microscope manager), on the camera a
microscope is set up with.

The window starts from **the file the fitter is set to**: its tags already
say which device took it, its serial number, which tags carry the
[EM gain](plugin:Panels/Glossary) (electron multiplication: how much an EMCCD
amplifies the signal before reading it out), the exposure and the
[camera ROI](plugin:Panels/Glossary) (also called the readout region: the
part of the chip that was read out), and which ones set the readout mode.  All of
that is filled in.  What is left is the conversion and the offset of this
file's readout mode, from the camera's data sheet or a calibration (the mean
of a dark movie is the offset).  Save, and every later file from that camera
in that mode is recognised.  A camera with several readout modes is given
them one file at a time: open a file taken in the next mode, and the editor
adds that mode to the camera and keeps the others.

The entries are saved in **your own** `microscopes.json`, in smappy's
settings folder (the window shows where), never in the list that comes with
smappy, which an update replaces.  The file holds your lab's cameras and its
microscopes; a `cameras.json` saved there by an older smappy is still read.  The database and the fitter's own camera fields
are the only two places a camera's numbers come from: smappy reads neither
camera YAML presets nor SMAP's `settings/*_cameras.mat`.  The cameras of the
Ries lab's SMAP file are in the list that comes with smappy.

## How it works

**1. A draft from the file.**  The device is the file's `Core-Camera`, and
its tags (`Evolve512-ReadoutRate`, ...) are written with the device name
replaced by `{prefix}`, so that a renamed device needs only the *device*
field changed.  The camera is recognised by its serial number where the
device reports one, else by its camera ID, model name or description.  The
EM gain is read from `MultiplierGain` on a camera that reports it there
(Photometrics), and from `Gain` otherwise (Andor).  Every tag the file has
among the usual readout-mode tags (`ReadoutMode`, `ReadoutRate`,
`Pre-Amp-Gain`, `Port`, `Output_Amplifier`, ...) is ticked under *readout
mode*, and the conversion and offset are set to *per readout mode*.  A
calibrated `PixelSizeUm` becomes the fixed pixel size.

**2. Editing.**  Each parameter comes from one of four places: *fixed* (the
same in every file), *from a tag* (read from the file, as a number, a list of
numbers, yes or no, compared with a value, or *per value*: a table from what
the tag reads to the value, with a default), *per readout mode* (a number
for this file's mode), or *not set*.  The readout mode is what the ticked
tags read in this file.

**3. Checking.**  Below, the window shows what the entry, as it stands, makes
of this file: every parameter, its value, and where it came from.  It is the
table of values a fitter's *microscope* section draws on, so the entry can be checked
against the file before it is saved.  Under the table, in orange, is
whatever is still wrong: a parameter the fit needs that is still unknown, a
recognising tag this file does not match, a file another camera already
claims.

**4. Saving.**  The camera goes into your `microscopes.json`, replacing one
of the same name there.  The database is then read again, and every open fitter
re-reads its file: its line disappears, its *auto* fields show the new
numbers, and the camera is offered under *camera* in each fitter's
*microscope* section.

## In detail

**Which mode is replaced.**  Saving puts this file's mode -- the ticked tags,
with the values this file has for them -- into the camera.  It replaces the
mode that already recognised this file (correcting its numbers), or one
recognised by exactly the same tags and values; otherwise it is added.  The
camera's other modes are kept untouched, because each was measured on its
own.  A mode with no ticked tag is never added: it would recognise nothing.

**A shipped camera.**  Editing a camera that came with smappy -- to give it a
mode -- saves a copy, under the same name, into your file.  When the database
is read, a camera of yours replaces the shipped one of the same name rather
than sitting beside it, so the copy is what is used from then on.  The window
says so before saving.

**Order.**  When two cameras would recognise the same file, the first in the
database wins, and yours come before the shipped ones.

## Controls

### start from
*a new camera, from this file* drafts an entry from the file's tags.  A
camera's name edits that camera instead: this is how a known camera gets a
mode, or a wrong number is corrected.  When the editor is opened for a file
whose camera was recognised, that camera is chosen already.

### name
What the camera is called under *camera* in the fitters' *microscope*
section and in the microscope manager.  An entry is saved
under its name, so saving under a new name makes a second camera.

### device
The Micro-Manager device name its tags start with.  If the device is renamed
in Micro-Manager, this is the one field to change.

### recognised by
The tag that names this one camera, and what it must read: *is* for the whole
value, *contains* for a part of it.  A serial number is best: two cameras of
the same model have the same model name.  Leave it empty for a camera that
should only ever be chosen by hand.

### readout mode
Tick the tags that set the conversion and the offset: on an EMCCD typically
the readout rate, the pre-amp gain and the amplifier.  Too few and two modes
with different numbers look the same; too many (the exposure, say) and every
change of a setting that does not matter needs its own mode.  The filter
above the list narrows it.

### parameters
Where each value comes from.  The conversion and the offset are usually *per
readout mode*; the pixel size *fixed* (two numbers, x and y, if the pixels
are not square); the EM gain and its switch, the exposure, the frame time and
the camera ROI *from a tag*.  A parameter the entry does not set is read from
the file the way any file is.

### conversion (e-/ADU)
Photoelectrons per count, for this file's readout mode.  From the camera's
data sheet, or measured from the mean and variance of a flat illumination.

### offset (ADU)
The count of a pixel that saw no light: the mean of a dark movie in this
readout mode.

### pixel size (µm)
The size of a camera pixel in the sample: the physical pixel divided by the
total magnification.  Where it depends on something in the light path rather
than on the camera, read it *from a tag*, *per value*: the tag of the device
that switches it, and what each of its states means, typed as
`In: 0.127 0.116; Out: 0.127`.  On the Ries lab's M2 the cylindrical lens of
the astigmatic 3D path stretches the image in one direction: 127 x 116 nm
with the lens in, 127 nm square without.  The value beside the table is the
default, for a file that does not record the tag (one taken before the
device was in Micro-Manager) or reads a state the table does not list.

### read noise (e-)
The camera's read noise, rms per pixel, in electrons.  The fit adds its
square to the Poisson noise of every pixel, which also keeps a pixel with
almost no signal from dominating the fit.  A modern sCMOS reads 0.7 to
1.4 e-, a CCD several; *not set* is 1.  With EM gain on it is not used: the
gain makes it a fraction of an electron.

### frame time (ms)
The time from one frame to the next -- the
[frame time](plugin:Panels/Glossary): the exposure plus the time the camera
needs between frames.  Everything that speaks of seconds reads it (*Time
Course*, *Blinking statistics*, the on-time of *Localization Statistics*).  A camera
that records its own cycle time (Micro-Manager's `ActualInterval-ms`, which
a draft takes *from a tag*) needs nothing here; *fixed* suits a camera that
always runs at one rate.  *Not set*, a file's frame time is the time-lapse
interval it records, if it is longer than the exposure, and else the
exposure.

### this file, read with this entry
What the entry, as it stands, makes of the file: value and source of every
parameter, red where a value the fit needs is missing.

### Save
Writes the camera into your `microscopes.json` and reads the database again.

### Close
Closes the window without saving what was changed since the last *Save*.

## Output

A camera in `microscopes.json`, in smappy's settings folder.  It is plain
JSON, one entry per camera in its `cameras` list, so it can be copied to another computer or handed to
a colleague.  An entry that every smappy installation should have can be
sent to the developers for the shipped list.
