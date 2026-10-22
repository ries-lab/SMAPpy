# The microscope database

A design note, 2026-10, not yet built, for discussion.  uiPSF needs the optics of the
microscope that took a file -- the NA, the immersion index, the emission wavelength, the
pixel per axis -- and today they are typed into *PSF calibration* every time
(`calibrate.pupil.OpticsSettings`).  Most of them are fixed per microscope; the rest follow
a few devices the acquisition records: the pixel follows the cylindrical lens, the
wavelength the emission filter.  This note proposes a microscope database beside the camera
database, filled by hand the way a camera is, with an EMU (ht-SMLM) configuration as an
optional start.  `docs/uipsf/microscopes.md` is what the lab's files record;
`docs/uipsf/plan.md` §7.5 and `docs/uipsf/roadmap.md` item 2 are where it was planned.

## What the files and the EMU configurations say

Read in October 2026 from the M2 and M5 bead stacks and movies in `smappy test` and on
`/Volumes/vbc/STRUCTBIO/RIES/2026`, with the configurations `M2-config.uicfg` and
`M5_config.uicfg` (EMU's export of the ht-SMLM plugin; `M2_config_old.uicfg` is an older
one, before the Mojo FPGA became MicroFPGA).

* **Micro-Manager records every device's state, and nothing about what it means.**
  `Servos-Position1 = 28000`, `Thorlabs Filter Wheel-State = 3`, `Thorlabs ELL9-Label =
  Position 1`.  Which of them is the cylindrical lens, and what sits in position 3, is
  nowhere in the file.
* **EMU's configuration says which device is which, and names the states.**  Three blocks
  per plugin configuration:
  * `parameters` -- names and roles: `Controls - Two-state device 2 name = 3DA`,
    `Filters - Filter names = 440/40,525/50,600/60,700/100,676/37,None` (one per state, in
    order), `2C - Slider 1 names`, `Laser 3 - Name = 638`, and roles EMU itself uses,
    `Acquisitions - BFP lens = Two-state device 1`.
  * `properties` -- each control's Micro-Manager property and the value of each of its
    states: `Two-state device 2 = Servos-Position1`, `- On value = 28000`, `- Off value =
    4000`; `Filter wheel position = Servos-Position3`, `… state 3 = 32000`.
  * `settings` -- which panels exist (a second filter wheel, the 2C tab).

  So a state is decoded by a chain: the property's value in the file → the control's
  state with that value → that state's name in the parameters.  M2's decoded:

  | acquisition | 3DA (`Servos-Position1`) | filter (`Servos-Position3`) | 2C reflected (ELL9) |
  |---|---|---|---|
  | 260105 beads and 3D movie | 28000, in | 32000, 700/100 | Position 3, 600/60 |
  | 260821 beads and 3D movie | 28000, in | 57000, None | Position 1, 676/37 |
  | 260829 beads | 28000, in | 32000, 700/100 | Position 0, Empty |
  | 260829 TIRF movies (`TIRF_1`, `_p05`, `sweep`, …) | **4000, out** | 32000, 700/100 | Position 0, Empty |
  | 260917 beads and 3D movie | 28000, in | 23500, 600/60 | not recorded |

  The 260829 movie of the in situ benchmark was taken without the lens, its beads with it
  (`docs/uipsf/architecture.md`, "In situ on M2") -- the failure a microscope database
  would have caught when the calibration was applied.  On M5 the lens, "3D", is a linear
  stage (`Thorlabs ELL20-Position (um)`, in at 0, out at 50000; the 260826 beads read 0),
  and there are two filter wheels with a list of names each (the 260826 beads: wheel 1 at
  state 3, 685/70, as the file's comment says).
* **Not everything decodes.**  On M2 the 10000 of `Servos-Position0` is both of two states
  of the 2C transmitted slider (685/70 and empty) in the configuration itself; M5's TIRF stage reads 18151 between its on
  (19839) and off (21094), because it is an angle, not a switch.
* **Devices are renamed, and their positions re-trimmed.**  The old M2 configuration names
  `Mojo-Servos-Position1` with 3DA on at 37000; the current, `Servos-Position1` at 28000.
  The ELL9 sliders are `Thorlabs ELL9` and `Thorlabs ELL9-1` in files up to September and
  `ELL9 - Reflected Slider` and `ELL9 BFP` in the current configuration; their serial
  numbers (`…-ID`) are in every file.
* **No configuration says what a name means physically.**  That "3DA" makes the pixel
  116 × 127 nm, that "700/100" is 700 nm, the NA, the immersion -- these are typed once,
  whatever the import.

## The decision: a microscope database beside the camera database

**It does not replace the camera database; it sits above it.**  What belongs to the camera
-- the conversion and the offset per readout mode, the read noise -- moves with the camera:
M2's iXon (8172) came from M1 in July 2026 and kept them.  What belongs to the microscope --
the objective, the filters, the lens, the magnification -- stays when the camera changes.
The pixel on the sample is the camera's pixel over the microscope's magnification, so it
belongs to the pair: the microscope entry lists the cameras it has had and carries the
pixel per camera; which camera took a file, the file says.

**One mechanism, the camera database's.**  `camera_db` already has everything needed: a
value that is fixed, read from a tag, or looked up per tag value (`map`); a `Source` with
every value; a shipped file joined with the user's by name; an editor that starts from a
file.  The microscope database reuses `Parameter`, `Source`, `Resolution` and the joining,
and adds the one thing the camera lacks: **devices**.

**A parameter depends on one named device, not on the whole state.**  The camera
database's `State` is a combination of tags that must all match, which is right for a
readout mode (the conversion depends on gain, rate and port together) and wrong for a
microscope: with a lens of two states, a filter wheel of six and a slider of four, every
combination would be an entry.  Instead each parameter names the device it follows, and
gives a value per state of that device: the pixel per state of the lens, the wavelength
per state of the filter wheel.  A parameter that follows nothing is fixed.  This is what
`camera_db.read_map`'s docstring already argued for the M2 lens.

## The entry

```json
{"name": "M2",
 "identify": {"MicroFPGA-Hub-MicroFPGA ID": "Au"},
 "devices": {
   "3D lens":        {"tag": "Servos-Position1", "read": "nearest",
                      "states": {"in": 28000, "out": 4000},
                      "emu": "Two-state device 2"},
   "filter":         {"tag": "Servos-Position3", "read": "nearest", "tolerance": 3000,
                      "states": {"440/40": 5000, "525/50": 12000, "600/60": 23500,
                                 "700/100": 32000, "676/37": 40000, "None": 57000},
                      "emu": "Filter wheel"},
   "2C transmitted": {"tag": "Servos-Position0", "read": "nearest",
                      "states": {"676/37": 20000, "700/100": 32000, "685/70": 10000},
                      "emu": "Slider 1"},
   "2C reflected":   {"tag": ["ELL9 - Reflected Slider-Label", "Thorlabs ELL9-Label"],
                      "states": {"600/60": "Position 3", "Empty": "Position 0",
                                 "525/50": "Position 2", "676/37": "Position 1"},
                      "emu": "Slider 2"}},
 "parameters": {
   "NA": 1.43,
   "n_immersion": 1.516,
   "wavelength_nm": {"device": "filter", "values": "from the name"},
   "modality": {"device": "3D lens", "values": {"in": "astigmatic", "out": "2D"}}},
 "channels": {"transmitted": {"filter": "2C transmitted", "side": "right"},
              "reflected":   {"filter": "2C reflected",   "side": "left"}},
 "cameras": {
   "Evolve 1": {"pixel_nm": {"device": "3D lens",
                             "values": {"in": [116, 127], "out": [127, 127]}}},
   "iXon 1":   {"pixel_nm": {"device": "3D lens",
                             "values": {"in": [127, 116], "out": [127, 127]}}}},
 "versions": [{"until": "2025-12-31", "devices": {"3D lens": {"tag": "Servos-Position1",
                                                             "read": "nearest",
                                                             "states": {"in": 37000, "out": 4000}}}}]}
```

* **`identify`**: tags that must all match, chosen from devices that stay with the
  microscope -- the FPGA's ID, a laser's or a stage's serial number -- not the camera,
  which moves.  As in the camera database, a microscope that cannot be recognised is
  chosen by hand, never guessed.
* **`devices`**: a name, the tag it is read from, and its states by name.  How a value is
  read: `exact` (a label: `Position 1`) or `nearest` (a servo or a stage).  *Nearest*
  splits between neighbouring states at the midpoint -- M2's lens is *in* above 16 000 and
  *out* below -- because a re-trimmed servo or a stage stopped a few µm short is still in
  the state it was sent to.  A third reading, `value`, keeps a continuous device's
  position as it is -- M5's TIRF angle -- recorded with the file and followed by no
  parameter until one needs it.  An optional `tolerance` rejects a value too far from every
  state, for a wheel where a value between two filters means it was set by hand or the
  configuration is wrong; without one every value decodes.  `tag` may be a list, tried in
  order: a device renamed in Micro-Manager keeps its entry.  States are keyed by **name**,
  values beside them, so a re-trimmed servo changes a number and not every parameter that
  follows the device.  A configuration that gives one value to two states (M2's 2C
  transmitted slider: 685/70 and empty both at 10000) is imported with the first and a
  warning.
* **`parameters`**: each one fixed, read from a tag, or `{"device": …, "values": {state:
  value}}` -- **one device per parameter**.  A parameter whose device is in a state the
  table does not have is missing, with a source that says which state, as a readout mode
  the camera database does not know is.  `"from the name"` reads a wavelength out of a
  filter's name (`676/37` → 676, `665LP` → nothing, to be typed): the names are already
  the data.
* **`channels`**, for a microscope with a split: each channel names the filter device it
  follows and the side of the chip it lands on (`left`, `right`, `top`, `bottom`), so a
  dual-channel file gets a wavelength per half -- *transmitted, right*.  The pairing is
  fixed per microscope.  **Whether the split is in is not recorded**: the splitting
  dichroic is put in by hand.  So single or dual channel is the choice the calibration
  and the fit already make (*dual* in *PSF calibration*, the dual-channel fitters), and
  the microscope answers it: dual, a wavelength per side; single, the *transmitted*
  channel's over the whole chip, the dichroic being out.  A microscope without a split
  has no `channels` and takes the `wavelength_nm` parameter.  Telling the two apart from
  the images (two copies of each spot, side by side) is possible and left out.
* **`cameras`**: the cameras this microscope has had, by their name in the camera
  database, each with the parameters that depend on the pair -- the pixel and its
  orientation.  They override the microscope's.  Which one applies is read from the file:
  the camera database recognises the camera by its serial number, so no dates are typed.
  A camera's name is ours to choose and is not a microscope's: *iXon 1*, *iXon 2*,
  *Evolve 1*.  The shipped entries named by microscope (*M1 Andor*, the iXon 8172 now on
  M2) are renamed so, the old name kept as an alias that saved settings and chains still
  find.
* **`versions`**: what the entry said before it was last changed, each with the date it
  applied until.  Saving a changed entry -- an EMU configuration imported again after a
  servo was re-trimmed, a device renamed, a filter swapped -- does not overwrite it: the
  entry as it was moves into `versions` with an `until` date, and only what differs is
  kept.  The date is the new EMU configuration file's modification date when the import
  reads one cleanly -- closer to when the microscope changed than the day smappy is told
  -- and otherwise the day it is saved; either is editable.  Midpoint splitting absorbs a small re-trim without any of this; a version is for
  the change it does not absorb.  (The example's version is the old M2 configuration's
  3DA; its date is illustrative -- the January 2026 files already read 28000.)

**What is resolved**: the NA, the immersion, sample and coverslip indices, the wavelength
(per channel), the pixel per axis, the modality (2D, astigmatic, later a phase mask), and
every device's state by name -- the last kept with the table (`metadata["microscope"]`),
because it is what a calibration is matched by.

## Resolving a file

1. **Which microscope**: the entry whose `identify` the file matches, or the one chosen;
   and **which version of it**: the current one, or the oldest whose `until` is after the
   file's `StartTime` (a file without a date gets the current one, and says so).
2. **Which camera**: the camera database's, as now; its name picks the entry's camera
   parameters.  A camera the entry does not list is a warning (the pixel is then the
   camera database's), not an error.
3. **Each device's state**, by name, with its source: `3D lens: in (Servos-Position1 =
   28000)`.
4. **Each parameter**, from its value or its device's state.
5. **Layers, as in `io.tiff.resolve_camera`**: the database, then the file, then the user,
   each winning over the one before.  **The file wins over the entry, the pixel size
   included.**  The consequence on M2 today: its files say `PixelSizeUm = 0.127`, square,
   with the lens in, so they resolve to 127 × 127 nm and the entry's 127 × 116 applies
   only to a file without a pixel size.  The lens-dependent pixel reaches M2's files when
   Micro-Manager's pixel-size configurations are tied to the 3DA state (a pixel
   configuration can be, with an affine transform for the two axes), or when the user
   types it.  The table shows the entry's value beside the file's, so the disagreement is
   visible.

The result is a `Resolution` with more rows; the camera parameter view shows it as it shows
a camera now, and `CameraMetadata.pixelsize_um` takes the pair it already accepts.

## Uses

* **PSF calibration**: `OpticsSettings` starts on *auto* for each field and shows what it
  resolved and from where; a value typed beats it, as a camera field does now.
* **Matching a calibration to data**: a calibration keeps the device states of its beads;
  a fit, and in situ learning, warn when the data's differ -- the 3D lens out where the
  beads had it in, another filter.  The camera's EM port is checked this way already.
* **The fit**: the pixel per axis, without a field per file.

## Filling it in

**The editor starts from a file**, as the camera editor does, and has five parts: the
microscope (name, identifying tags, offered from the file's serial-number tags); its
cameras (one row each, chosen from the camera database, with their pixel); the devices (a
table: name, tag, how read, states); the parameters (one row each: fixed, from a tag, or
*per device* -- a device chooser and a small table of its states, prefilled with the
device's state names); the channels (filter device and side).  Beside it, live, what the
entry makes of the file.

**Without EMU, a device is added from a tag of the file**: pick `Servos-Position1`, and the
editor offers the values that tag takes over the files opened in the session as states,
for the user to name.  Most microscopes need two or three devices.

**With EMU, *Import EMU configuration…***: each control the configuration allocates
becomes a device, its property the tag, its states the configuration's values, named from
the parameters block -- the filter wheels and sliders with their name lists (a second
wheel with `Filter names 2`), the two-state devices *on* and *off* under their name, the
lasers on and off.  Unallocated controls, the focus lock, QPD, trigger and linear-stage
controls are left out (a linear stage that is also a two-state device, M5's 3D lens, comes
in as the two-state device).  The roles EMU knows become device names
(`Acquisitions - BFP lens` → *BFP lens*).  EMU has no role for the 3D lens and is not
changed for one: the import asks which device is the 3D lens and which of its states is
*in*, and fills `modality` from the answer.  The 2C tab's titles (*Channel 1
(transmitted)*) offer the channels, the side to be chosen.  Importing again -- after a
servo is re-trimmed or a device renamed -- updates the tags and values of the devices it
made and keeps every parameter's values, because those are keyed by state name; a device
whose Micro-Manager name changed keeps its old tag as a second one to try.

**What is typed by hand either way**, once per microscope: the NA, the immersion index,
the pixel per camera and lens state, which device is the 3D lens, the side of each
channel, a wavelength for a filter whose name is not one.

## Where it lives

`smappy/microscope_db.py` beside `camera_db.py`, sharing its `Parameter`, `Source` and
joining (moved to a small common module if they need to be); `data/microscopes.json`
shipped, `<config dir>/microscopes.json` the user's; `io/emu.py` the importer, Qt-free;
the editor in `gui/microscope_editor.py`, opened from the camera parameter view and from
*PSF calibration*.  The camera editor stays and opens on its own as now.

**What ships**: the Ries lab's microscopes, all of them, in `data/microscopes.json`, as
its cameras are in `data/cameras.json` -- smappy is used by a few groups at first; others
keep theirs in their own file, which wins by name.  Later one of them may stay as the
example and the rest move out.

## Decided (2026-10-22)

1. **Channels** follow one filter device each and land on a side of the chip, set per
   microscope (*transmitted, right*).  Single or dual is chosen in the calibration or the
   fit, not read: with the dichroic out only the transmitted channel applies, over the
   whole chip.
2. **One device per parameter.**  M1 has a filter in the common path and one in each
   arm, which together set a channel's band; that waits until it is needed.
3. **The file wins**, the pixel size included (above, for what it means on M2).
4. **Two-state devices split at the midpoint.**  A continuous device such as a TIRF angle
   may be recorded (`value`); nothing follows it yet.
5. **EMU is not changed**: the import asks which device is the 3D lens.
6. **The camera comes from the file**, through the camera database, by its serial
   number.  Cameras are named for themselves (*iXon 1*), not for a microscope.
8. **Versions**: changing an entry keeps the old one, dated, for older files; the date
   is the EMU configuration file's modification date where it can be read, else the day
   of the change.
7. **All the lab's microscopes ship.**

## Still open

* **Checking a version's date against the files.**  A date set wrong -- the
  configuration changed in March, the file is dated May -- decodes the files in between
  with the wrong positions.  It shows as a device in no state (outside the tolerance), or,
  with midpoint splitting, silently as the wrong state.  The editor could list the
  session's files by date with the version each one gets and what its devices decode to
  under each version, so that the point where the old positions stop fitting is visible
  and the date can be moved to it.  Left until a wrong date is met.
