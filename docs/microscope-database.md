# The microscope database

A design note, 2026-10.  **Built**: `smappy/microscope_db.py`, the importer
`smappy/io/emu.py`, the lab's M2 and M5 in `data/microscopes.json`, the cameras in the same
file and named for themselves, the fit and *PSF calibration* reading it ("Wired in"), the
manager (`gui/microscope_editor.py`); tests in `tests/test_microscope_db.py`,
`tests/test_emu.py` and `tests/test_microscope_manager.py`.  **Proposed**: the channels'
geometry, the objective, the filters and a default PSF ("Third round", 2026-11-04).  uiPSF needs the optics of the
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

## Second round (2026-10-22): one microscope, one control

After using the first GUI, the user's direction, which replaces the per-camera tables of
the entry and the GUI below wherever they differ.  **Built**: one camera per setup and
versions (`microscope_db`), the control (`gui/microscope_form.MicroscopeForm`, a settings
part of kind `"microscope"`: the fitters' `camera` part and *PSF calibration*'s optics),
the last microscope used standing in (config `last_microscope`; it brings its camera and
pixel only when the file's Micro-Manager camera device is that camera's), the manager's
camera field, the Localize tab without its camera buttons:

* **The user sees one thing, the microscope**, and only the parts the function at hand
  uses.  The camera database stays, as a library of cameras the microscope is set up
  with; nobody chooses between a camera and a microscope in a fitter.
* **A microscope has one camera.**  A new camera is a reprogramming of the microscope --
  the camera, the pixel per state of the 3D lens, the sides of the split -- and the
  setup as it was is kept as a dated version, so older files are read as they were
  taken.  M2: the iXon now, the Evolve as a version until 2026-07-15.  The camera that
  took a file is still recognised by its serial number: when it is not the version's,
  the file's camera wins, with a warning (a camera swapped before the entry was).
* **The split lives in the microscope**: the arms, each one's filter and its half of the
  chip.  Single or dual is still the function's choice.
* **Read noise is the camera's**, shown only where a fit uses it: the fitters add it to
  the likelihood for a camera without EM gain (`CameraMetadata.readout_variance`, zero
  with EM gain on purpose); if a fit comes to use it with EM gain too, there as well.
  **Frame time is metadata**: kept with the table, no setting.
* **One control everywhere** -- *PSF calibration*, every fitter, in situ -- replacing the
  fitters' camera part: a line, *M2 · iXon 1 · 3DA on · 127 × 116 nm*, grey when all
  that the function uses is known, orange when something is missing, marked when no
  microscope was recognised and **the last one used** stands in.  *(Step 3, 2026-10-23:
  the window became the folded section itself, its title the line; no *auto* boxes, each
  field shows the value found and is typed over, ↺ back; the sources in the tooltips.)*
  *Microscope...* opened one window, the same everywhere: the microscope (recognised, chosen, or the last
  used), the camera (the microscope's, or chosen when the file's is not known), and
  **only the values the function uses**, each with its value and its source.  They are
  locked; **a lock opens the editing** -- one small lock, not a tick per value -- and an
  edited value is shown in another colour, kept with the run's settings (chains and
  batches), the rest still resolved.  *Microscope manager...* below.  Expanded in a
  fitter, the control shows the same rows.
* **What each function uses**, declared by it:

  | function | shown |
  |---|---|
  | Gaussian and other 2D fitters | conversion, offset, EM gain, pixel; read noise for a camera without EM gain |
  | Spline 3D, bead-average calibration | the same |
  | a fit with a uiPSF model | the same: the model carries the optics |
  | PSF calibration (beads) | NA, immersion, emission per arm, pixel, the camera's numbers |
  | in situ | as the beads |

  Every fit with a calibration -- spline or uiPSF model -- checks the calibration's
  microscope state against the data's and **warns, naming what differs**; it runs
  anyway.  A missing value blocks only a function that uses it.  Parameters to come
  (a microscope's default pupil magnitude for in situ) join this table when they exist.
* **The manager**, from the menu: per microscope its identifying tags, its **camera**
  (chosen from the library, *Edit camera...* a button away), devices, parameters, the
  split, versions.  The camera buttons leave the Localize tab.
* **Old settings still read**: a chain's or a GUI file's `camera.conversion` and the like
  become edited values of the new control, its `camera.camera` the camera chosen.

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
   "n_immersion": 1.518,
   "wavelength_nm": {"device": "filter", "values": "from the name"},
   "modality": {"device": "3D lens", "values": {"in": "astigmatic", "out": "2D"}},
   "pixel_nm": {"device": "3D lens", "values": {"in": [127, 116], "out": [127, 127]}}},
 "channels": {"transmitted": {"filter": "2C transmitted", "side": "right"},
              "reflected":   {"filter": "2C reflected",   "side": "left"}},
 "camera": "iXon 1",
 "versions": [{"until": "2026-07-15", "camera": "Evolve 1",
               "parameters": {"pixel_nm": {"device": "3D lens",
                                           "values": {"in": [116, 127], "out": [127, 127]}}},
               "channels": {"transmitted": {"filter": "2C transmitted", "side": "upper"},
                            "reflected": {"filter": "2C reflected", "side": "lower"}}}]}
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
  the data.  The objective's coverslip is two of them, `n_coverslip` (the glass it
  expects; absent, the immersion's, an oil objective's) and `coverslip_design_um` (the
  thickness it is corrected for, 170 without one): *PSF calibration* builds uipsf's layered
  model from them (`docs/uipsf/microscopes.md`, "The coverslip").
* **`channels`**, for a microscope with a split: each channel names the filter device it
  follows and the half of the chip it lands on (`left`, `right`, `upper`, `lower`, the dual
  calibration's words), so a
  dual-channel file gets a wavelength per half -- *transmitted, right*.  The pairing is
  fixed per microscope.  **Whether the split is in is not recorded**: the splitting
  dichroic is put in by hand.  So single or dual channel is the choice the calibration
  and the fit already make (*dual* in *PSF calibration*, the dual-channel fitters), and
  the microscope answers it: dual, a wavelength per side; single, the *transmitted*
  channel's over the whole chip, the dichroic being out.  A microscope without a split
  has no `channels` and takes the `wavelength_nm` parameter.  Telling the two apart from
  the images (two copies of each spot, side by side) is possible and left out.
* **`camera`** (second round; first `cameras`, a table per camera): the camera the
  microscope is set up with, by its name in the camera database.  The pixel per state of
  the 3D lens and the sides of the split are the microscope's own parameters, for that
  camera; a new camera is a new version.  A camera's name is ours to choose and is not a microscope's: *iXon 1*, *iXon 2*,
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
  the change it does not absorb.  A version may hold the camera too: the example's is M2
  with the Evolve, until the iXon came (between 2026-07-09 and 07-23).
  A channel can also say exactly where it lands, for a split into more than two halves --
  the four phase images of a 4Pi microscope: `region` is `[x, y, width, height]` on the
  chip, in whole pixels, or a table from camera to that, and `mirror` is `"x"`, `"y"` or
  `"xy"` for an arm the optics flip (`Microscope.channel_layout`).  That is the coarse
  geometry, stable as long as the optics are; the sub-pixel registration between the
  channels belongs to a calibration, and their drift to the acquisition
  (`smappy.channels`, `docs/4pi.md` §9).

**What is resolved**: the NA, the immersion, sample and coverslip indices, the wavelength
(per channel), the pixel per axis, the modality (2D, astigmatic, later a phase mask), the
astigmatism in situ learning starts from (`astigmatism_nm`, Noll 6 in nm per state of the 3D
lens, from bead models: the sign of z when no model starts it; 2026-10-23), and
every device's state by name -- the last kept with the table (`metadata["microscope"]`),
because it is what a calibration is matched by.

## Resolving a file

1. **Which microscope**: the entry whose `identify` the file matches, or the one chosen;
   and **which version of it**: the current one, or the oldest whose `until` is after the
   file's `StartTime` (a file without a date gets the current one, and says so).
2. **Which camera**: the version's.  The camera database still recognises the file's by
   its serial number; when that is another one, the file's wins, with a warning.
3. **Each device's state**, by name, with its source: `3D lens: in (Servos-Position1 =
   28000)`.
4. **Each parameter**, from its value or its device's state.
5. **Layers, as in `io.tiff.resolve_camera`**: the camera database, the file, the
   microscope's pixel, the user, each winning over the one before.  The microscope's pixel
   beats the file's because Micro-Manager's is not tied to the lens: M2's files say
   `PixelSizeUm = 0.127`, square, with the lens in, where the pixel is 127 × 116 nm.  Any
   other value of the file still wins over the entry.

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

**One file per lab, cameras and microscopes together**: `data/microscopes.json` shipped
(it replaced `data/cameras.json`), `<config dir>/microscopes.json` the user's, each with a
`cameras` list and a `microscopes` list.  `camera_db` reads and writes the one,
`microscope_db` the other, and each keeps the other's list when it saves
(`camera_db.write_section`).  A user's `cameras.json` from before is still read, below the
new file, and never written.

`smappy/microscope_db.py` beside `camera_db.py`, using its `Parameter` and `Source`;
`io/emu.py` the importer, Qt-free; the manager to be `gui/microscope_editor.py`, opened
from the menu (see "The GUI").  The camera editor stays and opens on its own as now.

**Camera names**: *iXon 1* (the iXon 8172, *M1 Andor* before), *Evolve 1* (M2's Evolve,
*M2 Evolve*), *Evolve 2* (*M3 Evolve*), *Fusion BT 1* (*M5 Fusion BT*); the old names are
`aliases`, which `CameraDatabase.get` answers to and which a user's copy saved under an
old name replaces, so saved settings, chains and user files keep working.

**What ships**: the Ries lab's microscopes, all of them -- smappy is used by a few groups
at first; others keep theirs in their own file, which wins by name.  Later one of them may
stay as the example and the rest move out.

## The shipped entries

Made by `emu.draft` from `M2-config.uicfg` and `M5_config.uicfg` (October 2026) and
completed by hand, checked against the bead stacks of 260105, 260821, 260829, 260917 and
261012 (M2) and 260901 (M5):

* **M2**, recognised by its focus-lock laser (`Focus-lock-Serial ID`), which stayed when the
  camera and the SmarAct stage changed.  NA 1.43, immersion 1.518 (Olympus, Leica and
  Zeiss type F oils are all specified at 1.518).  The cylindrical lens is EMU's *3DA*
  (`Servos-Position1`, on 28000, off 4000).  The pixel per camera, as learnt on beads
  (`benchmarks/uipsf/results/lab_beads_pixel.jsonl`, the lower negative log-likelihood):
  *iXon 1* 127 × 116 nm with the lens on, *Evolve 1* 116 × 127, both 127 square with it off.
  The transmitted arm lands on the right of the iXon's chip, the reflected on the left:
  the brighter half of the 260821 dual-channel beads (91 against 22 counts above
  background) and movie (1816 against 145), two far-red dyes.  The split runs between chip
  columns ~259 and ~275; every single-channel M2 acquisition since the iXon (the 561 nm
  beads of 260917 at x 279–479, the 638 nm ones of 260829 and 260904 from x 300) sits in
  the right field, so its light took the transmitted arm.  On the Evolve the split ran top
  and bottom (a dark band at chip row ~267, `260311_PD_BeadsCal_640LP_2D`), the dual-channel
  ROIs tall (`203-11-212-489`), and the transmitted arm on top: in the Album of
  `260401_AASJ_2837_pSun1-AF647_antiGFP-Zym2-CF660C` (two far-red dyes, the Evolve) the
  top half is the brighter, 444 against 115 counts above background -- the camera turned
  by 90° against the iXon, as its pixel is (116 × 127 against 127 × 116).  The ELL9 sliders keep the names they had until September (`Thorlabs ELL9`, the
  reflected arm; `Thorlabs ELL9-1`, the BFP lens: the same serial numbers) as second tags.
* **M2's transmitted slider has 685/70 and empty at the same value**, 10000, in EMU's
  configuration.  The entry puts *empty* first, so 10000 reads as empty and the filter
  wheel decides: the 561 nm beads of 260917 read 10000 behind a 600/60 wheel filter, and
  the single-mode stacks whose comment says 700/100 read 10000 too.  Where 685/70 really
  sits is to be found and EMU's configuration corrected.
* **M5**, recognised by its iChrome laser's serial number.  NA 1.50, immersion 1.518.  The
  3D lens is EMU's *3D*, the ELL20 stage (in at 0 µm, out at 50000); it does not change
  the pixel, which is the camera's.  Two filter wheels, the transmitted arm's on top in
  EMU; the sides of the split are not known.

## Two controls, two states, one value

* **Two states at one value** (M2's slider above): the state listed first wins, every
  time; the import warns, and the order is the entry's, so it is changed in the entry
  (or, better, the configuration is fixed).  What a file read cannot tell two states
  apart; nothing in the file says which button was pressed.
* **Two controls on one property** (M5's ELL20 is both the *3D* two-state device and the
  *3D Lens* linear stage; M2's UV pulse durations share a property): each control the
  import takes is a device of its own, named by its button.  The import leaves out the
  linear stages, the triggers and the pulse durations, so on M5 and M2 only one device
  reads each property.

## Decided (2026-10-22)

1. **Channels** follow one filter device each and land on a side of the chip, set per
   microscope (*transmitted, right*).  Single or dual is chosen in the calibration or the
   fit, not read: with the dichroic out only the transmitted channel applies, over the
   whole chip.
2. **One device per parameter.**  M1 has a filter in the common path and one in each
   arm, which together set a channel's band; that waits until it is needed.
3. **The file wins**, except for a pixel the entry ties to a device's state, which wins
   over the file's (changed the same day, "Wired in").
4. **Two-state devices split at the midpoint.**  A continuous device such as a TIRF angle
   may be recorded (`value`); nothing follows it yet.
5. **EMU is not changed**: the import asks which device is the 3D lens.
6. **The camera comes from the file**, through the camera database, by its serial
   number.  Cameras are named for themselves (*iXon 1*), not for a microscope.
7. **All the lab's microscopes ship.**
8. **Versions**: changing an entry keeps the old one, dated, for older files; the date
   is the EMU configuration file's modification date where it can be read, else the day
   of the change.  A configuration copied to another computer gets the day it was copied
   (the three in `smappy test` all read 2026-10-22), so the date stays editable.
9. **One file holds a lab's cameras and microscopes.**

## Wired in

* **The pixel**: `io.tiff.resolve_camera` has a layer between the file and the user: the
  pixel the entry gives for the devices' state (M2's 3DA) beats the file's `PixelSizeUm`
  -- decided 2026-10-22, replacing "the file wins" for the pixel, since Micro-Manager's
  pixel is not tied to the lens and M2's files say 127 square with it in.  A value typed
  still wins.  Everything the fit and the calibration resolve goes through it.
* **What travels**: `CameraMetadata.microscope`, the resolution's `summary()` -- name,
  devices' states, the devices a parameter follows (`followed`), values, the arms with
  their sides -- so the table keeps it (`metadata["camera"]["microscope"]`) and a
  calibration made from beads keeps the beads' (`parameters["microscope"]`, saved with it;
  in situ, the movie's).
* **PSF calibration**: *NA*, *emission*, *emission, 2nd* and *immersion index* are on
  *auto*, shown greyed from the first stack's microscope (`OpticsSettings.hints`); a run
  without a microscope that gives them refuses with the reason.  Dual colour: the main
  channel is the arm on the calibration's `main_channel` half (*auto*: left or upper, as the
  dual calibration picks it).  Single colour: the first arm, the transmitted one.
* **The warning**: every place that warns about the EM state warns too when the beads'
  microscope differs from the data's -- another microscope, or a followed device in
  another state ("3DA: on for the beads, off for the data"); lasers and lamps are not
  compared (`io.calibration.microscope_differences`).

## The GUI (first round, built 2026-10-22; replaced by the second round above)

Little space where files are opened, the manager elsewhere:

* **The microscope manager** is opened from the menu (beside the camera editor), not from
  *PSF calibration*: the entries -- devices, parameters, channels, cameras, versions --
  with *Import EMU configuration…* (asking which device is the 3D lens), started from a
  file, showing live what the entry makes of it.  Its own help page.
* **Wherever a file is opened** -- *PSF calibration*, the fitters -- the microscope and its
  state are found automatically and shown in one line: *M2 · 3DA on · 700/100 · 127 × 116
  nm*, or *no microscope recognised*.  A small button beside it opens what was found: the
  values with their sources, each one overridable, a microscope chosen by hand for a file
  none recognises, the manual elements below set, and a link to the manager.  What is
  chosen there is a setting of the run (as a camera chosen by hand is), not a change of
  the entry.
* **Manual elements** (later, when needed -- not in the next step): what is put in by
  hand and so never in the metadata -- above all
  the dichroic that splits the channels (M2's 665LP, 640LP), which EMU does not control
  and should not.  An entry lists them with their choices (*none*, *640LP*, *665LP*); the
  found settings show them to be set, and a calibration keeps what was set like any
  device's state.  The splitting dichroic answers single or dual too: *none* is one
  channel over the whole chip.  A long-pass sets no centre wavelength by itself: an arm
  whose band only it bounds (260821's transmitted arm, an empty slider behind 665LP) has
  its wavelength typed in the entry for that choice, or in the run.

Built as: *File > Microscope manager...* (`gui/microscope_editor.py`, its page
`docs/panels/Microscope manager.md`), the tables typed as the camera editor types a map;
*PSF calibration*'s line under the stacks with *Microscope...*
(`gui/microscope_view.FoundMicroscope`), a microscope chosen there handed to the run as its
camera; in the fitters, the microscope in the camera's folded title (*M2, astigmatic*), its
rows and a *Microscope manager...* button in *Camera parameters...*, and *microscope* under
the camera's advanced settings to choose one by hand.

**A missing value blocks only what reads it.**  A fit needs the camera's conversion, offset
and pixel, as before; the NA, the emission and the immersion are read by *PSF calibration*
and in situ learning alone, which refuse without them.  A microscope nobody recognises, or a
device in no state, blocks nothing: the pixel falls back to the camera's or the file's.

## Remembering calibrations and the objective (2026-10-23)

Decided with Jonas:

* **A list of calibrations, as links** (`smappy/calibration_list.py`, `calibrations.json` in the
  configuration folder): every calibration *PSF calibration* saves, with the beads' microscope summary
  (entry, setup version, device states, camera), EM gain and the beads' day.  The window describes it,
  since a bead average records no microscope in its file.
* **The spline fitters' *calibration* empty is *auto***: the listed calibration -- or one beside the data,
  its folder and the parent -- that matches the data's microscope, version, followed device states,
  camera, EM gain and channel count, nearest in time to the data; shown in grey in the field, all found in
  a menu (▾, right click).  Never one that does not match.  A fitted `.h5` stands for the calibration its
  provenance records.  No age warning.
* **Spline 3D 2C, *measure from this movie***: the calibration's PSFs with a registration measured as Gaussian
  2D 2C measures it, with the calibration's layout and main channel; another main channel is refused
  (`dual.with_registration`), so each PSF stays on its half.
* **The objective's pupil in the entry** (`pupils`, by the 3D lens's state, a section a version holds):
  every Zernike of phase (nm) and magnitude and the blur, from *Add to microscope...* after a pupil model
  or in situ run; in situ starts from it without *start from*, its Noll 6 over `astigmatism_nm`.  We do
  not swap objectives, so one per entry.
* **γ, the objective's apodisation** (`gamma_of`; the user's decision, November 2026): a property of the
  objective, perhaps a little of the colour, and of nothing else.  In situ without *start from* holds the
  entry's `gamma` parameter (a value, or one following the emission filter), else the γ of a minimal pupil
  kept under any state of the 3D lens, nearest in emission, with its learnt NA; only failing both a nominal
  0.36, with a warning.  M2's minimal bead models learnt 0.08 (Evolve, January 2026) and 0.16 (iXon, August
  2026), M5's 0.32–0.41 (`docs/uipsf/architecture.md`, "In situ on the minimal model").

## Third round (2026-11-04): the channels, the objective, the filters, a default PSF

Proposed, then decided (below); the proposal's parts that the decisions changed are kept
as they were written, the decisions win.  What started it: *measure from this movie* in the two-colour fitters
guesses the split -- which axis, which half is the main one, where the seam is
(`transform.detect_layout`, a provisional seam down the middle of the longer axis) -- and the
guess failed on real data.  The arrangement of the channels on the chip is a fact about the
microscope and its camera, measured once; it belongs in the entry, as the 4Pi quadrants
already are.

### Decided (2026-11-04, with Jonas)

* **No detection of the layout.**  `transform.detect_layout` and the bead calibration's
  `detect_bead_layout` go; the arrangement is a setting of the chip, in the two-colour
  fitters and the dual bead calibration: *layout* (right-left, up-down), *mirrored*, the
  *main* half, the split -- on *microscope* by default, the entry's arrangement, and chosen
  by hand for a microscope without one.  The registration only refines the map within it.
* **The arrangement in the entry** is the arms' `side`s, a `mirror` on the arm the optics
  flip, a named `main_channel` and an optional `split_position` per camera (the middle of
  the frame as acquired otherwise).  All the lab's microscopes are right-left, mirrored;
  which arm is on which side is read from ratiometric data, where the transmitted arm is
  the brighter.  M5 and M6 have the same arrangement.  The 4Pi quadrants stay `region`s.
* **The objective** is a section of the entry: name, NA, immersion and its index.  No
  magnification: the pixel on the sample is the camera's (per 3D-lens state where it
  differs), and the nominal magnification says little with the relay optics behind it.
  M2: *Leica 160x NA1.43*; M5: *Olympus 100x NA1.50*; M6: *Olympus 100x NA1.49*; all oil.
* **Filters**: a table of the filters by name with an effective band, typed for a filter
  whose name is not its band.  **No manual elements**: the dichroics are not needed (the
  ratiometric splitter is a 665LP, written in the entry's comment).
* **The default PSF** is what *PSF calibration*'s *Add to microscope* keeps: one per state
  of the 3D lens and per colour -- green, red, far red, not per filter -- so the key is
  ``3DA: on · far red``; one kept before colours were keyed serves every colour.
* **Cameras**: M2 has only its Andor (iXon 1) for this; M5 and M6 an Orca Fusion BT each, M6's
  pixel 102 x 108 nm.  No M6 file exists yet on the share (2025, 2026), so
  M6 is chosen by hand until one gives its camera's serial to `identify` by.

### Built: step 1 (2026-11-04)

The entry's `objective`, `filters` (`filter_wavelength`), `main_channel`, `split_position`,
`layout`, an arm's `mirror` in the resolution and its summary; `Microscope.arrangement` (the
chip settings' words) and `split_layout` (the two halves as a `ChannelLayout` on a camera ROI);
pupils keyed by lens state and colour (`colour_of`, `emission_of`); the manager's *objective*
table and *mirror* column, and everything else carried through it.  M2, M5 and the new M6 (with
its camera *Fusion BT 2*) filled in.  Nothing reads the arrangement yet: that is step 2.

### Built: step 2 (2026-11-04)

No detection: `transform.detect_layout`, `dual.detect_bead_layout` and the provisional seam down
the longer axis are gone.  `RegisterSettings.layout` and `main_channel` default to *microscope*
(an old *auto* reads as it), resolved by `dual.resolve_arrangement` from the file's microscope --
in the two-colour fitters' calibration pass (`_seam`), *Calibrate transform* (the table's
microscope) and the dual bead calibration (`beads_microscope`) -- and refused, asking for the
layout, where no entry says.  The vote pairs only across the seam (the split, else the middle of
the frame) and takes the axis from the layout.  Not done: checking a chosen registration file
against the entry's arrangement; 4Pi beads still read the current setup, not `as_of`.

### Built: step 2b (2026-11-04), so far

* **Stage 1**: every fit on the camera chip (below, "the coordinates").
* **Stage 2**: every fitter's *channels* part (`fit.ChannelsSettings`): *read* whole chip (a 1C
  fitter's default: no file records the dichroic) or two channels, and the arrangement --
  *layout*, *main channel*, *split position* -- on *microscope* or chosen.  On two channels a 1C
  fitter fits each channel on its own (`multichannel.SplitEngine`: one `LocalizationEngine` per
  channel, cut by `channels.split_halves`), each localization on the chip where its channel saw
  it, which channel in `camera_channel` (`channel` is the dye's colour).  A two-colour fitter
  shows only the arrangement and its calibration pass reads it there; the registration's copies
  of it are hidden in its form.
* **Stage 3, built**: the registration is between the channels of such a table
  (`calibrate.transform.register_channels(x, y, frame, channel, layout, ...)`): channel 1 put
  onto channel 0 by the layout's nominal map (`ChannelLayout.nominal_map`), the remaining shift
  voted for over the cross-channel pairs of a frame, then pair-and-refit as before.  Nothing
  measures a seam.  A two-channel fit records its rectangles on the chip
  (`metadata["channels"]`), which *Calibrate transform* reads; a table without `camera_channel`
  is refused.  The calibration pass of *measure from this movie* is a two-channel 1C fit plus
  that registration.  A registration's and a dual bead calibration's geometry hold the
  channels' rectangles and flips (`geometry["channels"]`, `dual.geometry_layout`,
  `dual.split_geometry`) in place of `split_position` and `mirror_axis_xy`; the bead
  calibration cuts its stacks by them.  Gone: `dualfit.which_channel`, `split_axis`,
  `reference_is_first`, `dual.chip_halves`, the seam estimate and the mirror modes of the
  vote.  Files written before (a halves geometry) are refused with a message to measure again.
  The map itself is still kept as a chip-to-chip matrix beside the layout, which
  `ChannelTransform.layout` turns into the layout's map between the channels' own pixels.

### The coordinates (decided 2026-11-04)

Every fitter reports positions on the **camera chip**, in chip pixels with the camera ROI's corner
added: a whole-chip fit as always; a fit through the channel layer where the main channel saw the
molecule -- its own pixels put back on the frame, its flip undone (`ChannelLayout.on_frame`),
the ROI's corner added (`MultiChannelEngine.flush`).  A 1C fit split into two channels reports
each channel's localizations on the chip where that channel saw them, until a registration maps
the second channel onto the main one.  So one molecule has one position whichever fitter found it,
and positions line up with the camera's own image.

**Images follow the same convention**: an image layer is placed on the chip.  The raw frames a
channel fit keeps are the main channel, its flip undone, on its rectangle
(`RawFrameKeeper.image`).  Still to do where it fits: an image of another channel (a channel's
average, a registered overlay) placed on the chip -- through the registration where it is mapped
onto the main channel, which a placement by offset cannot express for a mirrored channel.  The fit
region is drawn in frame pixels, as for a whole-chip fit (`fit.ChannelRegionFinder`).

### Where it stands

* **Built**: devices and their states, parameters following one device each, filters giving
  the wavelength by name, arms with a `side` (M2) or a `region` and `mirror` (the 4Pi
  example), dated versions, the pupil per state of the 3D lens (`pupils`, `gamma_of`), the
  manager, the found-microscope line.
* **The channel reader** (`channels.ChannelSource`) is what every multi-channel fitter reads
  through -- Gaussian 2D 2C, Spline 3D 2C, Spline 4Pi -- and 4Pi beads.  But where its layout
  comes from differs: Spline 4Pi and 4Pi beads take the entry's regions
  (`fit_4pi.microscope_layout`; 4Pi beads without `as_of`); the two-colour fitters only a
  registration's channels (`_TwoChannelFit.channel_layout`); the calibration pass of *measure
  from this movie* and the dual-colour bead calibration cut the chip by the arrangement
  (`channels.split_halves`).  The entry's `side` is read only to
  give each arm its wavelength (`OpticsSettings.resolved`).
* **Not in any file** the lab has (`Setups/M2`, `M5`, `M6` of the test data): the objective,
  the tube lens, the dichroics, which half is which arm, a mirror.  EMU's newer
  configurations record a chip split for activation feedback (M2 vertical at 218 px, M5
  horizontal at 251 px, a 10 px seam) -- a start for the seam, not the optics.

### 1. One channel geometry, two to four channels

An entry's arms say where each lands, per camera, in one form whatever their number (the
numbers are illustrative, the iXon's from EMU's split at 218 with its 10 px seam):

```json
"channels": {
  "transmitted": {"filter": "Channel 1 (transmitted)",
                  "region": {"iXon 1": [228, 0, 284, 512], "Evolve 1": [0, 0, 512, 251]}},
  "reflected":   {"filter": "Channel 2 (reflected)",
                  "region": {"iXon 1": [0, 0, 218, 512], "Evolve 1": [0, 261, 512, 251]},
                  "mirror": "x"}},
"main_channel": "transmitted"
```

* **`region`** on the chip, whole pixels, per camera -- as the 4Pi quadrants are.  A seam
  between the halves is simply not in either region.  `side` stays readable (old entries,
  and the coarse word the PSF calibration's `main_channel` speaks), and becomes derived: the
  side of the chip a region's centre is on.
* **`mirror`** per arm, against the main arm: what a mirror in the path flips.
* **`main_channel`**: the reference the registration maps onto and whose PSF a dual
  calibration's main is -- named, not *left or upper*.  Default the first arm.
* **`Microscope.channel_layout(camera)`** then serves two and four arms alike, and with
  `as_of` everywhere (4Pi beads too).

**Who reads it, in order**: a registration or calibration chosen for the run (its sub-pixel
map, its own halves); else the entry's layout; else, for a microscope with no regions, the
detection there is today, with a warning that says the entry would make it certain.  So:

* **Two-colour fitters**: `channel_layout` takes the entry's regions when the registration
  is measured from the movie, and checks a file's registration against them (a calibration
  of another arrangement is refused, as another main channel is now).
* **The calibration pass** (*measure from this movie*) reads the movie through
  `ChannelSource` on the entry's layout and registers channel 1's localizations onto
  channel 0's: no provisional seam, no layout vote -- the vote only finds the residual
  offset, in a window a few pixels wide.  This is the change that fixes the failure.
* **Dual-colour bead calibration** the same: the beads cut by the entry's layout.
* `RegisterSettings.layout` gains *microscope*, the default when the entry has regions.

**Filled in**: in the manager's *Channels* tab, on a frame of a file -- the regions drawn
over it, dragged or typed, a seam's width and the EMU split as a start
(`emu.chip_split`), a *from this registration* button that takes the halves of a
`_2ct.h5` or dual calibration that is known to be right.

### 2. The objective

Fixed per entry (we do not swap objectives), a section of its own rather than loose
parameters, so the manager can show it as a form and a version can hold it:

```json
"objective": {"name": "Olympus UPLAPO100XOHR", "magnification": 100, "NA": 1.5,
              "immersion": "oil", "n_immersion": 1.518, "n_coverslip": 1.523,
              "coverslip_design_um": 170, "tube_lens_mm": 180}
```

`NA`, `n_immersion`, `n_coverslip`, `coverslip_design_um` are what uiPSF reads today
(`OpticsSettings.resolved`), from `parameters`; they move here, with `parameters` still
read for old entries.  The magnification and tube lens give the nominal pixel (chip pixel
over the total magnification), shown beside the learnt `pixel_nm` as a check, never used in
its place.

### 3. The filters

The filter names are the data today (`676/37` -> 676).  What that cannot say -- a long
pass, a band whose name is not its band, an arm bounded by the splitting dichroic -- goes
in a table of the filters by name, shared by every device that holds them:

```json
"filters": {"665LP": {"from_nm": 665}, "700/100": {"centre_nm": 700, "width_nm": 100},
            "Empty": {}},
"manual": {"splitting dichroic": {"states": ["none", "640LP", "665LP"], "default": "665LP"}}
```

* An arm's **emission** is its band cut by what else is in its path -- the arm's filter, the
  common wheel, the splitting dichroic's side -- and the wavelength uiPSF takes is the
  centre of that, or one typed per filter where the dye matters more than the band
  (`wavelength_nm`).  A band with no upper bound (a long pass alone) is missing, as now.
* **Manual elements** (planned since 2026-10-22): a device no file records, with a default,
  set where the microscope is found (the found-settings line) and kept with a calibration
  like any state.  The splitting dichroic is the first: *none* is one channel over the
  whole chip -- the answer to single or dual, which nothing else records.

### 4. A default PSF

`pupils` already keeps a learnt pupil per state of the 3D lens (`keep_pupil`, *Add to
microscope...*), and in situ learning starts from it.  What is missing:

* **Shipped ones**: M2's, M5's and M6's minimal pupils from the lab's bead models
  (`benchmarks/uipsf/results/lab_beads.jsonl`), so a fresh install has them.
* **A PSF from the description**: `uipsf` builds a `VectorPSF` from a kept pupil and the
  resolved optics (a constructor beside `export.sample_psf`), so that what today needs a
  `.uipsf.h5` beside a spline can take the microscope's instead: *Simulate*'s PSF
  (*microscope's default*), Spline 3D's *focus* with no model file, and a first fit of
  astigmatic data with no calibration at all (a spline sampled from it, kept as a
  calibration in `calibration_list` marked *default*, never chosen by *auto* over a measured
  one).
* Per arm where the arms differ (the reflected arm's astigmatism), keyed `3DA: on /
  reflected`; the arm-less key stays the common one.

### 5. The manager

Tabs: *Objective* (the form above), *Filters*, *Devices* (as now, with manual elements),
*Channels* (the regions on a frame, per camera), *PSF* (the kept pupils, their source and
date, a drawing of each), *Versions*.  *Check a file* shows everything resolved, the
channels drawn on its frame.

### 6. The entries

* **M2**: regions for the iXon and the Evolve from its dual-colour data; the older EMU
  configurations as dated versions -- `ht-SMLM` (`Servos-Position0` the wheel,
  `Servos-Position3` slider 1, a 700/50, BFP on `Servos-Position2`) and the `Mojo-` era
  (other tags, 3DA on at 37000) -- when their dates are known.
* **M5**: regions (EMU's horizontal split at 251) and which arm is which.
* **M6**: new, from `Setups/M6/config.uicfg`: M5's layout with the serial MLE
  (`iChrome-MLE-…`), wheels `transmitted_ Filter Wheel-2-State` (transmitted, the `-2`)
  and `reflected_ Filter Wheel-State` (700/100 at 3), the 3D lens reversed (on 44317, off
  0), TIRF 19040/21040.  Nothing in its files identifies it (no serial); an `identify` tag
  is needed from the instrument.

### Order of work

1. The schema and its resolution (`main_channel`, regions per camera, `objective`,
   `filters`, `manual`), old entries read as before; tests.
2. The two-colour fitters, the calibration pass and the dual bead calibration on the
   entry's layout through `ChannelSource`; 4Pi beads with `as_of`; tests on simulated split
   data with a known arrangement, and on the test data's M2 dual-colour movie.
3. The manager's tabs; its page.
4. The default PSF: the builder, the shipped pupils, *Simulate* and *focus* reading it.
5. M2, M5, M6 filled in.

### Needed from the lab

The objective of each (name, NA, immersion), the tube lens or total magnification, the
camera on M5 and M6 (model, pixel), which arm lands on which half and whether one is
mirrored on each, the splitting dichroics in use, an identifying property for M6, and the
dates of M2's EMU configurations.

## Next

* Manual elements, when needed (above).
* **Files attached to an entry** (the user, 2026-10-22): what is measured once per
  microscope or camera and used by many runs -- an objective's pupil magnitude, learnt on
  beads and held for in situ learning; an sCMOS camera's offset, gain and noise maps.  An
  entry would name the file and the state it was measured in, kept beside the user's
  `microscopes.json`, and resolved like any parameter.
* **The last PSF calibration per state** (the user, 2026-10-22): a calibration already keeps
  its beads' microscope and device states (`parameters["microscope"]`).  Saving one could
  register it with the entry under that state, and a fitter's calibration on *auto* would
  take the newest one whose followed devices match the data's -- the 3D lens in and the
  same filter -- so the right calibration is found as the camera is, and a mismatch cannot
  happen by picking the wrong file.

## Still open

* **Checking a version's date against the files.**  A date set wrong -- the
  configuration changed in March, the file is dated May -- decodes the files in between
  with the wrong positions.  It shows as a device in no state (outside the tolerance), or,
  with midpoint splitting, silently as the wrong state.  The editor could list the
  session's files by date with the version each one gets and what its devices decode to
  under each version, so that the point where the old positions stop fitting is visible
  and the date can be moved to it.  Left until a wrong date is met.
