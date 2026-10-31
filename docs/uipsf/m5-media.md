# M5's beads in oil and buffer, two colours, with and without the 3D lens (study A)

31 October 2026 · Jonas Ries (written with Claude)

On 29 October 2026 M5 imaged 100 nm TetraSpeck beads in eight conditions, each a change of one thing:
the emission (600 nm under 561 nm excitation, 676 nm under 640 nm), the medium (dried and covered with
immersion oil, n 1.518, "match"; or in PBS, 1.334, "mismatch") and the 3D cylindrical lens (in or out).
`benchmarks/uipsf/real/m5_media.py` learns each with uiPSF through SMAPpy's calibration (its docstring has
the bead selection); `benchmarks/uipsf/real/m5_media_table.py` makes the tables below from
`benchmarks/uipsf/results/m5_media_window_studio.jsonl`, `m5_media_na151_studio.jsonl` and
`m5_media.jsonl`.  The pixel is
104.7 × 106.4 nm throughout, chosen by likelihood (`m5_media_pixel.jsonl`).  σ is the blur beyond the
pixel; phases are in nm of optical path (rad × λ / 2π), the unit two colours share; a bend is the slice
fits' systematic error against the stack fit, the rms over z of the median over beads.

**In short.**
- Learning on the full ±2 µm gives the same σ and phases as ±1 µm (to 1–3 nm) and a smaller radial
  magnitude, but no smaller bends.
- The tightest bends near focus come from learning on ±500 nm alone.
- What the lens adds is a path difference shared by both colours, as physics says it must be.
- The spherical aberration is not shared: in nm of path it is twice as large at 600 nm as at 676 nm.
- In matched oil, the bead-physics magnitude on a grid at NA 1.51 finds an effective NA of 1.49–1.52.
  It beats a free Zernike magnitude with fewer parameters (on the nominal 1.50 grid it does not), and finds
  the same cos θ exponent (≈ 0.4) in every condition but one.
- In buffer, the data want light out to the grid's edge and beyond, so the supercritical ring modelled
  for a 100 nm bead is too weak there.

## Setup

- **Machine:** an M1 Max with 32 GB, not the 64 GB the plan assumed.
- **Memory settings:** every run used `--chunk 1 --memory 0.5` (MPS capped at 12.5 GB) under `nice`, one
  job at a time.
- **What fitted:**
  - ±1 µm at grid NA 1.51 (208 pupil samples) fitted one bead per chunk.
  - ±2 µm at 50 nm slices (240 samples) fitted the four 600 nm sets.
  - The 676 nm in oil, 2D set ran out (10.6 GB held, 3 GB more asked), so all eight were run at 100 nm,
    and the four 600 nm sets at 50 nm as well.
- **Slice spacing doesn't matter:** on those four, 50 and 100 nm give the same σ and phases to 0.1 nm,
  the same magnitude to 0.001, and bends within 3 nm.  The report's "slice thinning 100 nm" holds at ±2 µm.
- **Times:** ±2 µm at 100 nm took 1–5 min per condition, at 50 nm 2.5–10 min; ±1 µm took 0.7–2.3 min.

**How the bends are measured on whole stacks.**  At ±2 µm a bead's window is the whole stack, centred on
the stack and not on the bead's focus, which drifted by up to 1 µm in oil.  `bead_slices.measure`
therefore takes each bead's z about its own focus (`centre_nm`, from the selection) and bins the curve in
z rather than by slice index.  A median per index had mixed beads a micrometre apart in z, and its edge
bins held a single bead (a −178 nm "bend").  On windows centred on the focus (±500, ±1000) the two give
the same numbers: one condition re-measured both ways is identical to 0.01 nm.

**Another session's runs**, under the same script, are on main: `results/m5_media.jsonl` (both magnitudes
at grid 1.50, ±1 µm, 50 nm) and `results/m5_media_window.jsonl` (the four 676 nm sets at ±0.5, ±1 µm at
100 nm, ±2 µm at 100 nm).  They reproduce this session's objectives exactly.  Their ±2 µm bends were taken
per slice index and are superseded: 676 nm oil 3D reads 5.5 nm there, 13.9 nm about each focus.  Their
±1 µm at 100 nm slices bends z as ±1 µm at 50 nm does here (10–15 nm).

## 1. The window: ±500, ±1000, ±2000 nm (Zernike magnitude, grid NA 1.50)

Per condition: σx, σy (nm); the magnitude's radial and non-symmetric rms; the main phase modes (nm of
path); and the z bend over ±500 nm, the range every window covers.

| λ | medium | lens | window | σx, σy | magn. radial / asym | Z5 | Z6 | Z11 | Z22 | z bend ±500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 600 | oil | 2D | ±500 | 32.5, 29.3 | 0.186 / 0.081 | −1.1 | 12.5 | 55.5 | 24.8 | 16.5 |
| | | | ±1000 | 33.2, 29.8 | 0.145 / 0.071 | −2.0 | 11.5 | 54.5 | 24.6 | 23.1 |
| | | | ±2000 | 33.6, 29.6 | 0.119 / 0.061 | −2.0 | 11.7 | 55.4 | 25.2 | 22.2 |
| 600 | buffer | 2D | ±500 | 24.9, 23.5 | 0.284 / 0.076 | 3.4 | 19.2 | 59.8 | 30.2 | 5.3 |
| | | | ±1000 | 22.9, 22.4 | 0.223 / 0.069 | 0.3 | 13.8 | 56.0 | 28.5 | 13.3 |
| | | | ±2000 | 21.3, 20.5 | 0.209 / 0.063 | 0.0 | 13.4 | 57.3 | 29.2 | 13.2 |
| 676 | oil | 2D | ±500 | 25.6, 0.0 | 0.202 / 0.068 | −0.2 | 15.3 | 26.3 | 15.9 | 7.4 |
| | | | ±1000 | 24.3, 0.0 | 0.152 / 0.054 | −0.7 | 13.7 | 27.0 | 17.5 | 11.4 |
| | | | ±2000 | 25.5, 0.0 | 0.124 / 0.049 | −0.8 | 13.7 | 28.6 | 19.4 | 16.7 |
| 676 | buffer | 2D | ±500 | 5.8, 12.3 | 0.279 / 0.067 | 1.1 | 22.8 | 33.7 | 24.0 | 6.4 |
| | | | ±1000 | 0.0, 11.4 | 0.212 / 0.059 | 1.5 | 18.1 | 32.3 | 24.8 | 13.5 |
| | | | ±2000 | 0.0, 13.5 | 0.205 / 0.057 | 1.6 | 18.1 | 34.4 | 26.6 | 13.6 |
| 600 | oil | 3D | ±500 | 45.1, 38.3 | 0.206 / 0.145 | −25.5 | 54.3 | 59.6 | 24.4 | 6.5 |
| | | | ±1000 | 49.8, 51.4 | 0.130 / 0.106 | −23.2 | 52.4 | 63.1 | 27.8 | 11.0 |
| | | | ±2000 | 50.9, 52.9 | 0.122 / 0.076 | −24.9 | 54.0 | 59.5 | 25.3 | 9.0 |
| 600 | buffer | 3D | ±500 | 29.9, 27.1 | 0.276 / 0.079 | −22.2 | 59.5 | 64.9 | 31.5 | 5.8 |
| | | | ±1000 | 28.4, 29.7 | 0.235 / 0.077 | −22.4 | 55.5 | 61.9 | 30.2 | 14.1 |
| | | | ±2000 | 26.2, 28.3 | 0.207 / 0.071 | −22.3 | 54.5 | 62.5 | 30.7 | 15.1 |
| 676 | oil | 3D | ±500 | 24.1, 20.1 | 0.196 / 0.097 | −19.7 | 61.6 | 35.8 | 20.8 | 6.4 |
| | | | ±1000 | 21.8, 22.8 | 0.157 / 0.062 | −21.9 | 56.0 | 32.4 | 19.4 | 11.1 |
| | | | ±2000 | 21.3, 24.3 | 0.139 / 0.061 | −21.7 | 55.7 | 32.5 | 20.0 | 13.4 |
| 676 | buffer | 3D | ±500 | 18.8, 18.3 | 0.268 / 0.078 | −20.7 | 63.7 | 42.0 | 27.2 | 6.4 |
| | | | ±1000 | 17.6, 22.8 | 0.203 / 0.063 | −20.7 | 57.7 | 38.0 | 26.9 | 13.0 |
| | | | ±2000 | 15.2, 23.3 | 0.188 / 0.058 | −20.6 | 57.2 | 38.7 | 27.3 | 15.3 |

±2000 is at 100 nm slices, the others at 50 nm.

- **±2000 against ±1000: the model holds, and there are no new errors.**
  - σ moves by ≤ 2.4 nm.
  - Astigmatism (Z5, Z6) moves by ≤ 1.7 nm, second spherical (Z22) by ≤ 2.5 nm, and first spherical
    (Z11) by ≤ 2.1 nm, except 600 nm oil 3D (−3.6), the cluttered outlier below.  The third spherical
    (Z37) moves by −0.5 to +2.9 nm, the one mode the extra defocus pins differently.
  - The x and y bends stay within 1–6.4 nm, and the z bends within ±500 are about the same
    (9–22 nm against 11–23).
  - Across the four colour × medium runs of one lens, the spread of σ and of each phase mode is about the
    same at ±2000 as at ±1000.
- **±2000 doesn't make the model better.**
  - The radial magnitude falls steadily with the window: 0.19–0.28 → 0.13–0.24 → 0.12–0.21.  More
    defocus constrains the pupil's rim, which the rings of defocused slices see and the focus does not.
  - This doesn't make the bends near focus any smaller.
  - The deviance per pixel falls too, 3.9 → 3.4 (2D mean) and 3.5 → 2.9 (3D).  But that is over
    different slices, mostly dim defocused ones, so it doesn't compare across windows.
  - Settling it would need every model judged on the same slices (held-out beads or a common ±500); this
    script does not save the models to do that.
- **±500 is best near focus.**
  - Its z bend over ±500 is 5–7 nm in seven of eight conditions (16.5 in 600 nm oil 2D), against
    9–23 nm for the wider windows.
  - Its Z6 astigmatism is 1–6 nm larger and its radial magnitude 0.04–0.08 larger: it fits the focus
    at the cost of a pupil that has never had to explain the far defocus.
  - What it does outside ±500 is untested.
- **The z bend is the open problem at every window**: 9–24 nm rms over ±600 for the wider windows
  (5–17 over ±500 for ±500), with each slice's scatter about it 3.3–6.8× its CRLB.  This is the same
  13–14 nm the pixel study found.  x and y are within 1–6.4 nm.

## 2. Physics: what agrees and what moves

From the ±2000 runs; ±1000 says the same.

| what | must agree because | 600 oil | 600 buffer | 676 oil | 676 buffer | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| lens: Z5 3D − 2D (nm path) | a lens is a path difference | −22.9 | −22.3 | −20.9 | −22.2 | **agrees** (±1.5) |
| lens: Z6 3D − 2D (nm path) | same | +42.3 | +41.1 | +42.0 | +39.1 | **agrees** (±1.6) |
| Z11 (nm path), 2D / 3D | an objective's aberration is a path | 55.4 / 59.5 | 57.3 / 62.5 | 28.6 / 32.5 | 34.4 / 38.7 | **moves with λ**: ×2 between colours |
| Z22 (nm path), 2D / 3D | same | 25.2 / 25.3 | 29.2 / 30.7 | 19.4 / 20.0 | 26.6 / 27.3 | moves with λ and medium |
| Z11 3D − 2D | the lens adds no spherical | +4.1 | +5.2 | +3.9 | +4.3 | **moves**, but consistently: +4–5 nm |
| Z11 buffer − oil | beads at the coverslip: the model's medium covers it | +1.9 / +3.0 (600) | | +5.8 / +6.2 (676) | | small, systematic |
| magnitude radial, 2D / 3D | oil has no supercritical light | 0.119 / 0.122 | 0.209 / 0.207 | 0.124 / 0.139 | 0.205 / 0.188 | **lens-independent**; buffer +0.08 |
| σx, σy 2D | the camera and optics | 33.6, 29.6 | 21.3, 20.5 | 25.5, 0.0 | 0.0, 13.5 | **moves**: by 34 nm, and which axis flips |
| σ 3D − 2D | the lens adds no blur | +17, +23 | +5, +8 | −4, +24 | +15, +10 | moves |

What agrees, and what moves:

- **The lens's astigmatism agrees.**  It adds −22 nm of Z5 and +41 nm of Z6 of path in all four
  colour × medium pairs, to ±1.6 nm.  Expressed in radians it would differ by 13 % between the colours, so
  path is the right unit, as for a thin lens it must be.  The 2D path's own astigmatism (Z6 12–18 nm) is
  M5's residual.
- **The spherical aberration doesn't agree.**  In nm of path, Z11 is 55–62 nm at 600 nm and 29–39 nm at
  676 nm.  In radians it is 0.58 against 0.27 rad, so it isn't a fixed phase either.
  - This is spherochromatism: the objective's spherical aberration changes with wavelength, which an
    apochromat corrects only at its design wavelengths.
  - Or the model's defocus mixes into Z11 differently at the two colours: n_imm is 1.518 for both, while
    oil's dispersion moves it by about 0.002, too little to explain a factor of 2.
  - The beads cannot tell the two apart.  A model learnt at one colour does not carry over to the other.
- **The lens also moves Z11, by +4–5 nm in every pair.**  A cylinder adds no spherical aberration, so this
  is either the fit trading Z11 against the larger astigmatism or a real difference between the two
  acquisitions (the lens is moved in and out between them).
- **Buffer against oil:** buffer carries 2–6 nm more Z11 and 4–7 nm more Z22.  The beads sit on the
  coverslip, so a depth-induced aberration should be absent, and what remains is small.
- **The radial magnitude depends on the medium, not on the lens.**  Oil gives 0.12–0.14: the objective's
  own fall towards the rim, as matched medium has no supercritical light.  Buffer gives 0.19–0.21; the
  extra 0.08 is the supercritical ring the model gets wrong.
- **σ doesn't behave like a property of the microscope.**
  - Oil is larger than buffer in three of the four colour × lens pairs, by up to 25 nm; at 676 nm in
    2D the blur moves from x (oil) to y (buffer) instead.
  - In oil, σ follows the clutter: 600 nm oil 3D has 18 beads among 333 dim objects and σ ≈ 51 nm, while
    676 nm oil 3D has 39 among 23 and σ ≈ 23 nm.  The faint objects in the oil samples, 10–50× dimmer
    and spread in z, likely sit in the beads' ROIs and are absorbed as blur.
  - The report's M5 σ (0–32 nm) was learnt on dried or buffer beads; buffer here gives 0–28 nm, in
    agreement.

## 3. The grid at NA 1.51 and the bead-physics magnitude (±1000, 50 nm)

The bead-physics magnitude: cos θ_imm^γ, the 100 nm sphere's supercritical ring (none when matched), a
learnt NA cutoff, and the m ≠ 0 Zernikes.  It is compared with the free Zernike magnitude on the grid at
the nominal NA 1.50 and at 1.51.  The runs at 1.50 are another session's, `results/m5_media.jsonl`; its
Zernike runs reproduce this session's objectives to 0.1 nats.  The beads are the same in all four, so the
objectives compare in nats (+: the second is better).  The cutoff is bounded at ρ 1.05: at the bound it
lies outside the grid, so the NA reads "≥ 1.51", not ρ × 1.51.

| λ | medium | lens | Zernike 1.51 over 1.50 | physics over Zernike, 1.50 | physics 1.51 over 1.50 | edge ρ at 1.50 | physics over Zernike, 1.51 | edge ρ at 1.51 | NA_eff | γ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 600 | oil | 2D | +1 309 | −1 456 | +3 200 | 1.050 | +435 | 1.008 | 1.521 | 0.40 |
| 676 | oil | 2D | −2 160 | −931 | +858 | 1.041 | +2 087 | 1.001 | 1.512 | 0.40 |
| 600 | oil | 3D | −278 | −7 884 | +2 026 | 0.995 | −5 581 | 0.988 | 1.492 | 0.13 |
| 676 | oil | 3D | −1 988 | −1 793 | +840 | 1.009 | +1 035 | 0.995 | 1.503 | 0.38 |
| 600 | buffer | 2D | +9 938 | +4 795 | +8 987 | 1.050 | +3 843 | 1.050 | ≥ 1.51 (bound) | 0.44 |
| 676 | buffer | 2D | +4 602 | +1 487 | +4 643 | 1.050 | +1 528 | 1.050 | ≥ 1.51 (bound) | 0.42 |
| 600 | buffer | 3D | +8 420 | −3 110 | +9 072 | 1.050 | −2 458 | 1.015 | 1.532 | 0.43 |
| 676 | buffer | 3D | +5 480 | +1 770 | +5 105 | 1.050 | +1 395 | 1.050 | ≥ 1.51 (bound) | 0.39 |

- **The physics needs the grid beyond the nominal NA.**
  - At 1.50 it loses to the Zernike magnitude in all four oil conditions (−0.9 to −7.9 k nats).  There
    its cutoff runs into the bound in five of eight: the grid's edge was cutting the light it wanted.
  - The grid at 1.51 gains it 0.8–9.1 k nats in every condition.
  - At 1.51 it wins in six of eight, by 0.4–3.8 k nats, with the radial Zernikes replaced by three
    numbers.
- **In oil, the effective NA is 1.49–1.52, at the nominal 1.50.**
  - With no supercritical light, the cutoff is the objective's own NA.  It reads 1.503 and 1.492 with the
    lens, where it lies inside the grid (ρ 0.995, 0.988), and 1.512 and 1.521 without, where it lies at
    the grid's edge (ρ 1.001, 1.008).  The last two are therefore lower bounds; a grid at 1.53 would
    settle them.
  - The 3D estimates move by ≤ 0.011 between the two grids.
  - The report's "effective NA below nominal" (M2: 1.40–1.42 of 1.43) does not hold for M5's objective.
- **In buffer, the data want light out to the grid's edge and beyond.**
  - The larger grid gains the Zernike fit 4.6–9.9 k nats and the physics 4.6–9.1 k.  The physics' cutoff
    goes to its bound in three of four and reads 1.53 in the fourth.
  - The objective is the same as in oil, where the cutoff is about 1.50, so the cutoff here is standing in
    for something else: more supercritical light than a 100 nm sphere's ring delivers.
  - This is the opposite of M2's beads in water, which wanted about half the modelled supercritical
    light.
- **What the physics fit agrees on.**  It matches the Zernike fit on σ to ≤ 1.7 nm and on the main phase
  modes to ≤ 1.4 nm (3.6 in the 600 nm oil 3D outlier).
  - Its γ, the cos θ exponent, is 0.38–0.44 in seven conditions at either grid, across both colours, both
    media and both lens settings.  That is an objective property, and one the physics model makes
    measurable.
  - It loses at 600 nm with the lens, in both media (−5.6 k in oil, −2.5 k in buffer).  The oil one is the
    cluttered, high-σ outlier (γ 0.12–0.13), so the radial shape there carries something that is not
    the objective.
- **The z bend is no better** with either magnitude (9.6–21.6 nm over ±500): the magnitude is not what
  bends z.

## What it means for the next steps

- **Learning window:** ±1000 nm at 100 nm slices is as good as ±2000 for the model and costs half.
  - ±2000 adds nothing measurable and needs 240 pupil samples.  On 32 GB that means one bead per chunk
    and 100 nm slices for the 676 nm sets.
  - ±500 gives the tightest z near focus.  A held-out comparison on common slices would settle whether
    it is also right beyond.
- **Magnitude:** the bead-physics model on a grid somewhat beyond the objective's NA (1.51 for 1.50)
  is the better model in matched oil, and its NA there is the objective's.  On the nominal grid it is
  worse than the free Zernikes.  In buffer it needs a stronger or free supercritical term rather than a
  larger NA.
- **The spherical aberration differs between colours.**  A two-colour model needs per-channel spherical
  aberration, not one shared pupil phase.
- **The 3D lens:** its astigmatism is reproducible to ±1.6 nm of path, and can be fixed from one
  calibration and shared across colours.
- **σ in oil follows the clutter.**  For the blur, the buffer samples (or the oil ones with the faint
  objects masked out of the ROIs) are the ones to read.
- **Still open:** the z bend (5–24 nm at every window and with either magnitude).
