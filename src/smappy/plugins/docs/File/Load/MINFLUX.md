---
version: "1"
covers: [smappy.io.formats._load_minflux, smappy.io.formats._minflux_records, smappy.io.formats._minflux_from_json]
---

## What it does

Opens the localizations exported by an Abberior MINFLUX microscope
([Gwosch et al. 2020](https://doi.org/10.1038/s41592-019-0688-0)), so that
they can be filtered, rendered and analysed like any other table.

MINFLUX ([Balzarotti et al. 2017](https://doi.org/10.1126/science.aak9913))
does not take camera frames.  It finds a molecule by probing it with a
beam that has a dark centre, in several iterations, each one closer and
more precise than the last, and records every iteration of every
localization.  The export is a list of those records.  This plugin takes the
last iteration of each valid one.

It reads the three shapes the export comes in: a `.npy` file (a NumPy
structured array), a `.json` file with the same nesting, and a `.zip` that
holds either.  It does not read MINFLUX data saved as a MATLAB `.mat` file.

## How it works

```figure-setup
import tempfile
from pathlib import Path
from smappy.simulate import simulate
from smappy.io.formats import _load_minflux
sim = simulate(n_frames=3000, seed=2)
x, y = np.asarray(sim["x_nm"]), np.asarray(sim["y_nm"])
near = (np.abs(x - 7214) < 500) & (np.abs(y - 6049) < 500)
rng = np.random.default_rng(0)
n_good, n_bad = int(near.sum()), 300
n = n_good + n_bad
# the export's shape: a record per localization, one entry per iteration
iteration = np.dtype([("loc", "f8", (3,)), ("eco", "i4"), ("efo", "f8"),
                      ("cfr", "f8"), ("dcr", "f8"), ("itr", "i4")])
record = np.zeros(n, [("itr", iteration, (4,)), ("tim", "f8"), ("tid", "i8"),
                      ("vld", "?")])
true_x = np.concatenate([x[near], rng.uniform(6714, 7714, n_bad)])
true_y = np.concatenate([y[near], rng.uniform(5549, 6549, n_bad)])
record["vld"] = np.arange(n) < n_good          # the scattered ones failed
record["tim"] = rng.uniform(0, 600, n)
record["tid"] = rng.integers(0, 200, n)
record["itr"]["itr"] = np.arange(4)
eco = rng.integers(40, 400, n)
for k in range(4):
    spread = 60.0 / (k + 1) if k < 3 else 0.0   # each iteration closer
    record["itr"]["loc"][:, k, 0] = (true_x + rng.normal(0, spread + 1e-6, n)) * 1e-9
    record["itr"]["loc"][:, k, 1] = (true_y + rng.normal(0, spread + 1e-6, n)) * 1e-9
    record["itr"]["eco"][:, k] = eco // (4 - k)
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "minflux.npy"
    np.save(path, record)
    locs, info = _load_minflux(path)
```

**1. The last iteration.**  Of the iterations of a localization only the
last, the most precise, is kept.  Its position is converted from metres to
nanometres (`x_nm`, `y_nm`, and `z_nm` if any localization has a z).

**2. Valid ones only.**  The instrument marks each localization valid or
not (`vld`), by criteria set in its own software.  Those that are not, and
any with no position in the last iteration, are left out.

**3. Sorted by time.**  MINFLUX has no frames.  The localizations are put in
the order they were taken (`tim`), and `frame` is the rank in that order: 0
for the first, 1 for the next, and so on.  The clock itself is kept in
`time_s`, in seconds, and the trace a localization belongs to in `tid`.

**4. Photons and precision.**  The photon count is the last iteration's
`eco`.  The precision is not in the export, so it is estimated from the
photons (see *In detail*), as SMAP does.

```figure Left: a simulated MINFLUX export, read.  The localizations the instrument marked valid are read (red); those it did not are left out (grey).  Right: the precision given to each localization, from its photon count alone.
fig.set_size_inches(7.5, 2.8)
left, right = fig.subplots(1, 2)
bad = ~record["vld"]
left.plot(record["itr"]["loc"][bad, 3, 0] * 1e9, record["itr"]["loc"][bad, 3, 1] * 1e9,
          ".", ms=2, color="0.7", label="not valid: left out")
left.plot(locs["x_nm"], locs["y_nm"], ".", ms=2, color="#d62728", label="read")
left.set_aspect("equal")
left.legend(fontsize=7, frameon=False, loc="lower center", markerscale=3,
            bbox_to_anchor=(0.5, 1.0), ncol=2)
left.set_xlabel("x (nm)", fontsize=8); left.set_ylabel("y (nm)", fontsize=8)
order = np.argsort(locs["photons"])
right.plot(locs["photons"][order], locs["xy_err_nm"][order], color="#1f77b4")
right.axhline(25, color="0.6", lw=0.8, ls="--")
right.text(400, 26, "the default filter", ha="right", fontsize=7, color="0.4")
right.set_ylim(0, 30)
right.set_xlabel("photons (eco)", fontsize=8)
right.set_ylabel("xy_err_nm (nm)", fontsize=8)
for ax in (left, right):
    ax.tick_params(labelsize=7)
```

## In detail

**The precision.**  With $N$ the photons of the last iteration, the
lateral precision is set to

$$\sigma_{xy} = \frac{150\,\mathrm{nm}}{\sqrt{N}} ,$$

SMAP's number for the json export.  It is a stand-in so that rendering and
the precision filter have something to work with, not the precision of a
MINFLUX localization, which depends on the beam pattern and is usually
several times better.  `sigma_nm` is set to 150 nm for every localization,
for the same reason.  With the default filter of 25 nm on `xy_err_nm`,
localizations with fewer than 36 photons are hidden when the file opens.

**The columns.**

| column | from the export |
| --- | --- |
| `x_nm`, `y_nm`, `z_nm` | `loc` of the last iteration, times $10^9$ |
| `frame` | the rank in time |
| `time_s` | `tim` |
| `tid` | `tid`, the trace |
| `photons` | `eco` |
| `efo`, `cfr`, `dcr`, `efc`, `ecc`, `fbg` | the same, where the export has them |
| `iteration` | `itr`, the number of the last iteration |

**Linking.**  The blinks are linked by *link blinks* as for any other file,
by distance (50 nm) and consecutive `frame`.  Here that joins consecutive
localizations of the same molecule -- usually the localizations of one
trace -- into one; turn *link blinks* off to keep every localization.

**Positions** are kept as the instrument gives them; they are not shifted to
start at zero.  The file's entry in the session (`metadata["files"]`)
records a nominal pixel size of 100 nm, as SMAP does; there is no camera.

## Output

The table, added to the session as a file of its own.  The text says how
many localizations were read.

## Differences from SMAP

Based on SMAP's `Loader_minflux_json`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).  The main
changes:

* The `.npy` and `.zip` exports are read, the MATLAB `.mat` export is not.
* Only the last iteration is read; SMAP can load all of them.
* Only valid localizations are read, and the positions are not moved to
  start at zero (SMAP's *center*).
* The precision is $150\,\mathrm{nm}/\sqrt{N}$ for every export.  SMAP uses
  it for the json export, and for a `.mat` export works it out from the
  size of the last iteration's beam pattern instead.

## References

* Balzarotti F, Eilers Y, Gwosch KC, et al. Nanometer resolution imaging
  and tracking of fluorescent molecules with minimal photon fluxes.
  *Science* 355, 606 (2017).
  [doi:10.1126/science.aak9913](https://doi.org/10.1126/science.aak9913)
  -- the method.
* Gwosch KC, Pape JK, Balzarotti F, et al. MINFLUX nanoscopy delivers 3D
  multicolor nanometer resolution in cells. *Nat Methods* 17, 217 (2020).
  [doi:10.1038/s41592-019-0688-0](https://doi.org/10.1038/s41592-019-0688-0)
  -- the iterative scheme whose records the export holds.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
