---
version: "2"
covers: [smappy.io.formats._load_sml, smappy.io.formats._has_saveloc, smappy.session.read_and_group]
---

## What it does

Opens the `_sml.mat` files that SMAP
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)) writes, so that
data fitted or analysed in SMAP can be analysed here.  It reads the
localizations and the camera's pixel size; nothing else of SMAP's session
comes across.

Use it rather than [Auto](plugin:File/Load/Auto) for a `.mat` file whose
name does not end in `_sml.mat`: this plugin reads it as SMAP's format
whatever it is called.  Both the older MATLAB files and the `-v7.3` ones
(which are HDF5 inside) are read.

## How it works

```figure-setup
import tempfile
from pathlib import Path
import scipy.io
from smappy.simulate import simulate
from smappy.io.formats import SML_COLUMNS, SML_DROP, _load_sml
sim = simulate(n_frames=2000, seed=2)
n = len(sim)
# a file shaped the way SMAP saves one: saveloc.loc, frames from 1
loc = {"xnm": sim["x_nm"], "ynm": sim["y_nm"], "frame": sim["frame"] + 1.0,
       "phot": sim["photons"], "bg": sim["background"],
       "locprecnm": sim["xy_err_nm"], "PSFxnm": np.full(n, 130.0),
       "LLrel": np.zeros(n), "filenumber": np.ones(n), "xnmf": sim["x_nm"],
       "numberInGroup": np.ones(n)}
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "demo_sml.mat"
    scipy.io.savemat(path, {"saveloc": {"loc": loc,
                                        "file": {"info": {"cam_pixelsize_um": 0.107}}}})
    locs, info = _load_sml(path)
```

**1. The localizations.**  SMAP keeps its table in `saveloc.loc`, one field
per column.  Each field becomes a column, renamed to this program's name
where there is one (the table under *In detail*) and kept under its own name
where there is not.  A few columns are left behind: `filenumber`, which is
given again when the file is added to the session, and the columns this
program never reads (`xnmf`, `ynmf`, `xpixf`, `ypixf`, `xpixerr`, `ypixerr`,
`PSFypix`).

**2. Frames from zero.**  MATLAB counts from 1, so the frame numbers are
shifted down by one: SMAP's frame 1 is frame 0 here.

**3. The camera.**  The pixel size (`cam_pixelsize_um`) is read from the
first file's `info`, and in a `-v7.3` file also the frame count, the
conversion, the offset and the EM gain.  They are kept in the table's
metadata under `smap`.

**4. Linking.**  As with every loader, the blinks are then linked, unless
*link blinks* says not to (see [Auto](plugin:File/Load/Auto)).

```figure What happened to the columns of a file shaped like SMAP's, as this plugin read it: renamed where this program has a name for the quantity, kept as they are where it has none, and dropped where nothing here reads them.
def mapping(ax, pairs):
    ax.set_xlim(0, 1); ax.set_ylim(len(pairs) - 0.4, -1.2); ax.axis("off")
    ax.text(0.30, -0.9, "in the _sml.mat", ha="right", fontsize=8, color="0.4")
    ax.text(0.62, -0.9, "in the table", ha="left", fontsize=8, color="0.4")
    for row, (source, target) in enumerate(pairs):
        ax.text(0.30, row, source, ha="right", va="center", family="monospace",
                fontsize=8, color="black" if target else "0.6")
        if target is None:
            ax.text(0.34, row, "dropped", ha="left", va="center", fontsize=7,
                    color="0.6", style="italic")
            continue
        ax.annotate("", (0.60, row), (0.32, row),
                    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8))
        ax.text(0.62, row, target, ha="left", va="center", family="monospace", fontsize=8)
pairs = [(name, None if name in SML_DROP else SML_COLUMNS.get(name, name))
         for name in loc]
assert all(target in locs for _, target in pairs if target)
pairs = [(s, t + ("   (minus 1)" if t == "frame" else "")) if t else (s, t)
         for s, t in pairs]
fig.set_size_inches(5.0, 3.0)
mapping(fig.subplots(), pairs)
```

## In detail

**The renames** (`SML_COLUMNS`):

| SMAP | here | | SMAP | here |
| --- | --- | --- | --- | --- |
| `xnm`, `ynm`, `znm` | `x_nm`, `y_nm`, `z_nm` | | `LLrel` | `logl_rel` |
| `phot` | `photons` | | `logLikelihood` | `logl` |
| `bg` | `background` | | `xnmerr`, `ynmerr` | `x_err_nm`, `y_err_nm` |
| `locprecnm` | `xy_err_nm` | | `zerr`, `locprecznm` | `z_err_nm` |
| `PSFxnm`, `PSFynm` | `sigma_nm`, `sigma_y_nm` | | `photerr` | `photons_err` |
| `xpix`, `ypix` | `x_pix`, `y_pix` | | `PSFxpix` | `sigma_pix` |

`frame`, `channel` and `iterations` keep their names.  Every other field --
`numberInGroup`, a column SMAP's own plugins added, a derived field --
comes across under its SMAP name.  In particular SMAP's `numberInGroup` is
not this program's `n_in_group`: that is written by the linking here.

**Types.**  The frame is

$$\mathrm{frame} = \mathrm{frame}_{\mathrm{SMAP}} - 1 ,$$

stored as a 64-bit integer; `channel` and `iterations` become 32-bit
integers, and every other double-precision column becomes single precision,
which halves the memory and loses nothing a localization can carry.

**Units.**  SMAP stores positions in nanometres, and the table is marked as
being in nanometres.  Its pixel columns (`xpix`, ...), if the file has them,
come across as `x_pix`, ... beside the nanometre ones.

**Which files.**  The reader looks for `saveloc` in the file: a `.mat`
without it is not a SMAP localization file and is refused.  A file that
held several SMAP files at once is read as one, since its `filenumber` is
among the columns left behind.

## Output

The table, added to the session as a file of its own (or appended, with
*add to the open files*).  The text says how many localizations were read
and, if they were linked, how many blinks they make.  The load is written
into the session's history with its settings.

## Differences from SMAP

Based on SMAP's `Loader_sml`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).  The main
changes:

* Only the localizations and the camera numbers are read.  SMAP also
  restores the file's history, the ROI manager's sites and, when asked, the
  GUI parameters saved with it; those describe SMAP's own windows and
  plugins, which have no counterpart here.
* A file that holds several SMAP files is read as one table.
* Frames count from 0, and the columns are renamed to this program's names.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
  -- the program whose files this reads.
