---
version: "1"
covers: [smappy.io.formats._load_csv, smappy.io.formats.guess_csv_mapping, smappy.io.formats.csv_columns]
---

## What it does

Opens a localization table written as text -- a `.csv`, `.txt` or `.tsv`
file with one localization per row -- by another program or by hand.
ThunderSTORM's export
([Ovesný et al. 2014](https://doi.org/10.1093/bioinformatics/btu202)) and
SMAP's own column names are recognised without being told; for anything
else, *columns* says which column is which.

What the file needs: a column for x and a column for y, **in nanometres**.
Everything else is optional.  A file with no frame column is read as if
every localization were in frame 0.

## How it works

```figure-setup
import tempfile
from pathlib import Path
from smappy.io.formats import _load_csv, guess_csv_mapping
headers = ["id", "frame", "x [nm]", "y [nm]", "sigma [nm]", "intensity [photon]",
           "offset [photon]", "bkgstd [photon]", "uncertainty [nm]", "chi2", "detector"]
row = "1,1,1520.3,880.1,131.2,2150,12.1,3.4,7.9,1.05,0"
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "thunderstorm.csv"
    path.write_text(",".join(headers) + "\n" + row + "\n")
    locs, info = _load_csv(path)
```

**1. The header.**  The first line is the header if any of its cells is
not a number; otherwise the file has none and its columns are called
*column 1*, *column 2*, ...  The cells are separated by commas, or by
semicolons if the first line has more of them.

**2. Which column is which.**  Unless *columns* says otherwise, each header
is cleaned -- lower case, the unit in square brackets taken off, spaces
made underscores -- and looked up in a list of the names other programs
use.  A header that is not in the list is not read.  x and y must be
found, or the file is refused with its headers listed.

**3. The numbers.**  Every row becomes a localization.  `frame`, `channel`
and `id` are read as whole numbers, everything else as single-precision
numbers, and a row whose x or y is missing or not a number is left out.

```figure A ThunderSTORM export, as this plugin reads it without being told anything: the headers it recognises, units taken off, become columns; the ones it does not are left behind.
def mapping(ax, pairs):
    ax.set_xlim(0, 1); ax.set_ylim(len(pairs) - 0.4, -1.2); ax.axis("off")
    ax.text(0.36, -0.9, "header in the file", ha="right", fontsize=8, color="0.4")
    ax.text(0.64, -0.9, "column in the table", ha="left", fontsize=8, color="0.4")
    for row, (source, target) in enumerate(pairs):
        ax.text(0.36, row, source, ha="right", va="center", family="monospace",
                fontsize=8, color="black" if target else "0.6")
        if target is None:
            ax.text(0.40, row, "not read", ha="left", va="center", fontsize=7,
                    color="0.6", style="italic")
            continue
        ax.annotate("", (0.62, row), (0.38, row),
                    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8))
        ax.text(0.64, row, target, ha="left", va="center", family="monospace", fontsize=8)
guessed = guess_csv_mapping(headers)
assert all(target in locs for target in guessed.values())
fig.set_size_inches(5.0, 3.0)
mapping(fig.subplots(), [(h, guessed.get(h)) for h in headers])
```

## In detail

**The names recognised**, after cleaning (`CSV_NAMES`):

| column | headers |
| --- | --- |
| `x_nm`, `y_nm`, `z_nm` | `x`, `y`, `z`; `xnm`, ...; `x_nm`, ... |
| `frame` | `frame`, `t` |
| `photons` | `intensity`, `photons`, `phot`, `n` |
| `xy_err_nm` | `uncertainty`, `uncertainty_xy`, `locprecnm` |
| `z_err_nm` | `uncertainty_z`, `locprecznm` |
| `sigma_nm`, `sigma_y_nm` | `sigma`, `sigma1`, `psfxnm`; `sigma2` |
| `background` | `offset`, `bg`, `background` |
| `background_std` | `bkgstd` |
| `channel`, `logl_rel`, `id` | `channel`; `llrel`; `id` |

A name used before a column was renamed is read under the new one, and when
two headers mean the same column the first one wins.

**Units are not converted.**  The unit in a header is taken off and not
looked at, so `x [px]` is read as if it were in nanometres.  A table in
camera pixels has to be converted first,

$$x_{\mathrm{nm}} = x_{\mathrm{px}}\, p ,$$

with $p$ the pixel size in nanometres -- in the program that wrote it, or
in a spreadsheet.  (The reader can do it when it is called from a script,
with `units="px"` and `pixelsize_nm`.)

**Frames are read as written.**  ThunderSTORM counts frames from 1, and
its frames keep those numbers here; nothing is shifted.  Only the linking of blinks looks
at the frame, and for it an offset of one does not matter.

**Separators.**  Commas and semicolons are recognised.  A tab-separated
file is not split into columns, so it is refused as having no x and y; save
it with commas.

## Parameters

### mapping
Needed only when the headers are not recognised, or to read a column the
list does not know.  Each pair is *header in the file* = *column in the
table*, separated by commas, and only the columns named are read -- so
name x and y too.  A file with no header has headers *column 1*,
*column 2*, ...  In the GUI, opening such a file from the File menu asks
for the mapping in a dialog instead.

## Output

The table, added to the session as a file of its own.  The text says how
many localizations were read; the history records the load with its
settings, *columns* among them.

## References

* Ovesný M, Křížek P, Borkovec J, Svindrych Z, Hagen GM. ThunderSTORM: a
  comprehensive ImageJ plug-in for PALM and STORM data analysis and
  super-resolution imaging. *Bioinformatics* 30, 2389 (2014).
  [doi:10.1093/bioinformatics/btu202](https://doi.org/10.1093/bioinformatics/btu202)
  -- the program whose export is read without a mapping.
