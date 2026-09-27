---
version: "1"
covers: [smappy.mathparse.parse, smappy.mathparse.names_in, smappy.mathparse.evaluate, smappy.mathparse.as_column, smappy.mathparse.remember, smappy.mathparse.apply_recipes, smappy.group.combine, smappy.plugins.math_parser.write_column, smappy.plugins.math_parser.check_field, smappy.plugins.math_parser.remember_expression]
---

## What it does

The Math Parser makes a new column of the localization table from an
expression in the columns it already has:

```
on_time_ms = n_in_group * 20
z_true_nm  = z_nm * 0.8
within     = (xy_err_nm < 25) & (sigma_nm > 100)
```

The result is an ordinary column.  The filter, the renderer, the colour
coding and every other plugin use it like any column the fitter wrote.  That
is what makes the plugin useful for everything there is no button for: a
refractive-index correction on z, an on-time in milliseconds, a ratio of two
fitted quantities, or a flag (0 or 1) marking the localizations that a range
filter on one column cannot express.

The expression is kept with the table, as a *recipe*.  That matters once the
localizations are grouped into blinks: the recipe says what the new column
means for a blink, so it is not averaged by accident, and it is still there,
with how it was computed, when the file is opened again.

## How it works

**1. Reading the expression.**  The text after *=* is read as a formula, not
run as a program: only the column names, numbers, operators and functions of
the table under *In detail* are allowed, and anything else is refused with a
message pointing at the piece that is wrong.  Every column the expression
names must be in the table; if one is missing, the message lists the columns
the table does have.

**2. Computing it.**  The formula is evaluated over whole columns at once,
one value per localization.  A comparison gives 1 where it holds and 0 where
it does not, so `within` above is a flag.  A result that is a single number
-- a constant, or `median(z_nm)` -- is copied to every localization.

**3. Writing it.**  With *apply to* set to *all localizations*, the column is
written for the whole table.  With *the selection*, only the localizations
of the current selection get the new value (the layer's filter, the ROI and
the slab together); the others keep what the column had, or get no value
(NaN) if the column is new.

**4. The recipe.**  The expression is stored in the table together with
*grouped*, the rule for what the column is on the grouped table.  Running
again into the same *new field* replaces its recipe.  An expression that
reads the field it writes -- `photons = photons * 2` -- is applied once and
kept as no recipe at all: re-evaluated at every grouping it would compound.
The column then goes on being combined per blink by its own rule (photons are
summed).

```figure-setup
from smappy.simulate import simulate
from smappy.group import group, combine
from smappy.mathparse import evaluate, remember
from smappy.plugins.math_parser import write_column
locs = simulate(n_frames=3000, seed=4)
grouped, group_index = group(locs)
flagged = write_column(locs, "bright", evaluate(locs, "photons > 1000"))
per_blink = {}
for rule in ("recompute", "any", "all", "mean"):
    remember(flagged, "bright", "photons > 1000", grouped=rule)
    per_blink[rule] = np.asarray(combine(flagged, group_index)["bright"])
n = np.asarray(grouped["n_in_group"])
```

**5. On the grouped table.**  When localizations are linked into blinks,
each ordinary column is combined by a fixed rule (positions averaged,
photons summed).  A computed column instead follows its recipe:

* *recompute the expression* evaluates the formula again on the grouped
  table, with the blink's own values.  This is right for a statement about a
  blink: `on_time_ms = n_in_group * 20` is then the blink's on-time.
* *mean*, *sum*, *minimum*, *maximum* combine the localizations' values, as
  for a measurement made per localization.
* *any localization* and *all localizations* are for a flag: the blink is 1
  if any (or all) of its localizations are.
* *leave it off* keeps the column off the grouped table.

The same column can mean quite different things per blink, which is why the
choice is asked for rather than guessed:

```figure The flag `bright = photons > 1000` on a simulated table, taken to the grouped table by four rules.  Left: the fraction of blinks that come out bright -- recomputing asks whether the blink's summed photons exceed 1000, which more blinks do than have all their frames above 1000.  Right: with *mean*, the flag becomes the fraction of a blink's frames that were bright, shown for the blinks of more than one frame.
fig.set_size_inches(7, 2.8)
left, right = fig.subplots(1, 2, gridspec_kw={"wspace": 0.35})
names = ["recompute", "any", "all", "mean"]
left.bar(range(4), [per_blink[r].mean() for r in names], color="#4c72b0")
left.set_xticks(range(4)); left.set_xticklabels(names, fontsize=8)
left.set_ylabel("fraction of blinks (mean value)", fontsize=8)
left.set_ylim(0, 1)
right.hist(per_blink["mean"][n > 1], bins=np.linspace(0, 1, 11),
           color="#dd8452", edgecolor="w")
right.set_xlabel("bright frames per blink, rule 'mean'", fontsize=8)
right.set_ylabel("blinks", fontsize=8)
for ax in (left, right):
    ax.tick_params(labelsize=7)
```

## In detail

**What an expression may contain.**  Column names are the table's own
(`x_nm`, `photons`, `xy_err_nm`, ...; the *new field* of an earlier run
counts as one).  Everything else:

| what | allowed |
| --- | --- |
| numbers | `20`, `0.8`, `1e3`; constants `pi`, `e`, `nan`, `inf`, `True`, `False` |
| arithmetic | `+`  `-`  `*`  `/`  `//` (integer division)  `%` (remainder)  `**` (power), unary `-` and `+` |
| comparisons | `<`  `<=`  `>`  `>=`  `==`  `!=`, one per bracket |
| combining flags | `&` (and), `^` (exclusive or), `~` (not), and the vertical bar for *or* |
| choice | `a if condition else b`, per localization |
| elementwise functions | `sqrt`, `abs`, `exp`, `log`, `log2`, `log10`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `arctan2`, `hypot`, `floor`, `ceil`, `round`, `sign`, `mod`, `rem`, `minimum`, `maximum`, `clip`, `where`, `isfinite`, `isnan`, `float`, `int` |
| reductions (one number) | `mean`, `median`, `std`, `sum`, `min`, `max`, `percentile`, `count` |

*Or* is written with a vertical bar: `(photons > 5000) | (sigma_nm > 150)`.
The functions are numpy's (`sqrt` is `numpy.sqrt`, `rem` is
`numpy.remainder`; `float` and `int` convert the type), called with
their arguments in order: `clip(photons, 0, 5000)`, `percentile(z_nm, 90)`,
`where(z_nm > 0, 1, -1)`.  The reductions ignore NaN, as `numpy.nanmedian`
does; `count` is the number of values, NaN included.  A reduction is taken
over the whole table even when *apply to* is *the selection*, and over the
grouped table when it is recomputed there.

**What is refused, and why.**  Each of these is a mistake that numpy would
answer with either a crash far from the cause or, worse, a plausible wrong
number, so it is caught with a message instead:

* `and`, `or`, `not` -- they ask a whole column whether it is true.  Write
  `&`, `|`, `~`.
* `0 < x_nm < 5`, a chained comparison -- the same question in disguise.
  Write `(0 < x_nm) & (x_nm < 5)`.
* `xy_err_nm < 25 & sigma_nm > 100` -- `&` binds tighter than `<`, so this
  would read as `xy_err_nm < (25 & sigma_nm) > 100`.  Put each comparison
  in brackets.  (In MATLAB, and so in SMAP, the precedence is the other way
  round.)
* `&`, `|`, `^` on a measured, non-integer column: a flag written by this
  plugin is stored as 0.0 and 1.0, so compare it first, `(within > 0) & ...`.
* A function that is not in the table above, a function named but not
  called, arguments given by name (`clip(a_min=...)`), text in quotes,
  `in`, `is`, `@`, `<<`, `>>`, and anything that is not arithmetic: an
  attribute (`x_nm.max()`), an index (`x_nm[0]`), a lambda, a list.

The expression is walked piece by piece rather than handed to Python's
`eval`.  It is saved with the table, in the history and in chain files, so it
travels between people, and evaluating a string from someone else's file
would let it do anything.

**The result as a column.**  A boolean result is stored as 0/1 in float32,
so the filter can take a range of it and grouping can reduce it.  A
floating-point result is stored as float32, like the rest of the table.  An
integer result -- `frame // 100` -- keeps its integer type, since a frame
number stops being exact in float32 above $2^{24}$.  A single number is
copied to every row; a result of the wrong length, or one that is not a
number, is refused.

**The new field's name.**  It must be a word of letters, digits and
underscores that does not start with a digit, so that a later expression can
use it.  The name of one of the functions is refused, and so are `group_id`
and `n_in_group`, which grouping writes and would overwrite.

**The recipe.**  Stored in the table's metadata, `metadata["derived"]`, in
the order the fields were defined (a field may be computed from an earlier
one), as the field, the expression and the rule.  It is saved in the file
with the table.  When the table is grouped, a computed column is left out of
the usual per-column rules and either reduced by its rule -- *mean* is the
plain average of the localizations' values, not weighted by their precision
as the positions are; *any* and *all* give 1 where the largest or smallest
value is not zero -- or recomputed from the expression on the grouped table.
Each time the table is grouped, the expressions are also evaluated again on
the ungrouped table, which is what keeps a field in `n_in_group` up to date
with the latest linking.  An expression in `n_in_group` or `group_id` needs
the table to have been grouped once, since grouping writes those columns;
before that the plugin says so.  A recipe that cannot be recomputed because a
column is missing is skipped with a message, and the table is still built.

**A field computed for the selection** is not a function of the table, so its
expression is never recomputed.  If *grouped* is *recompute the expression*,
the rule used instead is *mean*, and the text says so.

**The history.**  The last 20 pairs of field and expression are kept in
`math_parser.yaml` in the settings directory, most recent first, each pair
once, and offered under *history*.  A history that cannot be read or written
is treated as empty; it is never an error.

## Parameters

### field
Choose a name that says what the number is, with its unit, like
`on_time_ms`: it is what the filter and the colour coding will show.

### expression
Use *Preview* first: it computes the expression and reports the range and
median of the result without writing anything.  A precedence mistake gives
numbers, not an error.

### where
A field written only for the selection has NaN elsewhere, and is only
reduced, never recomputed, on the grouped table.

### grouped
*recompute* for an expression in `n_in_group` or in anything that is a
property of the blink; a combining rule for a measurement per localization;
*any* or *all* for a flag.

### recall
Picking an entry also restores the *grouped* rule it was used with.

## Output

* **The table**, with the new column.  The run is recorded in the file's
  history with its settings and can be undone (*File > Undo*), which also
  takes the recipe back.
* **The text**, the one line the history keeps: how many finite values were
  written, their range and their median, for example
  `on_time_ms: 24693 values, 20 to 360, median 60`.  With *the selection* it
  adds how large the table was.

A result whose range is not what was expected -- all zeros for a flag, or
values a thousand times too large -- is the sign of a mistake in the
expression; *Preview* shows it before the table has it.

## Differences from SMAP

SMAP's `Process/Modify/MathParser` does the same job.  What differs:

* **Parsed, not evaluated.**  SMAP puts `locs.` in front of every field name
  in the string and runs it with MATLAB's `eval`, so any MATLAB is allowed.
  Here the expression is read and only arithmetic over columns is run (see
  *In detail*), for the security reason above and so that a missing column
  or a precedence mistake is reported before anything is computed.
* **No dots.**  MATLAB needs `.*`, `./` and `.^` to stay elementwise; numpy
  is elementwise already, so `*`, `/` and `**` are written.  `&` and `|`
  bind tighter than a comparison here and looser in MATLAB, which is why an
  unbracketed comparison is refused.
* **Reductions** (`z_nm - median(z_nm)`) are allowed here as functions.
* **Where it is written.**  SMAP chooses one file or all files; here it is
  all localizations or the current selection (filter, ROI and slab).
* **Grouping.**  SMAP has a *regroup and filter* checkbox that links the
  whole table again after the calculation.  Here the column carries a rule
  for the grouped table instead, so recomputing an on-time or reducing a
  flag does not depend on relinking.
* **History.**  SMAP keeps 10 equations, and drops every earlier entry
  whose equation contains the new one, whatever field it went into; here it
  keeps 20, and drops only an entry with the same field *and* expression, so
  the same formula written into two fields is kept twice.
* **Precision.**  The result is float32, like the rest of the table, rather
  than double.
