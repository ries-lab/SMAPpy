---
version: "1"
covers: [smappy.table_edit.rename_column, smappy.table_edit.remove_columns, smappy.table_edit.check_name, smappy.mathparse.rename_in, smappy.mathparse.rename_recipes, smappy.session.Session._follow_renames, smappy.render.with_columns_of, smappy.plugins.edit_columns.EditColumns.run]
---

## What it does

Renames one column of the localizations, or removes columns.  A column
named as another program or a script expects it, a column a step wrote
under a name you regret, columns nobody reads that make every save and
every [merge of blinks](plugin:Panels/Glossary) slower: this is the tool
for them.  It changes the table, so it is one step of the undo.

Column selectors in the other plugins follow at once: they list the
columns the table has now.

## How it works

**Rename.**  The column keeps its values and its place under the new name.
What reads it by name follows where it can:

* a derived column -- one [Math Parser](plugin:Analysis/Process/Math Parser)
  wrote, kept as a recipe -- is renamed with its recipe, and an expression
  that reads the column reads the new name (`photons * 2` becomes
  `N * 2`), so the column is still recomputed when blinks are merged;
* a filter bound on the column, on every layer, is set on the new name;
* a layer coloured or weighted by the column is coloured by the new name;
* a bound the file keeps with the table (Remove Localizations' hide flag)
  moves to the new name.

What names the column in a setting does not follow: a chain step, a
plugin's saved settings, a region table measured by an id column.  They
fail the next time they run, with an error that names the column they
did not find, and the setting is yours to change.

**Remove.**  The ticked columns are gone from the table.  A derived column
whose expression reads one of them keeps its values but is no longer
recomputed when blinks are merged, and a filter bound the file kept on one
goes with it, and so does a layer's filter on a removed column; the run
says which, in a warning.  A layer coloured by a removed column is drawn
without colour.

## In detail

**What cannot be renamed or removed.**  The columns the program finds by
name to draw, merge or split the table are not offered and are refused by
name: the positions `x_nm`, `y_nm`, `z_nm`; `xy_err_nm`, the precision the
picture is drawn with; `frame`, `filenumber` and `channel`; `group_id`,
`n_in_group` and `group_use`, which merging blinks writes and reads; `tid`,
a MINFLUX trace, by which blinks are merged where it is there; and
`region_filter`, which the region filter computes again from its conditions.

**A new name** is letters, digits and underscores, not starting with a
digit, so that an expression can read it; not a function or a constant of
an expression (`max`, `pi`), not one of the columns above, and not a
column the table already has.

**An undo** brings the old table back, the old name with it.  The
layers' bounds on the new name do not go back to the old one -- a layer
keeps no bound on a column its table does not have -- so after an undo the
filter on the column is set again by hand, and a layer's colour column,
left on the new name, is drawn without colour until it is set again.

## Parameters

### operation
Renaming is one column per run.  Several columns to remove are ticked
together, one undo step.

### column
Only the columns that may be renamed are listed.

### new_name
Keep the field's convention, `<quantity>_<unit>` and `<quantity>_err_<unit>`
for a precision: plugins look up a precision by that name.

### remove
None is ticked to start with; tick the columns to remove.

## Output

The table, renamed or without the columns, and a line in the history
saying what changed and what followed: the recipes rewritten, the bounds
moved or lost.

## Differences from SMAP

Based on SMAP's `RenameFields` and `RemoveFields`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)), one plugin here
with an *operation*, one column renamed per run as in SMAP.  The main
changes:

* A column cannot be renamed onto one the table has; SMAP moves the one
  that was there to `<name>_old`.  Its *keep old* (a copy under the new
  name) is left out: a copy is a Math Parser expression, `new = old`.
* The columns the program finds by name are refused; SMAP let a position
  be renamed away.
* Filter bounds, recipes and the layers' colour column follow a rename;
  SMAP drops the layers' filter on the old name.
* Any number of columns is removed at once, where SMAP has five slots.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
