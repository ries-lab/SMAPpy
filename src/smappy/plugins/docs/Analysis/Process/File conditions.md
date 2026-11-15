---
version: "1"
covers: [smappy.plugins.file_conditions.conditions_of, smappy.plugins.file_conditions.as_column, smappy.plugins.file_conditions.FileConditions.run]
---

## What it does

Several measurements opened together usually differ in something -- the
laser power, the buffer, the sample, the day -- that the analysis should
group by: compare the traces at 10 µW against those at 40 µW, the cells in
one buffer against the other.  This plugin writes those conditions onto
each file's localizations, one attribute per condition, so that a filter,
a colour or *Compare groups* can tell them apart.

The conditions come from the **file names**, by a pattern -- names like
`cell1_10uW` and `cell2_40uW` give a `power` of 10 and 40 with the pattern
`(?P<power>\d+)uW` -- or are **typed**, one line per file: `cell1:
buffer=MEA, day=1`.

## How it works

**1. From the names.**  The *from the names* pattern is a regular
expression; each named group, `(?P<name>...)`, is a condition, its value
what the group matches in each file's name.  A file whose name does not
match has no value.

**2. Typed.**  Each line of *typed* is `part of a name: key=value, ...`; the
values go to every file whose name holds that part.  Typed values come
after the pattern's, and replace them where both give one.

**3. Onto the localizations.**  A condition whose values are all numbers
becomes a numeric attribute (NaN where a file has none).  One with a word
among its values becomes a whole number per file -- 0, 1, ... in the order
the words first appear, -1 where a file has none -- with the words kept in
the table's metadata, so *Compare groups* names its groups by them.

## Parameters

### pattern
A few patterns: `(?P<power>\d+)uW` for "…_40uW…", `_(?P<sample>[A-Z]\d+)_`
for "…_B3_…", `(?P<date>\d{6})-` for Imspector's dates.  The form shows,
below it, what each file gets.

### assign
For what the names do not say.  The part before the last colon is matched
anywhere in the name; keys must be usable as attribute names (letters,
digits, underscores).

## Output

An attribute per condition on the localizations; the text lists each
file's values.
