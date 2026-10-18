---
version: "1"
covers: [smappy.plugins.combine_files.combine, smappy.plugins.combine_files.id_columns, smappy.plugins.combine_files.incompatibilities, smappy.plugins.combine_files.CombineFiles.run, smappy.session.Session._follow_files]
---

## What it does

Files opened together are one table, but each stays a file of its own: it
keeps its own [frame](plugin:Panels/Glossary) numbers, and everything that
works along time works one file at a time. Drift correction estimates a
curve per file. [Merging blinks](plugin:Panels/Glossary) (also called
grouping or linking) and *Link tracks* never join two files. That is right
when the files are different acquisitions: two cells, two samples.

It is wrong when the files are **one acquisition saved in pieces**: a movie
split into several TIFF files, or a measurement stopped and started again on
the same field. Then the drift should be one curve over the whole time, and a
blink at the end of one file can continue into the next. *Combine files*
makes such files one file. The frames and the time of each file continue
from the files before it, and every localization gets file number 0.

Use it before drift correction and grouping, and before you draw ROIs: an
ROI belongs to one file. Do not use it to pool different samples. Open them
together and leave them as files, or analyse each one and pool the
results.

It reads the localizations, whatever the layer shows, and changes all of
them. Undo brings the files back.

## How it works

**1. Order.** The files are taken in the order of their numbers, as the
Render tab lists them.

**2. Frames continue.** Each file's frames move up by the frames of the
files before it. A file with localizations up to frame 9999 takes 10000
frames, so the next file's frame 0 becomes frame 10000, plus *gap*.

**3. Time continues.** A time column (`time_s`) moves up the same way, by
how long the files before it lasted.

**4. Ids stay apart.** Track 5 of the first file and track 5 of the second
are different tracks. So an id column, such as `track_id` or `cluster_id`,
is numbered on from the files before it. An id of 0 means "in none" and
stays 0.

**5. Merged blinks are dropped.** The merged blinks (`group_id`,
`n_in_group`) were found file by file. Across the old boundary there may now
be a blink that continues, so the next grouping finds them again.

**6. One file.** `filenumber` becomes 0 everywhere, and the session lists one
file, named after the files it was made from ("a.tif + b.tif").

**7. Warnings, not refusals.** Files whose pixel size, frame time, format or
columns differ are combined anyway. Each difference is a reason to doubt they
are one acquisition, and is said as a warning.

## In detail

With the files taken in order and $F_j$ the last frame of file $j$ (its
highest frame number), file $k$'s frames move up by

$$\Delta f_k = \sum_{j<k} \left( F_j + 1 + g \right)$$

where $g$ is *gap*. Frames after a file's last localization, recorded but
empty, are not known to the table and are not counted.

A time column moves up by the summed durations of the files before. A file's
duration is its frames, $F_j + 1 + g$, times its frame time. If its last time
stamp is later than that, or the frame time is not known, the last time stamp
is used instead, so the next file never starts inside this one. The frame
time is the file's own, as the session recorded it, else the camera's.

Id columns are the whole-number columns whose name ends in `_id`. The
numbers above 0 of file $k$ move up by the sum of the highest id of each file
before it. MINFLUX's trace id `tid` has no "none" value, so all its numbers
move up, by the highest id + 1 of each earlier file. Some id columns stay as
they are: `roi_id`, and the columns written from regions' results (*Regions
to column*). They are an ROI's number, which is the same whichever file a
localization came from.

The table's metadata (its camera, its fit) is the first file's. The new file
entry keeps each original file's entry (`combined_from`). Its pixel size is
the first file's. It has a frame time only if all the files share one, and
its columns are all the columns any file had.

The files' camera frames are kept for the first file only. Its frames keep
their numbers; the other files' frames would need theirs moved too. Layers
that showed some of the files show all of them again.

## Parameters

### gap

0 makes the files one movie with no break. Set it to at least the merge
setting's dark frames, so that grouping never joins the last frame of one
file to the first frame of the next. Do this when the files were recorded
with a pause between them, long enough for the fluorophores to change.

## Output

The table, now one file, and a line saying how many files went into how
many localizations and frames. Each incompatibility is a warning in the
same line. "a.tif was read without z_nm: its rows hold padding there"
means the file had no such column, so its rows hold NaN there, not a
measurement.
