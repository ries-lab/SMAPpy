# Performance

Measurements and the investigations behind them, moved out of `NOTES.md`
(which keeps the decisions).  Each number is for the machine named with it.

## Detection and fitting

M1 Pro, 46,005 frames of 200x200, spline fit, ROI 13: **35.9 s** (1283 frames/s,
~850k localizations).  Detection 12.9 s, fitting 22.2 s, I/O 0.8 s.

Findings worth keeping:

* the C++ single-pass maximum search beats `scipy.ndimage.maximum_filter` 5.2x
  (short-circuit evaluation rejects most pixels after one comparison)
* the 2-D DoG kernel has **rank 2**, so it is not separable -- subtracting the
  1-D kernels is wrong.  Subtracting the 2-D kernels and doing one convolution
  is exact and, below radius 4, faster than the fused separable passes
  (`_SEPARABLE_FROM_RADIUS`)
* SMAP's DoG window is `max(ceil(6*sigma-1), 3)`: radius 3 for sigma 1.2
* threading gives ~4x on filtering and maxima, ~6.5x on fitting
* the dynamic cutoff ran as a Python loop over frames (~2.7 s of the
  detection's 12.9 s, 80% of it numpy dispatch overhead).  It needs three order
  statistics, not a sort: `csrc/cutoff.hpp` finds them with nth_element, frames
  over threads, **9.3 -> 1.1 ms** per 200 frames.  A lexsort over the block,
  the obvious vectorisation, is slower than the loop it replaced.
* its quantiles are SMAP's nearest rank (`myquantilefast`); smappy had
  `np.quantile`'s interpolation only because it is numpy's default.  The
  cutoff rises slightly, dropping 0.01-0.14 % of the candidates, those at the
  threshold.  Note that clang fuses `a * b - c` into an FMA by default, so a
  kernel that has to match a reference bit for bit must prevent it
  (`cutoff.hpp`).

## Rendering

M1 Pro, 5M localizations into 2000x2000:

| | full field (100 nm/px) | zoomed 10x (10 nm/px) |
|---|---|---|
| histogram | 49 ms | 23 ms |
| fixed sigma | 156 ms | 40 ms |
| sigma from precision | 133 ms | 42 ms |
| the same, coloured | 314 ms | 48 ms |
| the same, one thread | 898 ms | 44 ms |

Filtering 5M localizations on four fields: 18 ms to build, 5 ms for the AND,
3 ms to compact.  Moving one slider costs 14 ms (one field recomputed, AND,
compact) plus 31 ms to render the 0.7M that survive -- live filtering at ~20 fps.

Grouping 844k localizations (dx = 50 nm, dt = 1): **0.31 s**, dominated by the
sequential linking.  The population median precision *rises* (17.5 -> 21.0 nm)
even though every group improves, because bright long-lived emitters collapse
into few rows and dim singles stay -- the per-group comparison above is the one
that means something.

End to end through the viewer, 5M localizations filtered to 3.7M, into a
1000x1000 canvas: **126 ms** for the full field, **3.9 ms** zoomed 10x, **0.4 ms**
zoomed 100x (select + render).  Without the index the zoomed cases cost ~42 ms,
because every thread walked all five million localizations to find the ones on
screen.

The spatial index over 5M localizations builds in 100 ms and answers in
0.02-0.12 ms.  It is a `lexsort` on 16-bit row and column keys: numpy radix-sorts
narrow integer keys and falls back to a comparison sort for wide ones, so the
obvious `argsort` on a combined cell id costs 1.5 s instead of 60 ms.

**The render threads draw a snapshot** (`LayerSnapshot`, 2026-10).  The 2D
worker used to be handed the view widget and the 3D worker the session, and
both walked the live layers from their own thread: a filter changed, a table
swapped or a block appended on the GUI thread meanwhile was read half-way
(the 3D worker caught "the table changed under us" and dropped the frame,
leaving the view waiting for good), a filter or a grouped table could be
*built* on the worker, and a widget referenced from a worker is how a Qt
object's last reference can be dropped off the GUI thread -- one route to the
deadlock behind `gui/collector.py`.  Now the GUI thread freezes what a render
needs and the worker gets that and nothing else (`PanelJob`, `Job3D`, one
`RenderWorker` class for both views):

* the shown table as `Localizations.snapshot()` -- a new dict over the same
  column arrays.  `extend` writes only past the end of a buffer or into a new
  one, and a live recipe replaces a column rather than writing into it, so
  nothing the snapshot holds is ever written;
* the filter's combined mask, by reference: `LocFilter` replaces its masks,
  never writes them.  Its compacted rows (`indices`, 10-20 ms at 5 M) are
  taken when the filter has them; when it has not, the worker compacts and
  the GUI thread hands them back on completion (`LocFilter.keep_indices`,
  kept only if the mask is still the filter's), so the next render does not;
* the spatial index: a `SpatialIndex` is immutable once built and is shared;
  a `GrowingIndex` is copied as new segment and offset lists over the same
  segments, because `append` and `_merge` change those lists one after the
  other and a query between the two paired a segment with another's offset;
* the median precision, a copy of `RenderSettings` and `DisplaySettings`
  (the panels edit them in place), an image layer's `ImageData` record (the
  pixels shared) and plane;
* for the 3D view, the table's `_gpu_token` is set on the session's table
  first (`table_token`), so every frame's snapshot shares the row cache and
  the GPU's buffers instead of uploading the table again.

`render_panels` is the whole 2D render over the snapshots (the old
`composite` merged into it), `render_3d` the 3D one; both are module-level
and run on any thread.

Measured on a shared 4-core container (load 4-12 throughout, so absolute times
swing by 30% between runs), 5 M localizations (a simulation tiled 30 x 30),
a 1000 x 800 canvas rendered as a 1.5x tile, 3D preview at 900 x 700.  Old
and new path alternated in one process, 15 rounds each, medians:

| | snapshot (GUI thread) | render wall, before -> after | render CPU, before -> after |
| --- | --- | --- | --- |
| 2D whole field, grouped (1.9 M kept) | 0.21 ms | 280 -> 293 ms | 596 -> 579 ms |
| 2D 3x zoom, grouped | 0.21 ms | 148 -> 155 ms | 225 -> 237 ms |
| 2D 30x zoom, grouped | 0.21 ms | 65 -> 67 ms | 58 -> 62 ms |
| 2D whole field, ungrouped (4.6 M kept) | 0.19 ms | 1962 -> 1880 ms | 1784 -> 1826 ms |
| 2D 3x zoom, ungrouped | 0.17 ms | 489 -> 487 ms | 640 -> 665 ms |
| 2D 30x zoom, ungrouped | 0.18 ms | 80 -> 79 ms | 80 -> 80 ms |
| 3D preview whole field, grouped | 0.22 ms | 378 -> 379 ms | 578 -> 586 ms |
| 3D preview 10 um crop, grouped | 0.21 ms | 19.7 -> 20.3 ms | 21.4 -> 21.6 ms |
| 3D preview whole field, ungrouped | 0.17 ms | 613 -> 632 ms | 778 -> 790 ms |
| 3D preview 10 um crop, ungrouped | 0.15 ms | 33.1 -> 31.1 ms | 32.6 -> 33.9 ms |

Before and after draw identical pictures (the planes compared exactly), and
the differences in the last two columns change sign between rows and between
runs (+-10%): they are the machine, not the snapshot, whose own cost is
0.03 ms on an idle moment and at most 0.25 ms under load -- under 0.5% of
the cheapest render in the table.  The one cost that moved threads is the mask's AND after a filter
change, 3 ms at 5 M, now on the GUI thread; the filter panel's count already
computed it there, so in the app it is not new.

## The 3D view queries the same index

The 3D view did not, and every frame walked the whole table to find the slab's
localizations, so cropping to a 300 nm box did not make it faster -- on a 46 M
localization file (35.4 M after the filter) a drag frame was ~0.5 s for a tiny
crop and ~1.2 s for the whole field.  Three things, in the order they cost:

* the **depth histogram** was a second full `project_layer` per layer per frame
  -- the projection the render had just done and dropped.  It is a shape, not a
  count (the panel hides the axis), so `depth_sample` takes `HIST_POINTS` of
  the slab's rows.  This one is paid by the GPU engines too: it runs on the CPU
  whatever `Projection.engine` says, which is why "slow in every mode" was the
  symptom.  (The histogram has since been removed from the 3D window
  altogether, 2026-10, in the redesign of its controls.)
* `slab_candidates` asks the index for the slab's bounding rectangle instead of
  reading every row.  The candidates are **not** sorted back into table order:
  sorting 3 M indices costs 126 ms, more than the tidier gather saves.
* `Projection.apply` and `Slab.to_local` promoted float32 tables to float64 and
  `apply` built an (N, 3) stack for the matmul -- 2.2 GB of temporaries per
  call.  Row by row in the table's own precision is 2.2x faster and half the
  memory; nanometres over a few hundred microns resolve to ~4 pm in float32,
  far below any rendered pixel.

A drag frame, 900x700, same file: whole field **1176 -> 543 ms**, a 10 um crop
**1123 -> 384 ms**, a 300 nm crop **519 -> 5.0 ms**.  The whole-field case is
now the render itself, which is the 2 M points it is asked to draw.

## The GPU engines, and why they were not faster

Three things kept them from being the fast proxy they are meant to be, all
found by running them on a 46 M localization file for the first time:

* the splat compute shader dispatched a **1-D** grid.  A dimension is capped at
  65535 workgroups, so at 256 threads it covered 16,776,960 points and a larger
  table failed validation outright -- "GPU" mode could not open this file at
  all.  The grid is 2-D now (`dispatch_grid`), and the count is compared as a
  u32: it used to ride in `depth.w` as an f32, which stops counting exactly at
  2^24 -- the same 16.7 M boundary, by coincidence of 256 x 65535.
* the selection was the whole filtered table, every frame, with the slab
  applied per point **inside** the shader.  So the GPU repeated what the CPU
  engine used to: a crop cost what the full field cost.  `selection_rows`
  narrows it through the index and memoises it on (filter, slab) -- neither of
  which a rotation changes, so a drag reuses the array *and* the GPU buffer
  built from it, which is keyed on that array's identity.
* points mode sorted back to front for the alpha, keyed on the three angles, so
  every frame of a drag re-sorted on the CPU and re-uploaded.  This made a
  small selection slower than a large one -- 1 M points ran at 8 fps where 2 M
  ran at 45, because 2 M was over `PREVIEW_POINTS` and skipped the sort.  The
  sort is now skipped while previewing, which is exactly when it does not
  matter.

Rotating a 900x700 preview, same file, after all three: a 10 um crop **3.3 fps
on the CPU, 33.6 on the GPU**; a 1 um crop 150 / 108.  Below ~100k points the
CPU wins again -- the GPU round trip is ~8 ms of readback whatever it drew, so
a 300 nm crop is 246 fps on the CPU and 117 on the GPU.

## The preview point budget

The full field was 8.4 fps because 18 M points were drawn into a 450x350
preview -- **116 points per rendered pixel**, so 99% of the work could not
change one.  A drag frame now draws a sample (`thinned`, `PreviewBudget`); a
final frame never does, so the image anyone measures or exports is still the
whole table.

**Thinned, not merged.**  An SMLM image is a density map, and dropping a fixed
fraction scales every pixel alike: a preview is what a shorter acquisition
would have looked like, which is a degradation that cannot be misread.
Merging neighbours into superpixels holds the total but *moves* it, and can
make a sparse filament look continuous -- an artifact that appears only while
the mouse is down and is gone before it can be checked.  Merging becomes right
when the ratio is 100x+ rather than the 9x wanted here, and then the criterion
is to merge only *below the rendered pixel*: the kernel is already floored at
`max(min_sigma, min_sigma_pixels * pixelsize)`, so points closer than that
cannot be told apart in the image.  That is an LOD octree, and it is the build
to do at 10^8 localizations, not at 10^7.

**Evenly spaced, not drawn.**  `choice(n, k, replace=False)` permutes n first
(73 ms at 35 M) and returns an unsorted list, which then scatters the gather.
`linspace` costs the k it returns, comes back ascending, and is deterministic,
so the same budget asks the GPU for the same selection and the buffer is
reused.  It is a systematic sample and only as good as row order being
unrelated to position -- measured against a random draw of the same size on a
3 M row file, sampled density over a 64 x 64 grid differed by L1 0.2305 against
the draw's 0.2349, on the grouped table as well as the raw one.

**Timed, not fixed.**  No constant serves a base M1 and a 3090, and the same
machine is an order of magnitude apart between a sub-pixel blob and a fat one.
The budget is the smaller of what can be seen (13 points per rendered pixel)
and what the machine got through last frame at `TARGET_FPS`.  The visible cap
applies before anything has been timed, and final frames are timed too, so the
first frame of the first drag already has a rate to work from.  The loop is a
contraction while the fixed per-frame overhead is under the target; past that
it falls to `minimum` and simply runs slower, which is the honest outcome.

Rotating a 900x700 preview on the 46 M file, before -> settled:

| slab | CPU | GPU planes | GPU points |
| --- | --- | --- | --- |
| whole field | 6.2 -> **25.8** | 9.0 -> **24.5** | 5.4 -> **23.3** |
| 10 um crop | 4.9 -> **25.5** | 31.1 -> **27.1** | 25.4 -> **28.1** |
| 1 um crop | 123 -> 122 | 102 -> 105 | 125 -> 116 |

The 1 um row does not move because the slab holds fewer points than the budget:
nothing is thinned, and a crop is drawn whole.  The whole field on the CPU
settles at 1.4 points per pixel, which is visibly sparse -- that is what a CPU
can do at 25 fps with 18 M points, and the GPU holds 11/px for the same rate.

Rendering at *display* resolution rather than a fixed nm/pixel is what bounds
this: zoomed out the kernels collapse onto the `mingausspix` floor, zoomed in
there are fewer localizations in view.  The zoomed numbers are floored by every
thread scanning all 5M localizations to find the ~50k in view; a spatial index
would remove that.

## Opening a 57 M localization file

**Opening a 57 M localization `_sml.mat`** was 134 s and is 64 s.  What
looked like 79 s of linking was 68 s of `np.lexsort` and 5.5 s of the walk:
the sort, not the link, was the cost.  It is a radix sort now -- 2.1 s, the
same permutation to the element.  numpy has a radix sort in it for uint8 and
uint16 and a comparison sort above that (0.18 s against 9.9 s on 20 M of the
same numbers), so the key goes in 16-bit digits, least significant first.
Everything above the frame's low 16 bits -- the block keys and the frame's
high bits -- is not sorted at all but computed per localization as a
partition, and the partitions sort independently across threads.  Nothing
assumes an input order: the partition is a key each localization carries and
the pass that gathers them is itself a counting sort.
Not taken: numpy's own stable sorts LSD-style over x, frame, block (exact,
72 s -- they are comparison sorts at those widths); block+frame first then
threaded lexsorts of x inside each bucket (exact, 5.1 s); discretizing x to
one 16-bit digit (0.4 s faster, 0.19% of localizations change places).

The linking ran in chunks for a while (`GroupSettings.link_chunks`,
`smappy._group_chunked`): 13 s to 6 s on this file, and not exact -- about 1
localization in 20,000 landed in another group, because the sequential walk
claims localizations as it goes and a chunk cannot know about the claims
reaching into it.  It is gone, since opening no longer waits for the merge
(see "Merging blinks in the background", below), and the walk is the exact
one again.

**`combine` does not parallelize, and that was worth finding out.**  The
columns are independent and numpy drops the GIL, but a column in flight holds
a float64 output over the groups and a float64 temporary over the table --
0.78 GB on that file, where the table is already 8 GB.  Threads against
seconds: 1: 12.5, 2: 10.5, 3: 12.9, 4: 14.8, 6: 23.1, 8: 31.8.  Two threads
win 2 s and each further one loses more, on a 34 GB machine; on the 8-16 GB a
microscope PC has, the second thread is what starts the swapping.  So it runs
one column at a time.  What did help there was the per-group sort:
`np.argsort(kind="stable")` on the group ids is 3.8 s, by 16-bit digits 2.1.

`combine` is compiled now (`csrc/combine.hpp`): 17.3 s to 9.1 s, and the peak
for the whole open from 20 GB to 9.7.  The memory is the point.  numpy wants
the column as float64 and the weighted product as an array, so a float32
column costs two 457 MB temporaries before `np.bincount` allocates its own
output; in C++ the cast, the weight and the sum are one pass and there are no
temporaries.  It also threads properly, which the numpy version could not:
every thread wanted its own accumulator over all 40 M groups, where
partitioning by *group* gives each thread a disjoint slice of one output and
nothing to coordinate.  All six modes (mean, sum, precision, quad, min, max)
and the per-coordinate weighting are unchanged -- C++ accumulates, Python
finishes, and the numpy path stays as the reference the tests compare to.

What was left of the 56 s: 32 s read, 9 s `combine`, 6 s linking, 2 s sort,
6 s indices and filters.  The file now opens after the read and the indices;
the merge -- about 24 s with the exact walk -- follows in the background.

## Merging blinks in the background

Every file was merged before it opened, so that its layer could open on
blinks, and that is what the chunked linker was for.  A layer still shows
merged blinks by default, but in the GUI the file opens at once and the
merge runs in a worker (`gui.merging`) with the stage and a clock over the
picture and in the status bar; the layer draws nothing until the blinks are
there, and a run that reads the table waits for them.  The worker gets the table read-only (`Tables.link_job`, a
snapshot of a live one) and hands the answer back to be kept on the GUI
thread (`Tables.take_grouped`), which writes `group_id` and `n_in_group`
back.  Below 300,000 localizations it merges in place, on first read.
Headless (a script, a chain, a batch) a layer merges on first read.

Measured on simulated tables (`smappy.simulate`, tiled side by side; a
4-core machine shared with other jobs, so the spread is wide), the whole
`group()` with the exact walk against the chunked one: 2 M at 5 a frame,
0.75-0.8 s against 0.6-0.8 s; 5 M at 50 a frame, 2.5-3.2 s against
2.8-3.4 s; 5 M at 500 a frame, 4.5-6.2 s (the walk 4.3-6.3 s, chunked
3.3-4.0 s); 20 M at 200 a frame, 21.8 s for both.  Done once, in the
background, a second or two of difference does not pay for an approximate
grouping and 360 lines.

## Large tables: 10 files of 10 M

Does one table for all files cost a plugin that runs on one of them, and
where does a table of 10^8 localizations spend its time?
`scripts/bench_large_table.py` times the session's own paths: the layer's
filter, a plugin's context and selection, the render, the linking.  A 2D
fit's nine columns (3.6 GB at 10^8), blinks of about two frames, 50 000
frames over a 40 um field per file; a 4-core Xeon at 2.1 GHz, 15 GB.
Against it, the same file open alone -- what a table per file would give:

| | 10 files, 10^8 | that file alone, 10^7 |
|---|---|---|
| open: table, spatial index, layer | 18.6 s (peak 6.4 GB) | 0.64 s (1.0 GB) |
| the layer's filter from its bounds, and the AND | 0.23 + 0.12 s | 0.02 + 0.01 s |
| show one file (`isin` over `filenumber`, AND) | 0.44 s | 0.08 s |
| move one slider, AND | 0.28 s | 0.02 s |
| plugin context | 0.1 ms | 0.1 ms |
| `selection.apply`, 8.8 M rows, every column | 0.50 s | 0.27 s |
| the same, three columns | 0.06 s | 0.05 s |
| Localization Statistics on them | 2.0 s | 2.9 s |
| render the file, full field 2000x2000 | 1.3 s | 1.1 s |
| render it zoomed 10x | 0.16 s | 0.12 s |
| render every file, full field / zoomed | 18.6 s / 0.42 s | -- |
| link the file into blinks | 5.7 s | 7.4 s |
| link the whole table (a layer that shows blinks) | **86 s, peak 13.1 GB** | 6.3 s |

(The Statistics and linking rows differ between the columns by run-to-run
noise on a shared machine, not by the table around them.)

**A plugin pays for its selection, not for the table.**  The context is
free, and the copy of the selected rows, the plugin and a zoomed render
cost the same with the other nine files beside the one it reads.  What one
table costs a plugin on one file is the filter: every bound and the file
choice is a pass over all 10^8 rows, 0.3-0.4 s instead of 0.02-0.08 s,
which makes a slider a few frames a second instead of live.

**The table's size is paid elsewhere, and most of it would be paid per file
too.**  Opening is the spatial index over every row (a `lexsort`, 18 s).
Linking is per file already (`block_fields`), so the whole table costs ten
files' linking, 86 s -- but all at once and before a layer that shows blinks
draws anything, which it does by default, and at a peak of 13 GB against a
3.6 GB table.  A full-field render of every file is every localization
drawn, 18.6 s; the GUI's drag frames are thinned (*The preview point
budget*), the final frame is not.

**Copying every column is not where a plugin's time goes.**  `apply` copies
nine columns where Statistics reads a few: 0.5 s of its 2 s.  A selection
that indexes a column when it is first read would save that, about a
quarter of a light plugin and less of a heavy one -- worth doing, not first.

What would help, in order:

* **Link only the files a layer shows, when it shows them**, cached per
  file -- linking already never crosses a file, so the per-file results are
  the same numbers.  It turns 86 s on opening into 6-7 s per file looked
  at, and the 13 GB peak into a 10^7 one.
* **Filter masks over a file's rows only.**  `add_file` appends, so a file
  is one contiguous block of rows; a layer showing one file could compute
  its bounds over that slice (a view, no copy: 0.0 ms above) and leave the
  rest False, which makes a slider cost what it costs on the file alone.
  It needs the rows to stay grouped by file, checked once per table change.
* **A lazy selection** (`Selection.apply` indexing on first read).
