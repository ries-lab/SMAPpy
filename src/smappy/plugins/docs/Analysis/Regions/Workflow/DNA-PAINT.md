---
version: "1"
covers: [smappy.chain.ChainPlugin.run, smappy.blocks.BlockPlugin.run]
---

## What it does

Takes a [DNA-PAINT](plugin:Panels/Glossary) measurement from its
localizations to its molecules in one run: the drift corrected on the
binding sites themselves, the sites found, each site's binding kinetics
measured, and the molecules -- one per site, at its centre -- written as a
new file.  It is the analysis of [Picasso](https://doi.org/10.1038/nprot.2017.024)
(the Jungmann lab's DNA-PAINT software) as SMAPpy's plugins do it.

It is a [chain](plugin:Panels/Glossary) (also called a pipeline or
workflow): the plugins it runs, one after the other, each with its own
settings in a section of this panel, folded behind *show the steps*.  Four
settings are shown at the top:

* *site radius* -- [Find clusters](plugin:Analysis/Regions/Find clusters)'
  radius for its *local maxima*, in both rounds;
* *localizations per site* -- their *minimum size*;
* *drift knot spacing* -- how finely in time
  [Fiducials](plugin:Analysis/Drift/Fiducials) follows the drift;
* *molecules by* -- [Molecular mapping](plugin:Analysis/Regions/Molecular mapping)'s
  *method*, centres or G5M.

The pages of the plugins explain the methods; this one what running them
together does.  Because it is a chain, it can be edited (*edit steps*),
saved under another name, and run over many files in a batch.

Use it on DNA-PAINT data whose sites are separate structures -- DNA
origami, proteins labelled sparsely enough that each site stands alone --
and whose localizations have a precision (`xy_err_nm`).

## How it works

**1. Localizations.**  The *layers* step shows the first layer's
localizations, not merged blinks: sites and molecules are made of
localizations.  Its filter stays as the Render tab has it.

**2. Coarse drift.**  [RCC](plugin:Analysis/Drift/RCC) corrects the drift
to a few nanometres, so that a site's localizations come together.

**3. Sites.**  Find clusters with *local maxima* finds each binding site,
and its *frame analysis* drops the clusters whose localizations bunch in
time: an imager stuck to the glass, dirt.

**4. Drift from the sites.**  Fiducials with *markers* set to *structures*
and the sites (`cluster_id`) as markers: every site is a marker, its
localizations fitted with one smooth drift curve and an offset for each
site.  This is the second round of Picasso's drift correction from DNA
origami, on the binding sites, and corrects the drift to a fraction of a
nanometre.

**5. Sites again.**  The sites found again on the corrected positions,
where each is tighter.

**6. Each site's kinetics.**  The *each site* block runs
[Binding kinetics](plugin:Analysis/Measure/Binding kinetics) once in every
site: its binding events, bright and dark times and qPAINT index, kept as a
row per site (`cluster_id`) for
[Filter regions](plugin:Analysis/Regions/Filter regions), the Regions tab
or [qPAINT](plugin:Analysis/Regions/Summarize/qPAINT).

**7. Molecules.**  Molecular mapping puts a molecule at the centre of each
site and adds them as a file, *molecules of* the file's name: RESI's step
(resolution enhancement by sequential imaging).  For RESI, run the chain
on each imaging round and register the rounds' molecules with
[Register layers](plugin:Analysis/Register/Register layers).

## In detail

**Why two rounds of sites.**  The sites of step 3 are found on positions
corrected only by RCC, which leaves a few nanometres -- enough to smear a
site over its neighbour's radius.  They are good enough as drift markers:
Fiducials leaves out the sites whose localizations disagree with the
others' drift.  After the drift from the sites, the clouds are as tight as
the precision, and step 5 finds them again.

**The site radius.**  About twice the localization precision, as Picasso
advises: on the simulated origami of the
*DNA-PAINT origami (20 nm grid)* preset (3 nm precision, sites 20 nm apart)
8 nm finds 818 of 823 sites; 10 nm, Picasso's default, joins neighbouring
sites.

**G5M.**  With *molecules by* set to G5M, each site's cluster is fitted
with a Gaussian mixture, which splits a cluster that holds two sites the
radius could not separate, and drops molecules Picasso's filter rejects.
It is slower: a few hundredths of a second a site.

**Nothing changes until the end.**  The chain runs on a copy of the
session; the corrected table, the sites' results and the molecules reach
the session only when the whole chain has succeeded.

## Output

* **The text**: one line per step -- the drift, the sites and how many
  failed the frame analysis, the markers left out, the sites measured, the
  molecules.
* **The figures** of each step.
* **The table**: drift-corrected, with `cluster_id` and `n_in_cluster`.
* **Per site**: the binding kinetics, a region table of `cluster_id`.
* **A file of molecules**, with each one's position, precision
  (`xy_err_nm`), localizations and binding events.
* **The log**: one entry for the chain, with every step's settings.

## References

* Schnitzbauer J, Strauss MT, Schlichthaerle T, Schueder F, Jungmann R.
  Super-resolution microscopy with DNA-PAINT. *Nat Protoc* 12, 1198-1228
  (2017). [doi:10.1038/nprot.2017.024](https://doi.org/10.1038/nprot.2017.024)
  -- Picasso, the analysis this chain follows.
* Reinhardt SCM, Masullo LA, Baudrexel I, Steen PR, et al.
  Ångström-resolution fluorescence microscopy. *Nature* 617, 711-716
  (2023). [doi:10.1038/s41586-023-05925-9](https://doi.org/10.1038/s41586-023-05925-9)
  -- RESI.
