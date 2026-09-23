---
name: batch
description: Write and run a smappy chain of plugins (.chain.yaml) and a batch job (.batch.yaml) that applies it to many localization files or image stacks, headless. Use when asked to analyse a set of files, build an analysis pipeline, or automate a sequence of smappy plugins.
---

# Chains and batch runs in smappy

`docs/batch.md` is the reference -- the file formats, the rules for grouping
and filtering, what gets written -- and its last section, "For agents", is
written for you.  Read it first.  In short:

1. `smappy-batch plugins [prefix]` lists plugins by path.
2. `smappy-batch describe <plugin path>` gives each field's flat dotted name,
   type, default, unit, range and choices.
3. Copy the closest of `docs/examples/*.chain.yaml` and
   `docs/examples/cells.batch.yaml`, and edit it.
4. `smappy-batch validate <job or chain>` until it reports no errors -- it is
   strict, and names the fields that do exist when one does not.
5. `smappy-batch run <job> --limit 1`, then read
   `<output>/<name>/results.json` and look at `<output>/<name>/figures/`.
6. Run the rest.  `summary.csv` has a row per file; a re-run skips files
   done with the same settings.

A step that needs a plugin which does not exist: write it following
`CLAUDE.md` ("Writing a plugin"), give it a `version`, return its numbers
as scalars in `Result.data`, and read `ctx.table()` if it should honour the
chain's grouping.  Then use it as a step.
