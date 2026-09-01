# Wiring smappy into microclaw

Nothing here is applied.  These are the microclaw-side changes that would make
`load_skill(name="smappy")` work in a session, kept in this repository so they
can be reviewed before anything lands in microclaw's own tree.

| file | what it is |
|---|---|
| `skills/smappy/SKILL.md` | the skill itself: copy to `microclaw/skills/smappy/SKILL.md` |
| `proposed/microclaw-wiring.patch` | two edits to microclaw's own files (below) |
| `proposed/test_smappy_skill.py` | tests for the skill: copy to microclaw's `tests/` |

The skill is written against microclaw's conventions -- frontmatter of exactly
`name` and `description`, directory name matching `name`, so
`microclaw/skills.py` accepts it -- and its catalog entry appears in the system
prompt automatically once the directory exists.

## What the patch changes

1. **`pyproject.toml`**: adds an `analysis` extra naming `smappy-smlm`.  Never a
   hard dependency: smappy ships compiled extensions, and microclaw drives a
   microscope perfectly well without ever fitting anything.  (The comment in the
   patch still says "not published yet"; smappy-smlm 0.1.0 is on PyPI now, so
   that sentence should go when the patch is applied.)
2. **`skills/smlm/SKILL.md`**: routes to the new skill.  Its "Microclaw does NOT
   perform localization fitting" becomes accurate rather than absolute --
   microclaw drives the microscope, and the fit is done either by smappy called
   from the session or by an external tool -- and the post-processing section,
   the software table and the per-technique suggestions gain smappy.

## Verified

Applied to microclaw at 5406e0f, the whole suite passed (2839 passed, 99
skipped), including the test that a built wheel ships the skill catalog.
