---
name: usage-logs
description: Analyse the usage-log bundles SMAPpy users send (smappy-usage-*.zip, from Preferences > Usage log) and write a report of what is hard to find, missing, badly explained or broken in SMAPpy, with evidence from what the AI assistant tried. Use when asked to analyse, summarise or read the usage logs, the users' logs or the assistant transcripts.
---

# Analysing SMAPpy's usage logs

Colleagues who turned on the usage log (`src/smappy/usage_log.py`,
`docs/ai-integration.md`, "The usage log") send zips to an upload-only u:cloud
folder.  Each holds what was new since the sender's last send:
`events.jsonl` (SMAPpy's MCP tool calls and the help pages opened),
`logs/` (plugin runs and failures), `assistant/own-code.md` with the scripts
it copied and `assistant/missing-tools.md`, `transcripts/claude|codex/`
(the conversations, pictures removed), and `about.json`.  The goal is a
short, ranked list of what to change in SMAPpy, each item backed by what
people and their assistants actually did.

**These are colleagues' private conversations.**  The report stays on this
computer, outside the repository.  Nothing from them goes into a commit, an
issue or a PR -- not a name, a file path, a sample or a quote -- unless the
user asks for that item, and then only the finding, in your own words.

## 1. Digest the bundles

The bundles are fetched from the u:cloud drop folder itself, over WebDAV
with the collector's own login (the share link they were sent to can add
files but read none), into `<output folder>/bundles`, and digested there:

    python .claude/skills/usage-logs/digest.py --fetch "<output folder>"

The output folder is `~/Documents/SMAPpy usage analysis` unless the user
names another.  The folder's WebDAV address and the user are kept in its
`fetch.json` (look in memory if it is missing, and pass `--url` and `--user`
once); the password is a u:cloud app password read from the macOS Keychain
(service `smappy-usage-logs`) and nowhere else -- never ask for it, type it
or store it.  A refused login is said plainly: then the user makes a new app
password (u:cloud > Settings > Security) and stores it with
`security add-generic-password -a <user> -s smappy-usage-logs -w`.
Nothing on the server is deleted or moved.  Bundles named `...-test.zip`
(someone trying the link) are skipped.

Bundles that are already on disk (a synced folder) are digested without
fetching:

    python .claude/skills/usage-logs/digest.py "<bundles folder>" "<output folder>"

It prints how many bundles are new.  It writes `overview.md` (per sender
and pooled: SMAPpy tools called and failed, every `find_plugins` query and
what it found, source read, help pages, plugin runs and failures, the
assistant's own-code and missing-tools notes, and every conversation ranked
by friction) and `sessions/<sender>/<client>-<id>.md`, one timeline per
conversation.  Read `overview.md` whole.  Read the previous `report-*.md`
in the output folder too, if there is one.

## 2. Read the conversations

Read the timelines, most friction first.  Read every conversation from a new
bundle, and the old ones only where the overview points at them.  If there
are more than about five, give them to subagents (`Explore` or
`general-purpose`), a few each.  Give each subagent the files and this
section, and ask it to return one block per episode of friction:

- **asked**: what the user wanted, in a line;
- **happened**: what the assistant did about it -- the tools, the searches
  and what they returned, the scripts it wrote, the errors -- and how it ended;
- **kind**: one of *not found* (a plugin existed and was not found),
  *missing* (no plugin does it), *unclear* (found, but the page, a setting
  or an error did not say enough), *bug* (SMAPpy failed or computed
  wrongly), *tool gap* (the assistant could not do through SMAPpy's tools
  what the GUI can), *assistant* (the model's own mistake, with SMAPpy
  blameless), or *fine* (it went well -- worth knowing too);
- **evidence**: the timeline file and the lines, quoted briefly;
- **feature**: the SMAPpy plugin, page, tool or setting involved, as far as
  can be told.

The signs to look for: a `find_plugins` that returned nothing or the wrong
plugins; `read_source` or `Read` of SMAPpy's code because a page did not say;
a Python script of its own (`Write` of a `.py`, `python` in the shell) where
a plugin, `read_values` or `run_plugin` might have served; `run_plugin` tried
again and again with other settings; a failed tool call or plugin run; the
user correcting the assistant; a question left unanswered.

## 3. Check every finding against the code

The logs say what someone tried with the version they had.  Before
anything goes in the report, check each finding in this checkout (pull
first), and settle its *kind*:

- Does a plugin do it?  `smappy-batch plugins`, `src/smappy/plugins/`, the
  pages under `src/smappy/plugins/docs/`.  If it does, the finding is
  *not found* or *unclear*: then why was it not found?  Look at its
  description, its page's words, and what `find_plugins` matches for the
  query that was used:
  `python -c "from smappy.assistant.tools import Assistant; print(Assistant().call('find_plugins', {'query': '...'}))"`.
- Was it fixed since the sender's version (`about.json`, `git log`)?  Then
  say so and leave it out of the ranking.
- A *tool gap* or an *assistant* mistake may be SMAPpy's to fix all the
  same: a tool's description (`src/smappy/assistant/tools.py`), its
  `INSTRUCTIONS`, or the `AGENTS.md` written into the assistant's folder
  (`assistant/launch.py`, `prepare_folder`).

## 4. Write the report

`<output folder>/report-YYYY-MM-DD.md`:

1. **In short** -- three to five lines: how many people, conversations and
   bundles, the period, and the two or three things most worth doing.
2. **Findings, ranked** -- by how many people and conversations hit it, then
   by how much it cost them.  For each: a title that is the fix
   ("A histogram of photons per frame: Statistics does it, but
   find_plugins does not find it"), the *kind*, how often and by whom
   (counts, not names), the evidence (timeline files and short quotes),
   and the change proposed: a new plugin, words for a page or a
   description, a tool, a fix -- with the files it touches.
3. **New, recurring, gone** -- against the previous report: what is new,
   what came back, and what no longer appears.
4. **What went well** -- briefly: what people did through SMAPpy without
   trouble, so it is not broken by the next change.

Keep it to what a person reads in ten minutes.  Then tell the user the
three findings at the top, give them the report's path, and offer to work
on one.  A fix becomes its own change, with its own plugin page and tests,
as `CLAUDE.md` describes.
