# AI integration

How an AI assistant works with SMAPpy: what it can do, through what, and
why it is built this way.  Decided in 2026-10; the first two steps are
built (`src/smappy/assistant/`, and the GUI's side in
`src/smappy/gui/assistant_server.py` and `assistant_pane.py`), the rest is
the plan.

## What it is for

Two uses, one set of tools:

1. **Beside the GUI.**  The assistant answers questions about the data and
   the methods, suggests the plugin that answers a question and opens it
   with its settings filled in, builds chains with the user, and -- on a
   switch -- runs them itself.
2. **On its own.**  The user says what they want to know; the assistant
   uses the whole of SMAPpy headless to find it out: picks the methods from
   their pages, checks them on simulated data with a known truth, runs them,
   judges the results against what the pages say a good and a bad one look
   like, scans parameters, and draws exactly the figure asked for.  What it
   did ends as a `.chain.yaml` that the GUI and `smappy-batch` run again.

## Architecture: the AI in its client, the tools in SMAPpy

SMAPpy contains no model and no API key.  It serves its tools over MCP (the
Model Context Protocol), and whatever AI the user already has connects to
it:

* **Model-agnostic.**  Any MCP client works -- the Claude Code app or CLI,
  Codex, and the clients that run DeepSeek, Qwen or Kimi models; local models
  later, with nothing to change here.
* **No keys, no cost in SMAPpy.**  A panel with a model of its own would need
  an API key: Anthropic does not let a third-party program sign in with a
  claude.ai subscription ("Unless previously approved, Anthropic does not
  allow third party developers to offer claude.ai login or rate limits for
  their products", the Agent SDK's overview), and most users have the
  subscription, not a key.
* **Offline stays offline.**  The server is started only when asked
  (`smappy-mcp`, or *Assistant > Allow connections* in the GUI); nothing else
  imports it, and SMAPpy works as before without a network.

The client runs on the same computer as SMAPpy: the server speaks over
stdio (headless) and a local socket (the running GUI).  A cloud session
cannot reach a GUI on a lab PC.

### The running GUI

*Assistant > Allow connections* makes the GUI listen on a `QLocalServer`: a
Unix socket on macOS and Linux, a named pipe on Windows, readable only by
the user who started it.  Not a TCP port: a port on 127.0.0.1 is open to
every program on the computer, a web page's POST included, and would have
needed a secret.  An MCP client launches `smappy-mcp --attach`, which finds
the window from a file per listening GUI in `<config dir>/assistant/gui/`
and copies stdio to the socket and back (`assistant/attach.py`, on Qt's
`QLocalSocket`, the one way to reach a Windows pipe in both directions at
once); `--attach NAME` picks a window by its session's name.  The forwarder
outlives the window: while SMAPpy is gone it answers the client's calls with
why they cannot be made, and when a SMAPpy listens again it attaches to it and
replays the client's `initialize`, so the conversation -- which knows what
was analysed -- goes on across a restart.

There is one assistant for all of SMAPpy's windows: *Start assistant* in a
second window writes a hand-over request beside the windows' files
(`attach.request_handover`), and the forwarder, which looks for one every
second, moves there once no call is in flight; the window it left stays
open.  The model is told -- the next tool answer starts with a line naming
the window it is in now and the one it left (`_Forwarder.note`), and
`session_state` names it (`window`).  SMAPpy closes the terminal when the
last window that allows connections quits (`launch.close_terminal`, the
setting *close its terminal when SMAPpy quits*, on by default), because the
forwarder, running from SMAPpy's Python, held an update or a restart of
SMAPpy up; while another listens the assistant goes there instead.

The tools are the same `Assistant`, over the GUI's session.  Every line from
the socket is queued and handled on the GUI thread, one per tick of a timer
whose `timeout` is a connected signal -- not `QTimer.singleShot`, and with
everything caught (`CLAUDE.md`) -- so the session and its listeners are
only ever touched where the render expects them.  An applied run goes
through the plugin's own panel -- its section shown in its tab, or its
window, the settings in its form, its Run pressed -- on the panel's thread,
and is answered when it ends (`tools.Later`): the user watches it run, can
stop it, sees its figures, and the window stays free; the client's next
lines wait in order.  Tries on copies stay out of sight and on the GUI
thread.  The settings' *show its runs in the plugins' panels* turns it off.

What is looked at every moment is kept apart from what is set once
(`gui/assistant_pane.py`, `docs/panels/Assistant.md`):

* **Every moment** -- is an assistant connected, may it act, what did it
  propose -- is a button in the render window's toolbar
  (`[● Claude · suggest · 2 ▾]`), and under it the list of proposals,
  dropped down inside the render window over the picture's corner
  (`ProposalOverlay`): it opens by itself when a proposal arrives, stays
  until collapsed -- the button highlighted while something waits -- and
  folds away when the last is dealt with.  Inside the window, not a window
  of its own (as it first was, beside the terminal): it moves with the
  picture and never takes the focus from the terminal.  A proposed run has *Open*, which
  shows the plugin's own window filled in (`PluginPanel.take_settings`); the
  user's Run there carries the proposal out (`AssistantServer.watch`): it
  becomes `done`, with the result kept and the settings it ran with, so the
  model can read what the user ran.  The rest has *Do it*; × dismisses.
* **Once** -- which client to start, the folder it starts in, the folder it
  may write into, the mode to start in, listening at start -- is a settings
  window, kept in the configuration.

*Assistant > Start assistant* (Ctrl+Shift+A) allows connections and starts
the chosen client already connected, in a column beside the picture
(`assistant/launch.py`): Claude Code or Codex in a terminal, the server on
their command line (`claude --mcp-config`, Codex's `-c`), so the user's
settings are not touched; the Claude app, which cannot be started on a
folder from outside, through the folder's `.mcp.json`; or a command of the
user's own.  It starts in a folder of its own, the same each time
(`launch.default_folder`, *SMAPpy assistant* in Documents, asked at the first
start), not the data's: the client keys its conversations, memory and trust
by that folder, a lab's `CLAUDE.md` is read from it, and scratch files stay
out of the raw data; Claude Code gets the output folder with `--add-dir`;
the folder holds SMAPpy's `AGENTS.md` (where the data, the code and the
user's plugins are, kept to the version) and the user's `CLAUDE.md`, which
imports it (`launch.prepare_folder`).
The server command is SMAPpy's own Python
(`-m smappy.assistant.server --attach PID`), which needs nothing on the PATH.
Typing to the assistant from SMAPpy itself stays step 5.

### Two tiers, by what the client can do

| tier | client | what it can do |
| --- | --- | --- |
| **coding agent** | Claude Code, Codex or another agent with files and a shell, and this server | everything: the tools, the source and pages directly, its own Python against `smappy` for scans and figures, new plugins, literature research |
| **tools** | any MCP chat client | the tools: the session, plugins and chains, figures and pictures; the pages and the source through `read_doc` and `read_source`, so the code stays within reach |

The coding agent's advantage comes from the agent, not from anything built
here, so everything built here serves both.  Two things help it along:
`session_state` and the connection's instructions say where the running
SMAPpy's code is (`tools.source_folder`, `site-packages/smappy` after pip or
uv) and which version, and *Start assistant* gives Claude Code that folder
with `--add-dir`, to read and grep without asking.  The C++ is installed
compiled only; its source is on GitHub at the version's tag.  A chat client
is told, should a task need files or commands, to say once that Claude Code
can do it.

### Why our own protocol code, not the `mcp` SDK

The server implements the part of MCP it uses -- `initialize`, `tools/list`,
`tools/call`, `ping` over newline-delimited JSON-RPC, about 200 lines in
`assistant/protocol.py` -- with no dependency.  The SDK was the first plan,
but its 2.0 renamed the server class and changed its API (`FastMCP` became
`MCPServer`), so supporting the versions users have installed would have cost
more code than the protocol, and the GUI's socket server needs the same
dispatch without the SDK's event loop.  The SDK's client is used in the
tests, when it is installed, to check that a real client talks to it.

## The tools

Plain functions on an `Assistant` over a `Session` (`assistant/tools.py`),
Qt-free, returning JSON; the protocol layer only lists and calls them.

* **reading** -- `session_state`, `list_plugins` (the catalog),
  `find_plugins` (the search, both under "The plugin index"),
  `describe_plugin` (fields, grouping, what the form says about them now),
  `read_doc` (a plugin's or a panel's page), `read_source` (a plugin's
  module or any `smappy` module), `find_functions` (the function index,
  below), `figure` (a result's figure as
  PNG), `picture` (the view as PNG).  The results `figure` draws are kept
  whole, so they are held to `MAX_RESULT_BYTES` (2 GB) of arrays as well as
  `MAX_RESULTS` of them, the newest always kept -- a PSF calibration's holds
  its bead stacks and model, and fifty of those had taken a GUI to 71 GB
  (issue #172); `session_state` says what they hold and `forget_results`
  lets them go.
* **trying** -- `run_plugin` with `apply=false` runs on a copy of the
  session (`Session.scratch`) and hands back its text, numbers and figures,
  changing nothing; `scan` does it over a list of values of one field.
* **changing** -- `run_plugin` with `apply=true`, `run_chain`, `load`,
  `set_filter`, `set_grouped`, `set_layer`, `undo`, `save`, `save_chain` (a
  workflow kept where the plugin tree, the GUI and `smappy-batch` find it).
  `load` sorts a file as File > Open does (`gui/drops.kind_of`): a table is
  opened, an image of a few frames is opened as sources, which no layer
  shows yet, and a long stack is refused as raw data for a fitter.  Where
  the count leaves it open (`drops.ASK_FRAMES`, 20 to 500 frames) the GUI
  asks a person; the assistant is never asked -- `load` and the actions
  *File/Open*, *File/Add* take `open_as` ("image", "stack"), and without it
  the count decides and the answer says that it did, so the model can say
  otherwise.
  `set_layer` is the Render tab's settings of one layer: an image put in an
  image layer (one past the last makes a new one), its display -- LUT,
  contrast and the rest of `DisplaySettings` -- and a localization layer's
  `RenderSettings`, by dotted name as a plugin's settings are given;
  `session_state` lists each layer's.
* **remembering** -- `remember`, `recall`, `forget`: the knowledge base below.
* `list_proposals` -- what *suggest* mode proposed, and what became of it.
* `open_window` -- the GUI only (`needs="windows"`): one of SMAPpy's own
  windows that is not a plugin -- the ROI manager (on an ROI by its
  number), the 3D view, the batch window, the calibrations, the microscope
  manager, the camera editor, the help -- by the names the menus give them
  (`ControlWindow._assistant_windows`).  Not the modal dialogs, which would
  hold the call.  The first tool built from a `note_missing_tool` note.
* `note_missing_tool` -- a gap, for SMAPpy's developers: what was asked,
  the detour taken, the tool that would have done it, appended to
  `<config dir>/assistant/missing-tools.md` on the user's yes (offered by
  the model in a sentence after a detour), and zipped into *Save a bug
  report*.  The tools are written by hand, through review; this is how a
  conversation's gap reaches whoever writes them.
* `record_own_code` -- a step the model computed with code of its own
  rather than a plugin: what, why no plugin did it, the plugins it built
  on, its script.  A line in the session's history (`Session.log`, "own
  code"), an entry in `<config dir>/assistant/own-code.md` and the script
  copied to `own-code/` beside it -- without asking, because it is a record
  of what was done, as the history is, where `note_missing_tool` is a wish.
  Not zipped into a bug report: it describes the user's analysis.  See
  "Plugins first" below.
* `list_actions`, `do_action` -- the GUI only (`needs="actions"`): what
  the menus and buttons do, by name -- *File/Open*, *File/Add* (the Load
  plugin's buttons), the Render tab's *+ > image layer* and layer buttons,
  the tabs, *View*, *Edit* -- each done by the method its click calls, with
  what the click would ask for (a file, a layer) as an argument, so that no
  dialog holds the queue (`gui/assistant_actions.py`, `tools.Action`).  One
  that changes only what is shown is made in suggest mode with the GUI
  attached; any other is a proposal.  Every menu item is an action or in
  `LEFT_OUT` with its reason (a modal dialog, a file written outside the
  output folder, the assistant's own connection), and a plugin's window
  button is in `WINDOW_BUTTONS`: the GUI tests walk the real menus and the
  shipped plugins, so a new item that is neither fails there.  Before them
  "add this image and show it in a layer" had no way in but the screen.
* **inside the windows** -- the same two tools reach the controls of the
  windows a button opens (`gui/window_actions.py`), named by the window:
  *Bead calibration/Add files*, */Settings*, */Detect and calibrate*,
  */Exclude beads*, */Save*; *PSF calibration/Choose* (sample, method,
  mode), */Settings* (one part at a time), */Detect*, */Calibrate*,
  */Mark beads*, */Save as*.  Each is the method the click calls, on the
  window the user sees; what a dialog would ask is an argument, and a
  window's */State* reads it back -- its files, settings by dotted name,
  status, the bead table.  A job runs in the window's worker and the action
  returns at once; one the assistant started fails into *State* and the
  status line, never a modal box, which would hold the queue
  (`quiet_job`).  Setting the inputs and computing inside the window is
  made in suggest mode; writing a file or handing a calibration to a fitter
  is a proposal there.  `open_window` alone had opened the bead
  calibration and left the assistant outside it (a user's report,
  2026-11-12), so every window `open_window` opens has actions or a reason
  in `WINDOWS_WITHOUT_ACTIONS` -- the batch window, DECODE training, the
  microscope manager and the camera editor so far -- and
  `tests/test_window_actions.py` fails on one that has neither.
  Without a window the same calibrations are `run_plugin`'s:
  *Localize/PSF calibration* (the old *Bead calibration* was retired in 2026-11) runs from
  their settings (`Plugin.form_in_window`: the window's settings, which the
  panel does not show), detect, calibrate and save as the window does with
  nothing marked by hand, through the same code (`calibrate/psf_run.py`),
  their pages the result's figures -- so headless `smappy-mcp`, a chain and
  a batch calibrate too.
* **what the windows show** -- `read_gui`, `set_gui`, `focus_gui`, the GUI
  only (`needs="gui"`, `gui/gui_read.py`'s `GuiAccess`).  Before them the
  assistant could open a form but not read it: asked which calibration the
  open Spline 3D form used, it searched the disk for one (a usage log,
  2026-11).  `read_gui` without a name lists the tabs with their pinned
  plugins, the open windows and what is in front; with a name -- *Localize >
  Spline 3D*, a plugin's path, *Render*, *ROI manager* -- it reads that one.
  A plugin's panel is read as its settings: the values the form shows by
  `run_plugin`'s dotted names, which differ from the defaults, what
  `Plugin.describe` says of them now (*auto* resolved, fields not read, the
  notice), the status line, and its last run, kept as a result for
  `figure`.  Every other tab and window is read generically, by walking its
  widgets in their layouts' order: a control is named by its label (a form
  row's, the label before it in its row, its own text, its tooltip) under
  the sections, groups and pages it is in, with its value, its choices and
  whether it is greyed; tables by their first rows, read-only text, notes,
  buttons; a `SettingsForm` met on the way by dotted names, a `PluginPanel`
  by what differs from its defaults.  Generic on purpose: a window written
  later is read without a line for it -- the price is that a name is only
  as good as the window's label.  `set_gui` sets fields by those names as
  typing does (the edit's own signal, so a plugin's `react` fills a preset's
  settings), leaving the rest; not buttons, which stay `do_action`'s, curated.
  A dialog a field opens is closed unanswered and said (`_NoDialogs`): it
  would hold the queue.  A plugin's form, the Render tab and the windows
  whose fields are inputs to a later run are set in suggest mode, as
  `open_plugin` fills a form; the ROI manager and the Regions tab change
  the ROIs as they are set, so there it is a proposal (`CHANGES_SESSION`).
  `focus_gui` brings a panel or window forward and, given fields, unfolds
  their sections, scrolls to them and outlines them for three seconds, taking no
  keyboard: the step a guided mode is built on -- propose a step, point at
  where it is set, read it back once the user has set it.
* **the drawn ROI and what is in it** -- `session_state` reports the ROI
  drawn in the 2D view (`Session.roi`, the one shape every plugin's
  selection is cut to) as `roi`, its shape as `set_roi` takes it and how
  many each layer shows inside it, and counts the ROI manager's apart
  (`roi_manager_rois`); the two were one number before, and the model could
  not tell "in the ROI" from "in the ROI manager".  `set_roi` draws or
  clears it, a change to what is shown and so made in suggest mode with the
  GUI attached.  `read_values` reads columns of the rows a plugin would read
  -- the layer's table as it shows it, its filter, the drawn ROI and the 3D
  box (`Context.table`); `where="all"` the whole table; `roi=` an ROI of the
  ROI manager through `ROIProject.region_context` -- as statistics, a
  histogram (a bin per value for an integer column with few) and the number
  of distinct values, which answers "how many clusters in the ROI" from
  `cluster_id`; `plot=true` draws the histograms.  Before it, "a histogram
  of the frames in the ROI" had no answer but a plugin that did not ask.
* `show` -- the GUI only (a tool `needs="viewer"`, not listed headless):
  moves the user's 2D view onto a point, an ROI or everything.
* `open_plugin` -- the GUI only (`needs="panels"`): opens a plugin's window
  with its settings filled in, for the user to run; made in suggest mode
  too.  Without it, "open Tracking analysis" was answered by a proposal to
  run it and an explanation of why it could not simply be opened.

A refused call -- an unknown setting, a file outside the output folder, a
plugin that does not exist -- answers with why and what exists instead, as
`smappy-batch validate` does, so the model corrects the call itself.

### Suggest and act

One switch, `mode`:

* **suggest** -- reading and trying are allowed; a change is not made but
  recorded as a proposal (`proposals`), which the GUI shows with the
  plugin's panel filled in for the user to run.  Exports, which write files
  even on a copy, are proposals too.  In the GUI the changes to what is
  shown rather than to the data -- `set_filter`, `set_grouped`,
  `set_layer`, `show`, an action that only shows, `set_gui` on a form or
  the Render tab -- are made at once: the user sees them, and a proposal to look somewhere
  would be a click to make a click.  Headless they stay proposals.
* **act** -- changes are made.  A file is written under the output folder
  (`--output`, the session's folder by default), or as a new file beside the
  data the call reads or the session has open -- a calibration beside its
  beads, where the plugin puts it anyway when no path is given.  An
  existing file there is never overwritten unless the assistant wrote it;
  elsewhere the call is refused and says why (`tools._allowed`).

Headless, the default is *act*: there is nobody to propose to.

### Guided suggestions (planned)

Decided in 2026-11, not built; it builds on the assistant's read and write
access to the open GUI, which is being built first.  A proposal today is one
change, accepted or not.  A larger task -- "analyse this DNA-PAINT
measurement" -- is several plugins and GUI actions in a row, each with
settings, and is better learnt by doing it in the GUI than by accepting a
result.

**The assistant writes the plan; SMAPpy runs it, without the assistant.**
The plan is a *guided chain*: the chain format (`docs/batch.md`) with, per
step, a few more keys -- the plugin or a GUI action (show a tab, set a
filter, draw an ROI: what `do_action` and the GUI tools do), its suggested
values, a sentence or two of why, what to look at afterwards and, where it
can be, a check.  A step is a unit of meaning ("correct the drift with
RCC"), not one per field.  Nothing in it waits for the model: MCP answers
calls and cannot be woken by a click, and a plan that runs in the GUI needs
no waking.

**In the GUI** it is a list beside the panels, the chain panel shown as a
guide: each step with its explanation, a tick when done and *Do next*,
which brings its tab or window forward, opens its panel filled in with the
values the step explains highlighted, and leaves the run to the user, who
may change any value first.  After the run the step's checks show, and the
list moves on.  A finished guide saves as an ordinary chain, which runs
again in the GUI and in a batch without guidance.

**Checks**, three kinds side by side:

* *text*, always: what a good result looks like, for the user to judge;
* *the plugin's own warnings* -- `ctx.warn`, its notice, a `preflight`
  question -- shown on the step, with nothing written for them;
* *a condition on the result's numbers* (`drift_max_nm < 50`, in Filter
  ROIs' syntax), only where the assistant has run the step on a copy and
  knows the result's names and typical values: a condition on a name the
  result lacks would only confuse.

And one before a step, which SMAPpy works out itself from what a plugin
needs: "this step needs `cluster_id`, which step 3 makes".  A failed check
shows its text and *Ask the assistant*, which sends the step, the user's
values and the result to the assistant's window: troubleshooting is the
assistant's, there.

**Dependencies between steps are left out of SMAPpy.**  Most settings are
local.  Where one step's result should set another's -- a cluster radius
from the measured precision, *merge blinks*' dark time from the blinking
statistics, the drift's time windows from how many localizations there are,
filter bounds from a histogram -- the plugin's *auto* usually does it from
the data at run time (`describe` says what it resolves to), so the
assistant leaves such settings on *auto* and fixes a value only with a
reason, and a value follows the data, the user's changes in earlier steps
included.  The few without an *auto* are said in the step's text ("about
twice the precision step 2 measured: 9 nm, so 18 nm").  When an earlier
change makes a later step clash, the user asks the assistant, which reads
the guide's state -- the steps run, their values, what differs from its
suggestion -- through the GUI tools and explains it.  References between
steps (`step2.precision_nm * 2`) come only if such links prove common.

## The plugin index

Two things, both generated from the plugins and their pages, so neither can
go stale; the pages and the code stay the authority, and the index only
says which page to read (`assistant/index.py`).

**The catalog** -- every plugin and chain on a line: its path as
`run_plugin` takes it, the name the GUI shows where it differs, its summary
and, where it holds several methods, its choice of them (*Segment regions*'
*method: circle template · beads · ... · like examples*; `index.catalog`).
About 17,000 characters for 88 plugins.  It is the overview a question is
matched against, by the model, which knows that "is my sample moving" may
be drift or vibration where no word is shared.  It is written into the
assistant folder's AGENTS.md (`launch.prepare_folder`), which a client
started from SMAPpy reads whole, and `list_plugins` returns it for every
other client.

**The search** -- `find_plugins` ranks every section of every plugin's and
panel's page, and every setting's label, help and choices, by BM25: a
plugin by its best section and a share of its next two, its name, path and
summary counting more.  The first index held only each page's opening, and a
plugin's modes are further down: "undrift from picked", Picasso's name for
Fiducials' *markers*, found nothing (2026-11).  Word endings are stripped
(`index.stem`), a Glossary entry's other names are searched at half weight
(`index.synonyms`), and words a query negates ("drift correction without
beads") count for none (`index.wanted_words`): they had put the bead
plugins above COMET.  It takes several phrasings at once -- the user's
words, the field's terms, the question's parts -- and says how many found
each plugin (`found_by`), with the sentences that matched, by section: the
model sees why a plugin was found without reading its page, and a word
those sentences use (*markers*, *picked*) is its next query.  Building it
takes about three seconds, once per process, most of it importing the
plugins for their settings.

Embeddings, which would find meaning where no word is shared, were left
out: a model of a hundred megabytes to ship and keep, against the model at
the other end of the call, which rephrases for nothing.  Whether the search
is good enough is measured, not argued: `tests/test_plugin_search.py` holds
questions -- from the usage logs and written to cover other programs' names,
modes inside plugins and vague questions -- each with the plugin that
answers it, which must be among the first five.  Questions the logs bring
are added there.

Every plugin a search finds comes with its main settings and defaults
(`tools._sketch`), enough for a first `run_plugin` without
`describe_plugin` -- see the next section; every one, because a plugin
cheaper to try than its neighbours in the list is the one tried.

### The instructions' length

Claude Code keeps only the first 4096 characters of a server's
instructions (`INSTRUCTIONS_LIMIT`).  `INSTRUCTIONS` with `source_note` and
the GUI's `GUI_NOTE` had grown to 5900, and the end -- where SMAPpy's code
is, and all the GUI's note -- never reached the model (2026-11).  So the
instructions say only how to work across the tools; what one tool needs is
in its description, and a test keeps the three under the limit.

### Plugins first

A coding agent has Python against `smappy` at hand, and in use (2026-11)
it often computed in numpy what a plugin already gives -- the analysis an
AI without SMAPpy would have written, untested and recorded nowhere.  Its
own code is wanted -- to explore, to combine plugins' results, for a
figure or a method SMAPpy lacks -- so there is no rule against it; three
things lean it towards the plugins and make the rest visible:

* the instructions (`INSTRUCTIONS`, and the assistant folder's AGENTS.md)
  say to ask `find_plugins` first and to build on the plugins' numbers, and
  to tell the user in two parts what SMAPpy's plugins did and what its own
  code did;
* `find_plugins` is a list of candidates, and its description says so: the
  model chooses among them from their pages, judges whether one is usable
  on this data at all (its needs against `session_state`), says why it
  chose it over a close alternative, and says so when none fits -- a
  ranking of words cannot know that drift without beads is COMET's;
* the path from a question to a run is shorter: `find_plugins` carries the
  settings, `run_plugin` answers with the result's `data` and `values`;
* code of its own builds on SMAPpy's functions: `find_functions` (below)
  finds the module-level functions behind the plugins, which it called
  rarely because only reading the code found them;
* `record_own_code` puts each step of its own into the history beside the
  plugins' runs and into `own-code.md`, whose recurring entries are the
  plugins to write next.  A script worth keeping is offered as a plugin of
  the user's (their plugin folder), written only on a yes.

What is not in the code -- a lab's preferences, typical values, failure modes
people have learnt to recognise -- is written down only once test tasks show
an assistant needs it, and then where possible in the plugins' pages.

### The function index

Every shipped plugin is a thin wrapper over module-level functions that
take arrays, and an agent writing code of its own rewrote them in numpy
(which beads to trust, a table collapsed per id) because nothing pointed
it at them.  `find_functions` searches an index of them
(`assistant/functions.py`): every public function and class at the top of a
`smappy` module -- not `gui`, `assistant`, `tutorial`, `docs`, `cli`, a
private module, nor the plugins' classes, which `find_plugins` finds --
with its dotted name (what `read_source` takes), its signature without the
parameters' annotations and its docstring's first sentence.  Built by
parsing the source with `ast`, never importing it, so torch and Qt stay
unloaded; about 2700 entries, two or three seconds once per process.  A
word counts by where it was found: the name, the module's name and first
sentence, the first sentence of the docstring, the rest of it, and the
paragraphs of the module's docstring that name the function in backticks --
in this code base the module's docstring often explains a function better
than its own (`beads`: "which ones to trust").  One call takes several
queries and answers each with its list.  The tool's description,
`INSTRUCTIONS` and the assistant folder's AGENTS.md say to ask it before
writing code of one's own.

## The knowledge base

Decisions, answers and values from a conversation that the user wants kept
are notes in a folder, one Markdown file each
(`<config dir>/assistant/knowledge`, or `--knowledge DIR` -- a lab's shared
folder, say), written by `remember` and searched by `recall`.  Plain files, so
they are read, edited and deleted without the assistant.  A note is written
only when the user has agreed to it: the tool's description says so, and a
client asks before a tool call unless told not to.  An unagreed note is worse
than none, because the next session takes it for the lab's practice.

A workflow is kept as a chain (`save_chain`), which is runnable, rather than
as a note that describes it.  Notes that hold solutions -- how a problem was
solved, with its chain -- come later.

## The usage log

What the assistant tried is the clearest record of what SMAPpy lacks: a
`find_plugins` that found nothing, a `read_source` because a page did not
say, a script of its own where a plugin existed.  `smappy/usage_log.py`
collects it from colleagues who agree, for the developers to read with a
model.  Off by default (Preferences > Usage log; *Help > Send usage log...*).
When on, `protocol.Server.tools_call` writes each tool call -- the tool,
its arguments, failure, time, the start of its answer -- and the Help
window each page opened, as JSON lines in `<config dir>/usage-log`.  The
rest is gathered when a bundle is made rather than recorded twice: the
plugin runs of `smappy.log`, `own-code.md` with its scripts,
`missing-tools.md`, and the client's own transcripts of conversations in
the assistant's folder (Claude Code's `~/.claude/projects/<folder>`,
Codex's `~/.codex/sessions` by their `cwd`), which hold what the user asked
-- the one part SMAPpy never sees.  Pictures are removed and long texts cut.

A bundle holds what is new since the last send and goes by one WebDAV `PUT`
to a Nextcloud upload-only share link (u:cloud's *file drop*): no account on
the sender's side, and a link that lets a lab add files but not read each
other's.  The link stays in the configuration until cleared; *Send now*, or
every seven days at startup in the background.  Nothing goes without a
link, and nothing is written while the log is off.

The bundles are read by the `usage-logs` skill (`.claude/skills/usage-logs`):
its `digest.py` turns them into an overview and one timeline per
conversation, standard library only, and the skill has a model read those,
check each finding against the code and write a ranked report, kept on the
developer's computer -- the conversations are colleagues' and stay out of
the repository.

## Steps

1. **Headless server** (built): `smappy-mcp` over a `Session`, the index,
   `read_doc` and `read_source`, suggest and act, the knowledge base,
   `tests/test_assistant.py` (with the `mcp` package's client when it is
   installed).
2. **The running GUI** (built): the same tools over a local socket, opt-in
   from a menu, every call queued onto the GUI thread with its exceptions
   caught; the *Assistant* pane with the proposals, the switch and the
   connection's state; view changes allowed in *suggest*; `smappy-mcp
   --attach`; `tests/test_assistant_gui.py`.  Then the everyday layout: the
   toolbar button, the proposals beside the terminal, *Start assistant*,
   proposals carried out by the user's run.
3. **Test tasks** with a coding agent and one non-Claude client; what they
   get wrong goes into the pages or a short note of lab knowledge.
4. **Solutions in the knowledge base**: a problem, how it was solved and
   the chain that solves it, found again by `recall`.
5. **Guided steps**: a suggest mode that proposes a list of steps and walks
   the user through them one at a time -- `focus_gui` on where each is set,
   `read_gui` to see that it was, the proposal's run as the step done.
   The reading, setting and pointing are built (`gui/gui_read.py`); the
   list and its pane are not.
6. **Typing in SMAPpy's pane**: Claude Code's *channels* can push a message
   from an MCP server into a running session (a research preview); the other
   route is running the user's own agent CLI per message.  Later.
6. **Read and write the open GUI**: every panel, window and control as it
   stands, read in any mode, set where only a form changes (being built,
   2026-11).
7. **Guided suggestions**: the assistant's plan as a guided chain the GUI
   runs step by step ("Guided suggestions (planned)" above), after 6.

## Using it now

    pip install -e .
    smappy-mcp --output ~/analysis              # what an MCP client launches

In Claude Code (`claude mcp add`, or the project's `.mcp.json`):

    claude mcp add smappy -- smappy-mcp --output /path/to/output

Other clients take the same command in their MCP settings.  `--mode suggest`
starts in suggest mode, `--open FILE` with a file loaded, `--knowledge DIR`
keeps the notes elsewhere.  Nothing to install beyond SMAPpy: the server has
no dependency of its own.

With the GUI: *Assistant > Start assistant* starts the client chosen in
its settings, connected.  For a client started by hand, *Assistant > Allow
connections*, then

    claude mcp add smappy-gui -- smappy-mcp --attach

(or `smappy-mcp --attach` in another client's settings).  The window's
mode, output folder and session apply; the client is (re)started after
connections are allowed.
