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

The tools are the same `Assistant`, over the GUI's session.  Every line from
the socket is queued and handled on the GUI thread, one per tick of a timer
whose `timeout` is a connected signal -- not `QTimer.singleShot`, and with
everything caught (`CLAUDE.md`) -- so the session and its listeners are
only ever touched where the render expects them.  The cost: a long applied
run holds the window while it runs.  Moving long runs onto a worker, as a
plugin's panel does, is left for when it is missed.

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
user's own.  The server command is SMAPpy's own Python
(`-m smappy.assistant.server --attach PID`), which needs nothing on the PATH.
Typing to the assistant from SMAPpy itself stays step 5.

### Two tiers, by what the client can do

| tier | client | what it can do |
| --- | --- | --- |
| **coding agent** | Claude Code, Codex or another agent with files and a shell, and this server | everything: the tools, the source and pages directly, its own Python against `smappy` for scans and figures, new plugins, literature research |
| **tools** | any MCP chat client | the tools: the session, plugins and chains, figures and pictures; the pages and the source through `read_doc` and `read_source`, so the code stays within reach |

The coding agent's advantage comes from the agent, not from anything built
here, so everything built here serves both.

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

* **reading** -- `session_state`, `list_plugins`, `find_plugins` (the
  generated index), `describe_plugin` (fields, grouping, what the form says
  about them now), `read_doc` (a plugin's or a panel's page), `read_source`
  (a plugin's module or any `smappy` module), `figure` (a result's figure as
  PNG), `picture` (the view as PNG).
* **trying** -- `run_plugin` with `apply=false` runs on a copy of the
  session (`Session.scratch`) and hands back its text, numbers and figures,
  changing nothing; `scan` does it over a list of values of one field.
* **changing** -- `run_plugin` with `apply=true`, `run_chain`, `load`,
  `set_filter`, `set_grouped`, `undo`, `save`, `save_chain` (a workflow kept
  where the plugin tree, the GUI and `smappy-batch` find it).
* **remembering** -- `remember`, `recall`, `forget`: the knowledge base below.
* `list_proposals` -- what *suggest* mode proposed, and what became of it.
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
  shown rather than to the data -- `set_filter`, `set_grouped`, `show` --
  are made at once: the user sees them, and a proposal to look somewhere
  would be a click to make a click.  Headless they stay proposals.
* **act** -- changes are made.  A file is written only under the allowed
  folder (`--output`, the session's folder by default); elsewhere the call is
  refused and says where it may write.

Headless, the default is *act*: there is nobody to propose to.

## The plugin index

`find_plugins` searches a generated index: each plugin's path, name, summary
and the first paragraph of its page's *What it does* -- where the page says
what the plugin is for and when not to use it.  Generated, not written, so
it cannot go stale; the pages and the code stay the authority, and the index
only says which page to read.

What is not in the code -- a lab's preferences, typical values, failure modes
people have learnt to recognise -- is written down only once test tasks show
an assistant needs it, and then where possible in the plugins' pages.

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
5. **Typing in SMAPpy's pane**: Claude Code's *channels* can push a message
   from an MCP server into a running session (a research preview); the other
   route is running the user's own agent CLI per message.  Later.

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
