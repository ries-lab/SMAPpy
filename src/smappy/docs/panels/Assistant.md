---
title: Assistant
summary: Starts your own AI assistant connected to the open session, shows in the render toolbar whether it is connected and what it may do, and lists what it proposed for you to open or dismiss.
widget: smappy.gui.assistant_pane.AssistantSettings
covers: [smappy.gui.assistant_pane.AssistantSettings, smappy.gui.assistant_pane.AssistantControls, smappy.gui.assistant_pane.ProposalList, smappy.gui.assistant_pane.AssistantButton, smappy.gui.assistant_server.AssistantServer, smappy.gui.assistant_server.RenderViewer, smappy.assistant.launch.start, smappy.assistant.attach.forward, smappy.assistant.attach.choose, smappy.assistant.tools.Assistant]
---

## What it does

An AI assistant -- Claude Code, Codex, or any program that speaks the Model
Context Protocol (MCP) -- can use SMAPpy's tools: read the plugins' pages
and their code, try a plugin on a copy of the data and look at its numbers
and figures, build a chain of plugins, and draw a picture of the
localizations.  The assistant runs in its own program, with your own
account; SMAPpy has no model and no key of its own, and nothing leaves the
computer through SMAPpy.

Connected to the open session, it looks at and changes what you see.
Nothing can connect until connections are allowed, and only programs
running as your user on this computer can.

## Starting it

*Start assistant* -- in the toolbar button's list, in the *Assistant* menu,
or Ctrl+Shift+A -- does it in one go: SMAPpy allows connections, starts your
assistant already connected to this window, and puts its window in a column
beside the picture with the list of proposals above it.  The first time it
asks one question, which assistant: Claude, Codex or the Claude app; *another
assistant...* opens these settings.

Nothing needs to be typed in a terminal.  If the assistant is not installed
yet, its window says so and asks *Install it now? [Y/n]*; Enter installs it
with its maker's own installer and starts it.  The first time, the assistant
asks you to sign in with your account, in the browser, and whether to trust
the folder it works in.  Then you type your questions there.

With an assistant connected already it
starts no second one -- two would share one list and one switch, and in
*act* mode change the table without knowing of each other -- but brings the
running one's terminal to the front (on a Mac).  To start another, *stop*
the first.

* **Claude Code in a terminal** is started with the connection on its
  command line, so your Claude Code settings are not changed.
* **Claude app (Code tab)**: the app cannot be started on a folder from
  outside.  SMAPpy adds the connection to the folder's `.mcp.json`, after
  asking, and opens the app; start a Code session in that folder and approve
  the `smappy-gui` server it offers.
* **Codex CLI in a terminal**, the same way as Claude Code.
* **A command of your own**, run in a terminal, for any other client.

A client SMAPpy does not start connects with the command *copy command*
copies (`smappy-mcp --attach`), or the settings entry *copy JSON* copies;
then *allow connections*.  On a Mac the terminal is Terminal, placed beside
the picture; elsewhere it opens where the system puts it.

## Every day: the toolbar button

The render window's toolbar has the assistant's button.  It shows a dot --
filled when an assistant is connected, hollow while SMAPpy only listens --
the assistant's name, the mode, and how many proposals wait; it is
highlighted while there are ones you have not looked at.  Clicking it opens
the list of proposals.  *Assistant > Proposals* shows the same list as a
window, which *Start assistant* places above the terminal.

In the list, the newest proposal is first; its reason is in its tooltip.

* A run of a plugin or a chain has **Open**: the plugin's own window with
  the proposed settings filled in.  Nothing runs until you press its Run --
  and that run carries the proposal out: it leaves the list, and the
  assistant can read the result and the settings you ran it with, changed
  or not.
* Opening a file, undoing or saving has **Do it**.
* **×** dismisses a proposal.  The assistant can see what became of each:
  proposed, opened, done or dismissed.

Under the list: *suggest* and *act*, *start* or *stop*, and *settings...*.

## Suggest and act

In *suggest* mode the assistant reads and tries on copies of the session,
and a change it would make -- running a plugin on the data, opening or
saving a file, undoing -- waits in the list for you.  What only changes what
is shown it makes at once: a layer's filter, grouping, and moving the 2D
view onto a place or an ROI.  In *act* mode it changes the session itself;
each change is in *Edit > Undo* as one from a panel is.  The assistant
cannot switch between the two; only you can.

Files are written only under the folder after *writes under*.

While the assistant's call runs, the window is busy: a long run it applies
-- a fit, a drift correction over a large file -- holds the window until it
ends, and the list says what is running.

## Controls

### allow connections
Listen for an assistant's client.  Unticked, nobody can connect, and a
client already connected is disconnected.  The line below says who is
connected.

### assistant
What *Start assistant* starts.  Changing it here starts nothing.

### command
For *a command of your own*: the command line, run in a terminal in the
folder after *works in*.  `{server}` is replaced by the command that reaches
this window and `{config}` by a JSON file naming it, the form most clients'
`--mcp-config` options take.

### works in
The folder the assistant starts in: what its own file tools see first, and
for the Claude app the folder whose `.mcp.json` gets the connection.  *same*
makes it the folder after *writes under*.

### writes under
The only folder the assistant may write files into; *choose...* sets
another.

### follow file
Back to the folder of the first open file, which changes as files are opened.

### start in suggest mode
The mode each time SMAPpy starts.  Unticked, it starts in *act*.

### allow connections when SMAPpy starts
Listen from the start, for an assistant you keep running.

### Claude Code uses SMAPpy's tools without asking
Claude Code asks before each tool call unless told otherwise.  What the
assistant may change in SMAPpy is decided by *suggest* and *act*, so this
lets it use SMAPpy's tools without that question; its shell and its own
file edits still ask.

### copy command
The command another client runs to reach this window.

### copy JSON
The same as the `mcpServers` entry most clients' settings take.

### Start assistant
Allows connections, starts the assistant chosen above, and places it and
the proposals beside the picture.
