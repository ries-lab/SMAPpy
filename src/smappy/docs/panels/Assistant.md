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

Connected to the open session, it looks at and changes what you see.  It
does not click: what the menus and buttons do -- *File > Open* and *Add*,
the Render tab's *+ > image layer* and a layer's colours, a tab brought to
the front, a window opened -- it asks SMAPpy to do by name, giving what a
click would ask you for (a file, a layer) itself.  A few items it leaves to
you: the dialogs that are yours (*Preferences*), saving over your file, and
quitting.  Nothing can connect until connections are allowed, and only
programs running as your user on this computer can.

### What to ask it for

Most often you know what you want to find out, not which plugin does it.
Tell the assistant that, in plain words, and it helps in one of three ways,
as you ask:

* **It explains** how to do it in SMAPpy: which plugin, which settings and
  why, and what the numbers that come out mean.  Nothing changes; you do it
  yourself and learn where it is.
* **It suggests**: it tries the plugin on a copy of the data, moves the view
  to show you where in the picture it means, and leaves the plugin with its
  settings as a proposal under the *Assistant* button.  *Open* fills the
  plugin's window; your Run carries it out.
* **It does it** for you, in *act* mode: each change applied at once, and
  undone with *Edit > Undo*.  Only you switch to *act*.

The tutorial [An AI assistant](https://ries-lab.github.io/SMAPpy/assistant/)
shows it on simulated nuclear pores: starting it, its proposals, suggest and
act; a whole analysis of a file without the window, kept as a chain; and
writing plugins with it.

## Starting it

*Start assistant* -- in the toolbar button's list, in the *Assistant* menu,
or Ctrl+Shift+A -- does it in one go: SMAPpy allows connections, starts your
assistant already connected to this window, and puts its window beside the
picture.  The first time it
asks which assistant -- Claude, Codex or the Claude app -- and which folder
it works in: one of its own, *SMAPpy assistant* in your Documents unless you
choose another (see *works in* below); *another assistant...* opens
these settings.

Nothing needs to be typed in a terminal.  If the assistant is not installed
yet, its window says so and asks *Install it now? [Y/n]*; Enter installs it
with its maker's own installer and starts it.  The first time, the assistant
asks you to sign in with your account, in the browser, and whether to trust
the folder it works in.  Then you type your questions there.

**One assistant for all of SMAPpy's windows.**  With several SMAPpy
windows open -- two datasets side by side, say -- there is one assistant,
working in one window at a time.  *Start assistant* in another window moves
it there, with its conversation; the window it leaves stays open and simply
has no assistant connected.  It finishes a call it is making before it
moves.  The assistant is told: its next answer from SMAPpy begins with a
line saying which window it is in now and which it left, and it says so to
you in a few words.  It can always ask SMAPpy which window it is in, and
which others are open.  Closing the window it works in moves it to another
window that allows connections, and it is told that too.

**Closing the last SMAPpy window closes the assistant's terminal**, which
ends the assistant and the connection it held: left open, that connection
is a program running from SMAPpy's installation, which holds up an update
or starting SMAPpy again (on Windows its files cannot be replaced while it
runs).  Windows that do not allow connections do not count: the assistant
could not go to them.  The conversation is not lost: Claude Code keeps it,
and `claude --resume` in the folder after *works in* picks it up again.

To keep the assistant open instead, untick *close its terminal when SMAPpy
quits* in the settings.  Then quitting SMAPpy leaves it running, with its
conversation: what was looked at and decided stays with it.  While SMAPpy
is gone, the assistant is told so when it tries to use SMAPpy; when SMAPpy
allows connections again -- *Start assistant* after a restart -- it
reconnects by itself, and *Start assistant* brings its window forward
instead of opening a new one.  To end it, quit the assistant in its window
(`/exit` in Claude Code).

With an assistant connected already it
starts no second one -- two would share one list and one switch, and in
*act* mode change the table without knowing of each other -- but brings the
running one's terminal to the front (on a Mac).  To start another, *stop*
the first.

* **Claude Code in a terminal** is started with the connection on its
  command line, so your Claude Code settings are not changed, and with
  SMAPpy's own code folder added, so it can read the code it is working with
  without asking.  It is the best choice: besides SMAPpy's tools it reads
  files, runs its own calculations and writes plugins.  A plain chat (the
  Claude app's Chat) can use SMAPpy's tools too, set up by hand with
  *copy JSON*, but none of that; it says so when a task needs it.
* **Claude app (Code tab)**: the app cannot be started on a folder from
  outside.  SMAPpy adds the connection to the folder's `.mcp.json`, after
  asking, and opens the app; start a Code session in that folder and approve
  the `smappy-gui` server it offers.
* **Codex CLI in a terminal**, the same way as Claude Code.
* **A command of your own**, run in a terminal, for any other client.
* **One I start myself**: SMAPpy only listens; start your client with the
  command *copy command* copies (`smappy-mcp --attach`), or give its settings
  the entry *copy JSON* copies.  On a Mac the terminal is Terminal, placed beside
the picture -- against its right edge, as tall, about 480 points wide,
narrower where the screen ends sooner and over the picture's edge where
there is no room at all.  *File > Workspace > Save* keeps where it is then,
and a workspace opened puts it back there.  On Linux the terminal is put
at the same place where it takes `--geometry` (xterm, gnome-terminal,
xfce4-terminal), at its own size; on Windows it opens where the system
puts it.

## Every day: the toolbar button

The render window's toolbar has the assistant's button.  It shows a dot --
filled when an assistant is connected, hollow while SMAPpy only listens --
the assistant's name, the mode, and how many proposals wait.  Under it, in
the picture's top-right corner, the list of proposals drops down: by itself
when the assistant proposes something, and on a click on the button.  It
stays open while you work; **—** collapses it, to look at the whole picture
first, and the button then stays highlighted while something waits.  When
the last proposal is dealt with, the list folds away by itself.
*Assistant > Proposals* opens it too.

The list shows one proposal at a time, with the assistant's reason under
it; with several, ‹ and › step through them (*2 / 4*).  A new one is shown
as it arrives, and one dealt with gives its place to the next.

* A run of a plugin or a chain has **Open**: the plugin's own window with
  the proposed settings filled in.  Nothing runs until you press its Run --
  and that run carries the proposal out: it leaves the list, and the
  assistant can read the result and the settings you ran it with, changed
  or not.
* Opening a file, undoing or saving has **Do it**.
* **×** dismisses a proposal.  The assistant can see what became of each:
  proposed, opened, done or dismissed.

Under the list: *suggest* and *act*, *Start assistant* or *stop*, and
*settings...*.

## Tools it was missing

Sometimes the assistant gets to what you asked by a detour, because none of
its tools does it directly.  It then says so in a sentence and offers to note
the missing tool; only if you agree does it write what was asked, how it got
there and what tool would have done it, to `missing-tools.md` in SMAPpy's
settings folder.  *Help > Save a bug report...* takes the file along, so the
tools that keep coming up can be built.

## Suggest and act

In *suggest* mode the assistant reads and tries on copies of the session,
and a change it would make -- running a plugin on the data, opening or
saving a file, undoing -- waits in the list for you.  What only changes what
is shown it makes at once: a layer's filter, grouping, moving the 2D view
onto a place or an ROI, opening a plugin's window with its settings
filled in ("open Tracking analysis"), which you then run yourself, and
opening one of SMAPpy's other windows -- the ROI manager on an ROI, the 3D
view, the batch window, the calibrations, the help.  In *act* mode it changes the session itself;
each change is in *Edit > Undo* as one from a panel is.  The assistant
cannot switch between the two; only you can.

Files are written only under the folder after *writes under*.

**In act mode you see it work.**  A plugin it runs on the data opens where
you would run it -- its section in its tab, or its own window -- with the
settings it chose filled in, and its Run is pressed: the progress line, its
figures, *Stop* and the undo step are as if you had clicked.  The window
stays free while it runs.  What it only tries, on copies, is not shown: it
often tries many settings before it chooses.  *show its runs in the plugins'
panels* in the settings switches this off, for a little more speed.

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
The folder the assistant starts in, the same every time, made when it is not
there yet: *SMAPpy assistant* in your Documents, or the one you chose
(*default* goes back).  It is one folder rather than the data's because the
assistant keeps its conversations (to pick one up again), what it has learnt
and which folders you trust by the folder it starts in, so a folder per
dataset would scatter them; a `CLAUDE.md` or `AGENTS.md` put there -- your
lab's cameras, the filters you always use -- is read at every start; and its
scripts and scratch files stay out of your raw data.  Claude Code gets the
folder after *writes under* added (`--add-dir`), so it reads your data there
without asking; a folder chosen there after it started is added in Claude
Code with `/add-dir`.  For the Claude app this is the folder whose
`.mcp.json` gets the connection.  Not shown for *one I start myself*, which
starts wherever you start it.

SMAPpy puts two files in it.  `AGENTS.md` tells the assistant where your
data, SMAPpy's code and your own plugins are; it is SMAPpy's, and follows the
installed version.  `CLAUDE.md` is yours, written once and then left alone:
what the assistant should know every time -- your microscopes and cameras,
the settings you always use -- goes there.  A folder of yours that has an
`AGENTS.md` or `CLAUDE.md` of its own keeps it.

### writes under
The only folder the assistant may write files into; *choose...* sets
another.

### follow file
Back to the folder of the first open file, which changes as files are opened;
your home folder while no open file is on disk (a simulation, say).

### starts in
The mode each time SMAPpy starts: *suggest* or *act*.  It is switched at
any time under the list of proposals.

### allow connections when SMAPpy starts
Listen from the start, for an assistant you keep running.

### show its runs in the plugins' panels
In act mode, each run the assistant makes on the data goes through the
plugin's section or window, filled in, as if you pressed its Run -- to be
watched and stopped.  Off, it runs straight on the session, the window
held until it ends.

### Claude Code uses SMAPpy's tools without asking
Shown for Claude Code.  Claude Code asks before each tool call unless told otherwise.  What the
assistant may change in SMAPpy is decided by *suggest* and *act*, so this
lets it use SMAPpy's tools without that question; its shell and its own
file edits still ask.

### close its terminal when SMAPpy quits
Shown for the assistants SMAPpy starts in a terminal; on by default.  When
the last SMAPpy window that allows connections quits, SMAPpy ends the
assistant and closes its terminal; while another window allows connections
the assistant moves there instead.  Untick it to keep the assistant and its
conversation running across a restart of SMAPpy; quit it yourself before
updating SMAPpy, as the update cannot replace files the assistant's
connection is using.

### connect it with
For *one I start myself*: *copy command* copies the command your client
runs to reach this window; *copy JSON* the same as the `mcpServers` entry
most clients' settings take.

### Start assistant
Allows connections, starts the assistant chosen above, and places it and
the proposals beside the picture.
