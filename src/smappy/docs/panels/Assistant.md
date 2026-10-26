---
title: Assistant
summary: Lets an AI assistant on this computer work with the open session, says who is connected and what it may do, and lists what it proposed for you to open or dismiss.
widget: smappy.gui.assistant_pane.AssistantPane
covers: [smappy.gui.assistant_pane.AssistantPane, smappy.gui.assistant_server.AssistantServer, smappy.gui.assistant_server.RenderViewer, smappy.assistant.attach.forward, smappy.assistant.attach.choose, smappy.assistant.tools.Assistant]
---

## What it does

An AI assistant -- Claude Code, Codex, or any program that speaks the Model
Context Protocol (MCP) -- can use SMAPpy's tools: read the plugins' pages
and their code, try a plugin on a copy of the data and look at its numbers
and figures, build a chain of plugins, and draw a picture of the
localizations.  The assistant runs in its own program, with your own
account; SMAPpy has no model and no key of its own, and nothing leaves the
computer through SMAPpy.

This pane connects such an assistant to the session open in this window, so
that what it looks at and what it changes is what you see.  Nothing can
connect until *allow connections* is ticked (or *Assistant > Allow
connections* is chosen), and only programs running as your user on this
computer can.

## Connecting a client

The client starts a small program that passes its messages on to this
window:

    smappy-mcp --attach

In Claude Code that is

    claude mcp add smappy-gui -- smappy-mcp --attach

and any other client takes the same command in its MCP settings.  With
several SMAPpy windows allowing connections, `smappy-mcp --attach NAME`
chooses one by its session's name (*File > Session name*); without a name it
takes the one that allowed connections last.  The client must be started,
or reconnected, after connections are allowed here.

## Suggest and act

In *suggest* mode the assistant reads and tries on copies of the session,
and a change it would make -- running a plugin on the data, opening or
saving a file, undoing -- waits in the list below for you.  What only
changes what is shown it makes at once: a layer's filter, grouping, and
moving the 2D view onto a place or an ROI (its *show* tool).  In *act* mode
it changes the session itself.  The assistant cannot switch between the two;
only this pane can.

Files are written only under the folder shown after *writes under*.

While the assistant's call runs, the window is busy: a long run it applies
-- a fit, a drift correction over a large file -- holds the window until it
ends, and the line above the proposals says what is running.

## Controls

### allow connections
Listen for an assistant's client.  Unticked, nobody can connect, and a
client already connected is disconnected.  The line below says who is
connected.

### copy
Copies the command that connects Claude Code to this window.

### suggest
The assistant proposes; you decide.  The default each time SMAPpy starts.

### act
The assistant runs plugins on the data and opens and saves files itself.
Each change is in *Edit > Undo* as one from a panel is.

### choose...
The folder the assistant may write into, instead of the first open file's.

### follow file
Back to the folder of the first open file, which changes as files are opened.

### proposals
What the assistant proposed in *suggest* mode, newest last, each with its
reason.  A run of a plugin or a chain has *Open*: the plugin's own window
with the proposed settings filled in, to check and run there with its Run --
nothing runs before you press it.  Opening a file, undoing or saving has
*Do it*.  *Dismiss* takes a proposal off the list.  The assistant can see
what became of each one: proposed, opened, done or dismissed.
