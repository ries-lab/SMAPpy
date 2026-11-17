---
title: Usage log
summary: A log of what the AI assistant tried, which plugins ran and which help pages were read, sent now and then to SMAPpy's developers so they learn what is hard to find or missing.
widget: smappy.gui.preferences.UsageLogPage
covers: [smappy.usage_log.bundle, smappy.usage_log.send, smappy.usage_log.upload, smappy.usage_log.tool_call]
---

## What it does

SMAPpy's developers learn most about what is missing from watching people
use it.  This log does the watching for them, if you agree.  It is **off**
until you tick *keep a usage log* (*File > Preferences > Usage log*, or
*Help > Send usage log...*), and nothing leaves your computer until you
also give a link to send it to.

The most telling part is the [AI assistant](plugin:Panels/Assistant).
When it searches for a plugin and finds none, reads SMAPpy's source
because a help page did not say enough, or writes code of its own for
something a plugin does, each is a sign that a feature is hard to find,
badly explained or missing.  The help pages you open say the same about
the documentation.

## How it works

**While you work.**  With the log on, SMAPpy writes down every call the
assistant makes to SMAPpy's tools -- which tool, with which settings,
whether it failed, how long it took, and the first few hundred characters
of its answer -- and every help page you open.  These go into the
configuration folder, under `usage-log`.

**When it is sent.**  SMAPpy makes one zip of everything new since the
last send:

- those events;
- the plugin runs from SMAPpy's own log (the one *Help > Show the log
  folder* opens), with their settings and any error;
- the assistant's notes of what it did with its own code, the copies of
  its scripts, and the tools it said it was missing;
- the assistant's conversations, as its program keeps them: Claude Code's
  and Codex's, from the folder the assistant works in
  (*Assistant > Settings...*).  These hold what you asked and what the
  assistant tried.

Pictures are taken out, and any single text longer than 8000 characters
is cut.  Your localizations and images are never in it.  File names and
folders are: what kind of data someone worked on is part of what is
learnt.

**Where it goes.**  The zip is uploaded to a cloud folder's
**upload-only link**: anyone with the link can add files to the folder,
and nobody can see what is in it.  So one link can be given to a whole lab.

## Making the link (for whoever collects the logs)

With u:cloud, the University of Vienna's Nextcloud, or any Nextcloud:

1. Make a folder, for example *SMAPpy usage logs*, in the u:cloud web
   page or in your synced u:cloud folder.
2. Share it: in the web page, the folder's share icon; in Finder or
   Explorer, the u:cloud entry of the folder's right-click menu, then
   *Share options*.
3. Add a **share link** (the **+** beside *Share link*).
4. In the link's **⋯** menu choose **File drop (upload only)**.  On some
   versions this is *Custom permissions* with only *Create* ticked.
5. If you like, set a password there.  Leave the expiry date empty, so the
   link keeps working.
6. Copy the link -- `https://ucloud.univie.ac.at/index.php/s/...` -- and
   give it, with the password, to the people who send logs.

## Controls

### keep a usage log

Off by default.  Turning it off stops new events being written.  What was
written stays in the configuration folder until you delete it, and is not
sent while the log is off.

### send to

The upload-only link.  It is kept in SMAPpy's configuration until you clear
the field, and an update of SMAPpy keeps it too.

### password

Only if the link was given one.  It is kept in SMAPpy's configuration file
as typed; it protects nothing but the right to drop files into the folder.

### your name

Goes into the names of the files sent, so the person who reads them knows
whose they are.  Left empty, a random name for this computer is used, the
same every time.

### send it every 7 days, at startup

When SMAPpy starts and seven days have passed since the last send, it sends
what is new in the background.  A failed send is shown on this page and
tried again at the next start.

### Send now

Sends what is new since the last send.  If nothing is new, nothing is sent.

### Save a copy...

Saves everything the log holds as a zip, without sending it, so you can look
inside first.
