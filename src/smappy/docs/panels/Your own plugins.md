---
title: Your own plugins
summary: How to make an analysis of your own -- by describing it to an AI assistant, or by writing it -- and where it lives.
---

## What it is

Every analysis in SMAPpy is a plugin: a form of settings, a Run button, and
a few lines of Python behind them.  When none of the shipped ones measures
what you need, you can add your own, and it behaves like any other: it has a
form, it runs on the localizations you are looking at (the layer's filter,
inside the ROI), it draws its figures, and it can be pinned to a tab or run
per ROI.

You do not need to be a programmer.  The usual way is to describe what you
want to an AI assistant -- "for each cluster, the number of localizations and
the radius of gyration, as a histogram" -- and let it write the plugin.  Your
plugin folder comes with the instructions an assistant needs.

## Where your plugins live

SMAPpy always reads one folder of your own: *Plugins > Show my plugin folder*
opens it, and the preferences show its path.  A Python file saved there is a
plugin the next time the Plugins menu or a tab's **+** is opened; nothing has
to be installed or registered, and SMAPpy does not need a restart.

* A file saved directly in the folder appears under **User** in the Plugins
  menu: `cluster_sizes.py` becomes *User > Cluster Sizes*.
* A file in subfolders takes its place from them, as the shipped plugins do:
  `Analysis/Measure/cluster_sizes.py` sits beside the shipped measurements.
* A plugin with the same place as a shipped one replaces it.
* A use-once plugin, made for one dataset, can simply be deleted afterwards.

Plugin folders of your own beyond this one -- a lab's shared folder, say --
are added in *File > Preferences*.

## How to make one

**Start from the example.**  *Plugins > New plugin...* asks for a name and
copies a working example plugin into your folder under that name.  It
measures how far each localization is from its nearest neighbour, which is
small enough to read and has every part a plugin needs: settings, a
measurement, a figure, and numbers per ROI.  The same from a terminal:
`smappy-plugins new "cluster sizes"`.

**With an AI assistant that works on files** (Claude Code, Codex and
similar): open a terminal in your plugin folder, start the assistant there,
and describe what you need -- what to measure, on which localizations, what
the result should look like.  The folder holds `AGENTS.md` (and `CLAUDE.md`,
which reads it), which tells the assistant how a SMAPpy plugin is written, so
it does not have to guess.  SMAPpy keeps that file up to date with the version
you have installed.

**With a chat assistant**: paste the contents of `AGENTS.md` into the chat,
then your description, and save the answer as a `.py` file in your folder.

**By hand**: copy `_template.py` (SMAPpy's example, which the menu skips
because its name starts with `_`) and change it.  A plugin is a list of
settings and a `run` function that reads the localizations and hands back a
text, numbers and figures.

**Trying it.**  Open it from the Plugins menu and press Run.  When you or the
assistant change the file, open it again: the new code is read.  If it does
not appear, `smappy-plugins check` in a terminal says why -- a typing error,
or a missing line the program requires.

## What is asked of your own plugins

Very little.  A plugin in your folder needs only to work: the program refuses
a plugin that does not say whether it changes the table (`changes_locs`), and
that is all.  A help page like this one, tests, and a version number are
yours to decide on.  An assistant will offer them, and for a plugin you mean
to keep they are worth having: a help page (`cluster_sizes.md` beside
`cluster_sizes.py`) opens with the **?** like any other, and a test on
simulated data is how you know the numbers are right.

## Sharing a plugin

A plugin other people would use can become part of SMAPpy.  Open a pull
request at [github.com/ries-lab/SMAPpy](https://github.com/ries-lab/SMAPpy)
that adds it to the shipped plugins.  What is shipped follows the
repository's rules, which your own plugins do not have to: a help page that
explains the method, tests that check the numbers on simulated data, and the
house style.  The repository's `CLAUDE.md` sets them out, and an assistant
working in a copy of the repository reads it.

To share with your lab without that, put the file in a folder everyone can
reach and add the folder in *File > Preferences*.
