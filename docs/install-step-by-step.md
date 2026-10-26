# Installing SMAPpy, step by step

This page is for you if you have never used a terminal or `pip`. It takes
about ten minutes, most of it waiting for downloads. You need an internet
connection, and nothing else: you do not need to install Python yourself, and
you do not need administrator rights.

The idea in one paragraph: SMAPpy is a Python program. We first install a
small helper called **uv**, which downloads the right version of Python and
installs SMAPpy into a place of its own, so it cannot interfere with anything
else on your computer. After that, you start SMAPpy by typing one word.

Follow the section for your computer: [Mac](#mac) or [Windows](#windows). Or,
if you use an AI assistant that can work on your computer, let it do the
steps for you.

## Let an AI assistant do it

Some AI assistants can run commands on your computer: Claude Code (also in the
*Code* tab of the Claude desktop app), OpenAI's Codex, Cursor and others. One
of these can do every step below for you, including the desktop icon, and
sort out what goes wrong on the way.

Use one of those, not a chat. A chat -- in a web browser, or the *Chat* tab
of the Claude desktop app -- either cannot run commands at all, or works in
a sandbox that cannot reach your home folder, where SMAPpy and uv are
installed and the desktop icon goes; it gets stuck on folder permissions
halfway through. In the Claude desktop app, open the *Code* tab and choose
your home folder as the folder to work in.

Give it this request:

> Install SMAPpy on this computer, with a desktop icon, following the
> section "For AI assistants" of
> https://github.com/ries-lab/SMAPpy/blob/main/docs/install-step-by-step.md

It will ask before running each command; the commands it should ask about
are the ones in [For AI assistants](#for-ai-assistants) below. When it is
done, double-click the new SMAPpy icon on your desktop.

## Mac

### 1. Open the Terminal

The Terminal is a window in which you type commands instead of clicking.

1. Press **Cmd ⌘ + Space**. A search field (Spotlight) appears.
2. Type `Terminal` and press **Return**.

A window opens with a line ending in `%` or `$`. This is the *prompt*: the
Terminal is waiting for you to type a command. In the steps below, you copy a
command from this page, paste it after the prompt with **Cmd ⌘ + V**, and
press **Return** to run it.

### 2. Install uv

Copy this line, paste it into the Terminal, and press **Return**:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

It prints a few lines and ends with something like
`everything's installed!`.

**Now close the Terminal window and open a new one** (step 1 again). The new
window knows where uv is; the old one does not.

### 3. Install SMAPpy

In the new Terminal window, paste this and press **Return**:

```bash
uv tool install --python 3.13 smappy-smlm
```

uv downloads Python 3.13 and SMAPpy with everything it needs (a few hundred
megabytes). When it is done, it prints `Installed 7 executables:` followed by names such as
`smappy-gui`.

If it also prints a warning that a directory `is not on your PATH`, run

```bash
uv tool update-shell
```

and then close the Terminal and open a new one.

### 4. Start SMAPpy

```bash
smappy-gui
```

The SMAPpy window opens. The first start takes a little longer than later
ones. Keep the Terminal window open while you work: closing it closes SMAPpy.
Messages that appear in the Terminal while SMAPpy runs are normal.

**Next time** you only need steps 1 and 4: open the Terminal, type
`smappy-gui`, press **Return**. Or make an icon for it once, as in step 5.

### 5. Optional: an icon on the desktop

To start SMAPpy with a double-click instead, paste this into the Terminal and
press **Return**:

```bash
echo "$(command -v smappy-gui)" > ~/Desktop/SMAPpy.command && chmod +x ~/Desktop/SMAPpy.command
```

A file `SMAPpy.command` appears on your desktop. Double-clicking it opens a
Terminal window and starts SMAPpy in it; as before, keep that window open
while you work. You can drag the file into the Dock to have it there too.

## Windows

### 1. Open PowerShell

PowerShell is a window in which you type commands instead of clicking.

1. Click the **Start** button (or press the **Windows key**).
2. Type `PowerShell`.
3. Click **Windows PowerShell** (not *Run as administrator*; you do not need
   it).

A blue or black window opens with a line ending in `>`, such as
`PS C:\Users\yourname>`. This is the *prompt*: PowerShell is waiting for you to
type a command. In the steps below, you copy a command from this page, paste it
after the prompt with **Ctrl + V** (or a right-click), and press **Enter** to
run it.

### 2. Install uv

Copy this line, paste it into PowerShell, and press **Enter**:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

It prints a few lines and ends with something like
`everything's installed!`.

**Now close the PowerShell window and open a new one** (step 1 again). The new
window knows where uv is; the old one does not.

### 3. Install SMAPpy

In the new PowerShell window, paste this and press **Enter**:

```powershell
uv tool install --python 3.13 smappy-smlm
```

uv downloads Python 3.13 and SMAPpy with everything it needs (a few hundred
megabytes). When it is done, it prints `Installed 7 executables:` followed by names such as
`smappy-gui`.

If it also prints a warning that a directory `is not on your PATH`, run

```powershell
uv tool update-shell
```

and then close PowerShell and open a new one.

### 4. Start SMAPpy

```powershell
smappy-gui
```

The SMAPpy window opens. The first start takes a little longer than later
ones; Windows may also ask whether to allow network access, which SMAPpy does
not need, so you can decline. Keep the PowerShell window open while you work:
closing it closes SMAPpy. Messages that appear in PowerShell while SMAPpy runs
are normal.

**Next time** you only need steps 1 and 4: open PowerShell, type
`smappy-gui`, press **Enter**. Or make an icon for it once, as in step 5.

### 5. Optional: an icon on the desktop

To start SMAPpy with a double-click instead, paste this into PowerShell and
press **Enter**:

```powershell
$s = (New-Object -ComObject WScript.Shell).CreateShortcut("$([Environment]::GetFolderPath('Desktop'))\SMAPpy.lnk"); $s.TargetPath = (Get-Command smappy-gui).Source; $s.Save()
```

A shortcut `SMAPpy` appears on your desktop. Double-clicking it opens a black
window and SMAPpy beside it; as before, keep the black window open while you
work. Right-click the shortcut and choose *Pin to Start* or *Pin to taskbar*
to have it there too (on Windows 11, under *Show more options*).

## Updating, and removing

The commands are the same on Mac and Windows. To get the newest version of
SMAPpy:

```bash
uv tool upgrade smappy-smlm
```

To remove SMAPpy from your computer:

```bash
uv tool uninstall smappy-smlm
```

Your settings and your own plugins are kept in SMAPpy's configuration folder
and are not removed by this.

## When something goes wrong

**`command not found: uv`** (Mac) or **`uv : The term 'uv' is not
recognized`** (Windows). The window was opened before uv was installed. Close
it, open a new one, and try again.

**`command not found: smappy-gui`** or **`'smappy-gui' is not recognized`**.
Run `uv tool update-shell`, close the window, open a new one, and try again.

**Windows: `DLL load failed`** when SMAPpy starts. Install the *Microsoft
Visual C++ Redistributable* (the x64 version) from
[Microsoft's download page](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist),
then start SMAPpy again.

**Anything else.** Copy everything the Terminal or PowerShell printed (select
it with the mouse, then Cmd ⌘ + C or Ctrl + C) and open an issue at
[github.com/ries-lab/SMAPpy/issues](https://github.com/ries-lab/SMAPpy/issues),
saying which computer you use and which step failed.

## If you already have Python

If Python 3.12, 3.13 or 3.14 is already installed and you prefer `pip`, this
does the same as steps 2 and 3:

```bash
python -m pip install smappy-smlm
```

(on a Mac, `python3` in place of `python`). If `smappy-gui` is then not found,
`python -m smappy.cli.gui` starts it as well.

## For AI assistants

This section is for an AI assistant installing SMAPpy on the user's behalf.
The user is likely not familiar with a terminal: explain briefly what each
command does when you ask to run it, and at the end tell them how to start
SMAPpy.

Rules:

* Install with uv as a uv tool, as below. Do not use the system Python,
  `sudo`, an administrator prompt, conda, or `pip` into any existing
  environment, unless the user asks for that.
* If `uv` is already installed, do not install it again. If `uv tool list`
  already shows `smappy-smlm`, run `uv tool upgrade smappy-smlm` instead of
  installing.
* Your shell will not see newly installed programs on its PATH. Call `uv` by
  its full path after installing it (`~/.local/bin/uv` on macOS,
  `$HOME\.local\bin\uv.exe` on Windows, unless the installer printed
  another location), and find SMAPpy's programs with `uv tool dir --bin`.
* Do not start the GUI to check the install: it blocks until the user closes
  it. Use the checks below.

Steps on **macOS** (zsh or bash):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh             # skip if uv exists
uv tool install --python 3.13 smappy-smlm
uv tool update-shell                                         # so the user's new terminals find smappy-gui
"$(uv tool dir)/smappy-smlm/bin/python" -c "import smappy, PySide6.QtWidgets; print('ok')"
echo "\"$(uv tool dir --bin)/smappy-gui\"" > ~/Desktop/SMAPpy.command && chmod +x ~/Desktop/SMAPpy.command
```

Steps on **Windows** (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # skip if uv exists
uv tool install --python 3.13 smappy-smlm
uv tool update-shell
& "$(uv tool dir)\smappy-smlm\Scripts\python.exe" -c "import smappy, PySide6.QtWidgets; print('ok')"
$s = (New-Object -ComObject WScript.Shell).CreateShortcut("$([Environment]::GetFolderPath('Desktop'))\SMAPpy.lnk"); $s.TargetPath = "$(uv tool dir --bin)\smappy-gui.exe"; $s.Save()
```

The check prints `ok`. If it fails on Windows with `DLL load failed`, the
Microsoft Visual C++ Redistributable (x64) is missing; tell the user, since
installing it needs them to confirm an administrator prompt. Make the desktop
icon only if the user asked for one. For anything else that fails, the
section [When something goes wrong](#when-something-goes-wrong) lists the
known problems; otherwise report the error to the user, and suggest opening
an issue at https://github.com/ries-lab/SMAPpy/issues.

## What next

[SMAPpy in three minutes](https://ries-lab.github.io/SMAPpy/quickstart/) is a
short guided tour on simulated data: a good first thing to do once the window
is open.
