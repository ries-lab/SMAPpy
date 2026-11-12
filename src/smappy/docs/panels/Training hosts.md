---
title: Training hosts
summary: The workstations DECODE trains and localizes on, reached over ssh -- and, step by step, how to connect one when you have never used ssh, or how to let an AI assistant do it with you.
widget: smappy.gui.preferences.HostsPage
covers: [smappy.training.jobs.Host, smappy.training.remote.check, smappy.training.remote.ssh]
---

## What it does

Training a [DECODE](plugin:Localize/DECODE) network takes hours on a
graphics card (a GPU) and days without one.  If your own computer has no
GPU, a **training host** -- a workstation or server with an NVIDIA GPU,
usually in a server room -- can do the work.  SMAPpy then sends it a
small folder, starts the training there, and copies the progress and
the finished model back.  The [DECODE fitter](plugin:Localize/DECODE)
can run its network there too (*run on*).  Your movies stay on your
computer.

SMAPpy talks to the workstation with **ssh**, the standard program for
logging in to another computer; it is part of macOS, Linux and Windows 10
and 11.  SMAPpy cannot type a password, so ssh has to log in without one.
That is done with a **key**: a pair of files on your computer, one secret,
one you give the workstation once.  After that, ssh proves who you are with
the key instead of your password.

You need, from whoever runs the workstation (your facility or IT):

- its **address**, for example `gpu1.institute.edu`;
- your **user name** and **password** on it (the password only once, in
  step 5);
- whether you must be on the institute's network or **VPN** to reach it.

Setting it up takes about fifteen minutes, once.  If you would rather not
type the commands yourself, see *Let an AI assistant help* below.

## How it works

The commands below are typed in a **terminal**, a window where you type
commands instead of clicking:

- **macOS:** *Applications > Utilities > Terminal*.
- **Windows 10 and 11:** the Start menu, type *PowerShell*, open *Windows
  PowerShell*.
- **Linux:** *Terminal* in the applications.

Type a command, or paste it, and press Enter.  In the examples, replace
`jdoe` by your user name and `gpu1.institute.edu` by the workstation's
address.

**1. Log in once by hand.**  Type

```
ssh jdoe@gpu1.institute.edu
```

The first time, ssh asks *Are you sure you want to continue connecting
(yes/no)?*: type `yes`.  Then it asks for your password; **nothing appears
while you type it** -- that is normal -- and press Enter.  You are now on
the workstation: the line where you type shows its name.  Type `exit` to come
back.  If it says *Connection timed out* or *Could not resolve hostname*,
you are not on the right network: connect to the VPN, or ask IT.

**2. Check that the workstation has a GPU.**  Log in again as in step 1 and
type

```
nvidia-smi
```

It draws a table naming the GPU, for example *NVIDIA H100*.  If it says
*command not found*, the workstation has no NVIDIA GPU, or its driver is
missing: ask whoever runs it.  Stay logged in for step 3.

**3. Install SMAPpy there.**  Still on the workstation, type

```
python3 -m pip install --user "smappy-smlm[decode]"
```

It downloads SMAPpy with PyTorch, a few GB, which takes some minutes.  Then
check it:

```
smappy-decode --help
```

It should print a short list of commands.  If it says *command not found*,
type `~/.local/bin/smappy-decode --help`; if that works, write
`~/.local/bin/smappy-decode` as the host's *command* in step 7.  Many
workstations keep Python in a conda environment instead; then ask whoever
runs it, or let an assistant help.  Install the same SMAPpy version as on
your computer (*Help > About SMAPpy* says which), for example
`"smappy-smlm[decode]==0.6.5"`.  Type `exit` to come back.

**4. Make a key** -- on *your* computer, not the workstation.  Type

```
ssh-keygen -t ed25519
```

and press Enter at each question: where to save it (the default is right),
and twice for no passphrase.  If it says the file *already exists*, you have
a key already: answer `n` and go on.  The key is two files in the folder
`.ssh` of your home folder: `id_ed25519` is the secret one -- never send it
to anyone, like a password -- and `id_ed25519.pub` is the one the
workstation gets.

**5. Give the workstation your key.**  On macOS and Linux:

```
ssh-copy-id jdoe@gpu1.institute.edu
```

On Windows, in PowerShell (one line):

```
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh jdoe@gpu1.institute.edu "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

It asks for your password once more: the last time.

**6. Check that no password is asked.**  Type

```
ssh jdoe@gpu1.institute.edu
```

It should log you in straight away.  Type `exit`.  If it still asks for the
password, step 5 did not work: do it again, and look at *When it does not
work* below.

**7. Enter it in SMAPpy.**  In this page (*File > Preferences*, *Training
hosts*), press *Add* and fill in:

- *name*: any short name, for example `gpu1`;
- *ssh*: `jdoe@gpu1.institute.edu`;
- *folder there*: where the job folders go on the workstation;
  `~/smappy_training` (a folder in your home folder there) is fine;
- *command*: `smappy-decode`, or the full path from step 3.

Press *Test*.  It should say *gpu1 answers; its GPU: NVIDIA ...*.  The
workstation now appears under *run on* in the training window and in the
DECODE fitter.

## Let an AI assistant help

An AI assistant that runs commands in a terminal -- for example Claude
Code, or the one *Assistant > Start assistant* opens -- can do steps 1 to 6
with you, check each result and explain what went wrong.  Start it, and
paste something like this, with your user name and address:

> I want SMAPpy on this computer to train DECODE on a workstation over
> ssh.  The workstation is gpu1.institute.edu and my user name is jdoe.
> Please check whether I already have an ssh key and make one if not, help
> me copy it to the workstation, check that ssh then logs in without a
> password, and check there that nvidia-smi and smappy-decode work
> (install smappy-smlm[decode] if not).  I will type my password myself
> when ssh asks.  Explain each step before you run it, and tell me what
> to enter in SMAPpy's Training hosts at the end.

**Never type your password into the assistant's chat.**  When ssh asks for
it, type it in the terminal yourself; an assistant does not need it.

## When it does not work

- **Permission denied (publickey,password)** or the password is asked
  again: the key was not copied (step 5), or the workstation takes no keys
  -- ask whoever runs it.
- **Connection timed out**, **No route to host**, **Could not resolve
  hostname**: the network.  Connect to the VPN, check the address, or ask
  IT whether the workstation can be reached from where you are.
- **Host key verification failed**: the workstation was reinstalled, or
  this is another computer with its address.  Ask IT before going on.
- *Test* says the **command did not answer**: SMAPpy is not installed there,
  or *command* is not where it is: step 3.
- *Test* says **no NVIDIA GPU seen there**: the training would run on its
  processor, far slower.  Step 2.
- A training refuses a job **prepared by another SMAPpy version**: install
  the same version there (step 3), or tick *train jobs another SMAPpy
  version prepared*.

## Controls

### How to connect a workstation...
This page.

### name
What the training window and the DECODE fitter call the workstation, in
*run on*.

### ssh
What `ssh` is given to reach it: `user@address`, or a short name from your
`~/.ssh/config`.  It must log in without a password (a key, step 4).

### folder there
Where the job folders go on the workstation; `~` is your home folder there.

### command
What runs the training there: `smappy-decode`, its full path, or a Python
with `-m decode_smlm.cli` (a conda environment's, say).

### train jobs another SMAPpy version prepared
A job records the SMAPpy version that prepared it, and a training refuses
one of another version unless this is ticked: for a workstation whose
SMAPpy is behind or ahead of this computer's.

### Add
A new workstation, to fill in.

### Remove
Takes the chosen workstation off the list; nothing on it is touched.

### Test
Logs in and asks the *command*, and says which GPU it found.
