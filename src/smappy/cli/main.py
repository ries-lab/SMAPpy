"""``smappy``: one command for everything SMAPpy runs from a terminal.

    smappy gui [FILE ...]          the windows
    smappy info FILE ...           what a localization file holds
    smappy convert IN [OUT]        a localization file in another format
    smappy fit DATA OUT            camera frames to localizations
    ...

Each subcommand is the module its ``smappy-xxx`` script runs, given the rest
of the line as its own: ``smappy fit -h`` is ``smappy-fit -h``.  Nothing is
imported until a subcommand is chosen, so ``smappy -h`` answers at once and
never loads Qt, numpy or torch.
"""
import difflib
import importlib
import sys

# name: (module, function, what it does), in the order the help lists them
COMMANDS = {
    "gui": ("smappy.cli.gui", "main", "the windows, on the localization files given"),
    "info": ("smappy.cli.info", "main", "what a localization file holds"),
    "convert": ("smappy.cli.convert", "main",
                "a localization file in another format, or as smappy HDF5"),
    "fit": ("smappy.cli.fit", "main", "camera frames to localizations"),
    "drift": ("smappy.cli.drift", "main", "drift correction of a localization file"),
    "calibrate": ("smappy.cli.calibrate", "main", "a spline PSF calibration from beads"),
    "batch": ("smappy.cli.batch", "main", "a chain of plugins over many files"),
    "batch-gui": ("smappy.gui.batch_window", "main", "the batch window"),
    "plugins": ("smappy.user_plugins", "main", "your own plugins: their folder, a new one"),
    "mcp": ("smappy.assistant.server", "main", "SMAPpy's tools for an AI assistant"),
    "decode": ("decode_smlm.cli", "main",
               "DECODE's training (needs smappy-smlm[decode])"),
}


def usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = ["usage: smappy COMMAND [ARGS ...]", "",
             "SMAPpy: single-molecule localization microscopy.", "", "commands:"]
    lines += [f"  {name:<{width}}  {what}" for name, (_, _, what) in COMMANDS.items()]
    lines += ["", "smappy COMMAND -h says what a command takes.",
              "smappy --version prints the version."]
    return "\n".join(lines)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        if args[1:2] and args[0] == "help":       # `smappy help fit` as `smappy fit -h`
            return main([args[1], "-h"])
        print(usage())
        return 0
    if args[0] in ("-V", "--version"):
        from .. import __version__
        print(f"smappy {__version__}")
        return 0
    name, rest = args[0], args[1:]
    if name not in COMMANDS:
        close = difflib.get_close_matches(name, COMMANDS, n=1)
        hint = f"; did you mean `smappy {close[0]}`?" if close else ""
        print(f"smappy: no command {name!r}{hint}\n\n{usage()}", file=sys.stderr)
        return 2
    module, function, _ = COMMANDS[name]
    try:
        run = getattr(importlib.import_module(module), function)
    except ModuleNotFoundError as e:     # an extra not installed: decode_smlm, its torch
        print(f"smappy {name}: {e}", file=sys.stderr)
        return 1
    # the subcommand reads sys.argv as its own script would: the usage line
    # says `smappy fit` (`command_name`), and a main that takes no argv works
    sys.argv = [f"smappy {name}", *rest]
    code = run()
    return code if isinstance(code, int) else 0


if __name__ == "__main__":
    sys.exit(main())
