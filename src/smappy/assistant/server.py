"""``smappy-mcp``: SMAPpy's tools for an AI assistant, over stdio.

The command an MCP client launches -- in Claude Code::

    claude mcp add smappy -- smappy-mcp --output ~/analysis

and the same command in any other client's MCP settings.  It holds one
session, headless, for as long as the client keeps it running.  With
``--attach`` it holds none: it forwards to the running GUI, whose session,
mode and output folder are used (`attach`).
`docs/ai-integration.md` says what it is for and what comes next.

stdout is the protocol's: the real one is kept for the messages and
``sys.stdout`` is pointed at stderr for as long as the server runs, so a
plugin that prints -- or a library warning on import -- cannot put a line in
the client's stream that is not JSON.  The client shows stderr as the
server's log.
"""
from __future__ import annotations

import argparse
import os
import sys


def utf8_protocol(reader, writer, log=None) -> None:
    """The protocol's streams in UTF-8, as MCP has it, and one ``\\n`` a line.

    On Windows ``sys.stdin`` and ``sys.stdout`` are in the ANSI code page, so
    a client's path with an ``ü`` arrived as two other letters, and an answer
    with a character outside cp1252 (the answers keep non-ASCII as it is)
    raised UnicodeEncodeError instead of being sent.  The log, stderr, the
    same, for the paths it names.  ``--attach`` forwards bytes and needs none.
    """
    for stream, newline in ((reader, None), (writer, "\n"), (log or sys.stderr, None)):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", **({"newline": newline} if newline else {}))


def main(argv=None) -> int:
    # the help is for the person setting a client up, not the module's notes
    p = argparse.ArgumentParser(
        prog="smappy-mcp",
        description="SMAPpy's tools for an AI assistant, over MCP (stdio).  "
                    "An MCP client launches this; in Claude Code: "
                    "claude mcp add smappy -- smappy-mcp --output ~/analysis.  "
                    "docs/ai-integration.md has the rest.")
    p.add_argument("--mode", choices=("act", "suggest"), default="act",
                   help="act: the assistant may change the session; suggest: "
                        "changes become proposals (default: act)")
    p.add_argument("--output", default=None,
                   help="the folder files may be written to (default: the "
                        "current folder)")
    p.add_argument("--knowledge", default=None,
                   help="the folder of remembered notes (default: in SMAPpy's "
                        "configuration folder)")
    p.add_argument("--open", default=None, metavar="FILE",
                   help="a localization file to open first")
    p.add_argument("--attach", nargs="?", const="", default=None, metavar="NAME",
                   help="use the running SMAPpy window instead (Assistant > Allow "
                        "connections there); NAME picks one of several by its "
                        "session's name.  The window's mode and output folder "
                        "apply")
    args = p.parse_args(argv)
    if args.attach is not None:
        if args.mode != "act" or args.output or args.knowledge or args.open:
            p.error("--attach takes the window's settings: no --mode, --output, "
                    "--knowledge or --open")
        from .attach import forward
        code = forward(args.attach)
        # out at once: the thread reading stdin is still blocked in a read,
        # and Python's own shutdown aborts on it ("Python quit unexpectedly")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)

    utf8_protocol(sys.stdin, sys.stdout, sys.stderr)
    protocol_out = sys.stdout
    sys.stdout = sys.stderr
    import matplotlib
    matplotlib.use("Agg")          # never a window: there is nobody to see it
    from .protocol import Server
    from .tools import INSTRUCTIONS, Assistant, source_note
    assistant = Assistant(mode=args.mode, output=args.output,
                          knowledge_folder=args.knowledge)
    if args.open:
        assistant.session.load(args.open)
    print(f"smappy-mcp: {args.mode} mode, writing under {assistant.output}",
          file=sys.stderr)
    try:
        Server(assistant, instructions=INSTRUCTIONS + source_note()).serve(
            sys.stdin, protocol_out)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
