"""``smappy-mcp``: SMAPpy's tools for an AI assistant, over stdio.

The command an MCP client launches -- in Claude Code::

    claude mcp add smappy -- smappy-mcp --output ~/analysis

and the same command in any other client's MCP settings.  It holds one
session, headless, for as long as the client keeps it running.
`docs/ai-integration.md` says what it is for and what comes next.

stdout is the protocol's: the real one is kept for the messages and
``sys.stdout`` is pointed at stderr for as long as the server runs, so a
plugin that prints -- or a library warning on import -- cannot put a line in
the client's stream that is not JSON.  The client shows stderr as the
server's log.
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="smappy-mcp", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
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
    args = p.parse_args(argv)

    protocol_out = sys.stdout
    sys.stdout = sys.stderr
    import matplotlib
    matplotlib.use("Agg")          # never a window: there is nobody to see it
    from .protocol import Server
    from .tools import INSTRUCTIONS, Assistant
    assistant = Assistant(mode=args.mode, output=args.output,
                          knowledge_folder=args.knowledge)
    if args.open:
        assistant.session.load(args.open)
    print(f"smappy-mcp: {args.mode} mode, writing under {assistant.output}",
          file=sys.stderr)
    try:
        Server(assistant, instructions=INSTRUCTIONS).serve(sys.stdin, protocol_out)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
