"""SMAPpy's tools for an AI assistant, served over MCP.

The assistant -- Claude Code, Codex, any client that speaks the Model
Context Protocol -- runs in its own program; this package is what it calls:
the session, the plugins, their pages and source, figures and pictures, and a
local store of what the user asked it to remember.  No model, no key and no
network here, so SMAPpy works offline as before and with any model.
`docs/ai-integration.md` is the design: why the AI is in the client, the two
tiers, *suggest* and *act*, and what comes next.

    tools.Assistant      the tools, plain functions over a `Session`
    index                the plugin index `find_plugins` searches
    knowledge            the notes `remember` keeps
    protocol             the part of MCP the server speaks, JSON-RPC
    server               ``smappy-mcp``, the command a client launches
    attach               ``smappy-mcp --attach``: a client to the running GUI

The GUI serves the same tools over a local socket (`gui/assistant_server.py`,
*Assistant > Allow connections*) and imports this package only then; nothing
else imports it.
"""
