"""The part of MCP the server speaks: JSON-RPC 2.0, a message per line.

MCP (the Model Context Protocol, https://modelcontextprotocol.io) is how a
client -- Claude Code, Codex, a chat program -- finds a server's tools and
calls them.  A tools-only server needs five messages of it: ``initialize``
and the client's ``notifications/initialized``, ``tools/list``,
``tools/call`` and ``ping``.  Those are here, over any pair of text streams,
so the same `Server` serves stdio now and the GUI's local socket later.

Written rather than taken from the `mcp` package: its 2.0 renamed the server
class and changed its API, and supporting both versions users have would be
more code than this (`docs/ai-integration.md`).  The package's client is used
in the tests, when installed, to check that a real client talks to this.
"""
from __future__ import annotations

import json
import traceback
from typing import Any, Dict, Optional, TextIO

from .tools import Assistant, Image, ToolError

# the revisions this server answers; a client asking for one gets it, any
# other the newest, and the client decides whether it can work with that
VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

PARSE_ERROR, INVALID_REQUEST, NO_METHOD, INVALID_PARAMS = -32700, -32600, -32601, -32602


def content(value) -> list:
    """A tool's answer as MCP content: a picture as an image, the rest as
    JSON text, which every client shows and every model reads."""
    if isinstance(value, Image):
        out = [{"type": "image", "data": value.base64(), "mimeType": "image/png"}]
        if value.caption:
            out.append({"type": "text", "text": value.caption})
        return out
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    return [{"type": "text", "text": json.dumps(value, indent=1, ensure_ascii=False,
                                                default=str)}]


class Server:
    """Answers MCP requests with an `Assistant`'s tools."""

    def __init__(self, assistant: Assistant, name: str = "smappy",
                 instructions: str = ""):
        from .. import __version__
        self.assistant = assistant
        self.info = {"name": name, "version": __version__}
        self.instructions = instructions
        self.client: Dict[str, Any] = {}

    # ------------------------------------------------------------ the methods
    def initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.client = params.get("clientInfo") or {}
        asked = params.get("protocolVersion")
        out = {"protocolVersion": asked if asked in VERSIONS else VERSIONS[0],
               "capabilities": {"tools": {"listChanged": False}},
               "serverInfo": self.info}
        if self.instructions:
            out["instructions"] = self.instructions
        return out

    def tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"tools": [{"name": spec.name, "description": spec.description,
                           "inputSchema": spec.schema()}
                          for spec in self.assistant.tools().values()]}

    def tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str):
            raise _Invalid("tools/call needs a tool's name")
        try:
            value = self.assistant.call(name, params.get("arguments") or {})
        except ToolError as error:
            # the model reads this and corrects the call: not a protocol error
            return {"content": content(str(error)), "isError": True}
        except Exception as error:
            return {"content": content(f"{type(error).__name__}: {error}\n"
                                       + traceback.format_exc(limit=6)),
                    "isError": True}
        return {"content": content(value), "isError": False}

    METHODS = {"initialize": initialize, "tools/list": tools_list,
               "tools/call": tools_call, "ping": lambda self, params: {},
               # asked for by some clients whatever the capabilities say
               "resources/list": lambda self, params: {"resources": []},
               "prompts/list": lambda self, params: {"prompts": []}}

    # ---------------------------------------------------------- the messages
    def handle(self, message: Any) -> Optional[Dict[str, Any]]:
        """The answer to one message; None for a notification."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, INVALID_REQUEST, "not a JSON-RPC 2.0 message")
        ident = message.get("id")
        method = message.get("method")
        if "id" not in message:
            return None                  # a notification: nothing is answered
        if not isinstance(method, str):
            return _error(ident, INVALID_REQUEST, "no method")
        function = self.METHODS.get(method)
        if function is None:
            return _error(ident, NO_METHOD, f"no method {method!r}")
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return _error(ident, INVALID_PARAMS, "params must be an object")
        try:
            return {"jsonrpc": "2.0", "id": ident, "result": function(self, params)}
        except _Invalid as error:
            return _error(ident, INVALID_PARAMS, str(error))
        except Exception as error:
            return _error(ident, -32603, f"{type(error).__name__}: {error}")

    def handle_line(self, line: str) -> Optional[str]:
        line = line.strip()
        if not line:
            return None
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            return json.dumps(_error(None, PARSE_ERROR, str(error)))
        if isinstance(message, list):
            # a batch, which revisions before 2025-06-18 allowed
            answers = [a for a in (self.handle(m) for m in message) if a is not None]
            return json.dumps(answers) if answers else None
        answer = self.handle(message)
        return None if answer is None else json.dumps(answer, ensure_ascii=False)

    def serve(self, reader: TextIO, writer: TextIO) -> None:
        """Answer every line until the reader ends."""
        for line in reader:
            answer = self.handle_line(line)
            if answer is not None:
                writer.write(answer + "\n")
                writer.flush()


class _Invalid(ValueError):
    pass


def _error(ident, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": message}}
