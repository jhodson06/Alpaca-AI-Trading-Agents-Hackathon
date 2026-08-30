"""
utils/mcp_bridge.py — Subprocess abstraction for the Alpaca MCP server lifecycle.

This module owns exactly one responsibility: standing up a local stdio
subprocess running the Alpaca MCP server, and handing back an initialized
`ClientSession` that Layer 2 (core/ampp_agent.py) can call tools against.

Design notes:
  - stdio, not SSE. The MCP spec supports both a local stdio transport and a
    remote Server-Sent Events transport. This bridge is hardcoded to stdio:
    it launches `uvx alpaca-mcp-server serve` as a child process and
    communicates over its stdin/stdout pipes. This avoids an extra network
    hop and matches the blueprint's requirement to run the official Alpaca
    MCP package directly.
  - Explicit environment injection. A stdio subprocess does not reliably
    inherit the parent process's environment variables on every OS. The
    Alpaca credentials and paper-trading flag are therefore explicitly
    copied into the child's `env` dict rather than assumed to be visible.
    Skipping this step is a common, silent cause of MCP subprocess auth
    failures.
  - Context-manager shaped. `create_alpaca_mcp_session` is an
    async-context-manager-yielding function (via
    `contextlib.asynccontextmanager`), so callers get automatic, exception-
    safe teardown of both the ClientSession and the underlying subprocess
    pipes via a single `async with` block — this matters a lot for Layer 2,
    which must be able to tear down and retry cleanly after any failure
    without leaking subprocesses.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "The 'mcp' package is required for utils/mcp_bridge.py but is not "
        "installed. Install it with: pip install 'mcp>=0.1.0'"
    ) from exc

import config


def _build_subprocess_env() -> dict[str, str]:
    """Construct the environment dict injected into the MCP subprocess.

    Starts from a copy of the current process environment (so PATH and
    other OS-level plumbing the subprocess needs to actually launch `uvx`
    are preserved) and then explicitly overlays the Alpaca credentials and
    paper-trading flag, since those are the values whose absence causes a
    silent, confusing authentication failure inside the child process
    rather than a clear error here.
    """
    env = os.environ.copy()
    env["ALPACA_API_KEY"] = config.ALPACA_API_KEY
    env["ALPACA_SECRET_KEY"] = config.ALPACA_SECRET_KEY
    # Stringify explicitly: subprocess environments are string-only, and we
    # want this to always be the literal "true" here, never a stray Python
    # bool repr, given config.py has already validated this is paper mode.
    env["ALPACA_PAPER_TRADE"] = "true" if config.ALPACA_PAPER_TRADE else "false"
    return env


@asynccontextmanager
async def create_alpaca_mcp_session() -> AsyncIterator[ClientSession]:
    """Launch the Alpaca MCP server over stdio and yield a ready ClientSession.

    Usage:
        async with create_alpaca_mcp_session() as session:
            result = await session.call_tool("get_option_chain", {...})

    On exit (normal or exceptional), both the ClientSession and the
    underlying subprocess/pipes are torn down automatically, because this
    function delegates to two nested async context managers
    (`stdio_client` and `ClientSession`) rather than manually managing
    their lifecycles.
    """
    server_params = StdioServerParameters(
        command=config.MCP_SERVER_COMMAND,
        args=config.MCP_SERVER_ARGS,
        env=_build_subprocess_env(),
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session
