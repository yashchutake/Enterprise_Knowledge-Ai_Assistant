"""
Thin async wrapper around the official MCP Filesystem Server
(@modelcontextprotocol/server-filesystem), launched over stdio.

This is the "external system" the assignment requires the assistant to
integrate with via MCP:
  - RetrieverAgent calls `mcp_read_file` to pull supplementary context
    (data/mcp_workspace/faq_extra.md) that lives OUTSIDE the vector store,
    simulating a live external knowledge source.
  - MCPLogAgent calls `mcp_write_file` to append every question/answer/score
    to data/mcp_workspace/query_log.md, simulating writing back to an
    external ticketing/logging system.

The server is restricted to config.MCP_WORKSPACE_DIR only (the directory
passed as its single allowed root), so it cannot read/write anywhere else
on disk.
"""
import os
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .. import config

os.makedirs(config.MCP_WORKSPACE_DIR, exist_ok=True)

SERVER_PARAMS = StdioServerParameters(
    command=config.MCP_FILESYSTEM_COMMAND,
    args=[*config.MCP_FILESYSTEM_ARGS, config.MCP_WORKSPACE_DIR],
)


@asynccontextmanager
async def mcp_session():
    """Open a fresh MCP session against the Filesystem Server."""
    async with stdio_client(SERVER_PARAMS) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


def _extract_text(result) -> str:
    parts = []
    for block in getattr(result, "content", []):
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


async def mcp_read_file(filename: str) -> str | None:
    """Read a file from the MCP workspace via the Filesystem MCP Server."""
    path = os.path.join(config.MCP_WORKSPACE_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        async with mcp_session() as session:
            result = await session.call_tool("read_file", {"path": path})
            return _extract_text(result)
    except Exception as exc:  # noqa: BLE001 - surface as inline error, don't crash the graph
        return f"[MCP read_file error: {exc}]"


async def mcp_write_file(filename: str, content: str) -> bool:
    """Write/overwrite a file in the MCP workspace via the Filesystem MCP Server."""
    path = os.path.join(config.MCP_WORKSPACE_DIR, filename)
    try:
        async with mcp_session() as session:
            await session.call_tool("write_file", {"path": path, "content": content})
        return True
    except Exception:  # noqa: BLE001
        return False


async def mcp_list_files() -> list[str]:
    """List files currently visible to the MCP Filesystem Server."""
    try:
        async with mcp_session() as session:
            result = await session.call_tool(
                "list_directory", {"path": config.MCP_WORKSPACE_DIR}
            )
            return _extract_text(result).splitlines()
    except Exception:  # noqa: BLE001
        return []
