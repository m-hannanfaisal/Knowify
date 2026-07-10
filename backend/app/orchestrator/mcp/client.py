import asyncio
import json
import os
import time
from typing import Any, Optional
import structlog
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = structlog.get_logger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "mcp_config.json")


def load_mcp_config() -> dict:
    """Loads standard MCP servers configurations block.

    Returns:
        dict: Standard mcpServers registry.
    """
    if not os.path.exists(CONFIG_PATH):
        logger.warn("mcp_config_missing", path=CONFIG_PATH)
        return {"mcpServers": {}}
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("mcp_config_load_failed", error=str(e))
        return {"mcpServers": {}}


def get_mock_mcp_response(server_name: str, tool_name: str, params: dict) -> str:
    """Generates mock responses for filesystem and web-search tools to support offline tests.

    Args:
        server_name (str): The MCP server identifier.
        tool_name (str): The tool name.
        params (dict): Inputs dictionary.

    Returns:
        str: Mock text response content.
    """
    if server_name == "filesystem":
        if "read" in tool_name or "view" in tool_name:
            return "Mock filesystem: Loaded file content from local mock tool execution."
        return f"Mock filesystem success: Operation '{tool_name}' executed."
    elif server_name == "web-search":
        query = params.get("query", "")
        return f"Mock Search results: Web page references and info matching query '{query}'."
    return f"Mock response: Invocation of {server_name}/{tool_name} completed."


async def call_mcp_tool(server_name: str, tool_name: str, params: dict) -> str:
    """Launches an MCP server subprocess, connects via stdio, and calls the specified tool.

    Includes automatic fallbacks to mock results if connections fail or dependencies are missing.

    Args:
        server_name (str): Target registered MCP server.
        tool_name (str): Target tool.
        params (dict): Arguments dictionary.

    Returns:
        str: Response text content.
    """
    start_time = time.perf_counter()
    config = load_mcp_config()
    server_configs = config.get("mcpServers", {})

    # Offline mock logic for sandbox compatibility
    if server_name not in server_configs:
        logger.info(
            "mcp_server_not_configured_using_mock",
            server=server_name,
            tool=tool_name,
        )
        return get_mock_mcp_response(server_name, tool_name, params)

    srv_cfg = server_configs[server_name]
    command = srv_cfg.get("command")
    args = srv_cfg.get("args", [])

    if not command:
        return get_mock_mcp_response(server_name, tool_name, params)

    try:
        server_params = StdioServerParameters(command=command, args=args, env=None)
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                response = await session.call_tool(tool_name, arguments=params)

                # Parse standard MCP tool call text contents
                contents = getattr(response, "content", [])
                text_outputs = []
                for content in contents:
                    if getattr(content, "type", "text") == "text":
                        text_outputs.append(getattr(content, "text", ""))

                result_text = "\n".join(text_outputs)
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                logger.info(
                    "mcp_tool_called_successfully",
                    server=server_name,
                    tool=tool_name,
                    latency_ms=latency_ms,
                )
                return result_text

    except Exception as e:
        logger.warn(
            "mcp_tool_execution_failed_falling_back_to_mock",
            server=server_name,
            tool=tool_name,
            error=str(e),
        )
        return get_mock_mcp_response(server_name, tool_name, params)
