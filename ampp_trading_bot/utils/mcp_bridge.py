"""
utils/mcp_bridge.py — Direct Alpaca SDK bridge for Layer 2.

This module replaces the original MCP subprocess approach with direct
alpaca-py SDK calls. The MCP server (alpaca-mcp-server) has a broken
upstream dependency (fastmcp.tools.tool no longer exists), so we bypass
it entirely and call the Alpaca API directly using the same SDK that
Layer 1 already uses successfully.

The interface is kept identical (async context manager yielding a
session-like object with `call_tool`, `list_tools`, `initialize`) so
that core/ampp_agent.py requires zero changes.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    LimitOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest

import config

logger = logging.getLogger("ampp.mcp_bridge")


# ---------------------------------------------------------------------------
# Fake MCP-compatible response objects
# ---------------------------------------------------------------------------
# These mimic the shape that ampp_agent.py expects from MCP tool results
# (content[0].text = JSON string) so the agent code needs zero changes.

@dataclass
class _TextBlock:
    text: str

@dataclass
class _ToolResult:
    content: list[_TextBlock]

@dataclass
class _ToolInfo:
    name: str
    description: str
    inputSchema: dict


def _make_result(data: dict) -> _ToolResult:
    """Wrap a dict as a fake MCP tool result."""
    return _ToolResult(content=[_TextBlock(text=json.dumps(data))])


# ---------------------------------------------------------------------------
# Direct SDK Session — drop-in replacement for MCP ClientSession
# ---------------------------------------------------------------------------

class AlpacaDirectSession:
    """Implements the same call_tool / list_tools interface as an MCP
    ClientSession, but calls the Alpaca SDK directly."""

    def __init__(self) -> None:
        self._trading_client = TradingClient(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            paper=config.ALPACA_PAPER_TRADE,
        )
        self._data_client = OptionHistoricalDataClient(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
        )

    async def initialize(self) -> None:
        """No-op for compatibility."""
        pass

    async def list_tools(self) -> Any:
        """Return the tool declarations that Gemini will see."""
        tools = [
            _ToolInfo(
                name="get_account",
                description="Get current account information including equity, buying power, and portfolio value.",
                inputSchema={"type": "object", "properties": {}},
            ),
            _ToolInfo(
                name="get_option_chain",
                description="Get the options chain for an underlying symbol. Returns available option contracts.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "underlying_symbol": {"type": "string", "description": "The underlying stock symbol (e.g. SPY)"},
                        "expiration_date": {"type": "string", "description": "Expiration date in YYYY-MM-DD format"},
                    },
                    "required": ["underlying_symbol"],
                },
            ),
            _ToolInfo(
                name="get_option_quote",
                description="Get the latest quote for a specific option contract symbol.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "The OCC option symbol (e.g. SPY260831C00650000)"},
                    },
                    "required": ["symbol"],
                },
            ),
            _ToolInfo(
                name="place_limit_order",
                description="Place a limit order for an option contract.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "The OCC option symbol"},
                        "qty": {"type": "integer", "description": "Number of contracts"},
                        "limit_price": {"type": "number", "description": "Limit price per contract"},
                        "side": {"type": "string", "enum": ["buy", "sell"], "description": "Order side"},
                        "time_in_force": {"type": "string", "enum": ["day", "gtc"], "description": "Time in force"},
                    },
                    "required": ["symbol", "qty", "limit_price", "side"],
                },
            ),
        ]
        return _ToolList(tools=tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> _ToolResult:
        """Route a tool call to the appropriate Alpaca SDK method."""
        if name == "get_option_quote":
            logger.debug("[MCP Bridge] Executing tool: %s(%s)", name, arguments)
        else:
            logger.info("[MCP Bridge] Executing tool: %s(%s)", name, arguments)

        if name == "get_account":
            return await self._get_account()
        elif name == "get_option_chain":
            return await self._get_option_chain(arguments)
        elif name == "get_option_quote":
            return await self._get_option_quote(arguments)
        elif name == "place_limit_order":
            return await self._place_limit_order(arguments)
        else:
            return _make_result({"error": f"Unknown tool: {name}"})

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _get_account(self) -> _ToolResult:
        account = self._trading_client.get_account()
        return _make_result({
            "equity": str(account.equity),
            "buying_power": str(account.buying_power),
            "cash": str(account.cash),
            "portfolio_value": str(account.portfolio_value),
            "status": str(account.status),
        })

    async def _get_option_chain(self, args: dict) -> _ToolResult:
        underlying = args.get("underlying_symbol", "SPY")
        exp_date_str = args.get("expiration_date")

        req_kwargs: dict[str, Any] = {
            "underlying_symbols": [underlying],
            "status": "active",
        }

        if exp_date_str:
            try:
                exp_date = date.fromisoformat(exp_date_str)
                req_kwargs["expiration_date_gte"] = exp_date
                req_kwargs["expiration_date_lte"] = exp_date
            except ValueError:
                pass
        else:
            # Default to 0DTE
            today = date.today()
            req_kwargs["expiration_date_gte"] = today
            req_kwargs["expiration_date_lte"] = today

        req = GetOptionContractsRequest(**req_kwargs)
        contracts = self._trading_client.get_option_contracts(req)

        chain_data = []
        for c in contracts.option_contracts[:50]:  # Limit to 50 for LLM context
            chain_data.append({
                "symbol": c.symbol,
                "type": str(c.type),
                "strike_price": str(c.strike_price),
                "expiration_date": str(c.expiration_date),
                "status": str(c.status),
            })

        return _make_result({"contracts": chain_data, "total": len(contracts.option_contracts)})

    async def _get_option_quote(self, args: dict) -> _ToolResult:
        symbol = args.get("symbol", "")
        try:
            req = OptionLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = self._data_client.get_option_latest_quote(req)

            if symbol in quotes:
                q = quotes[symbol]
                return _make_result({
                    "symbol": symbol,
                    "bid_price": str(q.bid_price) if q.bid_price else "0",
                    "ask_price": str(q.ask_price) if q.ask_price else "0",
                    "last_price": str(q.bid_price) if q.bid_price else "0",
                    "bid_size": q.bid_size if q.bid_size else 0,
                    "ask_size": q.ask_size if q.ask_size else 0,
                })
            else:
                return _make_result({"error": f"No quote found for {symbol}"})
        except Exception as exc:
            return _make_result({"error": f"Failed to get quote for {symbol}: {str(exc)}"})

    async def _place_limit_order(self, args: dict) -> _ToolResult:
        symbol = args.get("symbol", "")
        qty = int(args.get("qty", 0))
        limit_price = float(args.get("limit_price", 0))
        side_str = args.get("side", "buy")
        tif_str = args.get("time_in_force", "day")

        side = OrderSide.BUY if side_str == "buy" else OrderSide.SELL
        tif = TimeInForce.DAY if tif_str == "day" else TimeInForce.GTC

        try:
            order_req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                limit_price=limit_price,
                side=side,
                time_in_force=tif,
            )
            order = self._trading_client.submit_order(order_req)
            return _make_result({
                "id": str(order.id),
                "status": str(order.status),
                "symbol": str(order.symbol),
                "qty": str(order.qty),
                "limit_price": str(order.limit_price),
                "side": str(order.side),
            })
        except Exception as exc:
            return _make_result({"error": f"Order submission failed: {str(exc)}"})


@dataclass
class _ToolList:
    tools: list[_ToolInfo]


# ---------------------------------------------------------------------------
# Context manager (same interface as the original MCP bridge)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def create_alpaca_mcp_session() -> AsyncIterator[AlpacaDirectSession]:
    """Create a direct Alpaca SDK session — drop-in replacement for the
    MCP subprocess bridge.

    Usage:
        async with create_alpaca_mcp_session() as session:
            result = await session.call_tool("get_account", {})
    """
    session = AlpacaDirectSession()
    await session.initialize()
    logger.info("[MCP Bridge] Direct Alpaca SDK session initialized (bypassing MCP subprocess).")
    yield session
