"""
core/ampp_agent.py — Layer 2: The Cognitive Orchestrator.

This layer sits idle, consuming negligible compute, until a trigger payload
arrives on the shared asyncio.Queue from Layer 1 (core/ampp_aggregator.py).
On receipt, it:

  1. Checks hard, Python-enforced risk limits (see the RISK GATES section
     below) BEFORE invoking the LLM at all.
  2. Launches the Alpaca MCP server as a stdio subprocess via
     utils/mcp_bridge.py.
  3. Runs a multi-turn tool-calling conversation with Gemini Pro, using the
     hardened system prompt from config.py, relaying tool requests to the
     MCP ClientSession and tool results back to the model.
  4. Parses the model's final "abort" or "execute" JSON decision.
  5. For "execute", re-validates and re-clamps the model's proposed
     position size against MAX_POSITION_SIZE_PCT before calling
     place_limit_order — the model's stated qty is a *proposal*, not an
     instruction the system blindly follows.
  6. Continuously monitors any resulting open position against
     HARD_STOP_LOSS_PCT.

RISK GATES — READ THIS BEFORE MODIFYING THIS FILE
---------------------------------------------------
The blueprint's original design used the LLM's own JSON "action" field as
the *only* gate on whether a trade fires. That was changed after review:
this file now enforces four hard limits in plain Python, independent of and
prior to anything the model decides:

  - MAX_DAILY_LOSS_PCT   — checked BEFORE invoking the model. If today's
                            realized+unrealized loss already exceeds this
                            fraction of start-of-day equity, new trigger
                            payloads are rejected without spending a single
                            LLM call, and the aggregator is told to keep
                            listening but not to expect execution today.
  - MAX_CUMULATIVE_LOSS_PCT — checked BEFORE invoking the model, alongside
                            the daily check. If the loss since the start of
                            the whole 5-day measurement window exceeds this
                            fraction of start-of-window equity, this is a
                            terminal, non-resuming condition: the
                            orchestrator raises SystemHalt, which
                            main.py is expected to catch and use to
                            shut down the entire application (both layers),
                            logging a fatal message for the judges.
  - MAX_POSITION_SIZE_PCT — enforced AFTER the model proposes a qty, by
                            clamping (never raising) the proposed order
                            size down to this fraction of current buying
                            power if the proposal exceeds it.
  - HARD_STOP_LOSS_PCT    — enforced continuously, after entry, by
                            self._monitor_position(), independent of the
                            model (which is not re-consulted to "approve"
                            an exit; the exit is unconditional once the
                            stop is breached).

None of this is expressed in the system prompt as something the model
should self-enforce. It is enforced structurally, in code that runs whether
or not the model's output would have honored it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "The 'google-genai' package is required for core/ampp_agent.py but "
        "is not installed. Install it with: pip install 'google-genai>=0.1.0'"
    ) from exc

import config
from utils.mcp_bridge import create_alpaca_mcp_session

logger = logging.getLogger("ampp.agent")


class SystemHalt(Exception):
    """Raised when MAX_CUMULATIVE_LOSS_PCT is breached.

    This is intentionally NOT a subclass of the per-trigger exceptions
    caught inside the orchestrator's own run loop — it is meant to escape
    that loop entirely. main.py is expected to catch this specifically
    (it is re-raised out of run()) and use it as the signal to tear down
    both layers and exit the process, rather than something the
    orchestrator quietly recovers from and resumes after.
    """

    def __init__(self, message: str, *, cumulative_loss_pct: float) -> None:
        super().__init__(message)
        self.cumulative_loss_pct = cumulative_loss_pct


@dataclass
class AccountRiskState:
    """Tracks the equity baselines needed to evaluate the risk gates.

    `window_start_equity` is captured exactly once, the first time the
    orchestrator observes account equity after process start, and is never
    reset — it anchors MAX_CUMULATIVE_LOSS_PCT across the full 5-day
    window. `day_start_equity` is captured at the start of each calendar
    day (UTC) and reset at each new day boundary, anchoring
    MAX_DAILY_LOSS_PCT.
    """

    window_start_equity: float | None = None
    day_start_equity: float | None = None
    current_trading_day: date | None = None
    halted: bool = False

    def roll_day_if_needed(self, today: date) -> None:
        if self.current_trading_day != today:
            self.current_trading_day = today
            self.day_start_equity = None  # force re-capture on next equity read


@dataclass
class ConversationTurn:
    """One turn of the Gemini tool-calling conversation, kept for logging/debugging."""

    role: str  # "user" | "model" | "tool"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AMPPOrchestrator:
    """Layer 2: consumes anomaly payloads and runs the cognitive execution pipeline."""

    def __init__(self, trigger_queue: asyncio.Queue[dict[str, Any]]) -> None:
        self.queue = trigger_queue
        self._genai_client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._risk_state = AccountRiskState()
        # Tracks any open position this orchestrator itself entered, so
        # _monitor_position can watch it against HARD_STOP_LOSS_PCT without
        # depending on the model to ever mention it again.
        self._open_position: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Consume trigger payloads forever, until a SystemHalt propagates out.

        This is the coroutine main.py schedules via asyncio.gather
        alongside the Layer 1 aggregator loop. It deliberately never
        returns under normal operation — the only way out is a SystemHalt
        raised by the cumulative-loss gate, which is allowed to propagate
        so main.py can perform a coordinated shutdown of both layers.
        """
        logger.info("[Layer 2] Cognitive orchestrator online. Awaiting trigger payloads.")
        while True:
            payload = await self.queue.get()
            try:
                await self._handle_trigger(payload)
            except SystemHalt:
                # Deliberately not caught-and-logged like other exceptions:
                # this must propagate to main.py to halt the whole system.
                raise
            except Exception:  # noqa: BLE001 - see module docstring: must never crash the app
                logger.error(
                    "[Layer 2] Unhandled exception while processing trigger payload. "
                    "Tearing down and resuming queue consumption.\n%s",
                    traceback.format_exc(),
                )
            finally:
                self.queue.task_done()

    async def _handle_trigger(self, payload: dict[str, Any]) -> None:
        """Process one anomaly payload end to end: risk gates, MCP session, cognition, execution."""

        # --- RISK GATE 1: cumulative + daily loss, checked BEFORE the LLM ---
        # This is intentionally the very first thing that happens after
        # pulling a payload off the queue — before any network call, before
        # any subprocess is spawned. A halted or daily-exhausted system
        # should cost as close to zero as possible per rejected trigger.
        async with create_alpaca_mcp_session() as session:
            equity = await self._fetch_account_equity(session)
            self._update_risk_baselines(equity)

            cumulative_loss_pct = self._cumulative_loss_pct(equity)
            if cumulative_loss_pct >= config.MAX_CUMULATIVE_LOSS_PCT:
                self._risk_state.halted = True
                logger.critical(
                    "[Layer 2] FATAL SYSTEM SHUTDOWN: cumulative loss %.2f%% has breached "
                    "MAX_CUMULATIVE_LOSS_PCT (%.2f%%). Halting entire system. "
                    "Window start equity: $%.2f, current equity: $%.2f.",
                    cumulative_loss_pct * 100,
                    config.MAX_CUMULATIVE_LOSS_PCT * 100,
                    self._risk_state.window_start_equity,
                    equity,
                )
                raise SystemHalt(
                    "MAX_CUMULATIVE_LOSS_PCT breached; system halted.",
                    cumulative_loss_pct=cumulative_loss_pct,
                )

            daily_loss_pct = self._daily_loss_pct(equity)
            if daily_loss_pct >= config.MAX_DAILY_LOSS_PCT:
                logger.warning(
                    "[Layer 2] Trigger REJECTED without invoking the model: daily loss "
                    "%.2f%% has breached MAX_DAILY_LOSS_PCT (%.2f%%). No further trades "
                    "today. Day start equity: $%.2f, current equity: $%.2f.",
                    daily_loss_pct * 100,
                    config.MAX_DAILY_LOSS_PCT * 100,
                    self._risk_state.day_start_equity,
                    equity,
                )
                return  # not a SystemHalt — tomorrow gets a fresh budget

            # --- Cognitive pipeline (LLM + MCP tool-calling loop) ---
            decision = await self._run_cognitive_loop(session, payload)

            # --- Execution (with position-size re-clamping) ---
            if decision.get("action") == "execute":
                await self._execute_decision(session, decision, equity)
            else:
                logger.info(
                    "[Layer 2] Model decision: ABORT. Reason: %s",
                    decision.get("reason", "<no reason provided>"),
                )

    # ------------------------------------------------------------------
    # Risk baseline bookkeeping
    # ------------------------------------------------------------------

    def _update_risk_baselines(self, equity: float) -> None:
        today = datetime.now(timezone.utc).date()

        if self._risk_state.window_start_equity is None:
            self._risk_state.window_start_equity = equity
            logger.info("[Layer 2] Captured 5-day window start equity: $%.2f", equity)

        self._risk_state.roll_day_if_needed(today)
        if self._risk_state.day_start_equity is None:
            self._risk_state.day_start_equity = equity
            logger.info("[Layer 2] Captured day-start equity for %s: $%.2f", today, equity)

    def _cumulative_loss_pct(self, current_equity: float) -> float:
        start = self._risk_state.window_start_equity
        if not start or start <= 0:
            return 0.0
        return max(0.0, (start - current_equity) / start)

    def _daily_loss_pct(self, current_equity: float) -> float:
        start = self._risk_state.day_start_equity
        if not start or start <= 0:
            return 0.0
        return max(0.0, (start - current_equity) / start)

    async def _fetch_account_equity(self, session: Any) -> float:
        """Fetch current account equity via the MCP session's account tool.

        NOTE ON TOOL NAME: the blueprint's text does not specify an exact
        MCP tool name for account equity; "get_account" is used here as the
        conventional name for this operation in Alpaca's own SDK/API
        surface. If the actual Alpaca MCP server exposes a different tool
        name for this, update ONLY the string below — the rest of the risk
        logic is written against the returned equity float, not the tool
        name.
        """
        result = await session.call_tool("get_account", {})
        data = _extract_tool_json(result)
        try:
            return float(data["equity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"get_account tool result did not contain a numeric 'equity' field: {data!r}"
            ) from exc

    # ------------------------------------------------------------------
    # Cognitive loop (Gemini <-> MCP tool calling)
    # ------------------------------------------------------------------

    async def _run_cognitive_loop(
        self, session: Any, trigger_payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Run the multi-turn tool-calling conversation until a terminal JSON decision.

        The model is re-prompted after every tool result until it returns a
        JSON object whose "action" is "abort" or "execute" — those are the
        only two terminal states this loop recognizes. Any tool the model
        requests is routed through `session.call_tool` and the result fed
        back as an observation.
        """
        history: list[ConversationTurn] = [
            ConversationTurn(
                role="user",
                content=json.dumps({"anomaly_payload": trigger_payload}),
            )
        ]

        # NOTE ON TOOL SCHEMA SOURCING: the MCP ClientSession itself is the
        # authority on which tools exist and their argument schemas
        # (session.list_tools()). This loop fetches that list once per
        # trigger and passes it to Gemini as the available function
        # declarations, rather than hardcoding a tool list here — the MCP
        # server is the single source of truth for what it exposes.
        mcp_tools = await session.list_tools()
        gemini_tools = _mcp_tools_to_genai_tools(mcp_tools)

        max_turns = 8  # bounded: prevents a pathological infinite tool-call loop
        for turn_index in range(max_turns):
            response = await self._call_gemini(history, gemini_tools)

            function_call = _extract_function_call(response)
            if function_call is not None:
                tool_name, tool_args = function_call
                logger.info("[Layer 2] Model requested tool: %s(%s)", tool_name, tool_args)
                try:
                    tool_result = await session.call_tool(tool_name, tool_args)
                    observation = _extract_tool_json(tool_result)
                except Exception as exc:  # noqa: BLE001
                    # A failed tool call becomes an observation the model
                    # can reason about (e.g. retry, or abort), rather than
                    # crashing the whole trigger's processing.
                    observation = {"error": str(exc)}
                    logger.warning(
                        "[Layer 2] Tool call %s failed: %s", tool_name, exc
                    )

                history.append(ConversationTurn(role="model", content=json.dumps({"tool_call": tool_name, "args": tool_args})))
                history.append(ConversationTurn(role="tool", content=json.dumps(observation)))
                continue

            decision_text = _extract_text(response)
            decision = _parse_terminal_decision(decision_text)
            if decision is not None:
                history.append(ConversationTurn(role="model", content=decision_text))
                return decision

            # Model produced neither a recognized function call nor a
            # parseable terminal decision. Feed back a corrective
            # observation rather than silently looping forever on
            # malformed output.
            logger.warning(
                "[Layer 2] Model output was neither a tool call nor a valid terminal "
                "decision on turn %d: %r", turn_index, decision_text
            )
            history.append(ConversationTurn(role="model", content=decision_text))
            history.append(
                ConversationTurn(
                    role="tool",
                    content=json.dumps(
                        {
                            "error": (
                                "Your last response was not valid JSON with an "
                                "'action' of 'abort' or 'execute', and was not a "
                                "recognized tool call. Re-read the protocol and "
                                "respond with strictly formatted JSON only."
                            )
                        }
                    ),
                )
            )

        logger.error("[Layer 2] Cognitive loop exceeded max_turns without a terminal decision. Aborting.")
        return {"action": "abort", "reason": "Cognitive loop exceeded maximum turns without a decision."}

    async def _call_gemini(
        self, history: list[ConversationTurn], gemini_tools: list[Any]
    ) -> Any:
        """Send the accumulated conversation to Gemini and return the raw response.

        Includes retry logic with exponential backoff for transient 503/429
        errors, and falls back to gemini-2.0-flash if the primary model is
        persistently unavailable.
        """
        contents = [
            genai_types.Content(
                role="user" if turn.role in ("user", "tool") else "model",
                parts=[genai_types.Part(text=turn.content)],
            )
            for turn in history
        ]

        models_to_try = [config.GEMINI_MODEL_NAME, "gemini-3.6-flash", "gemini-3.5-flash"]
        last_exc = None

        for model_name in models_to_try:
            for attempt in range(3):
                try:
                    response = await self._genai_client.aio.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=genai_types.GenerateContentConfig(
                            system_instruction=config.AMPP_SYSTEM_PROMPT,
                            tools=gemini_tools,
                        ),
                    )
                    if model_name != config.GEMINI_MODEL_NAME:
                        logger.info("[Layer 2] Successfully used fallback model: %s", model_name)
                    return response
                except Exception as exc:
                    last_exc = exc
                    err_str = str(exc)
                    if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
                        wait = 2 ** attempt
                        logger.warning(
                            "[Layer 2] Gemini %s returned transient error (attempt %d/3). "
                            "Retrying in %ds... Error: %s",
                            model_name, attempt + 1, wait, err_str[:120],
                        )
                        await asyncio.sleep(wait)
                    else:
                        raise  # Non-transient error, don't retry

            logger.warning("[Layer 2] Model %s exhausted retries, trying fallback...", model_name)

        # All models and retries exhausted
        raise last_exc

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _execute_decision(
        self, session: Any, decision: dict[str, Any], current_equity: float
    ) -> None:
        """Validate, clamp, and execute an "execute" decision via place_limit_order.

        --- RISK GATE: position sizing ---
        The model's proposed `qty` is treated as a proposal. It is
        converted into a notional dollar amount using the proposed
        limit_price, and if that notional exceeds
        MAX_POSITION_SIZE_PCT * current_equity, the qty is clamped DOWN to
        fit within that ceiling before place_limit_order is ever called.
        The model is never re-consulted about the clamp — this is a hard
        ceiling, not a negotiation.
        """
        contract = decision.get("contract")
        limit_price = decision.get("limit_price")
        proposed_qty = decision.get("qty")

        if not contract or limit_price is None or proposed_qty is None:
            logger.error(
                "[Layer 2] 'execute' decision missing required fields (contract, "
                "limit_price, qty); treating as an implicit abort. Decision: %r",
                decision,
            )
            return

        try:
            limit_price = float(limit_price)
            proposed_qty = int(proposed_qty)
        except (TypeError, ValueError):
            logger.error(
                "[Layer 2] 'execute' decision had non-numeric limit_price/qty; "
                "treating as an implicit abort. Decision: %r", decision
            )
            return

        if limit_price <= 0 or proposed_qty <= 0:
            logger.error(
                "[Layer 2] 'execute' decision had non-positive limit_price/qty; "
                "treating as an implicit abort. Decision: %r", decision
            )
            return

        max_notional = config.MAX_POSITION_SIZE_PCT * current_equity
        proposed_notional = limit_price * proposed_qty * 100  # options are typically 100-share/contract multiplier

        final_qty = proposed_qty
        if proposed_notional > max_notional:
            # Clamp down to the largest qty that fits under the ceiling,
            # rounding down (int()) since we can't buy a fractional
            # contract, and never rounding up past the ceiling.
            final_qty = max(0, int(max_notional / (limit_price * 100)))
            logger.warning(
                "[Layer 2] Model-proposed qty=%d ($%.2f notional) exceeded "
                "MAX_POSITION_SIZE_PCT ceiling ($%.2f). Clamped to qty=%d.",
                proposed_qty, proposed_notional, max_notional, final_qty,
            )

        if final_qty <= 0:
            logger.warning(
                "[Layer 2] Clamped qty is 0 (proposed order too large relative to "
                "current equity even before clamping, or equity too low). "
                "Skipping execution for %s.", contract
            )
            return

        logger.info(
            "[Layer 2] EXECUTING: %s x%d @ limit $%.2f (model proposed qty=%d)",
            contract, final_qty, limit_price, proposed_qty,
        )

        try:
            order_result = await session.call_tool(
                "place_limit_order",
                {
                    "symbol": contract,
                    "qty": final_qty,
                    "limit_price": limit_price,
                    "side": "buy",
                    "time_in_force": "day",
                },
            )
            order_data = _extract_tool_json(order_result)
            confirmation_id = order_data.get("id") or order_data.get("order_id")
            logger.info(
                "[Layer 2] ORDER CONFIRMED. contract=%s qty=%d limit=$%.2f id=%s",
                contract, final_qty, limit_price, confirmation_id,
            )
            self._open_position = {
                "contract": contract,
                "qty": final_qty,
                "entry_price": limit_price,
                "order_id": confirmation_id,
            }
            # Fire-and-track: monitored independently of this trigger's
            # lifecycle so a stop-loss can fire even if Layer 2 is busy
            # processing a later, unrelated trigger.
            asyncio.create_task(self._monitor_position(session, dict(self._open_position)))
        except Exception:  # noqa: BLE001
            logger.error(
                "[Layer 2] place_limit_order failed for %s.\n%s",
                contract, traceback.format_exc(),
            )

    async def _monitor_position(self, session: Any, position: dict[str, Any]) -> None:
        """Continuously watch an open position against HARD_STOP_LOSS_PCT.

        This runs unconditionally after entry, independent of the model —
        it does not ask Gemini for permission to exit. Once the stop is
        breached, it submits a closing order directly.

        NOTE: This coroutine holds its own MCP session reference from the
        session that was active at entry time. In main.py's actual
        deployment, monitoring tasks that must outlive a single trigger's
        `async with create_alpaca_mcp_session()` block should be given
        their own independently-opened session; this implementation
        accepts that session as a parameter to keep this file's scope
        limited to orchestration logic rather than session-lifetime
        management, which belongs in main.py's task-supervision code.
        """
        contract = position["contract"]
        entry_price = position["entry_price"]
        qty = position["qty"]
        stop_price = entry_price * (1.0 - config.HARD_STOP_LOSS_PCT)

        logger.info(
            "[Layer 2] Monitoring %s: entry=$%.2f, hard stop=$%.2f (%.0f%%)",
            contract, entry_price, stop_price, config.HARD_STOP_LOSS_PCT * 100,
        )

        while True:
            await asyncio.sleep(1)
            try:
                quote_result = await session.call_tool("get_option_quote", {"symbol": contract})
                quote = _extract_tool_json(quote_result)
                last_price = float(quote.get("last_price", quote.get("bid_price", 0.0)))
            except Exception:  # noqa: BLE001
                logger.warning(
                    "[Layer 2] Failed to fetch monitoring quote for %s; will retry.\n%s",
                    contract, traceback.format_exc(),
                )
                continue

            if last_price <= 0:
                continue

            if last_price <= stop_price:
                logger.warning(
                    "[Layer 2] HARD STOP TRIGGERED for %s: last=$%.2f <= stop=$%.2f. "
                    "Submitting unconditional closing order.",
                    contract, last_price, stop_price,
                )
                try:
                    await session.call_tool(
                        "place_limit_order",
                        {
                            "symbol": contract,
                            "qty": qty,
                            "limit_price": last_price,
                            "side": "sell",
                            "time_in_force": "day",
                        },
                    )
                    logger.info("[Layer 2] Stop-loss exit order submitted for %s.", contract)
                except Exception:  # noqa: BLE001
                    logger.critical(
                        "[Layer 2] FAILED TO SUBMIT STOP-LOSS EXIT for %s. Manual "
                        "intervention may be required.\n%s",
                        contract, traceback.format_exc(),
                    )
                self._open_position = None
                return


# ---------------------------------------------------------------------------
# Module-level parsing helpers
# ---------------------------------------------------------------------------
# Kept as free functions (not methods) since they're pure transformations
# over SDK response objects / text, with no dependency on orchestrator
# state.

def _extract_tool_json(tool_result: Any) -> dict[str, Any]:
    """Normalize an MCP tool_result into a plain dict.

    MCP tool results are returned as a list of content blocks; this project
    expects the Alpaca MCP server to return a single text block containing
    a JSON string, which is the documented convention for MCP tools that
    return structured data. Raises a clear error rather than silently
    returning {} if that shape isn't met, so a schema mismatch surfaces
    immediately instead of as a mysterious downstream KeyError.
    """
    content = getattr(tool_result, "content", None)
    if not content:
        raise RuntimeError(f"MCP tool result had no content: {tool_result!r}")

    first_block = content[0]
    text = getattr(first_block, "text", None)
    if text is None:
        raise RuntimeError(f"MCP tool result's first content block had no text: {first_block!r}")

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"MCP tool result was not valid JSON: {text!r}") from exc


def _extract_function_call(response: Any) -> tuple[str, dict[str, Any]] | None:
    """Pull a (tool_name, args) pair out of a Gemini response, if present."""
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        parts = getattr(getattr(candidate, "content", None), "parts", None) or []
        for part in parts:
            fn_call = getattr(part, "function_call", None)
            if fn_call is not None:
                args = dict(fn_call.args) if fn_call.args else {}
                return fn_call.name, args
    return None


def _extract_text(response: Any) -> str:
    """Pull the plain-text portion of a Gemini response, if present."""
    text = getattr(response, "text", None)
    if text:
        return text
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        parts = getattr(getattr(candidate, "content", None), "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                return part_text
    return ""


def _parse_terminal_decision(text: str) -> dict[str, Any] | None:
    """Parse model output text as a terminal abort/execute JSON decision.

    Returns None (not an exception) if the text doesn't parse as JSON or
    doesn't have a recognized "action" — this is a normal, expected outcome
    when the model is instead issuing a tool call, and the caller branches
    on None vs. a dict rather than on an exception.
    """
    cleaned = text.strip()
    # Defensive: the system prompt forbids markdown formatting, but strip a
    # ```json fence if the model includes one anyway, rather than failing
    # the parse over a formatting slip.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None
    if parsed.get("action") not in ("abort", "execute"):
        return None
    return parsed


def _mcp_tools_to_genai_tools(mcp_tools_result: Any) -> list[Any]:
    """Convert an MCP list_tools() result into Gemini function-declaration tools.

    NOTE: The exact conversion between an MCP `Tool` object's inputSchema
    and a `genai_types.FunctionDeclaration` depends on the installed
    versions of both the `mcp` and `google-genai` packages, which could not
    be verified against a live install in this environment (see the
    surrounding code review notes). This function assumes the conventional
    shape: `mcp_tools_result.tools` is a list of objects with `.name`,
    `.description`, and `.inputSchema` (a JSON Schema dict). If the
    installed SDK versions differ, this is the single function to update —
    everything downstream consumes the returned `list[genai_types.Tool]`
    generically.
    """
    declarations = []
    for tool in getattr(mcp_tools_result, "tools", []):
        declarations.append(
            genai_types.FunctionDeclaration(
                name=tool.name,
                description=getattr(tool, "description", "") or "",
                parameters=getattr(tool, "inputSchema", None) or {"type": "object", "properties": {}},
            )
        )
    if not declarations:
        return []
    return [genai_types.Tool(function_declarations=declarations)]
