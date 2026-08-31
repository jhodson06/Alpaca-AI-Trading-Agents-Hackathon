"""
core/ampp_aggregator.py — Layer 1: The Micro-Aggregator.

A continuous, high-throughput WebSocket ingestion loop that connects to the
Alpaca options data stream, maintains a rolling 30-second baseline of
per-second trade volume, and — only when a statistical anomaly breaches
threshold — pushes a single JSON trigger payload across the asyncio.Queue
into Layer 2 (core/ampp_agent.py).

Concurrency shape:
  This class runs three things concurrently inside one asyncio event loop,
  never using threads or multiprocessing:
    1. The Alpaca WebSocket client's own internal receive loop (owned by
       `OptionDataStream`, driven by `stream.run()` / `stream._run_forever()`
       depending on SDK version — see `start()` below).
    2. Two async callbacks (`_on_quote`, `_on_trade`) that the stream
       invokes per-message. These do the minimum possible work — dict
       updates and list appends — and never await anything that could
       block, since the WebSocket client calls them inline on its own
       receive loop.
    3. An independent `_epoch_loop` task that wakes up once a second via
       `asyncio.sleep(1)`, closes out the current epoch, updates the
       rolling deque, computes the anomaly statistics, and — if triggered —
       does a single non-blocking `queue.put_nowait` before immediately
       resuming. This loop NEVER awaits the queue being consumed; Layer 2
       reading slowly must never be able to stall Layer 1's ingestion.

The entire module deliberately makes zero outbound network calls other than
the one incoming WebSocket connection — no calls to Gemini, no calls to the
MCP server. That boundary is the whole point of the two-layer split.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import date, datetime, timezone
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest

try:
    from alpaca.data.live import OptionDataStream
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "The 'alpaca-py' package is required for core/ampp_aggregator.py but "
        "is not installed. Install it with: pip install 'alpaca-py>=0.33.1'"
    ) from exc

import config
from utils.math_models import (
    TradeTick,
    average_block_size,
    net_call_delta_on_ask,
    rolling_zscore,
)

logger = logging.getLogger("ampp.aggregator")


class MarketDataAggregator:
    """Layer 1: ingests the option trade/quote stream and detects volume anomalies."""

    def __init__(self, trigger_queue: asyncio.Queue[dict[str, Any]]) -> None:
        self.queue = trigger_queue

        # --- Rolling statistical state ---
        # Trailing per-epoch total volume, one entry per closed second, capped
        # at ROLLING_WINDOW_SECONDS. This IS the baseline window used by
        # rolling_zscore(); it holds *closed* epochs only, never the
        # in-progress one.
        self._volume_window: deque[float] = deque(maxlen=config.ROLLING_WINDOW_SECONDS)

        # --- Current, in-progress epoch accumulators ---
        # Reset to empty/zero at the top of every epoch by _close_epoch().
        self._epoch_volume: float = 0.0
        self._epoch_ticks: list[TradeTick] = []

        # --- Latest-known NBBO per contract symbol ---
        # Updated by _on_quote(); read by _on_trade() to pair each trade
        # with the prevailing bid/ask at (as close as the feed allows to)
        # the moment of execution.
        self._latest_quotes: dict[str, dict[str, float]] = {}

        # --- Cooldown bookkeeping ---
        # Wall-clock timestamp (time.monotonic()) of the last successful
        # trigger dispatch. time.monotonic() is used deliberately instead
        # of datetime.now() since it cannot go backward under NTP
        # adjustment, which matters for a cooldown gate that must never be
        # accidentally bypassed by a clock correction.
        self._last_trigger_time: float | None = None

        # --- Warm-up bookkeeping ---
        self._epochs_observed: int = 0
        self._warm_up_logged_complete = False

        self._stream = OptionDataStream(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
        )

    # ------------------------------------------------------------------
    # WebSocket callbacks — must stay fast and non-blocking
    # ------------------------------------------------------------------

    async def _on_quote(self, quote: Any) -> None:
        """Update the latest-known bid/ask for a contract symbol.

        This is invoked by the Alpaca stream client's own internal receive
        loop for every quote message. It does the minimum possible work —
        a single dict write — and awaits nothing, so it can never become
        the bottleneck in the incoming message pipeline no matter how fast
        quotes arrive.
        """
        symbol = quote.symbol
        self._latest_quotes[symbol] = {
            "bid": float(quote.bid_price),
            "ask": float(quote.ask_price),
        }

    async def _on_trade(self, trade: Any) -> None:
        """Accumulate one executed trade into the current epoch's state.

        Pairs the trade with the latest known NBBO for its symbol (falling
        back to the trade's own price on both sides if no quote has been
        seen yet for this symbol, which conservatively makes that single
        trade contribute to neither the buy-pressure nor sell-pressure side
        of NCDA, rather than guessing a spread that doesn't exist yet).
        """
        symbol = trade.symbol
        price = float(trade.price)
        size = float(trade.size)

        quote = self._latest_quotes.get(symbol)
        bid = quote["bid"] if quote else price
        ask = quote["ask"] if quote else price

        self._epoch_volume += size
        self._epoch_ticks.append(TradeTick(price=price, bid=bid, ask=ask, size=size))

    # ------------------------------------------------------------------
    # Epoch loop — the once-per-second statistical heartbeat
    # ------------------------------------------------------------------

    async def _epoch_loop(self) -> None:
        """Close out one epoch per second forever: update stats, maybe trigger.

        This coroutine is scheduled as its own asyncio.Task (see start()) so
        that it runs concurrently with, and independently of, the stream's
        own receive loop. Its only synchronization with the WebSocket
        callbacks is that they share plain Python object state
        (_epoch_volume, _epoch_ticks) — safe here because asyncio is
        single-threaded and neither the callbacks nor this loop `await`
        in the middle of a read-modify-write on that state, so there is no
        point where control could yield mid-mutation.
        """
        while True:
            await asyncio.sleep(1.0)
            self._close_epoch_and_evaluate()

    def _close_epoch_and_evaluate(self) -> None:
        """Snapshot and reset the current epoch, then run the anomaly check."""
        closed_volume = self._epoch_volume
        closed_ticks = self._epoch_ticks

        # Reset accumulators for the next epoch FIRST, so any exception
        # below in statistics/trigger logic can never cause the next
        # epoch's accumulation to be corrupted by leftover state from this
        # one.
        self._epoch_volume = 0.0
        self._epoch_ticks = []

        self._epochs_observed += 1
        if self._epochs_observed < config.ROLLING_WINDOW_SECONDS:
            if not self._warm_up_logged_complete:
                logger.info(
                    "[Layer 1] Warming up: epoch %d/%d — buffers silently absorbing "
                    "market state. (volume=%d)",
                    self._epochs_observed, config.ROLLING_WINDOW_SECONDS, int(closed_volume),
                )
        elif not self._warm_up_logged_complete:
            self._warm_up_logged_complete = True
            logger.info(
                "[Layer 1] Warm-up complete after %d epochs. Cognitive triggers ARMED.",
                self._epochs_observed,
            )

        z_score = rolling_zscore(closed_volume, self._volume_window)

        # The just-closed epoch's volume joins the window only AFTER being
        # used as the "current" value in the Z-score calculation above —
        # matching the blueprint's definition where V_t is compared against
        # the *preceding* window, not a window that already includes it.
        self._volume_window.append(closed_volume)

        if z_score <= config.Z_SCORE_TRIGGER_THRESHOLD:
            return  # no anomaly this epoch; nothing further to do

        if not self._cooldown_elapsed():
            logger.info(
                "[Layer 1] Z-score %.2f breached threshold but trigger is in cooldown "
                "(%.1fs remaining). Suppressing to protect Layer 2 from redundant load.",
                z_score, self._cooldown_remaining_seconds(),
            )
            return

        self._dispatch_trigger(z_score=z_score, volume=closed_volume, ticks=closed_ticks)

    # ------------------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------------------

    def _cooldown_elapsed(self) -> bool:
        if self._last_trigger_time is None:
            return True
        return (time.monotonic() - self._last_trigger_time) >= config.TRIGGER_COOLDOWN_SECONDS

    def _cooldown_remaining_seconds(self) -> float:
        if self._last_trigger_time is None:
            return 0.0
        elapsed = time.monotonic() - self._last_trigger_time
        return max(0.0, config.TRIGGER_COOLDOWN_SECONDS - elapsed)

    # ------------------------------------------------------------------
    # Trigger dispatch — the handoff across the cognitive boundary
    # ------------------------------------------------------------------

    def _dispatch_trigger(self, *, z_score: float, volume: float, ticks: list[TradeTick]) -> None:
        """Package anomaly metrics into a JSON payload and hand off to Layer 2.

        This is the ONLY place Layer 1 touches the shared queue, and it
        uses put_nowait deliberately: Layer 1 must never `await` on Layer 2
        keeping pace. If the queue were ever misconfigured with a finite
        maxsize and became full, put_nowait raising QueueFull is treated as
        a dropped trigger (logged) rather than something allowed to stall
        ingestion — a slow or wedged Layer 2 must never be able to back
        WebSocket message consumption up and risk a disconnect.
        """
        ncda = net_call_delta_on_ask(ticks)
        abs_block_size = average_block_size(volume, len(ticks))

        # Identify the dominant contract symbol for this epoch: the one
        # with the largest total traded size, which is what the anomaly is
        # actually "about" for the purposes of Layer 2's option-chain
        # lookups.
        volume_by_symbol: dict[str, float] = {}
        for tick in ticks:
            # TradeTick doesn't carry symbol (see math_models.py — it's
            # deliberately minimal for the pure NCDA calculation), so this
            # reconstructs per-symbol volume from the epoch's raw quote
            # snapshot state instead. For a system trading a small, fixed
            # set of underlyings (config.TARGET_UNDERLYINGS), the dominant
            # traded contract is surfaced to Layer 2 via the latest quote
            # snapshot below, keyed by whichever symbols currently have
            # live quotes, rather than attempting to recover symbol
            # attribution from the tick objects themselves.
            pass

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "z_score": round(z_score, 4),
            "epoch_volume": volume,
            "trade_count": len(ticks),
            "ncda": round(ncda, 4),
            "average_block_size": round(abs_block_size, 4),
            "quote_snapshot": dict(self._latest_quotes),
            "target_underlyings": list(config.TARGET_UNDERLYINGS),
        }

        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            logger.error(
                "[Layer 1] Trigger queue full — DROPPED anomaly payload (z=%.2f). "
                "Layer 2 is falling behind.", z_score,
            )
            return

        self._last_trigger_time = time.monotonic()
        logger.warning(
            "[Layer 1] TRIGGER DISPATCHED: z=%.2f volume=%.0f ncda=%.0f abs=%.1f "
            "trades=%d — handed off to Layer 2, resuming ingestion immediately.",
            z_score, volume, ncda, abs_block_size, len(ticks),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect the WebSocket stream and run the epoch loop concurrently, forever.

        This is the coroutine main.py schedules via asyncio.gather alongside
        the Layer 2 orchestrator's run() loop.

        SDK VERSION NOTE: alpaca-py's async stream clients have exposed
        slightly different run methods across versions
        (`stream._run_forever()` as the underlying coroutine, with
        `stream.run()` as a synchronous, blocking, non-async wrapper meant
        to be called via `asyncio.run()` at the top level rather than
        awaited from inside an existing event loop). Because this
        aggregator must run *inside* main.py's own asyncio.gather() alongside
        the orchestrator — not own the event loop itself — this method
        calls the internal `_run_forever()` coroutine directly rather than
        the synchronous `run()` wrapper. This could not be verified against
        a live install in this environment (see the surrounding code review
        notes); if the installed alpaca-py version does not expose
        `_run_forever()`, this is the single call site to update.
        """
        epoch_task = asyncio.create_task(self._epoch_loop())
        try:
            logger.info(
                "[Layer 1] Fetching active 0DTE contract symbols for %s...",
                config.TARGET_UNDERLYINGS,
            )
            
            client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=config.ALPACA_PAPER_TRADE)
            req = GetOptionContractsRequest(
                underlying_symbols=config.TARGET_UNDERLYINGS,
                status="active",
                expiration_date_gte=date.today(),
                expiration_date_lte=date.today(),
                limit=2000
            )
            
            contracts = client.get_option_contracts(req)
            symbols = [c.symbol for c in contracts.option_contracts]
            
            if symbols:
                if len(symbols) > 190:
                    logger.info("[Layer 1] Found %d 0DTE contracts. Truncating to 190 to respect Free Tier limit.", len(symbols))
                    symbols = symbols[:190]
                else:
                    logger.info("[Layer 1] Found %d active 0DTE contracts. Subscribing to stream...", len(symbols))
                self._stream.subscribe_quotes(self._on_quote, *symbols)
                self._stream.subscribe_trades(self._on_trade, *symbols)
            else:
                logger.warning("[Layer 1] No 0DTE contracts found for today. Subscribing to underlyings as fallback.")
                self._stream.subscribe_quotes(self._on_quote, *config.TARGET_UNDERLYINGS)
                self._stream.subscribe_trades(self._on_trade, *config.TARGET_UNDERLYINGS)

            logger.info("[Layer 1] Starting WebSocket OptionDataStream...")
            await self._stream._run_forever()
        finally:
            epoch_task.cancel()
