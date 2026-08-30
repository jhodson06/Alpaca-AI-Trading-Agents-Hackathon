"""
utils/math_models.py — Pure, stateless quantitative functions.

Every function in this module is side-effect free: no I/O, no mutation of
arguments, no reliance on wall-clock time except where a timestamp is passed
in explicitly. This is deliberate. Layer 1 (core/ampp_aggregator.py) runs an
asyncio event loop that must never block, so all statistics used inside that
loop are pulled out here where they're trivially fast, trivially testable,
and impossible to accidentally turn into a blocking call.

Every public function returns a value; none of them raise on the "not enough
data yet" case. Callers get a defined value (typically 0.0) instead of an
exception during cold-start, matching the blueprint's requirement that the
first ~30 seconds of operation "silently absorb" market state without
producing spurious triggers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

import config


def rolling_zscore(current_volume: float, window: Sequence[float]) -> float:
    """Compute the Z-score of `current_volume` against a trailing window.

    Z_t = (V_t - mu_W) / sigma_W

    `window` is expected to hold the *preceding* epochs' volumes (i.e. it
    should not already include `current_volume`).

    Defensive behavior (both required by the blueprint's cold-start
    handling):
      - If fewer than config.MIN_WARM_UP_EPOCHS samples are available, the
        baseline is considered too thin to be meaningful, so this returns
        0.0 rather than an unstable or misleading statistic.
      - If the window's standard deviation is exactly zero (a degenerate,
        perfectly flat baseline — possible in low-liquidity pre-market
        conditions), this returns 0.0 rather than dividing by zero.
    """
    if len(window) < config.MIN_WARM_UP_EPOCHS:
        return 0.0

    arr = np.asarray(window, dtype=np.float64)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr))  # population std, matching the blueprint's 1/30 formulation

    if sigma == 0.0:
        return 0.0

    return (current_volume - mu) / sigma


@dataclass(frozen=True)
class TradeTick:
    """A single executed trade, paired with the prevailing NBBO at fill time.

    This is the minimal shape Layer 1 needs to compute NCDA. `price` is the
    trade's execution price; `bid`/`ask` are the best bid/ask quoted for
    that contract at (as close as the feed allows to) the moment of
    execution; `size` is the number of contracts in that print.
    """

    price: float
    bid: float
    ask: float
    size: float


def net_call_delta_on_ask(ticks: Sequence[TradeTick]) -> float:
    """Compute Net Call Delta on Ask (NCDA) for a set of trades.

    NCDA = sum(size where price >= ask) - sum(size where price <= bid)

    A trade executing at-or-above the prevailing ask is treated as
    aggressive buying (price-insensitive lifting of the offer); a trade
    executing at-or-below the prevailing bid is treated as aggressive
    selling (price-insensitive hitting of the bid). Trades strictly between
    bid and ask contribute to neither side, matching the blueprint's
    indicator-function definition exactly (no partial credit for
    "sort of" aggressive fills).

    Returns 0.0 for an empty sequence.
    """
    if not ticks:
        return 0.0

    buy_pressure = sum(t.size for t in ticks if t.price >= t.ask)
    sell_pressure = sum(t.size for t in ticks if t.price <= t.bid)
    return float(buy_pressure - sell_pressure)


def average_block_size(total_volume: float, trade_count: int) -> float:
    """Compute Average Block Size (ABS) = total_volume / trade_count.

    Used to help distinguish a few large institutional blocks from many
    small retail prints. Returns 0.0 if trade_count is 0, since there is no
    meaningful average of zero trades (and this avoids a ZeroDivisionError
    on an epoch that, for whatever reason, triggered statistics with no
    underlying trade objects).
    """
    if trade_count <= 0:
        return 0.0
    return float(total_volume) / float(trade_count)


def spread_ratio(bid: float, ask: float) -> float:
    """Compute the Spread Ratio: (ask - bid) / mid.

    Returns float('inf') if the midpoint is zero or negative (a degenerate/
    bad quote), which will always fail an "is this <= MAX_SPREAD_RATIO"
    check upstream — i.e. bad quote data fails safe toward "too illiquid to
    trade" rather than silently passing a spread check it can't actually
    evaluate.
    """
    mid = (bid + ask) / 2.0
    if mid <= 0.0:
        return float("inf")
    return (ask - bid) / mid


def slippage_adjusted_limit_price(bid: float, ask: float) -> float:
    """Compute a deliberately slippage-degraded limit price.

    L_exec = mid + SLIPPAGE_WEIGHT * (ask - mid)

    This intentionally prices the order closer to the ask than the
    midpoint, simulating the real-world cost of crossing a spread rather
    than assuming a fantasy midpoint fill.
    """
    mid = (bid + ask) / 2.0
    return mid + config.SLIPPAGE_WEIGHT * (ask - mid)
