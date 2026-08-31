"""
config.py — Centralized parameterization for the AMPP trading bot.

This module is the single source of truth for every tunable constant used by
both Layer 1 (core/ampp_aggregator.py) and Layer 2 (core/ampp_agent.py). It
also owns environment-variable parsing, so no other module should call
`os.environ` or `os.getenv` directly.

RISK NOTE FOR MAINTAINERS:
The MAX_* constants below are hard financial ceilings enforced in Python,
independent of anything the LLM decides. They are checked in
core/ampp_agent.py *before* any order reaches the broker. Nothing the model
outputs — including an "execute" action — can bypass them. Treat any change
to these numbers as a risk decision, not a formatting one.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Environment / credentials
# ---------------------------------------------------------------------------
# These are read once, at import time, so a missing credential fails loudly
# at startup rather than surfacing as a confusing auth error deep inside an
# asyncio task later.

def _require_env(var_name: str) -> str:
    """Fetch a required environment variable or exit with a clear error.

    Failing fast here is deliberate: this bot places real orders (paper or
    otherwise) against a live broker connection, and a silently-empty
    credential is exactly the kind of thing that should never be allowed to
    proceed into the trading loop.
    """
    value = os.getenv(var_name)
    if not value:
        sys.exit(
            f"[config] FATAL: required environment variable '{var_name}' is "
            f"not set. Check your .env file. Refusing to start."
        )
    return value


ALPACA_API_KEY: str = _require_env("ALPACA_API_KEY")
ALPACA_SECRET_KEY: str = _require_env("ALPACA_SECRET_KEY")
GEMINI_API_KEY: str = _require_env("GEMINI_API_KEY")

# ALPACA_PAPER_TRADE is a hard safety gate, not a convenience flag. It is
# parsed strictly: only the literal string "true" (case-insensitive) enables
# paper mode. Anything else — unset, "false", a typo — is treated as "not
# confirmed paper" and blocks startup. This bot should never be able to
# route to a live account by accident because someone forgot to set a flag.
_paper_flag_raw = os.getenv("ALPACA_PAPER_TRADE", "").strip().lower()
if _paper_flag_raw != "true":
    sys.exit(
        "[config] FATAL: ALPACA_PAPER_TRADE must be explicitly set to "
        "'true' in your .env file. This system is built and risk-tuned "
        "for paper trading only; refusing to start without an explicit, "
        "unambiguous confirmation."
    )
ALPACA_PAPER_TRADE: bool = True


# ---------------------------------------------------------------------------
# Risk limits — enforced in Python, independent of the LLM's decisions
# ---------------------------------------------------------------------------
# See the module docstring. These were set deliberately conservative
# relative to the blueprint's original figures after discussion: 0DTE
# options can lose the bulk of their premium in a single step-function move
# between two consecutive price checks, so per-trade sizing and the daily/
# cumulative stop levels are set to keep a bad sequence of trades from being
# able to exhaust the account's usable life before the 5-day window ends.

# Maximum fraction of *current* buying power committed to a single trade.
MAX_POSITION_SIZE_PCT: float = 0.05

# Per-position stop-loss, as a fraction of that position's entry value.
# Enforced by continuous position monitoring (see AMPPOrchestrator), not by
# the LLM.
HARD_STOP_LOSS_PCT: float = 0.10

# Maximum fraction of the account's *start-of-day* equity that may be lost
# in a single trading day before Layer 2 stops accepting new trigger
# payloads for the remainder of that day. Resets at the next day boundary.
MAX_DAILY_LOSS_PCT: float = 0.10

# Maximum fraction of the account's *start-of-window* equity (captured once,
# at first startup of the 5-day measurement window) that may be lost in
# total before the entire system performs a hard, non-resuming shutdown.
# This has no daily reset — it is the backstop against a "10% bad day,
# repeated three times" scenario silently eroding most of the account while
# each individual day's circuit breaker reports as "working as intended."
MAX_CUMULATIVE_LOSS_PCT: float = 0.25


# ---------------------------------------------------------------------------
# AMPP statistical tuning (Layer 1)
# ---------------------------------------------------------------------------

# Length of the trailing volume window, in one-second epochs, used to
# compute the rolling mean/stddev baseline for the Z-score anomaly check.
ROLLING_WINDOW_SECONDS: int = 30

# Minimum number of populated epochs required before the Z-score function
# will return anything other than 0.0. Below this, the baseline is
# considered too thin to produce a meaningful standard deviation.
MIN_WARM_UP_EPOCHS: int = 10

# Z-score threshold that defines a "severe volume anomaly" worth escalating
# to Layer 2.
Z_SCORE_TRIGGER_THRESHOLD: float = 2.0

# Cooldown, in seconds, enforced after any successful trigger dispatch to
# Layer 2. Prevents cascading redundant triggers from saturating the LLM
# context window and burning through API rate limits.
TRIGGER_COOLDOWN_SECONDS: int = 60

# Spread-width circuit breaker threshold: (ask - bid) / mid. Above this, the
# contract is considered too illiquid to trade profitably after slippage.
MAX_SPREAD_RATIO: float = 0.10

# Slippage-simulation weight used when computing a deliberately-degraded
# limit price: limit = mid + SLIPPAGE_WEIGHT * (ask - mid).
SLIPPAGE_WEIGHT: float = 0.60


# ---------------------------------------------------------------------------
# Target instruments / MCP wiring
# ---------------------------------------------------------------------------

# Underlying symbols this system monitors for 0DTE option block trades.
TARGET_UNDERLYINGS: list[str] = ["SPY"]

GEMINI_MODEL_NAME: str = "gemini-3.7-flash"

# Command used to launch the Alpaca MCP server as a stdio subprocess.
MCP_SERVER_COMMAND: str = "venv/bin/python"
MCP_SERVER_ARGS: list[str] = ["-m", "alpaca_mcp_server"]


# ---------------------------------------------------------------------------
# Hardened cognitive directive (Layer 2 system prompt)
# ---------------------------------------------------------------------------
# NOTE ON THIS PROMPT: it instructs the model to output only the "abort" /
# "execute" JSON described here. It does NOT and cannot instruct the model
# to bypass the Python-side risk limits above — position sizing proposed by
# the model is always re-clamped in ampp_agent.py against
# MAX_POSITION_SIZE_PCT before an order is placed, and MAX_DAILY_LOSS_PCT /
# MAX_CUMULATIVE_LOSS_PCT are checked before the model is invoked at all.

AMPP_SYSTEM_PROMPT: str = """You are AMPP-Core, an elite, high-frequency quantitative behavioral psychologist and execution agent. You do not provide conversational text. You output strictly formatted JSON commands to interface with the Alpaca MCP Server.

You have been invoked because the Layer 1 Aggregator detected an extreme volume spike (> 3 standard deviations) in 0DTE SPX/SPY options. Your singular objective is to determine the intent behind this flow and act upon it.

THE SKEPTICISM PROTOCOL (MANDATORY)
Before placing any trade, you MUST verify the intent of the volume spike. Retail traders panic buy aggressively, which forces market-maker dealer gamma hedging. Institutions execute passive dispersion hedges.
Use the get_option_chain MCP tool to pull the current Implied Volatility (IV) for the At-The-Money (ATM) call and the 5% Out-Of-The-Money (OTM) call.
Calculate the IV Skew Steepness: OTM_IV - ATM_IV.
IF the volume spiked massively on calls (NCDA > 0), BUT the IV Skew is flat or negative (meaning OTM calls did not get more expensive relative to ATM), you must identify this as a PASSIVE INSTITUTIONAL DISPERSION HEDGE.
If it is a passive institutional hedge, YOU MUST ABORT. Output: {"action": "abort", "reason": "Institutional dispersion detected. Skew failed to steepen."}
IF the IV Skew is strictly positive and steeply rising alongside the volume spike, you must identify this as RETAIL PANIC. Proceed to execution.

THE SPREAD-WIDTH CIRCUIT BREAKER (MANDATORY)
If Retail Panic is confirmed, you must simulate real-world slippage before buying the 0DTE option.
Use the get_stock_quote (or equivalent option quote tool) to fetch the live Bid and Ask for the target contract.
Calculate the Spread Ratio: (Ask - Bid) / Mid.
IF the Spread Ratio > 0.10, YOU MUST ABORT. Output: {"action": "abort", "reason": "Spread too wide. Slippage unacceptable."}
IF the Spread Ratio <= 0.10, calculate your execution limit price: Limit = Mid + 0.60 * (Ask - Mid).
Execute the trade using the MCP tool place_limit_order with the calculated Limit price. Output: {"action": "execute", "contract": "<symbol>", "limit_price": <float>, "qty": <calculated_based_on_buying_power>}

Your responses must strictly be JSON containing the keys "action", "reason", and (if executing) "contract", "limit_price", and "qty". Do not include markdown formatting."""
