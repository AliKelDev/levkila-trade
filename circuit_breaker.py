from __future__ import annotations

"""
circuit_breaker.py — Emergency shutdown module for DeepTrade.

Triggered in catastrophic failure scenarios where the decision system itself
is compromised (LLM unreachable, exchange unreachable). When active:
  1. All open positions are closed with market orders.
  2. All bracket orders (SL/TP) are cancelled.
  3. The auto-cycle scheduler is disabled.
  4. An emergency state file records the reason and timestamp.

Cases that trigger emergency mode:
  - DeepSeek LLM fails all retries (no decision engine available)
  - Exchange API totally unreachable during a critical operation

Cases that do NOT trigger (already handled by existing code):
  - A single order rejected (margin, min size) → logged, skipped, next cycle adapts
  - SL/TP hit automatically → reconciliation cleans up state
  - Individual API rate limits → backoff retry handles it
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import ccxt
from ccxt.base.errors import ExchangeError

LOGGER = logging.getLogger("deeptrade.circuit_breaker")

EMERGENCY_STATE_PATH = Path("emergency_state.json")


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def is_emergency_active() -> bool:
    """Return True if emergency mode is currently active."""
    if not EMERGENCY_STATE_PATH.exists():
        return False
    try:
        with EMERGENCY_STATE_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return bool(data.get("active", False))
    except Exception:  # noqa: BLE001
        return False


def get_emergency_info() -> Optional[Dict[str, Any]]:
    """Return the emergency state dict, or None if not active."""
    if not EMERGENCY_STATE_PATH.exists():
        return None
    try:
        with EMERGENCY_STATE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def clear_emergency() -> None:
    """Manually clear the emergency state (human operator action)."""
    if EMERGENCY_STATE_PATH.exists():
        EMERGENCY_STATE_PATH.unlink()
    LOGGER.info("Emergency state cleared by operator.")


def _write_emergency_state(reason: str, positions_closed: int, errors: list) -> None:
    state = {
        "active": True,
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "positions_closed": positions_closed,
        "errors_during_shutdown": errors,
    }
    temp = EMERGENCY_STATE_PATH.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)
    temp.replace(EMERGENCY_STATE_PATH)


# ---------------------------------------------------------------------------
# Core shutdown
# ---------------------------------------------------------------------------

def emergency_shutdown(
    exchange: ccxt.Exchange,
    reason: str,
    state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Close all open positions and cancel all bracket orders.
    Writes emergency_state.json so subsequent cycles refuse to run.

    Args:
        exchange: Active ccxt exchange instance (already connected).
        reason:   Human-readable description of what triggered the shutdown.
        state:    Optional bot_state dict — used to cancel tracked SL/TP order IDs
                  before fetching live positions (belt-and-suspenders approach).
    """
    LOGGER.critical("⚠️  EMERGENCY SHUTDOWN TRIGGERED: %s", reason)
    errors: list = []
    positions_closed = 0

    # Step 1 — Cancel all tracked SL/TP orders from state (faster than waiting for exchange)
    if state:
        state_positions = state.get("positions", {})
        for coin, pos_data in state_positions.items():
            symbol = f"{coin}/USDT"
            for oid_key in ("sl_oid", "tp_oid"):
                oid = pos_data.get(oid_key)
                if oid and oid != -1:
                    try:
                        exchange.cancel_order(oid, symbol)
                        LOGGER.info("Cancelled order %s on %s", oid, symbol)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"Cancel {oid} on {symbol}: {exc}")

    # Step 2 — Fetch live positions directly from exchange and close all of them
    try:
        raw_positions = exchange.fetch_positions()
        for entry in raw_positions:
            info = entry.get("info", {})
            raw_amt = info.get("positionAmt")
            contracts = float(raw_amt) if raw_amt is not None else float(entry.get("contracts") or 0)
            if not contracts:
                continue

            symbol = entry.get("symbol")
            if not symbol:
                continue

            side = "sell" if contracts > 0 else "buy"
            amount = abs(contracts)

            try:
                order = exchange.create_order(
                    symbol,
                    "market",
                    side,
                    amount,
                    None,
                    {"reduceOnly": True, "newOrderRespType": "RESULT"},
                )
                LOGGER.critical(
                    "EMERGENCY CLOSE: %s %s %.6f → order %s",
                    side.upper(), symbol, amount, order.get("id"),
                )
                positions_closed += 1
            except ExchangeError as exc:
                msg = str(exc)
                if "ReduceOnly Order is rejected" in msg:
                    LOGGER.info("Position already closed for %s (exchange rejected reduceOnly)", symbol)
                else:
                    LOGGER.error("Failed to close %s: %s", symbol, exc)
                    errors.append(f"Close {symbol}: {exc}")
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Unexpected error closing %s: %s", symbol, exc)
                errors.append(f"Close {symbol}: {exc}")

    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Could not fetch positions during emergency shutdown: %s", exc)
        errors.append(f"fetch_positions: {exc}")

    # Step 3 — Persist emergency state
    _write_emergency_state(reason, positions_closed, errors)

    LOGGER.critical(
        "Emergency shutdown complete. %d position(s) closed. %d error(s). "
        "Bot is now HALTED. Clear emergency_state.json to resume.",
        positions_closed,
        len(errors),
    )
