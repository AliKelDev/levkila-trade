from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ccxt
from ccxt.base.errors import ExchangeError
import pandas as pd
import pandas_ta as ta
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- Constants & Paths ---
COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"]
INTRADAY_TIMEFRAME = "3m"
LONGTERM_TIMEFRAME = "4h"
INTRADAY_BARS = 10
LONGTERM_BARS = 10
SYSTEM_PROMPT_FILENAME = "system_prompt.md"
STATE_PATH = Path("bot_state.json")
SENTIMENT_CACHE_PATH = Path("sentiment_snapshot.json")
DEFAULT_SLEEP = 1.5
MAX_GEMINI_RETRIES = 3
GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_MAX_TOKENS = 8192
GEMINI_TIMEOUT = 300  # seconds

# DeepSeek configuration
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_MAX_TOKENS = 8192
MAX_DEEPSEEK_RETRIES = 3

load_dotenv()

# --- Runtime Logging ---
LOG_BUFFER: List[Dict[str, Any]] = []
_progress_thread: Optional[Any] = None
_progress_stop = False


def _progress_indicator() -> None:
    """Background thread that prints animated dots while cycle executes."""
    global _progress_stop
    dot_count = 0
    while not _progress_stop:
        dots = "." * ((dot_count % 4) + 1)
        print(f"\r⏳ Executing{dots}   ", end="", flush=True)
        dot_count += 1
        time.sleep(2)  # Update every 2 seconds
    print("\r" + " " * 30 + "\r", end="", flush=True)  # Clear the line


# --- Data Classes ---
@dataclass
class LLMResponse:
    raw_text: str
    decisions: Dict[str, Any]
    chain_of_thought: Optional[str]
    final_content: str
    summary: Optional[str]


@dataclass
class RunCycleResult:
    user_prompt: str
    system_prompt: str
    llm_raw: str
    chain_of_thought: Optional[str]
    decisions: Dict[str, Any]
    final_content: str
    summary: Optional[str]
    account_prompt_before: str
    account_prompt_after: str
    positions_before: Dict[str, Dict[str, Any]]
    positions_after: Dict[str, Dict[str, Any]]
    balances_before: Dict[str, Any]
    balances_after: Dict[str, Any]
    logs: List[Dict[str, Any]]
    minutes_since_start: int
    invocation_count: int
    run_timestamp: str


# --- Utility Helpers ---
def log_section(title: str, content: str) -> None:
    entry = {
        "title": title,
        "content": content,
        "ts": time.time(),
    }
    LOG_BUFFER.append(entry)
    print(f"\n===== {title} =====")
    print(content)


def consume_logs() -> List[Dict[str, Any]]:
    global LOG_BUFFER
    logs = LOG_BUFFER[:]
    LOG_BUFFER = []
    return logs


def load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            log_section("STATE", "State file corrupted. Starting fresh.")
    return {"positions": {}, "leverage_applied": {}, "invocation_count": 0}


def save_state(state: Dict[str, Any]) -> None:
    with STATE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def symbol_to_coin(symbol: str) -> str:
    return symbol.split("/")[0]


def coin_to_symbol(coin: str) -> str:
    coin = coin.upper()
    if "/" in coin:
        return coin
    return f"{coin}/USDT"


def extract_json_block(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            candidate = match.group(0)
            return json.loads(candidate)
        raise


def extract_decision_json(final_content: str, reasoning: Optional[str]) -> Dict[str, Any]:
    if final_content:
        match = re.search(r"<FINAL_JSON>(.*?)</FINAL_JSON>", final_content, re.DOTALL)
        if match:
            final_content = match.group(1)
    try:
        return extract_json_block(final_content)
    except json.JSONDecodeError:
        if reasoning:
            match = re.search(r"<FINAL_JSON>(.*?)</FINAL_JSON>", reasoning, re.DOTALL)
            if match:
                return extract_json_block(match.group(1))
        raise
    return extract_json_block(final_content)


def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int, state: Dict[str, Any]) -> None:
    leverage_cache = state.setdefault("leverage_applied", {})
    if leverage_cache.get(symbol) == leverage:
        return
    try:
        exchange.set_leverage(leverage, symbol)
        leverage_cache[symbol] = leverage
        log_section("LEVERAGE", f"Set leverage {leverage}x for {symbol}")
    except Exception as exc:  # noqa: BLE001
        log_section("ERROR", f"Failed to set leverage for {symbol}: {exc}")


def cancel_order_if_exists(exchange: ccxt.Exchange, symbol: str, order_id: Optional[str]) -> None:
    if not order_id or order_id == -1:
        return
    try:
        exchange.cancel_order(order_id, symbol)
        log_section("ORDERS", f"Cancelled order {order_id} on {symbol}")
    except Exception as exc:  # noqa: BLE001
        log_section("WARNING", f"Could not cancel order {order_id} on {symbol}: {exc}")


def ensure_precision(exchange: ccxt.Exchange, symbol: str, amount: float) -> float:
    try:
        return float(exchange.amount_to_precision(symbol, amount))
    except Exception:
        return amount


def ensure_price_precision(exchange: ccxt.Exchange, symbol: str, price: float) -> float:
    try:
        return float(exchange.price_to_precision(symbol, price))
    except Exception:
        return price


# --- Market Data Builders ---
def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df


def build_market_prompt(exchange: ccxt.Exchange) -> str:
    master_prompt = "CURRENT MARKET STATE FOR ALL COINS\n"
    master_prompt += "ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST\n\n"

    public_derivatives = getattr(build_market_prompt, "_public_derivatives", None)
    if public_derivatives is None:
        public_derivatives = ccxt.binance({"options": {"defaultType": "future"}})
        try:
            public_derivatives.load_markets()
        except Exception as exc:  # noqa: BLE001
            log_section("WARNING", f"Failed to load mainnet markets for derivatives data: {exc}")
        build_market_prompt._public_derivatives = public_derivatives

    for symbol in COINS:
        coin = symbol_to_coin(symbol)
        try:
            intraday_df = fetch_ohlcv(exchange, symbol, INTRADAY_TIMEFRAME, 100)
            longterm_df = fetch_ohlcv(exchange, symbol, LONGTERM_TIMEFRAME, 100)
        except Exception as exc:  # noqa: BLE001
            log_section("WARNING", f"Failed to fetch OHLCV for {symbol}: {exc}")
            continue

        if intraday_df.empty or longterm_df.empty:
            continue

        open_interest_latest = None
        open_interest_avg = None
        funding_rate = None
        if public_derivatives:
            if hasattr(public_derivatives, "fetch_open_interest_history"):
                try:
                    oi_history = public_derivatives.fetch_open_interest_history(symbol, limit=10)
                    amounts = [
                        float(entry.get("openInterestAmount") or entry.get("openInterestValue") or 0)
                        for entry in oi_history or []
                        if entry
                    ]
                    if amounts:
                        open_interest_latest = amounts[-1]
                        open_interest_avg = sum(amounts) / len(amounts)
                except Exception as exc:  # noqa: BLE001
                    public_derivatives = None
            if public_derivatives:
                try:
                    funding_info = public_derivatives.fetch_funding_rate(symbol)
                    funding_rate = funding_info.get("fundingRate")
                except Exception as exc:  # noqa: BLE001
                    public_derivatives = None

        for df in (intraday_df, longterm_df):
            df.ta.ema(length=20, append=True)
            df.ta.ema(length=50, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.rsi(length=7, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.atr(length=3, append=True)
            df.ta.atr(length=14, append=True)

        latest_intraday = intraday_df.iloc[-1]
        latest_longterm = longterm_df.iloc[-1]
        intraday_series = intraday_df.tail(INTRADAY_BARS)
        longterm_series = longterm_df.tail(LONGTERM_BARS)

        coin_prompt = f"ALL {coin} DATA\n"
        coin_prompt += (
            "current_price = {price}, current_ema20 = {ema20:.3f}, "
            "current_macd = {macd:.3f}, current_rsi (7 period) = {rsi7:.3f}\n\n"
        ).format(
            price=latest_intraday["close"],
            ema20=latest_intraday["EMA_20"],
            macd=latest_intraday["MACDh_12_26_9"],
            rsi7=latest_intraday["RSI_7"],
        )

        if open_interest_latest is not None:
            if open_interest_avg is not None:
                coin_prompt += (
                    f"Open Interest: Latest: {open_interest_latest:.5f} "
                    f"Average: {open_interest_avg:.5f}\n"
                )
            else:
                coin_prompt += f"Open Interest: Latest: {open_interest_latest:.5f}\n"
        if funding_rate is not None:
            coin_prompt += f"Funding Rate: {funding_rate}\n"

        coin_prompt += "Intraday series (by minute, oldest → latest):\n"
        coin_prompt += f"Mid prices: {intraday_series['close'].tolist()}\n"
        coin_prompt += "EMA indicators (20‑period): {ema}\n".format(
            ema=[round(x, 3) for x in intraday_series["EMA_20"].tolist()]
        )
        coin_prompt += "MACD indicators: {macd}\n".format(
            macd=[round(x, 3) for x in intraday_series["MACDh_12_26_9"].tolist()]
        )
        coin_prompt += "RSI indicators (7‑Period): {rsi}\n\n".format(
            rsi=[round(x, 3) for x in intraday_series["RSI_7"].tolist()]
        )

        coin_prompt += "Longer‑term context (4‑hour timeframe):\n"
        coin_prompt += (
            "20‑Period EMA: {ema20:.3f} vs. 50‑Period EMA: {ema50:.3f}\n"
        ).format(ema20=latest_longterm["EMA_20"], ema50=latest_longterm["EMA_50"])
        coin_prompt += (
            "3‑Period ATR: {atr3:.3f} vs. 14‑Period ATR: {atr14:.3f}\n"
        ).format(atr3=latest_longterm["ATRr_3"], atr14=latest_longterm["ATRr_14"])
        coin_prompt += (
            "Current Volume: {volume} vs. Average Volume: {avg_volume:.3f}\n"
        ).format(volume=latest_longterm["volume"], avg_volume=longterm_df["volume"].mean())
        coin_prompt += "MACD indicators: {macd}\n".format(
            macd=[round(x, 3) for x in longterm_series["MACDh_12_26_9"].tolist()]
        )
        coin_prompt += "RSI indicators (14‑Period): {rsi}\n\n".format(
            rsi=[round(x, 3) for x in longterm_series["RSI_14"].tolist()]
        )

        master_prompt += coin_prompt

    return master_prompt


def fetch_account_position_map(exchange: ccxt.Exchange) -> Dict[str, Dict[str, Any]]:
    raw_positions = exchange.fetch_positions()
    positions: Dict[str, Dict[str, Any]] = {}
    for entry in raw_positions:
        contracts = float(entry.get("contracts") or entry.get("info", {}).get("positionAmt") or 0)
        if not contracts:
            continue
        symbol = entry.get("symbol")
        if not symbol:
            continue
        coin = symbol_to_coin(symbol)
        info = entry.get("info", {})
        positions[coin] = {
            "symbol": symbol,
            "contracts": contracts,
            "entry_price": float(entry.get("entryPrice") or info.get("entryPrice") or 0),
            "mark_price": float(entry.get("markPrice") or info.get("markPrice") or 0),
            "liquidation_price": float(info.get("liquidationPrice") or 0),
            "leverage": int(entry.get("leverage") or info.get("leverage") or 0),
            "unrealized_pnl": float(entry.get("unrealizedPnl") or info.get("unRealizedProfit") or 0),
        }
    return positions


def build_account_prompt(exchange: ccxt.Exchange, state: Dict[str, Any]) -> tuple[str, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    balance = exchange.fetch_balance()
    total_balance = balance.get("total", {}).get("USDT", 0)
    available_cash = balance.get("free", {}).get("USDT", 0)
    margin_balance = total_balance

    info = balance.get("info")
    # Binance futures returns a list of asset dicts; prefer their richer fields when present
    assets = None
    if isinstance(info, dict):
        assets = info.get("assets") or info.get("balances")
        if isinstance(assets, dict):  # Some CCXT versions return mapping keyed by asset
            assets = list(assets.values())
    elif isinstance(info, list):
        assets = info

    if isinstance(assets, list):
        for entry in assets:
            if not isinstance(entry, dict):
                continue
            if entry.get("asset") != "USDT":
                continue
            available_cash = float(
                entry.get("availableBalance")
                or entry.get("available")
                or entry.get("free")
                or available_cash
            )
            total_balance = float(
                entry.get("balance")
                or entry.get("walletBalance")
                or entry.get("crossWalletBalance")
                or total_balance
            )
            margin_balance = float(
                entry.get("marginBalance")
                or entry.get("margin_balance")
                or entry.get("crossMarginBalance")
                or margin_balance
            )
            break
    starting_capital = state.get("starting_capital", 5000)
    pnl_percent = 0.0
    if starting_capital:
        pnl_percent = ((total_balance - starting_capital) / starting_capital) * 100

    positions_map = fetch_account_position_map(exchange)
    state_positions = state.setdefault("positions", {})

    positions_str_chunks = []
    for coin, position in positions_map.items():
        state_entry = state_positions.get(coin, {})
        bundle = {
            "symbol": coin,
            "quantity": round(position["contracts"], 6),
            "entry_price": position["entry_price"],
            "current_price": position["mark_price"],
            "liquidation_price": position["liquidation_price"],
            "unrealized_pnl": position["unrealized_pnl"],
            "leverage": position["leverage"],
        }
        bundle.update({
            "exit_plan": state_entry.get("exit_plan", {}),
            "confidence": state_entry.get("confidence"),
            "risk_usd": state_entry.get("risk_usd"),
            "sl_oid": state_entry.get("sl_oid", -1),
            "tp_oid": state_entry.get("tp_oid", -1),
            "wait_for_fill": state_entry.get("wait_for_fill", False),
            "entry_oid": state_entry.get("entry_oid", -1),
            "notional_usd": state_entry.get("notional_usd"),
        })
        positions_str_chunks.append(str(bundle))

    positions_blob = " ".join(positions_str_chunks) if positions_str_chunks else "{}"

    account_prompt = "HERE IS YOUR ACCOUNT INFORMATION & PERFORMANCE\n"
    account_prompt += f"Current Total Return (percent): {pnl_percent:.2f}%\n"
    account_prompt += f"Current Account Value: {total_balance}\n"
    account_prompt += f"Current live positions & performance: {positions_blob}\n"
    account_prompt += "Sharpe Ratio: 0.0\n"

    balances = {
        "total_balance": total_balance,
        "available_cash": available_cash,
        "margin_balance": margin_balance,
        "pnl_percent": pnl_percent,
    }
    return account_prompt, positions_map, balances


# --- Gemini Client ---
class GeminiClient:
    def __init__(self, api_key: Optional[str]) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self.api_key = api_key
        # Reuse genai.Client
        self.client = genai.Client(api_key=api_key)

    def request(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        last_error: Optional[str] = None
        for attempt in range(1, MAX_GEMINI_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=GEMINI_MAX_TOKENS,
                    ),
                )
                final_content = response.text or ""
                chain = None # Flash-lite doesn't expose CoT in this way

                try:
                    decisions = extract_decision_json(final_content, chain)
                except json.JSONDecodeError as parse_error:
                    snippet = final_content[:200]
                    log_section(
                        "LLM",
                        (
                            "Failed to parse JSON from Gemini response. "
                            f"Error: {parse_error}. Snippet: {snippet!r}"
                        ),
                    )
                    raise

                return LLMResponse(
                    raw_text=final_content,
                    decisions=decisions,
                    chain_of_thought=chain,
                    final_content=final_content,
                    summary=None,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                log_section("LLM", f"Gemini attempt {attempt} failed: {exc}")
                time.sleep(2.0 * attempt)
        raise RuntimeError(f"Gemini request failed: {last_error}")


class DeepSeekClient:
    """DeepSeek LLM client using OpenAI-compatible API."""
    def __init__(self, api_key: Optional[str]) -> None:
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required")
        self.api_key = api_key
        self.base_url = DEEPSEEK_API_BASE

    def request(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        last_error: Optional[str] = None
        for attempt in range(1, MAX_DEEPSEEK_RETRIES + 1):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": DEEPSEEK_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": DEEPSEEK_MAX_TOKENS,
                    "temperature": 0.7,
                }
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
                
                final_content = result["choices"][0]["message"]["content"]
                chain = None  # DeepSeek doesn't expose reasoning separately

                try:
                    decisions = extract_decision_json(final_content, chain)
                except json.JSONDecodeError as parse_error:
                    snippet = final_content[:200]
                    log_section(
                        "LLM",
                        (
                            "Failed to parse JSON from DeepSeek response. "
                            f"Error: {parse_error}. Snippet: {snippet!r}"
                        ),
                    )
                    raise

                return LLMResponse(
                    raw_text=final_content,
                    decisions=decisions,
                    chain_of_thought=chain,
                    final_content=final_content,
                    summary=None,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                log_section("LLM", f"DeepSeek attempt {attempt} failed: {exc}")
                time.sleep(2.0 * attempt)
        raise RuntimeError(f"DeepSeek request failed: {last_error}")


def get_llm_client(model_name: str = "gemini") -> GeminiClient | DeepSeekClient:
    """Factory function to create the appropriate LLM client based on model name."""
    model_name = model_name.lower().strip()
    
    if model_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        log_section("LLM", f"Using Gemini model: {GEMINI_MODEL}")
        return GeminiClient(api_key)
    elif model_name == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment")
        log_section("LLM", f"Using DeepSeek model: {DEEPSEEK_MODEL}")
        return DeepSeekClient(api_key)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose 'gemini' or 'deepseek'.")


# --- Order Helpers ---
def calculate_order_amount(
    side: str,
    entry_price: float,
    stop_price: float,
    risk_usd: float,
) -> float:
    if side == "long":
        risk_per_unit = max(entry_price - stop_price, 0)
    else:
        risk_per_unit = max(stop_price - entry_price, 0)
    if risk_per_unit <= 0:
        raise ValueError("Invalid stop loss; risk per unit is non-positive")
    return risk_usd / risk_per_unit


def place_bracket_orders(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    amount: float,
    stop_loss: float,
    take_profit: float,
) -> Dict[str, Optional[str]]:
    orders = {"stop_loss": None, "take_profit": None}
    reduce_side = "sell" if side == "long" else "buy"

    # Stop loss
    try:
        sl_order = exchange.create_order(
            symbol,
            "STOP_MARKET",
            reduce_side,
            amount,
            None,
            {
                "stopPrice": ensure_price_precision(exchange, symbol, stop_loss),
                "reduceOnly": True,
                "workingType": "MARK_PRICE",
            },
        )
        orders["stop_loss"] = sl_order.get("id")
        log_section("ORDERS", f"Placed stop loss {orders['stop_loss']} at {stop_loss}")
    except Exception as exc:  # noqa: BLE001
        log_section("WARNING", f"Failed to place stop loss: {exc}")

    # Take profit
    try:
        tp_order = exchange.create_order(
            symbol,
            "TAKE_PROFIT_MARKET",
            reduce_side,
            amount,
            None,
            {
                "stopPrice": ensure_price_precision(exchange, symbol, take_profit),
                "reduceOnly": True,
                "workingType": "MARK_PRICE",
            },
        )
        orders["take_profit"] = tp_order.get("id")
        log_section("ORDERS", f"Placed take profit {orders['take_profit']} at {take_profit}")
    except Exception as exc:  # noqa: BLE001
        log_section("WARNING", f"Failed to place take profit: {exc}")

    return orders


def send_market_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    amount: float,
) -> Optional[Dict[str, Any]]:
    try:
        order = exchange.create_order(
            symbol,
            "market",
            side,
            amount,
            None,
            {
                "reduceOnly": False,
                "newOrderRespType": "RESULT",
            },
        )
        log_section("ORDERS", f"Executed market order {order.get('id')} on {symbol}")
        return order
    except ExchangeError as exc:
        message = str(exc)
        if "Margin is insufficient" in message:
            log_section("ERROR", f"Entry rejected for {symbol}: {message}")
            return None
        raise


def close_position(
    exchange: ccxt.Exchange,
    symbol: str,
    contracts: float,
) -> Optional[Dict[str, Any]]:
    if not contracts:
        return None
    side = "sell" if contracts > 0 else "buy"
    amount = abs(contracts)
    try:
        order = exchange.create_order(
            symbol,
            "market",
            side,
            amount,
            None,
            {
                "reduceOnly": True,
                "newOrderRespType": "RESULT",
            },
        )
        log_section("ORDERS", f"Closed position with order {order.get('id')} on {symbol}")
        return order
    except ExchangeError as exc:
        message = str(exc)
        if "ReduceOnly Order is rejected" in message:
            log_section("INFO", f"Close request ignored for {symbol}: {message}")
            return None
        raise


# --- Decision Processing ---
def process_decisions(
    exchange: ccxt.Exchange,
    decisions: Dict[str, Any],
    positions_map: Dict[str, Dict[str, Any]],
    balances: Dict[str, Any],
    state: Dict[str, Any],
) -> None:
    state_positions = state.setdefault("positions", {})
    available_cash = float(balances.get("available_cash", 0))

    for coin_key, payload in decisions.items():
        coin = coin_key.upper()
        symbol = coin_to_symbol(coin)
        args = payload.get("trade_signal_args", {})
        signal = args.get("signal")
        if not signal:
            log_section("WARNING", f"Missing signal for {coin}")
            continue

        log_section("ACTION", f"{coin} -> {signal}")

        if signal == "hold":
            state_positions.setdefault(coin, {})
            state_positions[coin].update({
                "exit_plan": args.get("exit_plan", state_positions[coin].get("exit_plan")),
                "confidence": args.get("confidence", state_positions[coin].get("confidence")),
                "risk_usd": args.get("risk_usd", state_positions[coin].get("risk_usd")),
                "wait_for_fill": False,
            })
            continue

        if signal == "close_position":
            position = positions_map.get(coin)
            if not position:
                log_section("INFO", f"No open position to close for {coin}")
                continue
            cancel_order_if_exists(exchange, symbol, state_positions.get(coin, {}).get("sl_oid"))
            cancel_order_if_exists(exchange, symbol, state_positions.get(coin, {}).get("tp_oid"))
            close_position(exchange, symbol, position["contracts"])
            state_positions.pop(coin, None)
            continue

        if signal in {"buy_to_enter", "sell_to_enter"}:
            side = "long" if signal == "buy_to_enter" else "short"
            existing = positions_map.get(coin)
            if existing and existing.get("contracts"):
                log_section("INFO", f"Position already open for {coin}, skipping entry")
                continue

            leverage = int(args.get("leverage", 10))
            set_leverage(exchange, symbol, leverage, state)

            ticker = exchange.fetch_ticker(symbol)
            price = ticker.get("last") or ticker.get("close")
            if not price:
                log_section("ERROR", f"Unable to determine price for {coin}")
                continue

            stop_loss = float(args.get("exit_plan", {}).get("stop_loss"))
            target = float(args.get("exit_plan", {}).get("profit_target"))
            risk_usd = float(args.get("risk_usd"))

            amount = calculate_order_amount(side, price, stop_loss, risk_usd)
            amount = ensure_precision(exchange, symbol, amount)
            if amount <= 0:
                log_section("ERROR", f"Computed amount invalid for {coin}")
                continue

            market = exchange.market(symbol)
            notional = amount * price
            min_cost = market.get("limits", {}).get("cost", {}).get("min")
            if min_cost and notional < min_cost:
                log_section(
                    "WARNING",
                    f"Notional {notional} below minimum {min_cost} for {symbol}",
                )
                continue

            order_side = "buy" if side == "long" else "sell"
            required_margin = notional / leverage if leverage else notional
            if required_margin > available_cash:
                log_section(
                    "WARNING",
                    (
                        f"Insufficient margin for {coin}: required {required_margin:.2f} "
                        f"but available {available_cash:.2f}"
                    ),
                )
                continue

            entry_order = send_market_order(exchange, symbol, order_side, amount)
            if not entry_order:
                continue
            entry_price = float(entry_order.get("average") or entry_order.get("price") or price)

            bracket = place_bracket_orders(exchange, symbol, side, amount, stop_loss, target)

            state_positions[coin] = {
                "exit_plan": args.get("exit_plan", {}),
                "confidence": args.get("confidence"),
                "risk_usd": risk_usd,
                "sl_oid": bracket.get("stop_loss") or -1,
                "tp_oid": bracket.get("take_profit") or -1,
                "wait_for_fill": False,
                "entry_oid": entry_order.get("id"),
                "notional_usd": notional,
                "leverage": leverage,
                "entry_price": entry_price,
                "side": side,
            }
            available_cash -= required_margin
            continue

        # AGGRESSIVE MODE: Add to existing position (pyramiding)
        if signal == "add_to_position":
            position = positions_map.get(coin)
            if not position or not position.get("contracts"):
                log_section("WARNING", f"No position to add to for {coin}")
                continue
            
            # Determine side from existing position
            current_contracts = position["contracts"]
            side = "long" if current_contracts > 0 else "short"
            order_side = "buy" if side == "long" else "sell"
            
            # Get additional size from args
            additional_size = float(args.get("additional_contracts", 0))
            if additional_size <= 0:
                log_section("WARNING", f"Invalid additional_contracts for {coin}")
                continue
            
            additional_size = ensure_precision(exchange, symbol, additional_size)
            
            ticker = exchange.fetch_ticker(symbol)
            price = ticker.get("last") or ticker.get("close")
            
            # Place market order to increase position
            add_order = send_market_order(exchange, symbol, order_side, additional_size)
            if add_order:
                log_section("PYRAMID", f"Added {additional_size} contracts to {coin} position")
            continue

        # AGGRESSIVE MODE: Adjust stop-loss and/or take-profit
        if signal == "adjust_exits":
            position = positions_map.get(coin)
            if not position:
                log_section("WARNING", f"No position to adjust for {coin}")
                continue
            
            state_entry = state_positions.get(coin, {})
            
            # Cancel existing SL/TP orders
            cancel_order_if_exists(exchange, symbol, state_entry.get("sl_oid"))
            cancel_order_if_exists(exchange, symbol, state_entry.get("tp_oid"))
            
            # Get new exit levels
            new_stop = args.get("new_stop_loss")
            new_target = args.get("new_profit_target")
            
            if not new_stop or not new_target:
                log_section("WARNING", f"Missing exit levels for {coin}")
                continue
            
            # Determine side from position
            contracts = position["contracts"]
            side = "long" if contracts > 0 else "short"
            amount = abs(contracts)
            
            # Place new bracket orders
            bracket = place_bracket_orders(exchange, symbol, side, amount, float(new_stop), float(new_target))
            
            # Update state
            state_entry["sl_oid"] = bracket.get("stop_loss") or -1
            state_entry["tp_oid"] = bracket.get("take_profit") or -1
            state_entry["exit_plan"] = {
                "stop_loss": float(new_stop),
                "profit_target": float(new_target),
                "invalidation_condition": state_entry.get("exit_plan", {}).get("invalidation_condition", "")
            }
            
            log_section("ADJUST", f"Updated exits for {coin}: SL={new_stop}, TP={new_target}")
            continue

        # AGGRESSIVE MODE: Partial close
        if signal == "partial_close":
            position = positions_map.get(coin)
            if not position:
                log_section("WARNING", f"No position to partially close for {coin}")
                continue
            
            contracts = position["contracts"]
            close_percentage = float(args.get("close_percentage", 0))
            
            if close_percentage <= 0 or close_percentage >= 1:
                log_section("WARNING", f"Invalid close_percentage {close_percentage} for {coin}")
                continue
            
            # Calculate amount to close
            close_amount = abs(contracts) * close_percentage
            close_amount = ensure_precision(exchange, symbol, close_amount)
            
            # Place reduceOnly order
            side = "sell" if contracts > 0 else "buy"
            try:
                order = exchange.create_order(
                    symbol,
                    "market",
                    side,
                    close_amount,
                    None,
                    {
                        "reduceOnly": True,
                        "newOrderRespType": "RESULT",
                    },
                )
                log_section("PARTIAL", f"Closed {close_percentage*100:.0f}% ({close_amount} contracts) of {coin} position")
            except ExchangeError as exc:
                message = str(exc)
                log_section("ERROR", f"Partial close failed for {coin}: {message}")
            continue

        log_section("WARNING", f"Unhandled signal {signal} for {coin}")


# --- Main Routine ---
def load_system_prompt(prompt_template: str = "strict") -> str:
    """Load system prompt based on selected template."""
    template_map = {
        "strict": "system_prompt.md",
        "aggressive": "system_prompt_aggressive.md",
        "quant": "system_prompt_quant.md",
    }
    
    filename = template_map.get(prompt_template.lower(), "system_prompt.md")
    prompt_path = Path(filename)
    
    if not prompt_path.exists():
        log_section("WARNING", f"Prompt template '{filename}' not found, using default")
        filename = "system_prompt.md"
    
    log_section("PROMPT", f"Loading {prompt_template.upper()} template: {filename}")
    with open(filename, "r", encoding="utf-8") as handle:
        return handle.read()


def build_exchange() -> ccxt.Exchange:
    api_key = os.getenv("TESTNET_API_KEY")
    secret = os.getenv("TESTNET_SECRET_KEY")
    if not api_key or not secret:
        raise ValueError("TESTNET_API_KEY and TESTNET_SECRET_KEY must be set in .env")

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
            "recvWindow": 10000,  # 10 second tolerance for timestamp sync
        },
    })
    exchange.enable_demo_trading(True)
    exchange.load_markets()
    return exchange


def fetch_and_cache_sentiment(coins: List[str] = COINS) -> None:
    """
    Background job that fetches qualitative data and stores it on disk.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return

    try:
        client = genai.Client(api_key=api_key)

        coin_names = ", ".join(symbol_to_coin(symbol) for symbol in coins)
        prompt = (
            f"Using Google Search, find the latest market-moving news for: {coin_names}. "
            "Summarize the sentiment (Bullish/Bearish/Neutral) and top 3 headlines. "
            "Be concise (max 200 words). Focus on the last 24 hours."
        )

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=contents,
            config=config,
        )
        sentiment_text = (response.text or "").strip()
        if not sentiment_text:
            sentiment_text = "No sentiment content returned."
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": sentiment_text,
            "status": "fresh",
        }

        temp_path = SENTIMENT_CACHE_PATH.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2)
        temp_path.replace(SENTIMENT_CACHE_PATH)

        log_section("SENTIMENT", "Gemini cache updated successfully.")
    except Exception as exc:  # noqa: BLE001
        log_section("ERROR", f"Sentiment fetch failed: {exc}")


def get_cached_sentiment() -> str:
    """
    Returns formatted qualitative intelligence if we have a recent snapshot.
    """
    if not SENTIMENT_CACHE_PATH.exists():
        return ""

    try:
        with SENTIMENT_CACHE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        timestamp = data.get("timestamp")
        content = data.get("content", "")
        if not timestamp or not content:
            return ""

        ts = datetime.fromisoformat(timestamp)
        age_minutes = (datetime.now(timezone.utc) - ts).total_seconds() / 60

        label = "Latest News"
        if age_minutes > 60:
            label = "Old News (Context Only)"

        return (
            f"\n### QUALITATIVE INTELLIGENCE ({label} - {int(age_minutes)}m ago)\n"
            f"{content}\n"
        )
    except Exception:  # noqa: BLE001
        return ""


def run_cycle(model_name: str = "gemini", prompt_template: str = "strict") -> RunCycleResult:
    global _progress_thread, _progress_stop

    consume_logs()
    state = load_state()
    exchange = build_exchange()

    # Start progress indicator
    _progress_stop = False
    _progress_thread = threading.Thread(target=_progress_indicator, daemon=True)
    _progress_thread.start()

    try:
        current_time = pd.Timestamp.utcnow()
        start_timestamp = state.get("start_timestamp")
        if not start_timestamp:
            start_timestamp = current_time.isoformat()
            state["start_timestamp"] = start_timestamp

        minutes_since_start = int(
            max(
                0,
                (current_time - pd.Timestamp(start_timestamp)).total_seconds(),
            )
            // 60
        )
        invocation_count = int(state.get("invocation_count", 0)) + 1
        state["invocation_count"] = invocation_count

        system_prompt = load_system_prompt(prompt_template)
        market_prompt = build_market_prompt(exchange)
        account_prompt_before, positions_before, balances_before = build_account_prompt(exchange, state)
        qualitative_info = get_cached_sentiment()

        if "starting_capital" not in state:
            total_balance = balances_before.get("total_balance")
            state["starting_capital"] = total_balance

        user_prompt = (
            "It has been {minutes} minutes since you started trading. "
            "The current time is {now} and you've been invoked {count} times. "
            "Below, we are providing you with a variety of state data, price data, and "
            "predictive signals so you can discover alpha. Below that is your current "
            "account information, value, performance, positions, etc.\n\n"
        ).format(minutes=minutes_since_start, now=current_time, count=invocation_count)
        if qualitative_info:
            user_prompt += qualitative_info + "\n"
        user_prompt += "ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST\n\n"
        user_prompt += market_prompt
        user_prompt += "\n"
        user_prompt += account_prompt_before

        log_section("USER PROMPT", user_prompt)

        client = get_llm_client(model_name)
        log_section(
            "LLM CALL",
            f"Submitting request to {model_name.upper()} model",
        )
        try:
            response = client.request(system_prompt, user_prompt)
            log_section("LLM CALL", f"{model_name.upper()} response received successfully")
        except Exception as exc:  # noqa: BLE001
            log_section("LLM ERROR", f"{model_name.upper()} request failed after retries: {exc}")
            save_state(state)
            logs = consume_logs()
            return RunCycleResult(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                llm_raw="",
                chain_of_thought=None,
                decisions={},
                final_content="",
                summary=None,
                account_prompt_before=account_prompt_before,
                account_prompt_after=account_prompt_before,
                positions_before=positions_before,
                positions_after=positions_before,
                balances_before=balances_before,
                balances_after=balances_before,
                logs=logs,
                minutes_since_start=minutes_since_start,
                invocation_count=invocation_count,
                run_timestamp=current_time.isoformat(),
            )
        finally:
            log_section("LLM CALL", "LLM request finished")

        log_section("LLM RAW", response.raw_text)
        log_section(
            "LLM REASONING",
            response.chain_of_thought or "<no reasoning content>",
        )
        log_section("LLM DECISIONS", json.dumps(response.decisions, indent=2))

        process_decisions(exchange, response.decisions, positions_before, balances_before, state)
        save_state(state)

        time.sleep(DEFAULT_SLEEP)
        account_prompt_after, positions_after, balances_after = build_account_prompt(exchange, state)
        log_section("ACCOUNT SUMMARY", account_prompt_after)

        logs = consume_logs()

        return RunCycleResult(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            llm_raw=response.raw_text,
            chain_of_thought=response.chain_of_thought,
            decisions=response.decisions,
            final_content=response.final_content,
            summary=response.summary,
            account_prompt_before=account_prompt_before,
            account_prompt_after=account_prompt_after,
            positions_before=positions_before,
            positions_after=positions_after,
            balances_before=balances_before,
            balances_after=balances_after,
            logs=logs,
            minutes_since_start=minutes_since_start,
            invocation_count=invocation_count,
            run_timestamp=current_time.isoformat(),
        )
    finally:
        # Stop progress indicator
        _progress_stop = True
        if _progress_thread and _progress_thread.is_alive():
            _progress_thread.join(timeout=1)

        try:
            exchange.close()
        except Exception:  # noqa: BLE001
            pass


def get_account_snapshot() -> Dict[str, Any]:
    state = load_state()
    exchange = build_exchange()
    try:
        account_prompt, positions_map, balances = build_account_prompt(exchange, state)
    finally:
        try:
            exchange.close()
        except Exception:  # noqa: BLE001
            pass
    return {
        "account_prompt": account_prompt,
        "positions": positions_map,
        "balances": balances,
        "state": state,
    }


def main() -> None:
    run_cycle()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_section("EXIT", "Shutdown by user")
    except Exception as exc:  # noqa: BLE001
        log_section("FATAL", str(exc))
