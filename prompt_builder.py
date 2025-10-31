from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import ccxt
import pandas as pd
import pandas_ta as ta
import requests
from dotenv import load_dotenv

# --- Constants & Paths ---
COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"]
INTRADAY_TIMEFRAME = "3m"
LONGTERM_TIMEFRAME = "4h"
INTRADAY_BARS = 10
LONGTERM_BARS = 10
SYSTEM_PROMPT_FILENAME = "system_prompt.md"
STATE_PATH = Path("bot_state.json")
DEFAULT_SLEEP = 1.5
MAX_DEEPSEEK_RETRIES = 3
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

load_dotenv()

# --- Data Classes ---
@dataclass
class DeepSeekResponse:
    raw_text: str
    decisions: Dict[str, Any]
    chain_of_thought: Optional[str]


# --- Utility Helpers ---
def log_section(title: str, content: str) -> None:
    print(f"\n===== {title} =====")
    print(content)


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
    account_prompt += f"Available Cash: {available_cash}\n"
    account_prompt += f"Current Account Value: {total_balance}\n"
    account_prompt += f"Current live positions & performance: {positions_blob}\n"
    account_prompt += "Sharpe Ratio: 0.0\n"

    balances = {
        "total_balance": total_balance,
        "available_cash": available_cash,
        "pnl_percent": pnl_percent,
    }
    return account_prompt, positions_map, balances


# --- DeepSeek Client ---
class DeepSeekClient:
    def __init__(self, api_key: Optional[str]) -> None:
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required")
        self.api_key = api_key
        self.session = requests.Session()

    def request(self, system_prompt: str, user_prompt: str) -> DeepSeekResponse:
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        backoff = 2.0
        last_error: Optional[str] = None
        for attempt in range(1, MAX_DEEPSEEK_RETRIES + 1):
            try:
                response = self.session.post(
                    DEEPSEEK_URL,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=60,
                )
                if response.status_code >= 500:
                    last_error = f"Server error {response.status_code}: {response.text}"
                    raise RuntimeError(last_error)
                if response.status_code == 429:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"]["content"]
                chain = None
                if "<chain_of_thought>" in content:
                    chain_match = re.search(
                        r"<chain_of_thought>(.*?)</chain_of_thought>",
                        content,
                        re.DOTALL,
                    )
                    if chain_match:
                        chain = chain_match.group(1).strip()
                decisions = extract_json_block(content)
                return DeepSeekResponse(raw_text=content, decisions=decisions, chain_of_thought=chain)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                log_section("LLM", f"DeepSeek attempt {attempt} failed: {exc}")
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError(f"DeepSeek request failed: {last_error}")


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
) -> Dict[str, Any]:
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


def close_position(
    exchange: ccxt.Exchange,
    symbol: str,
    contracts: float,
) -> Optional[Dict[str, Any]]:
    if not contracts:
        return None
    side = "sell" if contracts > 0 else "buy"
    amount = abs(contracts)
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


# --- Decision Processing ---
def process_decisions(
    exchange: ccxt.Exchange,
    decisions: Dict[str, Any],
    positions_map: Dict[str, Dict[str, Any]],
    state: Dict[str, Any],
) -> None:
    state_positions = state.setdefault("positions", {})

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
            entry_order = send_market_order(exchange, symbol, order_side, amount)
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
            continue

        log_section("WARNING", f"Unhandled signal {signal} for {coin}")


# --- Main Routine ---
def load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_FILENAME, "r", encoding="utf-8") as handle:
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
        },
    })
    exchange.enable_demo_trading(True)
    exchange.load_markets()
    return exchange


def main() -> None:
    state = load_state()
    exchange = build_exchange()

    current_time = pd.Timestamp.utcnow()
    start_timestamp = state.get("start_timestamp")
    if not start_timestamp:
        start_timestamp = current_time.isoformat()
        state["start_timestamp"] = start_timestamp

    # Compute run metadata for prompt context
    minutes_since_start = int(
        max(
            0,
            (current_time - pd.Timestamp(start_timestamp)).total_seconds(),
        )
        // 60
    )
    invocation_count = int(state.get("invocation_count", 0)) + 1
    state["invocation_count"] = invocation_count

    system_prompt = load_system_prompt()
    market_prompt = build_market_prompt(exchange)
    account_prompt, positions_map, balances = build_account_prompt(exchange, state)

    user_prompt = (
        "It has been {minutes} minutes since you started trading. "
        "The current time is {now} and you've been invoked {count} times. "
        "Below, we are providing you with a variety of state data, price data, and "
        "predictive signals so you can discover alpha. Below that is your current "
        "account information, value, performance, positions, etc.\n\n"
    ).format(minutes=minutes_since_start, now=current_time, count=invocation_count)
    user_prompt += "ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST\n\n"
    user_prompt += market_prompt
    user_prompt += "\n"
    user_prompt += account_prompt

    log_section("USER PROMPT", user_prompt)

    client = DeepSeekClient(os.getenv("DEEPSEEK_API_KEY"))
    response = client.request(system_prompt, user_prompt)

    log_section("LLM RAW", response.raw_text)
    if response.chain_of_thought:
        log_section("LLM CHAIN OF THOUGHT", response.chain_of_thought)
    log_section("LLM DECISIONS", json.dumps(response.decisions, indent=2))

    process_decisions(exchange, response.decisions, positions_map, state)
    save_state(state)

    time.sleep(DEFAULT_SLEEP)
    # Refresh view post-execution
    account_prompt_after, _, _ = build_account_prompt(exchange, state)
    log_section("ACCOUNT SUMMARY", account_prompt_after)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_section("EXIT", "Shutdown by user")
    except Exception as exc:  # noqa: BLE001
        log_section("FATAL", str(exc))
