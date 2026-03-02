"""
auditeur.py — Agent IA "Auditeur"

Déclenché chaque jour à 1h05 heure française.
Tourne sur DeepSeek Reasoner (deepseek-reasoner).
Son rapport est destiné à l'analyse personnelle du CIO — il n'est pas consommé
par les autres agents.

Les rapports sont persistés dans auditeur_cache.json (liste, plus récent en tête,
max AUDITEUR_MAX_ENTRIES entrées).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from prompt_builder import DEEPSEEK_URL, build_exchange, STATE_PATH

load_dotenv()

# ── Constantes ──────────────────────────────────────────────────────────────

AUDITEUR_CACHE_PATH = Path("auditeur_cache.json")
AUDITEUR_MAX_ENTRIES = 90           # ~3 mois de rapports quotidiens
AUDITEUR_MODEL = "deepseek-reasoner"
AUDITEUR_MAX_TOKENS = 8192

CYCLE_HISTORY_PATH = Path("cycle_history.json")
SENTINELLE_DAILY_LOG_PATH = Path("sentinelle_daily_log.json")
DAILY_TRADE_LOG_PATH = Path("daily_trade_log.json")
DAILY_TRADE_LOG_WINDOW_HOURS = 48   # Rolling window — keeps 2 full days so the
                                    # Auditeur reading at 00:05 UTC always finds
                                    # the complete previous UTC day.

AUDITEUR_SYSTEM_PROMPT = """You are the Chief Performance Officer (CPO) and Lead Quantitative Auditor for DeepTrade, an algorithmic crypto hedge fund.
Your role is to analyze the daily trading ledger, audit the decisions made by the Chief Investment Officer (CIO) and the Executor agent, and report directly to the human CEO.

YOUR ANALYTICAL DIRECTIVES:
1. The Fee & Slippage Audit: Compare Gross PnL to Net PnL. If fees consume more than 30% of the Gross Profit, or if slippage is unusually high, you must flag this as a critical inefficiency (Overtrading or Timeframe too short).
2. The CIO Logic Audit: Analyze the "cio_rationale_at_entry" for closed trades. Did the CIO make a logical decision that was simply invalidated by the market, or was the logic inherently flawed/fomo-driven?
3. The "Neutrality" Audit: If 0 trades were taken, compare this to the macro context (BTC/ETH daily change). Was staying neutral a brilliant preservation of capital during a chop/range, or a missed opportunity during a massive trend?
4. Technical Health: Investigate any API/CCXT errors. Identify if margin limits were hit due to the €100 account size constraints.

YOUR TONE & BEHAVIOR:
- Be cold, ruthless, and purely data-driven. Do not sugarcoat losses. Do not praise lucky wins if the logic was flawed.
- Use precise quantitative language.
- Think deeply about the correlation between the CIO's rationale and the actual duration/outcome of the trade.

YOUR OUTPUT FORMAT:
You must output a highly structured daily report strictly in Markdown, using the following template. Do not add conversational fluff before or after the report.

# 📊 DeepTrade Daily Post-Mortem

**Date:** [Extract from input] | **Net PnL:** [X] EUR ([Y]%) | **Win Rate:** [Z]%

## 1. Executive Summary
[Write 2-3 hard-hitting sentences summarizing the day's performance. Was it a success, a fee-drain, a technical failure, or a masterclass in capital preservation?]

## 2. Fee & Execution Analysis
- **Gross vs Net:** [Brief analysis of how much of the profit was eaten by Binance fees].
- **Slippage & Duration:** [Are we holding trades long enough to justify the fees? Are we getting chopped out too fast?]

## 3. CIO Strategy Audit
[Analyze the specific "cio_rationale_at_entry" from the ledger. Did the CIO's thesis play out? Identify any cognitive biases or repeating errors in the CIO's logic based on today's trades. If no trades, analyze the decision to hold.]

## 4. Technical & Risk Alerts
[Highlight any API errors, margin issues, Sentinel alarms triggered, or stop-losses that were too tight/bypassed. Write "🟢 All systems nominal" if none.]

## 5. Actionable Directives for the CEO
[Provide 1 to 3 concrete recommendations. E.g., "Increase minimum SL distance by 0.5% to avoid chop", "Pause trading on SOL due to erratic slippage", "No changes needed, strategy is solid".]"""


# ── Collecte de données ───────────────────────────────────────────────────────

def _load_daily_trade_log() -> List[Dict[str, Any]]:
    """Charge le journal glissant 48h des cycles de trading."""
    if not DAILY_TRADE_LOG_PATH.exists():
        return []
    try:
        with DAILY_TRADE_LOG_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        return []


def append_to_daily_trade_log(cycle: Dict[str, Any]) -> None:
    """
    Ajoute un cycle terminé au journal glissant (daily_trade_log.json).
    Purge automatiquement les entrées de plus de DAILY_TRADE_LOG_WINDOW_HOURS.
    Écriture atomique : .tmp → rename.
    Appelé depuis dash_app.py après chaque cycle terminé.
    """
    try:
        entries = _load_daily_trade_log()
        entries.append(cycle)

        # Purge des entrées trop anciennes — JAMAIS pendant la fenêtre 48h
        # → les données d'hier (UTC) sont toujours disponibles à 00h05 UTC
        cutoff = datetime.now(timezone.utc) - timedelta(hours=DAILY_TRADE_LOG_WINDOW_HOURS)
        filtered: List[Dict[str, Any]] = []
        for e in entries:
            ts_raw = e.get("run_timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_raw)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    filtered.append(e)
            except Exception:  # noqa: BLE001
                filtered.append(e)  # Sans timestamp lisible : on garde

        temp = DAILY_TRADE_LOG_PATH.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as fh:
            json.dump(filtered, fh, indent=2, ensure_ascii=False, default=str)
        temp.replace(DAILY_TRADE_LOG_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"[AUDITEUR] Erreur écriture daily_trade_log : {exc}")


def _load_cycle_history() -> List[Dict[str, Any]]:
    """Charge les 10 derniers cycles (cycle_history.json) — usage interne uniquement."""
    if not CYCLE_HISTORY_PATH.exists():
        return []
    try:
        with CYCLE_HISTORY_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        return []


def _load_bot_state() -> Dict[str, Any]:
    """Charge le bot_state.json."""
    if not STATE_PATH.exists():
        return {}
    try:
        with STATE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def _load_sentinelle_daily_log() -> List[Dict[str, Any]]:
    """Charge le log quotidien des alarmes Sentinelle."""
    if not SENTINELLE_DAILY_LOG_PATH.exists():
        return []
    try:
        with SENTINELLE_DAILY_LOG_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        return []


def _get_yesterday_range() -> Tuple[datetime, datetime]:
    """Retourne (début_hier_UTC, fin_hier_UTC)."""
    now = datetime.now(timezone.utc)
    # Hier 00:00:00 UTC → 23:59:59 UTC
    yesterday_start = (now - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    yesterday_end = yesterday_start.replace(hour=23, minute=59, second=59)
    return yesterday_start, yesterday_end


def _filter_cycles_for_yesterday(
    cycles: List[Dict[str, Any]],
    start: datetime,
    end: datetime,
) -> List[Dict[str, Any]]:
    """Filtre les cycles appartenant à la journée d'hier (UTC)."""
    result = []
    for c in cycles:
        ts_raw = c.get("run_timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if start <= ts <= end:
                result.append(c)
        except Exception:  # noqa: BLE001
            continue
    return result


def _fetch_daily_ohlcv(symbol: str, exchange) -> Optional[Dict[str, float]]:
    """
    Récupère l'OHLCV quotidien d'hier pour un symbole via l'exchange.
    Retourne dict avec open, close, change_percent ou None si erreur.
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1d", limit=2)
        if not ohlcv or len(ohlcv) < 1:
            return None
        # La barre d'hier est la penultième (index -2) si disponible, sinon -1
        bar = ohlcv[-2] if len(ohlcv) >= 2 else ohlcv[-1]
        open_price = float(bar[1])
        close_price = float(bar[4])
        change_pct = ((close_price - open_price) / open_price * 100) if open_price else 0.0
        return {
            "open": round(open_price, 4),
            "close": round(close_price, 4),
            "change_percent": round(change_pct, 2),
        }
    except Exception:  # noqa: BLE001
        return None


def _compute_trade_metrics(
    cycles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calcule les métriques agrégées de la journée à partir des cycles :
    - balance_start / balance_end
    - net_pnl_usdt / net_pnl_percent
    - total_trades (nombre de signaux non-hold/do_nothing)
    - wins / losses → win_rate
    - total_fees (estimés si dispo dans logs)
    - closed_trades (liste formatée)
    - errors (messages d'erreur extraits des logs)
    """
    if not cycles:
        return {
            "balance_start": "N/A",
            "balance_end": "N/A",
            "net_pnl_usdt": "N/A",
            "net_pnl_percent": "N/A",
            "total_trades": 0,
            "win_rate_percent": "N/A",
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "total_fees_paid": "N/A",
            "avg_slippage_percent": "N/A",
            "closed_trades": [],
            "errors": [],
        }

    # Balance start = premier cycle du jour, balance end = dernier
    balance_start_raw = (cycles[0].get("balances_before") or {}).get("total_balance")
    balance_end_raw = (cycles[-1].get("balances_after") or {}).get("total_balance")

    balance_start = round(float(balance_start_raw), 2) if balance_start_raw is not None else "N/A"
    balance_end = round(float(balance_end_raw), 2) if balance_end_raw is not None else "N/A"

    if isinstance(balance_start, float) and isinstance(balance_end, float):
        net_pnl = round(balance_end - balance_start, 2)
        net_pnl_pct = round((net_pnl / balance_start * 100) if balance_start else 0.0, 2)
    else:
        net_pnl = "N/A"
        net_pnl_pct = "N/A"

    # Trades et win/loss
    trade_entries: List[Dict[str, Any]] = []
    errors: List[str] = []
    trade_id_counter = 1

    for cycle in cycles:
        decisions = cycle.get("decisions") or {}
        ts_raw = cycle.get("run_timestamp", "")
        try:
            cycle_ts = datetime.fromisoformat(ts_raw)
            if cycle_ts.tzinfo is None:
                cycle_ts = cycle_ts.replace(tzinfo=timezone.utc)
            cycle_time_str = cycle_ts.strftime("%H:%M UTC")
        except Exception:  # noqa: BLE001
            cycle_time_str = ts_raw

        for coin, payload in decisions.items():
            args = payload.get("trade_signal_args") or {}
            signal = args.get("signal", "")
            if signal in {"hold", "do_nothing", ""}:
                continue

            gross_pnl = None
            net_pnl_trade = None

            # Tenter d'extraire le PnL du cycle si dispo dans balances
            if signal == "close_position":
                bal_before = (cycle.get("balances_before") or {}).get("total_balance")
                bal_after = (cycle.get("balances_after") or {}).get("total_balance")
                if bal_before is not None and bal_after is not None:
                    gross_pnl = round(float(bal_after) - float(bal_before), 2)
                    net_pnl_trade = gross_pnl  # fees non séparés

            direction = "LONG" if signal == "buy_to_enter" else (
                "SHORT" if signal == "sell_to_enter" else signal.upper()
            )
            leverage = args.get("leverage", "N/A")
            exit_plan = args.get("exit_plan") or {}

            trade_entries.append({
                "trade_id": str(trade_id_counter),
                "asset": f"{coin.upper()}USDT",
                "direction": direction,
                "leverage_used": f"{leverage}x" if leverage != "N/A" else "N/A",
                "cio_rationale_at_entry": args.get("justification", "N/A")[:120] if args.get("justification") else "N/A",
                "entry_time": cycle_time_str,
                "entry_price": args.get("entry_price", "N/A"),
                "exit_time": "N/A",
                "exit_price": exit_plan.get("profit_target", "N/A"),
                "exit_reason": "OPEN_OR_PENDING",
                "duration_minutes": "N/A",
                "gross_pnl_usdt": gross_pnl if gross_pnl is not None else "N/A",
                "fees_usdt": "N/A",
                "net_pnl_usdt": net_pnl_trade if net_pnl_trade is not None else "N/A",
            })
            trade_id_counter += 1

        # Extraire les erreurs des logs
        for log_entry in (cycle.get("logs") or []):
            title = log_entry.get("title", "")
            if title.upper() in {"ERROR", "WARNING"}:
                content = log_entry.get("content", "").strip()
                if content and content not in errors:
                    errors.append(f"[{title}] {content[:200]}")

    total_trades = len(trade_entries)
    closed = [t for t in trade_entries if isinstance(t.get("gross_pnl_usdt"), float)]
    wins = [t for t in closed if t["gross_pnl_usdt"] > 0]
    losses = [t for t in closed if t["gross_pnl_usdt"] <= 0]
    win_rate = round(len(wins) / len(closed) * 100, 1) if closed else "N/A"
    gross_profit = round(sum(t["gross_pnl_usdt"] for t in wins), 2)
    gross_loss = round(sum(abs(t["gross_pnl_usdt"]) for t in losses), 2)

    return {
        "balance_start": balance_start,
        "balance_end": balance_end,
        "net_pnl_usdt": net_pnl,
        "net_pnl_percent": net_pnl_pct,
        "total_trades": total_trades,
        "win_rate_percent": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "total_fees_paid": "N/A (non tracées séparément)",
        "avg_slippage_percent": "N/A",
        "closed_trades": trade_entries,
        "errors": errors,
    }


def _format_open_positions(bot_state: Dict[str, Any], live_positions: Optional[Dict] = None) -> str:
    """
    Formate les positions ouvertes depuis bot_state.json.
    live_positions peut compléter avec le PnL flottant actuel.
    """
    positions = bot_state.get("positions") or {}
    if not positions:
        return "None"

    lines = []
    for coin, pos in positions.items():
        side_raw = pos.get("side", "")
        direction = "LONG" if side_raw == "long" else "SHORT" if side_raw == "short" else side_raw.upper()
        entry = pos.get("entry_price", "N/A")
        leverage = pos.get("leverage", "N/A")

        # PnL flottant depuis live_positions si disponible
        unrealized = "N/A"
        if live_positions and coin in live_positions:
            raw_pnl = live_positions[coin].get("unrealized_pnl")
            if raw_pnl is not None:
                unrealized = f"{round(float(raw_pnl), 2)} USDT"

        lines.append(
            f"Asset: {coin}USDT | Direction: {direction} | Entry: {entry} | "
            f"Leverage: {leverage}x | Current Floating P&L: {unrealized}"
        )
    return "\n".join(lines)


def _format_sentinelle_alarms(alarms: List[Dict[str, Any]], start: datetime, end: datetime) -> Tuple[int, str]:
    """
    Filtre et formate les alarmes Sentinelle déclenchées hier.
    Retourne (count, description_str).
    """
    triggered = []
    for alarm in alarms:
        ts_raw = alarm.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if start <= ts <= end and alarm.get("trigger_recalculation"):
                triggered.append(alarm)
        except Exception:  # noqa: BLE001
            continue

    if not triggered:
        return 0, ""

    lines = []
    for a in triggered:
        ts_raw = a.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_raw)
            time_str = ts.strftime("%H:%M UTC")
        except Exception:  # noqa: BLE001
            time_str = ts_raw
        reason = a.get("reasoning_for_logs", "N/A")[:120]
        lines.append(f"  - {time_str}: {reason}")
    return len(triggered), "\n".join(lines)


# ── Construction du prompt ────────────────────────────────────────────────────

def build_auditeur_prompt() -> str:
    """
    Construit la user prompt complète de l'Auditeur en collectant
    les données réelles du système.
    """
    yesterday_start, yesterday_end = _get_yesterday_range()
    date_label = yesterday_start.strftime("%Y-%m-%d")

    # --- Cycles du jour (depuis le journal 48h) ---
    all_cycles = _load_daily_trade_log()
    yesterday_cycles = _filter_cycles_for_yesterday(all_cycles, yesterday_start, yesterday_end)
    metrics = _compute_trade_metrics(yesterday_cycles)

    # --- BTC/ETH prix ---
    btc_data: Optional[Dict] = None
    eth_data: Optional[Dict] = None
    sol_data: Optional[Dict] = None
    bnb_data: Optional[Dict] = None
    xrp_data: Optional[Dict] = None
    doge_data: Optional[Dict] = None
    live_positions: Optional[Dict] = None

    try:
        exchange = build_exchange()
        btc_data = _fetch_daily_ohlcv("BTC/USDT", exchange)
        eth_data = _fetch_daily_ohlcv("ETH/USDT", exchange)
        sol_data = _fetch_daily_ohlcv("SOL/USDT", exchange)
        bnb_data = _fetch_daily_ohlcv("BNB/USDT", exchange)
        xrp_data = _fetch_daily_ohlcv("XRP/USDT", exchange)
        doge_data = _fetch_daily_ohlcv("DOGE/USDT", exchange)
        # Positions live pour le PnL flottant
        try:
            from prompt_builder import fetch_account_position_map
            live_positions = fetch_account_position_map(exchange)
        except Exception:  # noqa: BLE001
            pass
        try:
            exchange.close()
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass

    # --- Sentinelle alarms ---
    sentinelle_log = _load_sentinelle_daily_log()
    alarm_count, alarm_list = _format_sentinelle_alarms(sentinelle_log, yesterday_start, yesterday_end)

    # --- Positions ouvertes ---
    bot_state = _load_bot_state()
    open_positions_str = _format_open_positions(bot_state, live_positions)

    # --- Format des valeurs BTC/ETH ---
    def _fmt_price(data: Optional[Dict], key: str) -> str:
        return str(data[key]) if data else "N/A"

    btc_open = _fmt_price(btc_data, "open")
    btc_close = _fmt_price(btc_data, "close")
    btc_chg = _fmt_price(btc_data, "change_percent")
    eth_open = _fmt_price(eth_data, "open")
    eth_close = _fmt_price(eth_data, "close")
    eth_chg = _fmt_price(eth_data, "change_percent")
    sol_open = _fmt_price(sol_data, "open")
    sol_close = _fmt_price(sol_data, "close")
    sol_chg = _fmt_price(sol_data, "change_percent")
    bnb_open = _fmt_price(bnb_data, "open")
    bnb_close = _fmt_price(bnb_data, "close")
    bnb_chg = _fmt_price(bnb_data, "change_percent")
    xrp_open = _fmt_price(xrp_data, "open")
    xrp_close = _fmt_price(xrp_data, "close")
    xrp_chg = _fmt_price(xrp_data, "change_percent")
    doge_open = _fmt_price(doge_data, "open")
    doge_close = _fmt_price(doge_data, "close")
    doge_chg = _fmt_price(doge_data, "change_percent")

    # --- Section alarmes ---
    if alarm_count > 0:
        alarm_section = f"{alarm_count}\n{alarm_list}"
    else:
        alarm_section = "0"

    # --- Section erreurs ---
    errors = metrics["errors"]
    if errors:
        errors_section = "\n".join(f"- {e}" for e in errors)
    else:
        errors_section = "- No errors logged."

    # --- Section trades fermés ---
    closed_trades = metrics["closed_trades"]
    if not closed_trades:
        trades_section = "No trades were closed today."
    else:
        trades_section = json.dumps(closed_trades, indent=2, ensure_ascii=False)

    prompt = f"""DAILY TRADING OPERATIONS REPORT
Date: {date_label} (00:00 UTC to 23:59 UTC)
Account Type: LIVE REAL MONEY
Starting Balance: {metrics['balance_start']} USDT
Ending Balance: {metrics['balance_end']} USDT
Net Daily P&L: {metrics['net_pnl_usdt']} USDT ({metrics['net_pnl_percent']}%)

--- SECTION 1: MACRO MARKET CONTEXT ---
(To evaluate if our bot underperformed or outperformed the baseline market)
BTC Daily Open: {btc_open} | BTC Daily Close: {btc_close} | 24h Change: {btc_chg}%
ETH Daily Open: {eth_open} | ETH Daily Close: {eth_close} | 24h Change: {eth_chg}%
SOL Daily Open: {sol_open} | SOL Daily Close: {sol_close} | 24h Change: {sol_chg}%
BNB Daily Open: {bnb_open} | BNB Daily Close: {bnb_close} | 24h Change: {bnb_chg}%
XRP Daily Open: {xrp_open} | XRP Daily Close: {xrp_close} | 24h Change: {xrp_chg}%
DOGE Daily Open: {doge_open} | DOGE Daily Close: {doge_close} | 24h Change: {doge_chg}%
Major CRO (Sentinel) Alarms Triggered Today: {alarm_section}

--- SECTION 2: COST & EFFICIENCY METRICS ---
Total Trades Executed: {metrics['total_trades']}
Win Rate: {metrics['win_rate_percent']}%
Total Gross Profit: {metrics['gross_profit']} USDT
Total Gross Loss: {metrics['gross_loss']} USDT
TOTAL FEES PAID (Binance Taker/Maker): {metrics['total_fees_paid']}
Average Slippage Estimated: {metrics['avg_slippage_percent']}

--- SECTION 3: CLOSED TRADES LEDGER ---
{trades_section}

--- SECTION 4: POSITIONS CURRENTLY OPEN (FLOATING) ---
{open_positions_str}

--- SECTION 5: SYSTEM & API ERRORS ---
{errors_section}

---
TASK: Based on this exact data, execute your System Prompt instructions and provide the Daily Post-Mortem Analysis."""

    return prompt


# ── Persistance ──────────────────────────────────────────────────────────────

def _load_cache() -> List[Dict[str, Any]]:
    """Charge le cache des rapports depuis le disque."""
    if not AUDITEUR_CACHE_PATH.exists():
        return []
    try:
        with AUDITEUR_CACHE_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
    except Exception:  # noqa: BLE001
        pass
    return []


def _save_cache(entries: List[Dict[str, Any]]) -> None:
    """Sauvegarde atomique de la liste des rapports (plus récent en tête)."""
    try:
        temp_path = AUDITEUR_CACHE_PATH.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2, ensure_ascii=False)
        temp_path.replace(AUDITEUR_CACHE_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"[AUDITEUR] Erreur sauvegarde cache : {exc}")


def _prepend_entry(entry: Dict[str, Any]) -> None:
    """Ajoute un rapport en tête du cache, en respectant la limite max."""
    entries = _load_cache()
    entries.insert(0, entry)
    entries = entries[:AUDITEUR_MAX_ENTRIES]
    _save_cache(entries)


# ── Agent principal ───────────────────────────────────────────────────────────

def run_auditeur() -> Dict[str, Any]:
    """
    Construit le prompt avec les données réelles du jour précédent,
    appelle DeepSeek Reasoner, et persiste le rapport dans auditeur_cache.json.
    Retourne le rapport sous forme de dict.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        error = {"error": "DEEPSEEK_API_KEY non définie."}
        print("[AUDITEUR] DEEPSEEK_API_KEY manquante.")
        return error

    now = datetime.now(timezone.utc)
    print(f"\n===== AUDITEUR — {now.strftime('%Y-%m-%d %H:%M:%S UTC')} =====")

    user_prompt = build_auditeur_prompt()

    payload = {
        "model": AUDITEUR_MODEL,
        "messages": [
            {"role": "system", "content": AUDITEUR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": AUDITEUR_MAX_TOKENS,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            DEEPSEEK_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        message = data.get("choices", [{}])[0].get("message", {})
        content: str = message.get("content") or ""
        reasoning: str = message.get("reasoning_content") or ""

        print(f"Rapport reçu ({len(content)} caractères, raisonnement : {len(reasoning)} car.)")

        entry: Dict[str, Any] = {
            "timestamp": now.isoformat(),
            "user_prompt": user_prompt,
            "content": content,
            "reasoning_content": reasoning,
        }
        _prepend_entry(entry)
        return entry

    except requests.RequestException as exc:
        error_entry: Dict[str, Any] = {
            "timestamp": now.isoformat(),
            "error": f"Erreur réseau : {exc}",
            "content": "",
            "reasoning_content": "",
        }
        _prepend_entry(error_entry)
        print(f"[AUDITEUR] Erreur réseau : {exc}")
        return error_entry

    except Exception as exc:  # noqa: BLE001
        error_entry = {
            "timestamp": now.isoformat(),
            "error": f"Erreur inattendue : {exc}",
            "content": "",
            "reasoning_content": "",
        }
        _prepend_entry(error_entry)
        print(f"[AUDITEUR] Erreur inattendue : {exc}")
        return error_entry


# ── Lecture du cache (pour le dashboard) ─────────────────────────────────────

def get_auditeur_history() -> List[Dict[str, Any]]:
    """Retourne la liste des rapports depuis le cache (plus récent en tête)."""
    return _load_cache()


# ── Logging des alarmes Sentinelle (appelé depuis dash_app.py) ────────────────

def log_sentinelle_alarm(decision: Dict[str, Any]) -> None:
    """
    Enregistre une alarme Sentinelle dans sentinelle_daily_log.json si déclenchée.
    Conserve les 500 dernières entrées (50 jours à 10 alarmes max/jour).
    À appeler depuis dash_app.py après chaque décision Sentinelle.
    """
    if not decision.get("trigger_recalculation"):
        return
    try:
        log: List[Dict[str, Any]] = []
        if SENTINELLE_DAILY_LOG_PATH.exists():
            with SENTINELLE_DAILY_LOG_PATH.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, list):
                log = raw

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger_recalculation": True,
            "confidence_in_alarm": decision.get("confidence_in_alarm"),
            "reasoning_for_logs": decision.get("reasoning_for_logs", ""),
            "message_to_macro_strategist": decision.get("message_to_macro_strategist"),
        }
        log.insert(0, entry)
        log = log[:500]

        temp = SENTINELLE_DAILY_LOG_PATH.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as fh:
            json.dump(log, fh, indent=2, ensure_ascii=False)
        temp.replace(SENTINELLE_DAILY_LOG_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"[AUDITEUR] Erreur log Sentinelle : {exc}")
