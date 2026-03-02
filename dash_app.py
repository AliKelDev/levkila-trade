from __future__ import annotations

import copy
import json
import logging
import threading
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State

from prompt_builder import (
    SENTIMENT_CACHE_PATH,
    fetch_and_cache_sentiment,
    get_account_snapshot,
    run_cycle,
    build_exchange,
    build_market_prompt,
    build_account_prompt,
    load_state,
)
from macro_strategist import (
    MACRO_STRATEGY_CACHE_PATH,
    fetch_and_cache_macro_strategy,
    get_cached_macro_strategy,
)
from sentiment_analyst import get_cached_sentiment, get_cached_sentiment_json, fetch_and_cache_sentiment as _fetch_sentiment
from sentinelle import run_sentinelle
from circuit_breaker import (
    is_emergency_active,
    get_emergency_info,
    clear_emergency,
)
from auditeur import run_auditeur, get_auditeur_history, log_sentinelle_alarm, append_to_daily_trade_log


LOGGER = logging.getLogger("deeptrade.dash")
logging.basicConfig(level=logging.INFO)
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)

DEFAULT_HISTORY_LIMIT = 20
CYCLE_HISTORY_LIMIT = 10        # number of cycles persisted to disk
CYCLE_HISTORY_PATH = Path("cycle_history.json")
DEFAULT_SNAPSHOT_REFRESH = 60  # seconds
DEFAULT_LOOP_INTERVAL = 60     # seconds between auto-cycle runs
EMOJI_PATTERN = re.compile(r"[\U00010000-\U0010FFFF]")


def _persist_cycle_history() -> None:
    """Write the last CYCLE_HISTORY_LIMIT cycles to disk for external agent consumption."""
    try:
        history = list(STATE.history)[-CYCLE_HISTORY_LIMIT:]
        temp_path = CYCLE_HISTORY_PATH.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2, ensure_ascii=False, default=str)
        temp_path.replace(CYCLE_HISTORY_PATH)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to persist cycle history: %s", exc)


def _load_cycle_history() -> List[Dict[str, Any]]:
    """Load persisted cycle history from disk at startup."""
    if not CYCLE_HISTORY_PATH.exists():
        return []
    try:
        with CYCLE_HISTORY_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to load cycle history from disk: %s", exc)
    return []

class AppState:
    """Thread-safe container for dashboard state."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        _prior = _load_cycle_history()
        self.history: deque[Dict[str, Any]] = deque(_prior, maxlen=DEFAULT_HISTORY_LIMIT)
        self.last_snapshot: Optional[Dict[str, Any]] = None
        self.last_snapshot_refreshed_at: Optional[str] = None
        self.snapshot_status: str = ""
        self.cycle_status: str = ""
        self.loop_enabled: bool = False
        self.loop_interval: int = DEFAULT_LOOP_INTERVAL
        self.loop_running: bool = False
        self.loop_primed: bool = False
        self.grace_period_end: float = 0.0
        self.snapshot_auto_refresh: bool = True
        self.snapshot_refresh_interval: int = DEFAULT_SNAPSHOT_REFRESH

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "history": list(self.history),
                "last_snapshot": copy.deepcopy(self.last_snapshot),
                "last_snapshot_refreshed_at": self.last_snapshot_refreshed_at,
                "snapshot_status": self.snapshot_status,
                "cycle_status": self.cycle_status,
                "loop_enabled": self.loop_enabled,
                "loop_interval": self.loop_interval,
                "loop_running": self.loop_running,
                "loop_primed": self.loop_primed,
                "grace_period_end": self.grace_period_end,
                "snapshot_auto_refresh": self.snapshot_auto_refresh,
                "snapshot_refresh_interval": self.snapshot_refresh_interval,
            }

    def append_history(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.history.append(payload)


STATE = AppState()
STATE_LOCK = threading.RLock()
SCHEDULER = BackgroundScheduler(job_defaults={"max_instances": 1})
SCHEDULER.start()

MANUAL_CYCLE_THREAD: Optional[threading.Thread] = None
SNAPSHOT_LOCK = threading.Lock()


def _format_local_timestamp(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    local_ts = ts.astimezone()
    return local_ts.strftime("%Y-%m-%d %H:%M:%S %Z")





def _record_result(result) -> None:
    STATE.append_history(asdict(result))
    STATE.update(
        last_snapshot={
            "account_prompt": result.account_prompt_after,
            "positions": result.positions_after,
            "balances": result.balances_after,
        }
    )
    _persist_cycle_history()
    # Feed the 48h rolling log for the Auditeur's daily report
    append_to_daily_trade_log(asdict(result))


def _cycle_worker(source: str) -> None:
    global MANUAL_CYCLE_THREAD
    start_ts = datetime.now(timezone.utc)
    start_message = f"Start {source} trading cycle { _format_local_timestamp(start_ts) }"
    LOGGER.info(start_message)
    STATE.update(cycle_status=f"{source} trading cycle started {_format_local_timestamp(start_ts)}")

    try:
        result = run_cycle()
        _record_result(result)
        STATE.update(loop_primed=True)
        if STATE.loop_enabled:
            STATE.update(grace_period_end=time.time() + max(1, STATE.loop_interval))
            configure_auto_cycle_job()
        finished_ts = datetime.now(timezone.utc)
        STATE.update(
            cycle_status=f"{source} trading cycle completed {_format_local_timestamp(finished_ts)}"
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Error during trading cycle")
        STATE.update(cycle_status=f"{source} trading cycle failed: {exc}")
        # If the cycle triggered emergency mode, halt the auto-loop immediately
        if is_emergency_active():
            STATE.update(loop_enabled=False, loop_primed=False)
            LOGGER.critical(
                "Auto-loop disabled: emergency mode activated during cycle."
            )
    finally:
        STATE.update(loop_running=False)
        if MANUAL_CYCLE_THREAD and threading.current_thread() is MANUAL_CYCLE_THREAD:
            MANUAL_CYCLE_THREAD = None


def _start_cycle(source: str) -> bool:
    global MANUAL_CYCLE_THREAD
    with STATE_LOCK:
        if STATE.loop_running:
            return False
        STATE.update(loop_running=True)
        if source == "Manual" and STATE.loop_enabled:
            STATE.update(grace_period_end=time.time() + max(1, STATE.loop_interval))

    worker = threading.Thread(target=_cycle_worker, args=(source,), daemon=True)
    if source == "Manual":
        MANUAL_CYCLE_THREAD = worker
    worker.start()
    return True


def manual_cycle() -> bool:
    return _start_cycle("Manual")


def auto_cycle_tick() -> None:
    # Refuse to run if emergency mode is active
    if is_emergency_active():
        STATE.update(loop_enabled=False, loop_primed=False)
        return
    snapshot = STATE.snapshot()
    if not snapshot["loop_enabled"] or not snapshot["loop_primed"]:
        return
    if snapshot["loop_running"]:
        return
    if time.time() < snapshot["grace_period_end"]:
        return
    started = _start_cycle("Auto")
    if started:
        LOGGER.info("Auto cycle triggered")


def _refresh_snapshot(source: str) -> None:
    if not SNAPSHOT_LOCK.acquire(blocking=False):
        return

    try:
        start_ts = datetime.now(timezone.utc)
        STATE.update(snapshot_status=f"{source} snapshot refresh started {_format_local_timestamp(start_ts)}")
        snapshot = get_account_snapshot()
        finished_ts = datetime.now(timezone.utc)
        STATE.update(
            last_snapshot=snapshot,
            last_snapshot_refreshed_at=finished_ts.isoformat(),
            snapshot_status=f"{source} snapshot refreshed {_format_local_timestamp(finished_ts)}",
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Snapshot refresh failed")
        STATE.update(snapshot_status=f"{source} snapshot refresh failed: {exc}")
    finally:
        SNAPSHOT_LOCK.release()


def manual_snapshot_refresh() -> bool:
    if not SNAPSHOT_LOCK.acquire(blocking=False):
        return False

    def worker() -> None:
        try:
            start_ts = datetime.now(timezone.utc)
            STATE.update(snapshot_status=f"Manual snapshot refresh started {_format_local_timestamp(start_ts)}")
            snapshot = get_account_snapshot()
            finished_ts = datetime.now(timezone.utc)
            STATE.update(
                last_snapshot=snapshot,
                last_snapshot_refreshed_at=finished_ts.isoformat(),
                snapshot_status=f"Manual snapshot refreshed {_format_local_timestamp(finished_ts)}",
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Manual snapshot refresh failed")
            STATE.update(snapshot_status=f"Manual snapshot refresh failed: {exc}")
        finally:
            SNAPSHOT_LOCK.release()

    threading.Thread(target=worker, daemon=True).start()
    return True


def auto_snapshot_job() -> None:
    _refresh_snapshot("Auto")


def configure_auto_cycle_job() -> None:
    snapshot = STATE.snapshot()
    job = SCHEDULER.get_job("auto_cycle_job")

    if snapshot["loop_enabled"] and snapshot["loop_primed"]:
        interval = snapshot["loop_interval"] if snapshot["loop_interval"] > 0 else 1
        trigger = IntervalTrigger(seconds=interval)
        if not job:
            SCHEDULER.add_job(
                auto_cycle_tick,
                trigger=trigger,
                id="auto_cycle_job",
                replace_existing=True,
                next_run_time=datetime.now(timezone.utc) + timedelta(seconds=interval),
            )
        else:
            current = job.trigger.interval.total_seconds()
            if current != interval:
                SCHEDULER.reschedule_job(
                    "auto_cycle_job",
                    trigger=trigger,
                    next_run_time=datetime.now(timezone.utc) + timedelta(seconds=interval),
                )
    else:
        if job:
            SCHEDULER.remove_job("auto_cycle_job")


def configure_snapshot_job() -> None:
    snapshot = STATE.snapshot()
    job = SCHEDULER.get_job("snapshot_refresh_job")

    if snapshot["snapshot_auto_refresh"]:
        interval = max(15, int(snapshot["snapshot_refresh_interval"]))
        trigger = IntervalTrigger(seconds=interval)
        if not job:
            SCHEDULER.add_job(
                auto_snapshot_job,
                trigger=trigger,
                id="snapshot_refresh_job",
                replace_existing=True,
                next_run_time=datetime.now(timezone.utc) + timedelta(seconds=interval),
            )
        else:
            current = job.trigger.interval.total_seconds()
            if current != interval:
                SCHEDULER.reschedule_job(
                    "snapshot_refresh_job",
                    trigger=trigger,
                    next_run_time=datetime.now(timezone.utc) + timedelta(seconds=interval),
                )
    else:
        if job:
            SCHEDULER.remove_job("snapshot_refresh_job")


def sentiment_then_sentinelle_job() -> None:
    """
    Wrapper scheduled every 5 minutes:
    1. Refresh the sentiment cache (Gemini + Google Search).
    2. Run La Sentinelle (CRO audit).
    3. If the Sentinelle triggers a recalculation, immediately run the macro strategist.
    """
    # Step 1 — refresh sentiment
    _fetch_sentiment()

    # Step 2 — audit with La Sentinelle
    try:
        exchange = build_exchange()
        state = load_state()
        try:
            market_data = build_market_prompt(exchange)
            account_prompt, _, _ = build_account_prompt(exchange, state)
        finally:
            try:
                exchange.close()
            except Exception:  # noqa: BLE001
                pass

        sentiment = get_cached_sentiment()
        macro_strategy = get_cached_macro_strategy()

        # Check if the Sentiment Analyst already flagged a black swan — add it as context
        sentiment_json = get_cached_sentiment_json()
        black_swan_pre_flag = ""
        if sentiment_json:
            bsa = sentiment_json.get("black_swan_alert", {})
            if bsa.get("triggered"):
                urgency = bsa.get("urgency_message", "Black swan detected by Sentiment Analyst.")
                black_swan_pre_flag = f"⚠️ PRE-ALERT FROM SENTIMENT ANALYST: {urgency}"
                LOGGER.warning("Sentiment Analyst flagged a black swan: %s", urgency)

        sentinelle_market = (black_swan_pre_flag + "\n\n" + market_data) if black_swan_pre_flag else market_data

        decision = run_sentinelle(
            market_data=sentinelle_market,
            sentiment=sentiment,
            macro_strategy=macro_strategy,
        )

        # Log the alarm for the Auditeur's daily report
        log_sentinelle_alarm(decision)

        # Step 3 — trigger emergency macro recalculation if needed
        if decision.get("trigger_recalculation"):
            LOGGER.warning(
                "Sentinelle triggered emergency macro recalculation! "
                "Confidence: %s | Reason: %s",
                decision.get("confidence_in_alarm"),
                decision.get("reasoning_for_logs"),
            )
            # Pass the Sentinelle's emergency message to the macro strategist
            emergency_note = decision.get("message_to_macro_strategist") or ""
            fetch_and_cache_macro_strategy(
                market_data,
                account_prompt,
                sentiment,
                emergency_note=emergency_note,
            )
            # Reset the hourly scheduler: next normal run = 1h from NOW
            # so the CIO gets a full fresh hour after the emergency recalculation.
            macro_job = SCHEDULER.get_job("macro_strategy_job")
            if macro_job:
                SCHEDULER.reschedule_job(
                    "macro_strategy_job",
                    trigger=IntervalTrigger(hours=1),
                    next_run_time=datetime.now(timezone.utc) + timedelta(hours=1),
                )
                LOGGER.info("Macro strategy job rescheduled: next run in 1h from emergency trigger.")

    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Sentinelle job failed: %s", exc)


def configure_sentiment_job() -> None:
    job_id = "gemini_sentiment_job"
    if not SCHEDULER.get_job(job_id):
        SCHEDULER.add_job(
            sentiment_then_sentinelle_job,
            "interval",
            minutes=5,
            id=job_id,
            next_run_time=datetime.now(timezone.utc),
        )
        LOGGER.info("Sentiment+Sentinelle job scheduled (every 5m).")


def macro_strategy_job() -> None:
    """Wrapper executed by the scheduler: builds live market data and calls the macro strategist."""
    try:
        exchange = build_exchange()
        state = load_state()
        try:
            market_data = build_market_prompt(exchange)
            account_prompt, _, _ = build_account_prompt(exchange, state)
        finally:
            try:
                exchange.close()
            except Exception:  # noqa: BLE001
                pass
        sentiment = get_cached_sentiment()
        fetch_and_cache_macro_strategy(market_data, account_prompt, sentiment)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Macro strategy job failed: %s", exc)


def configure_macro_strategy_job() -> None:
    job_id = "macro_strategy_job"
    if not SCHEDULER.get_job(job_id):
        SCHEDULER.add_job(
            macro_strategy_job,
            "interval",
            hours=1,
            id=job_id,
            next_run_time=datetime.now(timezone.utc),
        )
        LOGGER.info("Macro strategy job scheduled (every 1h).")


def auditeur_job() -> None:
    """Wrapper scheduled daily at 01:05 Europe/Paris: runs the Auditeur agent."""
    try:
        entry = run_auditeur()
        if entry.get("error"):
            LOGGER.error("Auditeur job returned an error: %s", entry["error"])
        else:
            LOGGER.info("Auditeur job completed successfully.")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Auditeur job failed: %s", exc)


def configure_auditeur_job() -> None:
    job_id = "auditeur_job"
    if not SCHEDULER.get_job(job_id):
        SCHEDULER.add_job(
            auditeur_job,
            CronTrigger(hour=1, minute=5, timezone="Europe/Paris"),
            id=job_id,
        )
        LOGGER.info("Auditeur job scheduled (daily at 01:05 Europe/Paris).")


configure_auto_cycle_job()
configure_snapshot_job()
configure_sentiment_job()
configure_macro_strategy_job()
configure_auditeur_job()


app = Dash(__name__)
app.title = "DeepTrade Control Panel"
server = app.server


def _build_balances_view(snapshot: Dict[str, Any]) -> List[Any]:
    balances = (snapshot.get("last_snapshot") or {}).get("balances", {})
    if not balances:
        return [html.Div("No balances available", className="muted")]

    rows = []
    for label, value in balances.items():
        rows.append(
            html.Div([
                html.Span(_strip_emoji(label.replace("_", " ").title()), className="metric-label"),
                html.Span(f"{value:,.2f}", className="metric-value"),
            ], className="metric-row")
        )
    return rows


def _positions_table(snapshot: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    positions = (snapshot.get("last_snapshot") or {}).get("positions", {})
    if not positions:
        return [], []

    rows: List[Dict[str, Any]] = []
    for coin, payload in positions.items():
        row = {"coin": coin}
        row.update(payload)
        for key, value in list(row.items()):
            if isinstance(value, (dict, list)):
                try:
                    row[key] = json.dumps(value, indent=2, ensure_ascii=False)
                except TypeError:
                    row[key] = str(value)
            if isinstance(row[key], str):
                row[key] = _strip_emoji(row[key])
        rows.append(row)

    columns = [{"name": key.replace("_", " ").title(), "id": key} for key in rows[0].keys()]
    return rows, columns


def _equity_figure(history: List[Dict[str, Any]]) -> go.Figure:
    line_color = "#f4f1ea"
    marker_color = "#f7f4ed"
    fill_color = "rgba(247, 244, 237, 0.12)"
    grid_color = "rgba(255, 255, 255, 0.1)"
    font_color = "#f5f2eb"
    if not history:
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=40, r=20, t=30, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        fig.add_annotation(text="No history yet", showarrow=False, x=0.5, y=0.5)
        return fig

    rows = []
    for entry in history:
        ts = entry.get("run_timestamp")
        balances = entry.get("balances_after") or {}
        total = balances.get("total_balance")
        if ts and total is not None:
            try:
                rows.append({
                    "timestamp": pd.to_datetime(ts),
                    "total_balance": float(total),
                })
            except Exception:  # noqa: BLE001
                continue

    if not rows:
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=40, r=20, t=30, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        fig.add_annotation(text="No history yet", showarrow=False, x=0.5, y=0.5)
        return fig

    df = pd.DataFrame(rows).sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["total_balance"],
            mode="lines+markers",
            name="Total Balance",
            line=dict(color=line_color, width=3),
            marker=dict(color=marker_color, size=6),
            fill='tozeroy',
            fillcolor=fill_color,
        )
    )
    fig.update_layout(
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Time",
        yaxis_title="Total Balance ($)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=font_color),
        xaxis=dict(
            gridcolor=grid_color,
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor=grid_color,
            showgrid=True,
        ),
        hovermode='x unified',
    )
    return fig


def _build_trade_feed(history: List[Dict[str, Any]]) -> List[Any]:
    if not history:
        return [html.Div("No runs recorded yet.", className="muted")]

    feed: List[Any] = []
    for idx, entry in enumerate(reversed(history)):
        latest = idx == 0
        timestamp = entry.get("run_timestamp")
        readable = "Unknown time"
        if timestamp:
            try:
                readable = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                readable = timestamp

        invocation = entry.get("invocation_count", "?")
        minutes = entry.get("minutes_since_start", "?")

        summary = _strip_emoji(f"Run #{invocation} — {readable}")
        if latest:
            summary += " • latest"

        feed.append(
            html.Details([
                html.Summary(summary),
                html.P(_strip_emoji(f"{minutes} minutes since start")),
                html.H4("User Prompt"),
                html.Pre(_strip_emoji(entry.get("user_prompt") or ""), className="code-block"),
                html.H4("System Prompt"),
                html.Pre(_strip_emoji(entry.get("system_prompt") or ""), className="code-block"),
                html.H4("LLM Decisions"),
                html.Pre(_strip_emoji(_format_dict(entry.get("decisions"))), className="code-block"),
                html.H4("Final Content"),
                html.Pre(_strip_emoji(entry.get("final_content") or ""), className="code-block"),
                html.H4("Logs"),
                html.Pre(_format_logs(entry.get("logs")), className="code-block"),
            ], open=latest)
        )

    return feed


def _format_dict(payload: Optional[Dict[str, Any]]) -> str:
    if payload is None:
        return "{}"
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(payload)


def _format_logs(logs: Optional[List[Dict[str, Any]]]) -> str:
    if not logs:
        return "No logs."
    lines = []
    for entry in logs:
        title = _strip_emoji(entry.get("title", "LOG"))
        content = _strip_emoji(entry.get("content", ""))
        lines.append(f"[{title}]\n{content}\n")
    return "\n".join(lines)


def _strip_emoji(value: Optional[Any]) -> str:
    if value is None:
        return ""
    return EMOJI_PATTERN.sub("", str(value))


def _build_ai_explanation(history: List[Dict[str, Any]]) -> List[Any]:
    """Build a clean AI reasoning view from the latest cycle."""
    if not history:
        return [html.Div("No cycles run yet. Start a trading cycle to see the AI reasoning.", className="muted")]

    entry = history[-1]  # Most recent cycle
    timestamp = entry.get("run_timestamp", "")
    readable = "Unknown time"
    if timestamp:
        try:
            readable = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            readable = timestamp

    decisions = entry.get("decisions") or {}
    chain = _strip_emoji(entry.get("chain_of_thought") or "")

    # --- Decision cards per coin ---
    decision_cards = []
    signal_colors = {
        "buy_to_enter": "#22c55e",
        "sell_to_enter": "#fb7185",
        "close_position": "#facc15",
        "hold": "rgba(255,255,255,0.4)",
        "do_nothing": "rgba(255,255,255,0.2)",
    }
    for coin, payload in decisions.items():
        args = payload.get("trade_signal_args", {})
        signal = args.get("signal", "unknown")
        color = signal_colors.get(signal, "rgba(255,255,255,0.3)")
        confidence = args.get("confidence")
        leverage = args.get("leverage")
        risk_pct = args.get("risk_percentage")
        justification = _strip_emoji(args.get("justification") or "")
        exit_plan = args.get("exit_plan") or {}

        meta_items = []
        if confidence is not None:
            meta_items.append(f"Confidence: {float(confidence):.0%}")
        if leverage is not None:
            meta_items.append(f"Leverage: {leverage}×")
        if risk_pct is not None:
            meta_items.append(f"Risk: {risk_pct}%")
        if exit_plan.get("profit_target"):
            meta_items.append(f"TP: {exit_plan['profit_target']}")
        if exit_plan.get("stop_loss"):
            meta_items.append(f"SL: {exit_plan['stop_loss']}")

        card_children = [
            html.Div([
                html.Span(coin.upper(), style={"fontWeight": "700", "fontSize": "1.05rem"}),
                html.Span(
                    signal.replace("_", " ").upper(),
                    style={
                        "color": color,
                        "fontWeight": "600",
                        "fontSize": "0.8rem",
                        "letterSpacing": "0.12em",
                        "marginLeft": "auto",
                        "padding": "0.25rem 0.75rem",
                        "borderRadius": "999px",
                        "border": f"1px solid {color}",
                    }
                ),
            ], style={"display": "flex", "alignItems": "center", "gap": "1rem", "marginBottom": "0.75rem"}),
        ]
        if meta_items:
            card_children.append(
                html.Div(" · ".join(meta_items), style={"fontSize": "0.8rem", "color": "rgba(255,255,255,0.55)", "marginBottom": "0.5rem", "letterSpacing": "0.05em"})
            )
        if justification:
            card_children.append(
                html.P(justification, style={"fontSize": "0.9rem", "color": "rgba(255,255,255,0.75)", "margin": "0.5rem 0 0"})
            )
        if exit_plan.get("invalidation_condition"):
            card_children.append(
                html.P(
                    [html.Strong("Invalidation: "), _strip_emoji(exit_plan["invalidation_condition"])],
                    style={"fontSize": "0.85rem", "color": "rgba(251,113,133,0.9)", "marginTop": "0.5rem"}
                )
            )

        decision_cards.append(
            html.Div(card_children, style={
                "background": "rgba(255,255,255,0.03)",
                "border": "1px solid rgba(255,255,255,0.08)",
                "borderRadius": "16px",
                "padding": "1.25rem 1.5rem",
            })
        )

    if not decision_cards:
        decision_cards = [html.Div("No decisions in this cycle.", className="muted")]

    sections = [
        html.Div([
            html.Span(f"Latest cycle — {readable}", style={"fontWeight": "600", "fontSize": "1rem"}),
            html.Span(f"Cycle #{entry.get('invocation_count', '?')} · {entry.get('minutes_since_start', '?')}m since start",
                      style={"fontSize": "0.8rem", "color": "rgba(255,255,255,0.45)", "marginLeft": "1rem"}),
        ], style={"marginBottom": "1.5rem", "paddingBottom": "1rem", "borderBottom": "1px solid rgba(255,255,255,0.08)"}),

        html.H3("Decisions", style={"marginBottom": "1rem"}),
        html.Div(decision_cards, style={"display": "grid", "gap": "1rem",
                                       "gridTemplateColumns": "repeat(auto-fill, minmax(300px, 1fr))",
                                       "marginBottom": "2rem"}),
    ]

    if chain:
        sections += [
            html.H3("Chain of Thought", style={"marginBottom": "0.75rem"}),
            html.Pre(chain, className="code-block",
                     style={"maxHeight": "500px", "overflowY": "auto", "fontSize": "0.82rem", "lineHeight": "1.7"}),
        ]

    return sections


app.layout = html.Div(
    id="app-shell",
    className="app-shell theme-dark",
    children=[
        html.Div([
            html.Div([
                html.H1("DeepTrade Control Panel"),
            ], className="header-bar"),
            # ── Emergency Banner (hidden when no emergency) ───────────────
            html.Div(
                id="emergency-banner",
                style={"display": "none"},
                className="emergency-banner",
                children=[
                    html.Span("⛔ EMERGENCY MODE ACTIVE — Bot halted. All positions closed."),
                    html.Span(id="emergency-reason", style={"marginLeft": "1rem", "fontStyle": "italic"}),
                    html.Button(
                        "✅ Reset & Resume",
                        id="clear-emergency-btn",
                        n_clicks=0,
                        className="action-btn",
                        style={"marginLeft": "2rem", "background": "#fff", "color": "#b00020"},
                    ),
                ],
            ),
            html.Div(id="clear-emergency-feedback", className="feedback"),
            # ─────────────────────────────────────────────────────────────
            dcc.Interval(id="refresh-interval", interval=4000, n_intervals=0),
            dcc.Tabs(
                id="tabs",
                value="dashboard",
                className="deeptrade-tabs-nav",
                parent_className="deeptrade-tabs",
                children=[
                    dcc.Tab(
                        label="Dashboard",
                        value="dashboard",
                        className="deeptrade-tab",
                        selected_className="deeptrade-tab--active",
                        children=[
                            html.Div([
                                html.Div([
                                    html.H2("Control Center"),
                                    html.Div([
                                        html.Div(id="snapshot-status", className="status-pill"),
                                        html.Div(id="cycle-status", className="status-pill"),
                                    ], className="status-cluster"),
                                    html.Div([
                                        html.Button("Start Trading Cycle", id="run-cycle-btn", n_clicks=0, className="action-btn action-btn--primary"),
                                        html.Button("Refresh Snapshot", id="refresh-snapshot-btn", n_clicks=0, className="action-btn"),
                                    ], className="flex-gap-1", style={"marginTop": "1.5rem", "flexWrap": "wrap"}),
                                    html.Div(id="manual-cycle-feedback", className="feedback"),
                                    html.Div(id="snapshot-feedback", className="feedback"),
                                    html.Div(id="cycle-hint", className="muted"),
                                    html.Div(id="next-snapshot-info", className="muted"),
                                ], className="panel panel--control panel--wide"),

                                html.Div([
                                    html.H2("Account Overview"),
                                    html.Div(id="balances-container", className="balances"),
                                ], className="panel panel--stats"),

                                html.Div([
                                    html.H3("Open Positions"),
                                    dash_table.DataTable(
                                        id="positions-table",
                                        data=[],
                                        columns=[],
                                        style_table={"overflowX": "auto"},
                                        style_cell={"textAlign": "left", "whiteSpace": "pre-line"},
                                    ),
                                ], id="positions-container", className="panel panel--table panel--wide"),

                                html.Div([
                                    html.H3("Account Prompt"),
                                    html.Pre(id="account-prompt", className="code-block"),
                                ], className="panel panel--code"),

                                html.Div([
                                    html.H3("Equity Curve"),
                                    dcc.Graph(id="equity-curve"),
                                    html.Div(id="history-empty", className="muted"),
                                ], className="panel panel--chart panel--wide"),
                                html.Div([
                                    html.H3("Market Intelligence"),
                                    html.Div(
                                        id="sentiment-display",
                                        className="markdown-body",
                                        style={"fontSize": "0.9rem"},
                                    ),
                                    html.Div(
                                        id="sentiment-timestamp",
                                        className="muted",
                                        style={"marginTop": "0.5rem"},
                                    ),
                                ], className="panel panel--wide"),
                            ], className="dashboard-grid"),
                        ],
                    ),
                    dcc.Tab(
                        label="AI Explanation",
                        value="ai-explanation",
                        className="deeptrade-tab",
                        selected_className="deeptrade-tab--active",
                        children=[
                            html.Div(
                                id="ai-explanation-content",
                                className="panel panel--wide",
                                children=[html.Div("Loading...", className="muted")],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Auditeur",
                        value="auditeur",
                        className="deeptrade-tab",
                        selected_className="deeptrade-tab--active",
                        children=[
                            html.Div(
                                id="auditeur-content",
                                className="panel panel--wide",
                                children=[html.Div("Chargement...", className="muted")],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Trade History",
                        value="trades",
                        className="deeptrade-tab",
                        selected_className="deeptrade-tab--active",
                        children=[
                            html.Div([
                                html.H2("Trade Execution History"),
                                html.Div(id="trades-feed", className="trades-feed"),
                            ], className="panel panel--wide"),
                        ],
                    ),
                    dcc.Tab(
                        label="Settings",
                        value="settings",
                        className="deeptrade-tab",
                        selected_className="deeptrade-tab--active",
                        children=[
                            html.Div([
                                html.Div([
                                    html.H2("Auto Loop Configuration"),
                                    dcc.Checklist(
                                        id="auto-loop-toggle",
                                        options=[{"label": "Enable automated trading cycles", "value": "enabled"}],
                                        value=[],
                                        className="checkbox-row",
                                    ),
                                    dcc.Input(
                                        id="loop-interval-input",
                                        type="number",
                                        min=10,
                                        max=3600,
                                        step=1,
                                        placeholder=f"Loop interval (seconds, default {DEFAULT_LOOP_INTERVAL})",
                                        className="input-field",
                                    ),
                                    html.Div(id="grace-period-info", className="muted"),
                                ], className="panel section-block"),

                                html.Div([
                                    html.H2("Snapshot Refresh Settings"),
                                    dcc.Checklist(
                                        id="snapshot-auto-toggle",
                                        options=[{"label": "Auto refresh account snapshot", "value": "enabled"}],
                                        value=["enabled"],
                                        className="checkbox-row",
                                    ),
                                    dcc.Input(
                                        id="snapshot-interval-input",
                                        type="number",
                                        min=15,
                                        max=900,
                                        step=15,
                                        placeholder="Snapshot interval (seconds)",
                                        className="input-field",
                                    ),
                                    html.Div(id="last-snapshot-info", className="muted"),
                                ], className="panel section-block"),

                                html.Button("Save Settings", id="save-settings-btn", n_clicks=0, className="action-btn action-btn--primary"),
                                html.Div(id="settings-feedback", className="feedback"),
                            ], className="settings-stack"),
                        ],
                    ),
                ],
            ),
        ], className="app-content", style={"minHeight": "100vh"}),
    ]
)


@app.callback(
    Output("emergency-banner", "style"),
    Output("emergency-reason", "children"),
    Input("refresh-interval", "n_intervals"),
)
def update_emergency_banner(n_intervals):
    """Show / hide the emergency banner based on circuit-breaker state."""
    if is_emergency_active():
        info = get_emergency_info() or {}
        reason = info.get("reason", "Unknown reason")
        triggered_at = info.get("triggered_at", "")
        return (
            {"display": "flex", "alignItems": "center", "padding": "1rem",
             "background": "#b00020", "color": "#fff", "fontWeight": "bold",
             "borderRadius": "8px", "margin": "1rem"},
            f"{reason} (at {triggered_at})",
        )
    return {"display": "none"}, ""


@app.callback(
    Output("clear-emergency-feedback", "children"),
    Input("clear-emergency-btn", "n_clicks"),
    prevent_initial_call=True,
)
def handle_clear_emergency(n_clicks: int) -> str:
    """Clear emergency mode — allows the operator to resume trading."""
    clear_emergency()
    return "✅ Emergency cleared. You can restart trading manually."


@app.callback(
    Output("manual-cycle-feedback", "children"),
    Input("run-cycle-btn", "n_clicks"),
    prevent_initial_call=True,
)
def handle_manual_cycle(n_clicks: int) -> str:
    started = manual_cycle()
    if started:
        return "Manual trading cycle queued."
    return "Trading cycle already running."


@app.callback(
    Output("snapshot-feedback", "children"),
    Input("refresh-snapshot-btn", "n_clicks"),
    prevent_initial_call=True,
)
def handle_manual_snapshot(n_clicks: int) -> str:
    success = manual_snapshot_refresh()
    if success:
        return "Manual snapshot refresh requested."
    return "Snapshot refresh already in progress."


@app.callback(
    Output("settings-feedback", "children"),
    Input("save-settings-btn", "n_clicks"),
    State("auto-loop-toggle", "value"),
    State("loop-interval-input", "value"),
    State("snapshot-auto-toggle", "value"),
    State("snapshot-interval-input", "value"),
    prevent_initial_call=True,
)
def handle_save_settings(
    n_clicks: int,
    auto_loop_values: List[str],
    loop_interval_value: Optional[int],
    snapshot_auto_values: List[str],
    snapshot_interval_value: Optional[int],
) -> str:
    loop_enabled = "enabled" in (auto_loop_values or [])
    loop_interval = int(loop_interval_value or 0)
    snapshot_auto = "enabled" in (snapshot_auto_values or [])
    snapshot_interval = int(snapshot_interval_value or DEFAULT_SNAPSHOT_REFRESH)

    STATE.update(
        loop_enabled=loop_enabled,
        loop_interval=loop_interval,
        snapshot_auto_refresh=snapshot_auto,
        snapshot_refresh_interval=max(15, snapshot_interval),
    )

    if not loop_enabled:
        STATE.update(loop_primed=False, grace_period_end=0)

    configure_auto_cycle_job()
    configure_snapshot_job()

    return "Settings saved."


@app.callback(
    Output("snapshot-status", "children"),
    Output("cycle-status", "children"),
    Output("cycle-hint", "children"),
    Output("balances-container", "children"),
    Output("positions-table", "data"),
    Output("positions-table", "columns"),
    Output("account-prompt", "children"),
    Output("equity-curve", "figure"),
    Output("history-empty", "children"),
    Output("trades-feed", "children"),
    Output("next-snapshot-info", "children"),
    Output("grace-period-info", "children"),
    Output("last-snapshot-info", "children"),
    Output("auto-loop-toggle", "value"),
    Output("loop-interval-input", "value"),
    Output("snapshot-auto-toggle", "value"),
    Output("snapshot-interval-input", "value"),
    Output("sentiment-display", "children"),
    Output("sentiment-timestamp", "children"),
    Output("ai-explanation-content", "children"),
    Input("refresh-interval", "n_intervals"),
)
def update_ui(n_intervals: int):
    snapshot = STATE.snapshot()

    balances_view = _build_balances_view(snapshot)
    positions_data, positions_columns = _positions_table(snapshot)
    account_prompt = _strip_emoji((snapshot.get("last_snapshot") or {}).get("account_prompt", ""))
    history = snapshot.get("history") or []
    equity_figure = _equity_figure(history)
    history_hint = _strip_emoji("" if history else "Run at least one cycle to populate performance history.")
    trades_feed = _build_trade_feed(history)

    next_snapshot = ""
    job = SCHEDULER.get_job("snapshot_refresh_job")
    if job and job.next_run_time:
        next_snapshot = f"Next auto snapshot scheduled at {_format_local_timestamp(job.next_run_time)}"
    elif not snapshot.get("snapshot_auto_refresh", True):
        next_snapshot = "Auto snapshot refresh is disabled."
    next_snapshot = _strip_emoji(next_snapshot)

    cycle_hint = ""
    settings_hint = ""
    if snapshot.get("loop_enabled"):
        remaining = max(0, int(snapshot.get("grace_period_end", 0) - time.time()))
        interval = snapshot.get("loop_interval", 0)
        if snapshot.get("loop_running"):
            cycle_hint = "Trading cycle is executing in the background."
        elif remaining > 0:
            cycle_hint = f"Grace period: {remaining}s remaining."
        elif interval > 0:
            cycle_hint = f"Auto loop interval: {interval} seconds."
        else:
            cycle_hint = "Auto loop set to continuous mode."

        settings_hint = cycle_hint
        if not snapshot.get("loop_primed"):
            suffix = " Auto loop arms after a manual trading cycle."
            cycle_hint = (cycle_hint + suffix).strip()
            settings_hint = (settings_hint + suffix).strip()
    else:
        cycle_hint = "Auto loop disabled."
        settings_hint = "Auto loop disabled."
    cycle_hint = _strip_emoji(cycle_hint)
    settings_hint = _strip_emoji(settings_hint)

    last_snapshot_info = ""
    if snapshot.get("last_snapshot_refreshed_at"):
        last_snapshot_info = f"Last snapshot update: {snapshot['last_snapshot_refreshed_at']}"
    last_snapshot_info = _strip_emoji(last_snapshot_info)

    auto_loop_value = ["enabled"] if snapshot.get("loop_enabled") else []
    snapshot_auto_value = ["enabled"] if snapshot.get("snapshot_auto_refresh", True) else []

    sentiment_display = dcc.Markdown("Loading or waiting for next cycle...")
    sentiment_time = ""
    if SENTIMENT_CACHE_PATH.exists():
        try:
            with SENTIMENT_CACHE_PATH.open("r", encoding="utf-8") as handle:
                sentiment_data = json.load(handle)
            sentiment_content = sentiment_data.get("content") or "No content"
            sentiment_display = dcc.Markdown(sentiment_content)

            timestamp = sentiment_data.get("timestamp")
            if timestamp:
                ts = datetime.fromisoformat(timestamp)
                sentiment_time = f"Updated: {ts.astimezone().strftime('%H:%M')}"
        except Exception:  # noqa: BLE001
            sentiment_display = dcc.Markdown("Unable to load sentiment snapshot.")

    ai_explanation = _build_ai_explanation(history)

    return (
        _strip_emoji(snapshot.get("snapshot_status", "")),
        _strip_emoji(snapshot.get("cycle_status", "")),
        cycle_hint,
        balances_view,
        positions_data,
        positions_columns,
        account_prompt,
        equity_figure,
        history_hint,
        trades_feed,
        next_snapshot,
        settings_hint,
        last_snapshot_info,
        auto_loop_value,
        snapshot.get("loop_interval"),
        snapshot_auto_value,
        snapshot.get("snapshot_refresh_interval"),
        sentiment_display,
        sentiment_time,
        ai_explanation,
    )


# ── Callback : onglet Auditeur ────────────────────────────────────────────────

@app.callback(
    Output("auditeur-content", "children"),
    Input("refresh-interval", "n_intervals"),
)
def update_auditeur_tab(n_intervals):
    """Affiche l'historique des rapports de l'Auditeur (plus récent en tête)."""
    history = get_auditeur_history()

    if not history:
        return html.Div([
            html.H2("Auditeur — Rapports Quotidiens", style={"marginBottom": "1.5rem"}),
            html.Div(
                "Aucun rapport disponible. L'Auditeur se déclenchera automatiquement à 1h05 chaque matin (heure française).",
                className="muted",
            ),
        ])

    cards = []
    for entry in history:
        ts_raw = entry.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_raw).astimezone()
            ts_label = ts.strftime("%A %d %B %Y — %H:%M %Z")
        except Exception:  # noqa: BLE001
            ts_label = ts_raw

        error = entry.get("error")
        content = _strip_emoji(entry.get("content") or "")
        reasoning = _strip_emoji(entry.get("reasoning_content") or "")

        card_children = [
            html.Div(ts_label, style={
                "fontWeight": "700",
                "fontSize": "1rem",
                "marginBottom": "1rem",
                "paddingBottom": "0.75rem",
                "borderBottom": "1px solid rgba(255,255,255,0.08)",
            }),
        ]

        if error:
            card_children.append(
                html.Div(f"Erreur : {error}", style={"color": "#fb7185", "fontSize": "0.9rem"})
            )
        else:
            if reasoning:
                card_children.append(
                    html.Details([
                        html.Summary(
                            "Raisonnement DeepSeek (thinking)",
                            style={"cursor": "pointer", "color": "rgba(255,255,255,0.45)", "fontSize": "0.85rem", "marginBottom": "0.5rem"},
                        ),
                        html.Pre(reasoning, className="code-block", style={
                            "maxHeight": "350px",
                            "overflowY": "auto",
                            "fontSize": "0.8rem",
                            "lineHeight": "1.65",
                            "marginTop": "0.5rem",
                        }),
                    ], style={"marginBottom": "1rem"})
                )

            if content:
                card_children.append(
                    dcc.Markdown(content, style={"fontSize": "0.92rem", "lineHeight": "1.8"})
                )
            else:
                card_children.append(
                    html.Div("Contenu vide.", className="muted")
                )

        cards.append(
            html.Div(card_children, style={
                "background": "rgba(255,255,255,0.03)",
                "border": "1px solid rgba(255,255,255,0.08)",
                "borderRadius": "16px",
                "padding": "1.5rem",
                "marginBottom": "1.25rem",
            })
        )

    return html.Div([
        html.H2(
            f"Auditeur — Rapports Quotidiens ({len(history)} entrée{'s' if len(history) > 1 else ''})",
            style={"marginBottom": "1.5rem"},
        ),
        html.Div(cards),
    ])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
