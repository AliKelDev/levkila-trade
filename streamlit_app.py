from dataclasses import asdict
from datetime import datetime
import json
import logging
import time
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from prompt_builder import get_account_snapshot, run_cycle

# Suppress APScheduler logging noise
logging.getLogger("apscheduler").setLevel(logging.CRITICAL)

st.set_page_config(page_title="DeepTrade Control Panel", layout="wide")
st.title("DeepTrade Control Panel")

DEFAULT_HISTORY_LIMIT = 20

# --- Global scheduler and grace period tracking ---
_scheduler: BackgroundScheduler = None
_grace_period_end: float = 0  # Track grace period outside of session state
_loop_enabled: bool = False  # Track autoloop status outside of session state
_loop_running: bool = False  # Track cycle execution status outside of session state

def _get_scheduler() -> BackgroundScheduler:
    """Get or create the global background scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler()
        # CRITICAL: Only allow 1 instance at a time to prevent concurrent trades
        _scheduler.configure(job_defaults={'max_instances': 1})
        _scheduler.start()
    return _scheduler

# --- Session defaults ---
for key, default in (
    ("last_snapshot", None),
    ("history", []),
    ("loop_enabled", False),
    ("loop_running", False),
    ("loop_interval", 0),
    ("loop_primed", False),
):
    if key not in st.session_state:
        st.session_state[key] = default


def _positions_to_dataframe(positions: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    if not positions:
        return pd.DataFrame(columns=["coin"])
    rows: List[Dict[str, Any]] = []
    for coin, payload in positions.items():
        row = {"coin": coin}
        row.update(payload)
        for key, value in list(row.items()):
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, indent=2)
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except (TypeError, ValueError):
                continue
    return df


def _decisions_to_dataframe(decisions: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for coin, payload in decisions.items():
        args = payload.get("trade_signal_args", {})
        exit_plan = args.get("exit_plan", {})
        rows.append(
            {
                "coin": coin,
                "signal": args.get("signal"),
                "leverage": args.get("leverage"),
                "confidence": args.get("confidence"),
                "risk_usd": args.get("risk_usd"),
                "justification": args.get("justification"),
                "exit_plan": json.dumps(exit_plan, indent=2) if exit_plan else "",
            }
        )
    if not rows:
        return pd.DataFrame(columns=["coin"])
    return pd.DataFrame(rows)


def _render_logs(logs: List[Dict[str, Any]]) -> None:
    if not logs:
        st.caption("No log entries captured.")
        return
    for entry in logs:
        ts = datetime.fromtimestamp(entry.get("ts", time.time()))
        timestamp = ts.strftime("%Y-%m-%d %H:%M:%S")
        title = entry.get("title", "INFO")
        content = entry.get("content", "")
        st.markdown(
            f"<div style='margin-bottom:0.5rem;'>"
            f"<span style='font-weight:600;'>{timestamp} — {title}</span>"
            f"<pre style='background:#11111120;padding:0.5rem;border-radius:0.25rem;'>"
            f"{content}\n"
            f"</pre>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _record_result(result) -> None:
    result_dict = asdict(result)
    st.session_state["last_snapshot"] = {
        "account_prompt": result.account_prompt_after,
        "positions": result.positions_after,
        "balances": result.balances_after,
    }
    history = st.session_state["history"]
    history.append(result_dict)
    if len(history) > DEFAULT_HISTORY_LIMIT:
        st.session_state["history"] = history[-DEFAULT_HISTORY_LIMIT:]


def _auto_run_cycle_background() -> None:
    """Background job to run trading cycle. Updates session state to trigger UI refresh.

    CRITICAL: This function must NEVER run concurrently. The scheduler is configured
    with max_instances=1 to ensure only one cycle executes at a time.

    The grace period is set BEFORE this function is called, preventing overlapping executions.
    """
    global _loop_running
    _loop_running = True
    st.session_state["loop_running"] = True
    try:
        result = run_cycle()
        _record_result(result)
    except Exception as exc:
        print(f"Error in background cycle: {exc}")
    finally:
        _loop_running = False
        st.session_state["loop_running"] = False


def _auto_run_cycle_with_grace_period() -> None:
    """Background job that auto-reruns the cycle if autoloop is enabled.

    Strategy:
    - If grace period is active: return silently (avoid log spam)
    - If grace period expired AND cycle is NOT running: trigger next cycle
    - If autoloop is disabled: do nothing

    Uses global variables instead of session state to avoid ScriptRunContext warnings.
    """
    global _grace_period_end, _loop_enabled, _loop_running

    # If grace period is still active, return quickly (silently)
    if time.time() < _grace_period_end:
        return

    # Grace period expired - check if autoloop is still enabled
    if not _loop_enabled:
        return

    # Autoloop is on and grace period expired - if cycle not running, start it
    if not _loop_running:
        _auto_run_cycle_background()
        # Set new grace period for the next cycle
        _grace_period_end = time.time() + 180


def _sync_scheduler() -> None:
    """Sync scheduler state with session state settings.

    Strategy:
    - Scheduler checks every loop_interval seconds
    - If grace period is active, the job function returns immediately (no execution)
    - If grace period expired, the job function runs the cycle and sets new grace period
    - Grace period: 3 minutes to let cycle complete
    - If loop_interval is 0: Check every 1 second (continuous mode)
    """
    scheduler = _get_scheduler()
    loop_enabled = st.session_state.get("loop_enabled", False)
    loop_primed = st.session_state.get("loop_primed", False)
    loop_interval = st.session_state.get("loop_interval", 0)

    existing_job = scheduler.get_job("auto_cycle_job")

    if loop_enabled and loop_primed:
        # Check interval: use loop_interval, or 1 second if in continuous mode (0)
        check_interval = loop_interval if loop_interval > 0 else 1

        if not existing_job:
            # Create new job
            scheduler.add_job(
                _auto_run_cycle_with_grace_period,
                trigger=IntervalTrigger(seconds=check_interval),
                id="auto_cycle_job",
                replace_existing=True,
            )
        else:
            # Update interval if it changed
            current_interval = existing_job.trigger.interval.total_seconds()
            if current_interval != check_interval:
                scheduler.reschedule_job(
                    "auto_cycle_job",
                    trigger=IntervalTrigger(seconds=check_interval)
                )
    else:
        # Disable scheduler
        if existing_job:
            scheduler.remove_job("auto_cycle_job")


# Initialize scheduler on app load
_sync_scheduler()

left_col, right_col = st.columns([1, 1.3], gap="large")

with left_col:
    st.subheader("Controls")
    if st.button("Run Trading Cycle", type="primary", use_container_width=True):
        # If autoloop is enabled, set grace period BEFORE cycle starts to avoid log spam
        if st.session_state.get("loop_enabled", False):
            _grace_period_end = time.time() + 180

        with st.spinner("Executing trading cycle..."):
            result = run_cycle()
            _record_result(result)
        st.session_state["loop_primed"] = True
        st.success("Cycle complete")

        # If autoloop is enabled and scheduler not yet started, start it now
        if st.session_state.get("loop_enabled", False):
            _sync_scheduler()

    if st.button("Refresh Account Snapshot", use_container_width=True):
        with st.spinner("Fetching account state..."):
            st.session_state["last_snapshot"] = get_account_snapshot()
        st.success("Snapshot refreshed")

    st.markdown("---")
    st.subheader("Auto Loop")
    st.checkbox("Enable auto loop", key="loop_enabled")

    # Sync global _loop_enabled with session state (for background thread)
    _loop_enabled = st.session_state["loop_enabled"]

    if st.session_state["loop_enabled"]:
        interval = st.number_input(
            "Delay between cycles (seconds)",
            min_value=0,
            max_value=3600,
            step=1,
            value=st.session_state.get("loop_interval", 0),
        )
        st.session_state["loop_interval"] = interval

        # Only sync scheduler if a cycle has been run (grace_period_end > 0)
        # This prevents autoloop from triggering before the first manual cycle
        if st.session_state.get("loop_primed") and _grace_period_end > 0:
            _sync_scheduler()

        # Show grace period countdown if active
        time_remaining = max(0, int(_grace_period_end - time.time()))

        if time_remaining > 0:
            st.caption(f"⏸️ Grace period: {time_remaining}s remaining (job paused)")
        else:
            if interval > 0:
                st.caption(f"⏱️ Running every {interval} seconds")
            else:
                st.caption("⚡ Auto loop: continuous (waits for cycle to finish)")

            if st.session_state.get("loop_running"):
                st.caption("🔄 Currently executing cycle...")

        if not st.session_state.get("loop_primed"):
            st.caption("✅ Auto loop armed after the next manual trading cycle.")
    else:
        # Sync scheduler to disable
        _sync_scheduler()
        st.session_state["loop_primed"] = False
        st.caption("Auto loop disabled.")
    
    st.markdown("---")
    snapshot = st.session_state.get("last_snapshot")
    if snapshot:
        balances = snapshot.get("balances", {})
        if balances:
            st.subheader("Balances")
            for label, value in balances.items():
                st.metric(label.replace("_", " ").title(), f"{value:,.2f}")

        st.subheader("Open Positions")
        positions_df = _positions_to_dataframe(snapshot.get("positions", {}))
        if not positions_df.empty:
            st.dataframe(positions_df, width="stretch")
        else:
            st.caption("No open positions")

        st.subheader("Account Prompt")
        st.code(snapshot.get("account_prompt", ""), language="markdown")
    else:
        st.info("Run a trading cycle or refresh the snapshot to populate account data.")

with right_col:
    st.subheader("Conversation Feed")
    history: List[Dict[str, Any]] = st.session_state.get("history", [])
    if not history:
        st.info("No decisions recorded yet.")
    else:
        for idx, entry in enumerate(reversed(history)):
            latest = idx == 0
            timestamp = entry.get("run_timestamp")
            readable_time = (
                datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                if timestamp
                else "Unknown time"
            )
            minutes_since_start = entry.get("minutes_since_start")
            invocation = entry.get("invocation_count")

            st.markdown(
                f"### Run {invocation} — {readable_time}"
                + (" _(latest)_" if latest else "")
            )
            st.caption(f"{minutes_since_start} minutes since start")

            with st.expander("User Prompt", expanded=latest):
                st.code(entry.get("user_prompt", ""), language="markdown")

            with st.expander("System Prompt", expanded=False):
                st.code(entry.get("system_prompt", ""), language="markdown")

            with st.expander("LLM Raw Output", expanded=False):
                st.code(entry.get("llm_raw", ""), language="json")

            chain = entry.get("chain_of_thought")
            if chain:
                with st.expander("Chain of Thought", expanded=False):
                    st.code(chain, language="markdown")

            st.markdown("**Decisions**")
            decisions_df = _decisions_to_dataframe(entry.get("decisions", {}))
            if not decisions_df.empty:
                st.dataframe(decisions_df, width="stretch")
            else:
                st.caption("No decisions parsed.")

            st.markdown("**Execution Logs**")
            _render_logs(entry.get("logs", []))

            st.markdown("---")
