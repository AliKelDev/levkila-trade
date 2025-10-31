from dataclasses import asdict
from datetime import datetime
import json
import time
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from prompt_builder import get_account_snapshot, run_cycle

st.set_page_config(page_title="DeepTrade Control Panel", layout="wide")
st.title("DeepTrade Control Panel")

DEFAULT_HISTORY_LIMIT = 20

# --- Session defaults ---
for key, default in (
    ("last_snapshot", None),
    ("history", []),
    ("loop_enabled", False),
    ("loop_interval", 120),
    ("next_run_time", time.time()),
    ("loop_running", False),
    ("loop_initialized", False),
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


def _maybe_auto_run_cycle() -> None:
    if not st.session_state["loop_enabled"]:
        st.session_state["loop_initialized"] = False
        return

    if not st.session_state["loop_initialized"]:
        st.session_state["next_run_time"] = time.time()
        st.session_state["loop_initialized"] = True

    if st.session_state["loop_running"]:
        return

    now = time.time()
    next_run = st.session_state.get("next_run_time", now)
    if now >= next_run:
        st.session_state["loop_running"] = True
        try:
            result = run_cycle()
            _record_result(result)
        finally:
            st.session_state["loop_running"] = False
            st.session_state["next_run_time"] = time.time() + st.session_state["loop_interval"]
        st.experimental_rerun()


_maybe_auto_run_cycle()

left_col, right_col = st.columns([1, 1.3], gap="large")

with left_col:
    st.subheader("Controls")
    if st.button("Run Trading Cycle", type="primary", use_container_width=True):
        with st.spinner("Executing trading cycle..."):
            result = run_cycle()
            _record_result(result)
            if st.session_state["loop_enabled"]:
                st.session_state["next_run_time"] = time.time() + st.session_state["loop_interval"]
        st.success("Cycle complete")

    if st.button("Refresh Account Snapshot", use_container_width=True):
        with st.spinner("Fetching account state..."):
            st.session_state["last_snapshot"] = get_account_snapshot()
        st.success("Snapshot refreshed")

    st.markdown("---")
    st.subheader("Auto Loop")
    previously_enabled = st.session_state["loop_enabled"]
    st.checkbox("Enable auto loop", key="loop_enabled")
    st.number_input(
        "Loop interval (seconds)",
        min_value=30,
        max_value=3600,
        step=30,
        key="loop_interval",
    )

    if st.session_state["loop_enabled"] and not previously_enabled:
        st.session_state["next_run_time"] = time.time()
        st.session_state["loop_initialized"] = True
    if not st.session_state["loop_enabled"] and previously_enabled:
        st.session_state["loop_initialized"] = False

    next_run_in = max(0, int(st.session_state.get("next_run_time", time.time()) - time.time()))
    if st.session_state["loop_enabled"]:
        st.caption(f"Next auto run in {next_run_in} seconds")
    else:
        st.caption("Auto loop disabled")

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

# auto rerun polling
if st.session_state["loop_enabled"] and not st.session_state["loop_running"]:
    remaining = max(0, st.session_state.get("next_run_time", time.time()) - time.time())
    sleep_for = min(1.0, remaining) if remaining > 0 else 0.5
    time.sleep(sleep_for)
    if st.session_state["loop_enabled"]:
        st.experimental_rerun()
