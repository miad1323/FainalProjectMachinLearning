import time
import traceback
from pathlib import Path
import pandas as pd
import requests
import streamlit as st

st.set_page_config(layout="wide")
st.title("Live Football Prediction Dashboard (Bonus)")
st.write("Loading data...")

SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[1]
    / "phase1" / "outputs" / "in_play_snapshots_all_seasons.csv"
)
PREMATCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "phase1" / "outputs" / "pre_match_features_all_seasons.csv"
)

try:

    @st.cache_data
    def load_data():
        if not SNAPSHOT_PATH.exists():
            st.error(
                f"Snapshot file not found at {SNAPSHOT_PATH}. "
                "Please run Phase 1 first."
            )
            st.stop()
        snap = pd.read_csv(SNAPSHOT_PATH)
        if snap.empty:
            st.error("Snapshot file is empty.")
            st.stop()

        if "season" not in snap.columns or snap["season"].isna().all():
            if PREMATCH_PATH.exists():
                pre = pd.read_csv(PREMATCH_PATH)
                if "match_id" in pre.columns and "season" in pre.columns:
                    season_map = pre[["match_id", "season"]].drop_duplicates("match_id")
                    snap = snap.merge(season_map, on="match_id", how="left")
                    if "season_y" in snap.columns:
                        snap["season"] = snap["season_y"]
                        snap.drop(
                            columns=["season_x", "season_y"],
                            inplace=True,
                            errors="ignore"
                        )
                    elif "season_x" in snap.columns:
                        snap["season"] = snap["season_x"]
                        snap.drop(columns=["season_x"], inplace=True)
                else:
                    snap["season"] = "Unknown"
            else:
                snap["season"] = "Unknown"

        snap["season"] = snap["season"].fillna("Unknown")
        return snap

    snapshots = load_data()
    seasons = sorted(snapshots["season"].unique())
    selected_season = st.selectbox("Select season", seasons)
    season_snapshots = snapshots[snapshots["season"] == selected_season]
    match_ids = season_snapshots["match_id"].unique()

    st.write(f"Season {selected_season}: {len(match_ids)} matches available.")
    match_id = st.selectbox("Select match to replay", match_ids)
    match_data = season_snapshots[
        season_snapshots["match_id"] == match_id
    ].sort_values("snapshot_minute")

    if st.button("Replay match"):
        st.subheader("Current Match State")
        progress_bar = st.progress(0)
        status_text = st.empty()
        margin_placeholder = st.empty()

        col1, col2, col3 = st.columns(3)
        home_placeholder = col1.empty()
        draw_placeholder = col2.empty()
        away_placeholder = col3.empty()
        shap_placeholder = st.empty()

        st.subheader("Match Timeline")
        timeline_placeholder = st.empty()
        history_log = []

        for idx, row in match_data.iterrows():
            minute = int(row.snapshot_minute)
            status_text.text(f"Minute {minute}")

            try:
                response = requests.post(
                    "http://localhost:8001/predict",
                    json={"match_id": int(match_id), "snapshot_minute": minute},
                )
                if response.status_code == 200:
                    pred = response.json()
                    probs = pred["probabilities"]
                    margin = pred["expected_margin"]

                    margin_placeholder.metric("Expected margin", f"{margin:+.2f}")
                    home_placeholder.metric("Home win", f"{probs['H']:.2%}")
                    draw_placeholder.metric("Draw", f"{probs['D']:.2%}")
                    away_placeholder.metric("Away win", f"{probs['A']:.2%}")

                    log_entry = (
                        f"**Min {minute:02d}** | Home: {probs['H']:.1%} | "
                        f"Draw: {probs['D']:.1%} | Away: {probs['A']:.1%} | "
                        f"Margin: {margin:+.2f}"
                    )
                    history_log.append(log_entry)
                    timeline_placeholder.markdown("\n\n".join(history_log))

                    if pred.get("top_shap"):
                        shap_placeholder.write("Top SHAP features:")
                        shap_placeholder.json(pred["top_shap"])
                    else:
                        shap_placeholder.empty()
                else:
                    try:
                        err_detail = response.json().get("detail", response.text)
                    except Exception:
                        err_detail = response.text
                    st.warning(f"API error: {err_detail}")
            except Exception as e:
                st.error(f"Error calling API: {e}")

            progress_bar.progress(min(minute / 90, 1.0))
            time.sleep(0.3)

        st.success("Match replay complete!")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.code(traceback.format_exc())