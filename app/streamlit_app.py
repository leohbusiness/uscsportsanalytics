"""Sixth Man Advantage dashboard: "Who do I bring in right now?"

Pick a team and a game situation; the chart ranks that team's bench players by how much
win probability they have added in comparable situations this season (shrunk toward zero
for small samples).

Run locally:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without pip install
from sixthman import config  # noqa: E402
from sixthman.metrics import MARGIN_BUCKETS, bench_stints, rank_players  # noqa: E402

st.set_page_config(page_title="Sixth Man Advantage", layout="wide")


@st.cache_data
def load_stints() -> pd.DataFrame | None:
    path = config.PROCESSED_DIR / "stints.parquet"
    return bench_stints(pd.read_parquet(path)) if path.exists() else None


stints = load_stints()
st.title("Who do I bring in right now?")
if stints is None:
    st.warning("No processed data yet. Run `python -m sixthman.pipeline` first (see README).")
    st.stop()

with st.sidebar:
    st.header("Game situation")
    team = st.selectbox("Your team", sorted(stints["team"].unique()))
    margin = st.multiselect("Score margin at entry", [b[2] for b in MARGIN_BUCKETS],
                            default=[b[2] for b in MARGIN_BUCKETS])
    minutes_left = st.slider("Minutes left in game at entry", 0, 48, (0, 48))
    opp_starters = st.slider("Opposing starters on floor", 0, 5, (0, 5))
    high_lev = st.checkbox("High-leverage moments only", value=False,
                           help=f"Entry leverage index >= {config.HIGH_LEVERAGE_THRESHOLD}")
    games = st.multiselect("Restrict to games (optional)",
                           sorted(stints.loc[stints["team"] == team, "game_id"].unique()))

f = stints[(stints["team"] == team)
           & stints["margin_bucket"].isin(margin)
           & stints["entry_seconds_remaining"].between(minutes_left[0] * 60, minutes_left[1] * 60)
           & stints["opp_starters_on"].between(*opp_starters)]
if high_lev:
    f = f[f["high_leverage"]]
if games:
    f = f[f["game_id"].isin(games)]

ranked = rank_players(f)
if ranked.empty:
    st.info("No bench stints match this situation. Widen the filters.")
    st.stop()

ranked["Win prob. added per stint (%)"] = 100 * ranked["shrunk_wpa_per_stint"]
fig = px.bar(ranked.sort_values("shrunk_wpa_per_stint"), x="Win prob. added per stint (%)",
             y="player", orientation="h", hover_data=["stints", "minutes", "avg_entry_leverage"])
fig.update_layout(yaxis_title=None, height=max(300, 40 * len(ranked)))
st.plotly_chart(fig, width="stretch")
st.caption(f"{len(f):,} matching bench stints. Values shrink toward 0 by adding "
           f"{config.SHRINKAGE_STINTS} zero-WPA stints per player.")

with st.expander("Details"):
    st.dataframe(ranked.drop(columns=["team"]).round(3), width="stretch")
