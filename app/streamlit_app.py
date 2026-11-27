"""Sixth Man Advantage dashboard: "Who do I bring in right now?"

Set the game situation; the chart ranks bench players by how much win probability they
added, beyond what their teammates and opponents predict, in comparable situations this
season (2025-26), shrunk toward zero for small samples.

Run locally:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without pip install
from sixthman import config  # noqa: E402
from sixthman.metrics import (MARGIN_BUCKETS, METRICS, OPP_LABELS, TIME_BUCKETS,  # noqa: E402
                              bench_stints, rank_players)

POS, NEG = "#257de0", "#ec4647"  # diverging poles: helps / hurts
ALL = "All teams"

st.set_page_config(page_title="Sixth Man Advantage", page_icon="🏀", layout="wide")


@st.cache_data
def load() -> pd.DataFrame | None:
    for path in (config.APP_DATA, config.PROCESSED_DIR / "stints.parquet"):
        if path.exists():
            return bench_stints(pd.read_parquet(path))
    return None


data = load()
if data is None:
    st.title("Who do I bring in right now?")
    st.warning("No processed data yet. Run `python -m sixthman.pipeline` first (see README).")
    st.stop()

GAME_LABELS = (data.groupby("game_id").agg(date=("date", "first"), team=("team", lambda t: " vs ".join(sorted(pd.unique(t)))))
               .astype(str).agg(lambda r: f"{r['date'][:10]} {r['team']}", axis=1).to_dict())


def _game_label(g: str) -> str:
    return GAME_LABELS.get(g, g)


# ---------------------------------------------------------------- sidebar: the situation
with st.sidebar:
    st.header("Game situation")
    teams = [ALL] + sorted(data["team"].unique())
    team = st.selectbox("Your team", teams, index=teams.index("OKC") if "OKC" in teams else 0)
    phase = st.multiselect("Season phase", sorted(data["phase"].unique()), default=["Regular season"])
    margin = st.multiselect("Score when he checks in (your team)", [b[2] for b in MARGIN_BUCKETS],
                            default=[b[2] for b in MARGIN_BUCKETS])
    time = st.multiselect("Game time", [b[2] for b in TIME_BUCKETS], default=[b[2] for b in TIME_BUCKETS])
    opp = st.multiselect("Opponent lineup on the floor", OPP_LABELS, default=OPP_LABELS,
                         help="Terciles of the five opponents' average season on-court net rating.")
    high_lev = st.toggle("Only high-leverage moments", value=False,
                         help=f"A point is worth at least {config.HIGH_LEVERAGE_THRESHOLD}x "
                              "its season-average win-probability value.")
    reserves_only = st.toggle("True reserves only", value=True,
                              help=f"Hide regular starters who came off the bench now and then "
                                   f"(started more than {config.RESERVE_MAX_START_RATE:.0%} of their games).")
    st.divider()
    metric = st.radio("Measure", list(METRICS), format_func=METRICS.get,
                      help="Over expected removes what his teammates' and opponents' strength predicts.")
    min_stints = st.slider("Minimum matching stints", 1, 100, 10 if team != ALL else 40)
    team_games = data if team == ALL else data[data["team"] == team]
    games = st.multiselect("Specific games (optional)", sorted(team_games["game_id"].unique()),
                           format_func=_game_label)

# ---------------------------------------------------------------- filter + rank
f = team_games[team_games["phase"].isin(phase) & team_games["margin_bucket"].isin(margin)
               & team_games["time_bucket"].isin(time) & team_games["opp_strength"].isin(opp)]
if high_lev:
    f = f[f["high_leverage"]]
if reserves_only and "true_reserve" in f:
    f = f[f["true_reserve"]]
if games:
    f = f[f["game_id"].isin(games)]

ranked = rank_players(f, metric)
ranked = ranked[ranked["stints"] >= min_stints]

title = "Who do I bring in right now?" if team != ALL else "Which reserves win games in this situation?"
st.title(title)
st.caption(f"{'All teams' if team == ALL else team} · {len(f):,} matching bench stints · "
           "2025-26 NBA play-by-play · bars = win probability added per stint, shrunk for sample size")

if ranked.empty:
    st.info("No bench player has enough stints in this situation. Widen the filters or lower the minimum.")
    st.stop()

top = ranked.iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Best option", top["player"], help=top["team"])
c2.metric("Expected gain per stint", f"{100 * top['score']:+.1f} pts of win %")
c3.metric("Based on", f"{int(top['stints'])} stints", help=f"{top['minutes']:.0f} minutes on court")

show = ranked.head(15).iloc[::-1]
label = show["player"] + ("  (" + show["team"] + ")" if team == ALL else "")
fig = go.Figure(go.Bar(
    x=100 * show["score"], y=label, orientation="h",
    marker=dict(color=[POS if v >= 0 else NEG for v in show["score"]], cornerradius=4),
    error_x=dict(type="data", array=100 * show["score_se"], thickness=1, width=0, color="rgba(128,128,128,.6)"),
    customdata=show[["stints", "minutes", "avg_entry_leverage", "raw_wpa", "plus_minus"]].to_numpy(),
    hovertemplate="<b>%{y}</b><br>%{x:+.2f} pts of win % per stint<br>"
                  "%{customdata[0]} stints · %{customdata[1]:.0f} min<br>"
                  "avg entry leverage %{customdata[2]:.2f}<br>raw WPA total %{customdata[3]:+.2f} · "
                  "+/- %{customdata[4]:+.0f}<extra></extra>",
))
fig.update_layout(height=max(320, 34 * len(show) + 80), margin=dict(l=10, r=20, t=10, b=40),
                  xaxis=dict(title=f"{METRICS[metric]} (percentage points)", zeroline=True,
                             zerolinecolor="rgba(128,128,128,.7)", gridcolor="rgba(128,128,128,.15)"),
                  yaxis=dict(title=None), bargap=0.35)
st.plotly_chart(fig, width="stretch")
st.caption("Blue helps, red hurts. Whiskers show about one standard error. Shrinkage adds "
           f"{config.SHRINKAGE_STINTS} zero-value stints per player.")

# ---------------------------------------------------------------- drill-down: one player
with st.expander("Player profile: where does he help?"):
    who = st.selectbox("Player", ranked["player"].tolist())
    p = team_games[(team_games["player"] == who) & team_games["phase"].isin(phase)]
    grid = (p.pivot_table(index="time_bucket", columns="margin_bucket", values=metric, aggfunc="mean")
            .reindex(index=[b[2] for b in TIME_BUCKETS], columns=[b[2] for b in MARGIN_BUCKETS]) * 100)
    counts = p.pivot_table(index="time_bucket", columns="margin_bucket", values=metric, aggfunc="size") \
        .reindex(index=grid.index, columns=grid.columns).fillna(0).astype(int)
    lim = float(max(1.0, abs(grid).max().max())) if grid.notna().any().any() else 1.0
    heat = go.Figure(go.Heatmap(z=grid.to_numpy(), x=grid.columns, y=grid.index, zmid=0, zmin=-lim, zmax=lim,
                                colorscale=[[0, NEG], [0.5, "#efefef"], [1, POS]], customdata=counts.to_numpy(),
                                hovertemplate="%{y} · %{x}<br>%{z:+.2f} pts of win % per stint"
                                              "<br>%{customdata} stints<extra></extra>",
                                colorbar=dict(title="pts")))
    heat.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(heat, width="stretch")
    st.caption("Unshrunk averages by entry situation. Cells with few stints are noisy; hover for counts.")

with st.expander("Table"):
    st.dataframe(ranked[["player", "team", "stints", "minutes", "score", "per_stint", "raw_wpa",
                         "plus_minus", "avg_entry_leverage"]].round(4), width="stretch", hide_index=True)

with st.expander("How this works"):
    st.markdown(f"""
- **Win probability** is modeled from this season's own play-by-play (score margin and time left,
  logistic regression, validated on held-out games).
- A **stint** runs from when a reserve checks in until he checks out (cut at quarter breaks).
  **WPA** is his team's change in win probability over that stint, so late, close-game points count more.
- **Over expected** subtracts what the gap between his four teammates' and the five opponents'
  season on-court ratings predicts, so a reserve isn't credited just for playing beside stars.
- **Shrinkage**: score = total / (stints + {config.SHRINKAGE_STINTS}).
- "Bench" means he did not start that game.
""")
