"""Identify starters/bench players and split each player's floor time into stints.

A stint is a run of consecutive events in one period during which the player is on court.
Stints are cut at period boundaries (documented limitation): a reserve who plays across a
quarter break gets two stints, each judged on its own win-probability change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Pseudo-minutes of zero plus-minus added when estimating a player's on-court net rating,
# so low-minute players don't look like superstars or disasters.
NET_RATING_PRIOR_MINUTES = 250


def _on_court_long(events: pd.DataFrame) -> pd.DataFrame:
    """One row per (event, player on court): event row position, side, player."""
    parts = []
    for side, cols in (("home", config.HOME_LINEUP), ("away", config.AWAY_LINEUP)):
        names = events[cols].to_numpy().ravel()  # row-major: event 0 slots 1-5, event 1 ...
        rows = np.repeat(np.arange(len(events)), 5)
        part = pd.DataFrame({"row": rows, "player": names, "side": side})
        parts.append(part[pd.notna(part["player"])])
    long = pd.concat(parts, ignore_index=True)
    long["game_id"] = events["game_id"].to_numpy()[long["row"]]
    long["period"] = events["period"].to_numpy()[long["row"]]
    return long


def starters(events: pd.DataFrame) -> pd.DataFrame:
    """Starting five for each team in each game = lineups on the first event of period 1."""
    first = events[events["period"] == 1].groupby("game_id").head(1)
    rows = []
    for cols in (config.HOME_LINEUP, config.AWAY_LINEUP):
        part = first[["game_id"] + cols].melt(id_vars="game_id", value_name="player")
        rows.append(part[["game_id", "player"]])
    return pd.concat(rows).dropna().drop_duplicates()


def build_stints(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate on-court runs into stints with entry context and win-probability change."""
    events = events.reset_index(drop=True)
    long = _on_court_long(events).sort_values(["game_id", "player", "row"], kind="stable")
    same = (long["game_id"].eq(long["game_id"].shift()) & long["player"].eq(long["player"].shift())
            & long["period"].eq(long["period"].shift()) & long["row"].sub(long["row"].shift()).eq(1))
    long["stint_key"] = (~same).cumsum()
    st = long.groupby("stint_key").agg(game_id=("game_id", "first"), player=("player", "first"),
                                       side=("side", "first"), period=("period", "first"),
                                       row_start=("row", "min"), row_end=("row", "max"),
                                       events=("row", "size")).reset_index(drop=True)

    s, e = st["row_start"].to_numpy(), st["row_end"].to_numpy()
    home = (st["side"] == "home").to_numpy()
    sign = np.where(home, 1, -1)
    col = lambda name: events[name].to_numpy()  # noqa: E731

    st["team"] = np.where(home, col("home_team")[s], col("away_team")[s])
    st["opponent"] = np.where(home, col("away_team")[s], col("home_team")[s])
    st["phase"] = col("phase")[s]
    st["date"] = col("date")[s] if "date" in events else ""
    st["entry_clock"] = col("clock_before")[s]
    st["entry_seconds_remaining"] = col("seconds_remaining")[s] + (col("clock_before")[s] - col("clock_seconds")[s])
    st["entry_margin"] = sign * col("margin_home_before")[s]
    st["entry_leverage"] = col("leverage")[s]
    st["duration_seconds"] = np.clip(col("clock_before")[s] - col("clock_seconds")[e], 0, None)
    st["wpa"] = sign * (col("wp_home")[e] - col("wp_home_before")[s])
    st["plus_minus"] = sign * (col("margin_home")[e] - col("margin_home_before")[s])

    # Opposing lineup at entry.
    home_l, away_l = events[config.HOME_LINEUP].to_numpy(), events[config.AWAY_LINEUP].to_numpy()
    opp = np.where(home[:, None], away_l[s], home_l[s])
    st["opp_lineup"] = [" | ".join(sorted(str(x) for x in r if pd.notna(x))) for r in opp]

    starter_set = set(zip(*starters(events)[["game_id", "player"]].to_numpy().T))
    gids = st["game_id"].to_numpy()
    st["is_starter"] = [(g, p) in starter_set for g, p in zip(gids, st["player"])]
    st["opp_starters_on"] = [sum((g, p) in starter_set for p in r if pd.notna(p)) for g, r in zip(gids, opp)]

    # Opponent lineup strength: mean on-court net rating (per 48, shrunk) of the five opponents.
    net = player_net_rating(st)
    lookup = net.set_index("player")["net_per48"].to_dict()
    st["opp_lineup_net48"] = [
        float(np.mean(vals)) if (vals := [lookup.get(p, 0.0) for p in r if pd.notna(p)]) else 0.0
        for r in opp
    ]
    own = np.where(home[:, None], home_l[s], away_l[s])
    st["teammates_net48"] = [
        float(np.mean(vals)) if (vals := [lookup.get(p, 0.0) for p in r if pd.notna(p) and p != me]) else 0.0
        for r, me in zip(own, st["player"])
    ]
    return st.drop(columns=["row_start", "row_end"])


def player_net_rating(st: pd.DataFrame) -> pd.DataFrame:
    """Season on-court plus-minus per 48 minutes for every player, shrunk toward 0."""
    g = st.groupby("player").agg(pm=("plus_minus", "sum"), secs=("duration_seconds", "sum")).reset_index()
    g["minutes"] = g["secs"] / 60
    g["net_per48"] = 48 * g["pm"] / (g["minutes"] + NET_RATING_PRIOR_MINUTES)
    return g[["player", "minutes", "net_per48"]]
