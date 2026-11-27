"""Identify starters/bench players and split each player's floor time into stints.

A stint is a run of consecutive events in one period during which the player is on court.
Stints are cut at period boundaries for simplicity (documented limitation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def long_lineups(events: pd.DataFrame) -> pd.DataFrame:
    """One row per (event, player on court) with side, team and opponent context."""
    parts = []
    for side, cols, team_col, opp_cols in [
        ("home", config.HOME_LINEUP, "home_team", config.AWAY_LINEUP),
        ("away", config.AWAY_LINEUP, "away_team", config.HOME_LINEUP),
    ]:
        base = events[["game_id", "period", "seq", team_col] + cols + opp_cols].copy()
        base["opp_lineup"] = base[opp_cols].astype(str).apply(lambda r: " | ".join(sorted(r)), axis=1)
        melted = base.melt(id_vars=["game_id", "period", "seq", team_col, "opp_lineup"] + opp_cols,
                           value_vars=cols, value_name="player_on").drop(columns="variable")
        melted = melted.rename(columns={team_col: "team_on"})
        melted["side"] = side
        parts.append(melted)
    out = pd.concat(parts, ignore_index=True).dropna(subset=["player_on"])
    return out.rename(columns={"player_on": "player"})


def starters(events: pd.DataFrame) -> pd.DataFrame:
    """Starting five for each team in each game = lineup on the first event of period 1."""
    first = events[events["period"] == 1].sort_values("seq").groupby("game_id").head(1)
    rows = []
    for _, r in first.iterrows():
        for side, cols in [("home", config.HOME_LINEUP), ("away", config.AWAY_LINEUP)]:
            rows += [{"game_id": r["game_id"], "player": r[c], "is_starter": True} for c in cols]
    return pd.DataFrame(rows).drop_duplicates()


def build_stints(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate on-court runs into stints with entry context and win probability added."""
    on = long_lineups(events).sort_values(["game_id", "player", "seq"])
    new_stint = (on.groupby(["game_id", "player"])["seq"].diff() != 1) | \
                (on.groupby(["game_id", "player"])["period"].diff() != 0)
    on["stint_no"] = new_stint.groupby([on["game_id"], on["player"]]).cumsum()

    ev = events.set_index(["game_id", "seq"])
    grp = on.groupby(["game_id", "player", "stint_no"])
    st = grp.agg(team=("team_on", "first"), side=("side", "first"), period=("period", "first"),
                 seq_start=("seq", "min"), seq_end=("seq", "max"),
                 opp_lineup=("opp_lineup", "first")).reset_index()

    start = ev.loc[list(zip(st["game_id"], st["seq_start"]))].reset_index(drop=True)
    end = ev.loc[list(zip(st["game_id"], st["seq_end"]))].reset_index(drop=True)
    sign = np.where(st["side"] == "home", 1, -1)

    st["entry_seconds_remaining"] = start["seconds_before"].to_numpy()
    st["entry_margin"] = sign * start["margin_home_before"].to_numpy()
    st["entry_leverage"] = start["leverage"].to_numpy()
    st["duration_seconds"] = np.clip(start["seconds_before"].to_numpy()
                                     - end["seconds_remaining"].to_numpy(), 0, None)
    st["wpa"] = sign * (end["wp_home"].to_numpy() - start["wp_home_before"].to_numpy())
    st["plus_minus"] = sign * (end["margin_home"].to_numpy() - start["margin_home_before"].to_numpy())

    # Opposing starters on court at entry: a simple proxy for opponent lineup strength.
    s = starters(events)
    starter_set = set(zip(s["game_id"], s["player"]))
    st["opp_starters_on"] = [
        sum((g, p) in starter_set for p in lineup.split(" | "))
        for g, lineup in zip(st["game_id"], st["opp_lineup"])
    ]
    st["is_starter"] = [(g, p) in starter_set for g, p in zip(st["game_id"], st["player"])]
    return st
