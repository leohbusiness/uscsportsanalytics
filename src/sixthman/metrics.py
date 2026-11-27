"""Bench player metrics built on stints.

Core idea ("One Sharp Thing"): a bench player's value is the win probability his team
gains while he is on the floor, judged in the situations a coach actually faces. Raw WPA
already weights each point by game leverage; we add situation filters and shrinkage so
small samples don't top the rankings.
"""
from __future__ import annotations

import pandas as pd

from . import config

MARGIN_BUCKETS = [(-99, -11, "Down 11+"), (-10, -4, "Down 4-10"), (-3, 3, "Within 3"),
                  (4, 10, "Up 4-10"), (11, 99, "Up 11+")]


def margin_bucket(margin: pd.Series) -> pd.Series:
    out = pd.Series("", index=margin.index)
    for lo, hi, label in MARGIN_BUCKETS:
        out[(margin >= lo) & (margin <= hi)] = label
    return out


def bench_stints(stints: pd.DataFrame) -> pd.DataFrame:
    st = stints[~stints["is_starter"]].copy()
    st["margin_bucket"] = margin_bucket(st["entry_margin"])
    st["high_leverage"] = st["entry_leverage"] >= config.HIGH_LEVERAGE_THRESHOLD
    return st


def rank_players(st: pd.DataFrame, k: float = config.SHRINKAGE_STINTS) -> pd.DataFrame:
    """Rank players on the given (already filtered) stints.

    shrunk_wpa_per_stint = sum(WPA) / (n_stints + k): equivalent to adding k stints of
    zero WPA, i.e. shrinking toward a replacement-level (zero) prior.
    """
    g = st.groupby(["team", "player"])
    out = g.agg(stints=("wpa", "size"), minutes=("duration_seconds", lambda s: s.sum() / 60),
                total_wpa=("wpa", "sum"), plus_minus=("plus_minus", "sum"),
                avg_entry_leverage=("entry_leverage", "mean")).reset_index()
    out["wpa_per_stint"] = out["total_wpa"] / out["stints"]
    out["shrunk_wpa_per_stint"] = out["total_wpa"] / (out["stints"] + k)
    return out.sort_values("shrunk_wpa_per_stint", ascending=False).reset_index(drop=True)
