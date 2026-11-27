"""Bench player metrics built on stints.

Core metric ("One Sharp Thing"): **Bench WPA over expected per stint, shrunk.**
For each time a reserve checks in, we measure the change in his team's win probability
until he checks out (WPA, which already weights every point by game leverage), subtract
what the strength of his four teammates and five opponents predicts (see expected.py), and
average over the stints that match the coach's current situation, shrinking small samples
toward zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

MARGIN_BUCKETS = [(-99, -11, "Down 11+"), (-10, -4, "Down 4-10"), (-3, 3, "Within 3"),
                  (4, 10, "Up 4-10"), (11, 99, "Up 11+")]
TIME_BUCKETS = [(1440, 9999, "1st half"), (720, 1439, "3rd quarter"),
                (300, 719, "4th: 12-5 min"), (0, 299, "Last 5 min / OT")]
OPP_LABELS = ["Weaker", "Average", "Stronger"]
METRICS = {
    "wpa_over_expected": "Win prob. over expected per stint",
    "wpa": "Raw win prob. added per stint",
}


def _bucket(values: pd.Series, buckets) -> pd.Series:
    out = pd.Series("", index=values.index, dtype=object)
    for lo, hi, label in buckets:
        out[(values >= lo) & (values <= hi)] = label
    return out


def margin_bucket(margin: pd.Series) -> pd.Series:
    return _bucket(margin, MARGIN_BUCKETS)


def bench_stints(stints: pd.DataFrame) -> pd.DataFrame:
    st = stints[~stints["is_starter"]].copy() if "is_starter" in stints else stints.copy()
    st["margin_bucket"] = margin_bucket(st["entry_margin"])
    secs = st["entry_seconds_remaining"].where(st["period"] <= config.REGULATION_PERIODS, 0)
    st["time_bucket"] = _bucket(secs, TIME_BUCKETS)
    st["high_leverage"] = st["entry_leverage"] >= config.HIGH_LEVERAGE_THRESHOLD
    if "player_start_rate" in st:
        st["true_reserve"] = st["player_start_rate"] <= config.RESERVE_MAX_START_RATE
    if "opp_lineup_net48" in st:
        st["opp_strength"] = pd.qcut(st["opp_lineup_net48"], 3, labels=OPP_LABELS).astype(str)
    return st


def rank_players(st: pd.DataFrame, metric: str = "wpa_over_expected",
                 k: float = config.SHRINKAGE_STINTS) -> pd.DataFrame:
    """Rank players on the given (already filtered) stints.

    score = sum(metric) / (n_stints + k): equivalent to adding k stints of zero value,
    i.e. shrinking toward a replacement-level prior so tiny samples can't top the list.
    """
    metric = metric if metric in st else "wpa"
    g = st.groupby("player")
    out = g.agg(team=("team", lambda s: "/".join(pd.unique(s))), stints=(metric, "size"),
                minutes=("duration_seconds", lambda s: s.sum() / 60), total=(metric, "sum"),
                raw_wpa=("wpa", "sum"), plus_minus=("plus_minus", "sum"),
                avg_entry_leverage=("entry_leverage", "mean")).reset_index()
    out["per_stint"] = out["total"] / out["stints"]
    out["score"] = out["total"] / (out["stints"] + k)
    # Rough 1-sigma band on the per-stint mean, shrunk the same way, for the chart's error bars.
    sd = g[metric].std().reindex(out["player"]).fillna(0).to_numpy()
    out["score_se"] = sd * np.sqrt(out["stints"]) / (out["stints"] + k)
    out["shrunk_wpa_per_stint"] = out["score"]  # backward-compatible name
    return out.sort_values("score", ascending=False).reset_index(drop=True)
